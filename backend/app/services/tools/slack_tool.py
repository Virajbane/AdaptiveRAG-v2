"""
Slack tool - Post messages to Slack channels using Slack Bot API

2026-08-09: Created to support Slack integration in RAG system

2026-08-22 STAGE 16+ FIX (root cause 6.5):
  Added search_messages() to support Slack message-search routing.
  The planner can now route "find/search Slack messages" questions to
  the concrete tool "slack_search" backed by this method.  post_message()
  and post_formatted_message() are preserved exactly as they were.

  Implementation notes:
  - search.messages requires a user token (xoxp-...) with search:read scope,
    not just a bot token.  If the configured token lacks this scope the method
    returns a structured error dict (never raises) so callers always get a
    clean response.
  - Result normalisation: each match is reduced to
    {text, channel, author, timestamp, permalink} -- a stable internal
    representation independent of Slack API version.
  - All known error codes (invalid_auth, missing_scope, ratelimited, etc.)
    are mapped to human-readable messages with an error_code field so the
    tool layer can surface them cleanly.
"""

from typing import Dict
from app.config.settings import settings


class SlackTool:
    """
    Slack bot tool.
    Posts messages to Slack channels and searches message history.
    """

    def __init__(self):
        self.bot_token = settings.SLACK_BOT_TOKEN
        self.client = None

    async def _init_client(self):
        """Initialize Slack client (lazy load)"""
        if self.client is None:
            if not self.bot_token:
                raise ValueError("Slack bot token not configured")
            
            from slack_sdk.web.async_client import AsyncWebClient
            self.client = AsyncWebClient(token=self.bot_token)

    async def post_message(self, channel: str, message: str) -> Dict:
        """
        Post a message to a Slack channel.

        Args:
            channel: Channel name (e.g., '#new-channel' or 'C1234567890')
            message: Message text to post

        Returns:
            {
                "success": bool,
                "channel": str,
                "ts": str (message timestamp),
                "message": str,
                "error": str (if failed)
            }
        """
        try:
            await self._init_client()

            # Ensure channel starts with # if it's a channel name
            if not channel.startswith('#') and not channel.startswith('C'):
                channel = f"#{channel}"

            # Post the message
            response = await self.client.chat_postMessage(
                channel=channel,
                text=message
            )

            if not response["ok"]:
                error_msg = response.get('error', 'Unknown error')
                return {"error": f"Slack API error: {error_msg}"}

            return {
                "success": True,
                "channel": response["channel"],
                "ts": response["ts"],
                "message": "Message posted successfully"
            }

        except Exception as e:
            return {"error": f"Slack post failed: {str(e)}"}

    async def post_formatted_message(
        self,
        channel: str,
        title: str,
        body: str,
        sections: list = None
    ) -> Dict:
        """
        Post a formatted message to Slack (using Slack blocks).

        Args:
            channel: Channel name
            title: Message title
            body: Message body
            sections: Optional list of {name: str, value: str} sections

        Returns:
            Same as post_message()
        """
        try:
            await self._init_client()

            if not channel.startswith('#') and not channel.startswith('C'):
                channel = f"#{channel}"

            # Build blocks for rich formatting
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": title
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": body
                    }
                }
            ]

            # Add sections if provided
            if sections:
                for section in sections:
                    blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*{section['name']}*\n{section['value']}"
                        }
                    })

            # Post the message
            response = await self.client.chat_postMessage(
                channel=channel,
                blocks=blocks
            )

            if not response["ok"]:
                error_msg = response.get('error', 'Unknown error')
                return {"error": f"Slack API error: {error_msg}"}

            return {
                "success": True,
                "channel": response["channel"],
                "ts": response["ts"],
                "message": "Formatted message posted successfully"
            }

        except Exception as e:
            return {"error": f"Slack post failed: {str(e)}"}

    async def search_messages(
        self,
        query: str,
        channel: str | None = None,
        user: str | None = None,
        oldest: str | None = None,
        latest: str | None = None,
        count: int = 10,
    ) -> Dict:
        """
        Search Slack messages using the Slack API search.messages endpoint.

        NOTE: search.messages requires a *user* token (xoxp-...) with the
        ``search:read`` scope, not just a bot token. If only a bot token is
        configured the API will return a ``missing_scope`` error and this
        method will return a structured error dict -- it never raises.

        Args:
            query:   Free-text search query (Slack search modifiers like
                     ``in:#channel`` and ``from:@user`` are also accepted
                     as part of the query string).
            channel: Optional channel name/ID to restrict the search to.
                     When supplied it is appended to `query` as
                     ``in:<channel>``.
            user:    Optional Slack user ID to filter by sender.
                     When supplied it is appended to `query` as
                     ``from:<user>``.
            oldest:  Optional Unix timestamp (string) -- include messages
                     sent after this time.
            latest:  Optional Unix timestamp (string) -- include messages
                     sent before this time.
            count:   Maximum number of results to return (default 10,
                     capped at 100 by the Slack API).

        Returns on success::

            {
                "success": True,
                "query": str,          # effective query sent to Slack
                "total": int,          # total matching messages (Slack's count)
                "messages": [
                    {
                        "text":      str,
                        "channel":   str,   # channel name or ID
                        "author":    str,   # Slack user ID of sender
                        "timestamp": str,   # Slack message ts (Unix float as str)
                        "permalink": str,   # direct link to message
                    },
                    ...
                ]
            }

        Returns on any error::

            {
                "error": str,      # human-readable description
                "error_code": str  # Slack API error code if available
            }
        """
        # ── Guard: token configured? ──────────────────────────────────────
        if not self.bot_token:
            return {
                "error": (
                    "Slack token not configured. Set SLACK_BOT_TOKEN in "
                    "your environment. Note: search.messages requires a "
                    "user token (xoxp-...) with search:read scope."
                ),
                "error_code": "no_token",
            }

        # ── Build effective query ─────────────────────────────────────────
        effective_query = query.strip()
        if channel:
            ch = channel.lstrip("#")
            effective_query += f" in:#{ch}"
        if user:
            effective_query += f" from:{user}"

        # ── Execute search ────────────────────────────────────────────────
        try:
            await self._init_client()

            kwargs: dict = {
                "query": effective_query,
                "count": min(max(1, count), 100),
                "sort": "timestamp",
                "sort_dir": "desc",
                "highlight": False,
            }
            if oldest:
                kwargs["oldest"] = oldest
            if latest:
                kwargs["latest"] = latest

            response = await self.client.search_messages(**kwargs)

        except Exception as exc:
            err = str(exc)
            # Map common authentication / permission errors to clear messages.
            if "invalid_auth" in err or "token_revoked" in err:
                return {
                    "error": "Slack authentication failed. Check SLACK_BOT_TOKEN.",
                    "error_code": "invalid_auth",
                }
            if "missing_scope" in err:
                return {
                    "error": (
                        "Slack token is missing the search:read scope. "
                        "search.messages requires a user token (xoxp-...) "
                        "with search:read enabled in your Slack App settings."
                    ),
                    "error_code": "missing_scope",
                }
            if "ratelimited" in err:
                return {
                    "error": "Slack API rate limit hit. Retry after a moment.",
                    "error_code": "ratelimited",
                }
            return {
                "error": f"Slack search request failed: {err}",
                "error_code": "request_error",
            }

        # ── Check API-level ok flag ───────────────────────────────────────
        if not response.get("ok"):
            error_code = response.get("error", "unknown_error")
            _known_errors = {
                "invalid_auth": "Slack authentication failed. Check SLACK_BOT_TOKEN.",
                "token_revoked": "Slack token has been revoked.",
                "missing_scope": (
                    "Slack token is missing the search:read scope. "
                    "search.messages requires a user token (xoxp-...) "
                    "with search:read enabled."
                ),
                "not_authed": "Slack token not provided or invalid.",
                "ratelimited": "Slack API rate limit hit. Retry after a moment.",
            }
            return {
                "error": _known_errors.get(
                    error_code, f"Slack API error: {error_code}"
                ),
                "error_code": error_code,
            }

        # ── Normalise results ─────────────────────────────────────────────
        raw_messages = response.get("messages", {})
        matches = raw_messages.get("matches", [])
        total = raw_messages.get("total", len(matches))

        if not matches:
            return {
                "success": True,
                "query": effective_query,
                "total": 0,
                "messages": [],
            }

        normalised = []
        for m in matches:
            # channel may be a dict (with name/id keys) or a bare string
            ch_raw = m.get("channel", {})
            if isinstance(ch_raw, dict):
                channel_name = ch_raw.get("name") or ch_raw.get("id", "unknown")
            else:
                channel_name = str(ch_raw) if ch_raw else "unknown"

            normalised.append(
                {
                    "text": m.get("text", ""),
                    "channel": channel_name,
                    "author": m.get("user", m.get("username", "unknown")),
                    "timestamp": m.get("ts", ""),
                    "permalink": m.get("permalink", ""),
                }
            )

        return {
            "success": True,
            "query": effective_query,
            "total": total,
            "messages": normalised,
        }


# Global Slack tool instance
slack_tool = SlackTool()
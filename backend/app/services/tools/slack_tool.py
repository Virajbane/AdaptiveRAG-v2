"""
Slack tool - Post messages to Slack channels using Slack Bot API

2026-08-09: Created to support Slack integration in RAG system
"""

from typing import Dict
from app.config.settings import settings


class SlackTool:
    """
    Slack bot tool.
    Posts messages to Slack channels.
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


# Global Slack tool instance
slack_tool = SlackTool()
"""
Email tool - Send emails via Gmail SMTP

2026-08-09: Created to support email integration in RAG system
"""

from typing import Dict
from app.config.settings import settings


class EmailTool:
    """
    Email sending tool.
    Sends emails via Gmail SMTP.
    """

    def __init__(self):
        self.smtp_host = settings.SMTP_HOST
        self.smtp_port = settings.SMTP_PORT
        self.smtp_user = settings.SMTP_USER
        self.smtp_password = settings.SMTP_PASSWORD

    async def send_email(
        self,
        subject: str,
        body: str,
        to_email: str,
        html_body: str = None
    ) -> Dict:
        """
        Send an email via Gmail SMTP.

        Args:
            subject: Email subject line
            body: Email body (plain text)
            to_email: Recipient email address
            html_body: Optional HTML version of body

        Returns:
            {
                "success": bool,
                "message": str,
                "error": str (if failed)
            }
        """
        try:
            if not self.smtp_password or not self.smtp_user:
                return {"error": "SMTP credentials not configured"}

            import aiosmtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            # Create email message
            msg = MIMEMultipart("alternative")
            msg["From"] = self.smtp_user
            msg["To"] = to_email
            msg["Subject"] = subject

            # Attach plain text version
            msg.attach(MIMEText(body, "plain"))

            # Attach HTML version if provided
            if html_body:
                msg.attach(MIMEText(html_body, "html"))

            # Send via SMTP
            async with aiosmtplib.SMTP(
                hostname=self.smtp_host,
                port=self.smtp_port
            ) as smtp:
                await smtp.starttls()
                await smtp.login(self.smtp_user, self.smtp_password)
                await smtp.send_message(msg)

            return {
                "success": True,
                "message": f"Email sent successfully to {to_email}"
            }

        except Exception as e:
            return {"error": f"Email send failed: {str(e)}"}

    async def send_html_email(
        self,
        subject: str,
        html_body: str,
        to_email: str,
        text_fallback: str = None
    ) -> Dict:
        """
        Send an HTML email via Gmail SMTP.

        Args:
            subject: Email subject line
            html_body: Email body in HTML format
            to_email: Recipient email address
            text_fallback: Plain text fallback (optional)

        Returns:
            Same as send_email()
        """
        text_body = text_fallback or html_body  # Fallback to HTML if no text provided
        return await self.send_email(
            subject=subject,
            body=text_body,
            to_email=to_email,
            html_body=html_body
        )

    async def send_batch_email(
        self,
        subject: str,
        body: str,
        to_emails: list,
        html_body: str = None
    ) -> Dict:
        """
        Send the same email to multiple recipients.

        Args:
            subject: Email subject line
            body: Email body (plain text)
            to_emails: List of recipient email addresses
            html_body: Optional HTML version of body

        Returns:
            {
                "success": bool,
                "sent_count": int,
                "failed_count": int,
                "message": str,
                "errors": list
            }
        """
        import aiosmtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        sent_count = 0
        failed_count = 0
        errors = []

        try:
            if not self.smtp_password or not self.smtp_user:
                return {"error": "SMTP credentials not configured"}

            # Connect once for all emails
            async with aiosmtplib.SMTP(
                hostname=self.smtp_host,
                port=self.smtp_port
            ) as smtp:
                await smtp.starttls()
                await smtp.login(self.smtp_user, self.smtp_password)

                for to_email in to_emails:
                    try:
                        # Create email message
                        msg = MIMEMultipart("alternative")
                        msg["From"] = self.smtp_user
                        msg["To"] = to_email
                        msg["Subject"] = subject

                        # Attach plain text version
                        msg.attach(MIMEText(body, "plain"))

                        # Attach HTML version if provided
                        if html_body:
                            msg.attach(MIMEText(html_body, "html"))

                        # Send
                        await smtp.send_message(msg)
                        sent_count += 1

                    except Exception as e:
                        failed_count += 1
                        errors.append(f"{to_email}: {str(e)}")

            return {
                "success": failed_count == 0,
                "sent_count": sent_count,
                "failed_count": failed_count,
                "message": f"Sent {sent_count}/{len(to_emails)} emails successfully",
                "errors": errors if errors else None
            }

        except Exception as e:
            return {"error": f"Batch email send failed: {str(e)}"}


# Global email tool instance
email_tool = EmailTool()
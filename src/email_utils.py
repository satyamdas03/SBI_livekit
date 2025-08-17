# src/email_utils.py
import os
import smtplib
import asyncio
import contextlib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587  # TLS

def _send_blocking(
    gmail_user: str,
    gmail_app_password: str,
    to_email: str,
    subject: str,
    message: str,
    cc_email: Optional[str] = None,
) -> str:
    msg = MIMEMultipart()
    msg["From"] = gmail_user
    msg["To"] = to_email
    msg["Subject"] = subject[:180] if subject else "(no subject)"
    recipients = [to_email]
    if cc_email:
        msg["Cc"] = cc_email
        recipients.append(cc_email)

    msg.attach(MIMEText(message or "", "plain"))

    server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30)
    try:
        # STARTTLS upgrades the plaintext connection to TLS on port 587.
        server.starttls()
        server.login(gmail_user, gmail_app_password)
        server.sendmail(gmail_user, recipients, msg.as_string())
    finally:
        # Quit cleanly; ignore any error during shutdown.
        with contextlib.suppress(Exception):
            server.quit()
    return f"Email sent to {to_email}"

async def send_email_async(
    to_email: str,
    subject: str,
    message: str,
    cc_email: Optional[str] = None,
) -> str:
    gmail_user = os.getenv("GMAIL_USER")
    gmail_app_password = os.getenv("GMAIL_APP_PASSWORD")
    if not gmail_user or not gmail_app_password:
        return "Email sending failed: GMAIL_USER or GMAIL_APP_PASSWORD not configured."

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        _send_blocking,
        gmail_user,
        gmail_app_password,
        to_email,
        subject,
        message,
        cc_email,
    )

# api/app/services/email.py
import os
from resend import Emails

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
FROM_EMAIL = os.getenv("FROM_EMAIL", "no-reply@charteredsportsbetting.com")

if not RESEND_API_KEY:
    # You can swap this for a logger.warning if you prefer
    raise RuntimeError("RESEND_API_KEY not set in environment")

Emails.api_key = RESEND_API_KEY


def send_email(to: str, subject: str, html: str):
    """
    Thin wrapper around Resend's Emails.send.

    Usage:
        send_email("user@example.com", "Welcome", "<h1>Hi</h1>")
    """
    return Emails.send(
        {
            "from": f"CSB ✅ <{FROM_EMAIL}>",
            "to": [to],
            "subject": subject,
            "html": html,
        }
    )
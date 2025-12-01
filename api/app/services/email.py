# api/app/services/email.py
import os
from resend import Emails

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
FROM_EMAIL = os.getenv("FROM_EMAIL", "no-reply@charteredsportsbetting.com")

if RESEND_API_KEY:
    Emails.api_key = RESEND_API_KEY
else:
    print("⚠️ WARNING: RESEND_API_KEY not set — emails will NOT send.")


def send_email(to: str, subject: str, html: str):
    """
    Thin wrapper around Resend's Emails.send.
    """
    if not RESEND_API_KEY:
        print("❌ Email NOT sent — RESEND_API_KEY missing")
        return {"error": "emails_disabled"}

    return Emails.send(
        {
            "from": f"CSB <{FROM_EMAIL}>",
            "to": [to],
            "subject": subject,
            "html": html,
        }
    )
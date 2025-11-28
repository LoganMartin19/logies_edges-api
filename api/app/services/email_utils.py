# api/app/services/email_utils.py
import os
import smtplib
from email.message import EmailMessage
from typing import Optional

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("SMTP_FROM", "no-reply@charteredsportsbetting.com")

def _send_raw_email(
    to_email: str,
    subject: str,
    text_body: str,
    html_body: Optional[str] = None,
) -> None:
    """
    Very simple SMTP-based sender.
    If SMTP_* env vars are not set, this becomes a no-op (just prints).
    Plug in SendGrid / Resend here later if you prefer.
    """
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS):
        # Fallback: just log for now so we don't crash
        print(f"[email_utils] Would send email to {to_email}: {subject}")
        print(text_body)
        return

    msg = EmailMessage()
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(text_body)

    if html_body:
        msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)


def send_welcome_email(email: str) -> None:
    """
    One-time welcome email for new users.
    """
    subject = "Welcome to Chartered Sports Betting (CSB) 🎯"
    text = f"""
Hi,

Thanks for signing up to Chartered Sports Betting (CSB).

You can now:
  • Explore the dashboard and today’s edges
  • Follow tipsters and track their ROI
  • Apply for your own verified tipster profile

Remember: CSB is an analytics platform, not a bookie. Always gamble responsibly.

Cheers,
CSB
"""

    html = f"""
<p>Hi,</p>

<p>Thanks for signing up to <strong>Chartered Sports Betting (CSB)</strong>.</p>

<p>You can now:</p>
<ul>
  <li>Explore the dashboard and today’s edges</li>
  <li>Follow tipsters and track their ROI</li>
  <li>Apply for your own verified tipster profile</li>
</ul>

<p>Remember: CSB is an analytics platform, not a bookie. Always gamble responsibly.</p>

<p>Cheers,<br />CSB</p>
"""

    _send_raw_email(email, subject, text, html)


def send_new_pick_email(
    *,
    to_email: str,
    tipster_name: str,
    tipster_username: str,
    market: str,
    price: float,
    fixture_label: str,
    fixture_path: str | None,
) -> None:
    """
    Notify a follower when a tipster they follow posts a new pick.
    """
    subject = f"New pick from {tipster_name} (@{tipster_username})"

    line = f"{fixture_label} – {market} @ {price:.2f}"
    link_part = (
        f"\nView game: https://logies-edges-site.vercel.app{fixture_path}"
        if fixture_path
        else ""
    )

    text = f"""
New pick from {tipster_name} (@{tipster_username}):

{line}{link_part}

You’re receiving this because you follow this tipster on CSB.
To unfollow, open their profile and tap “Unfollow”.

Remember: always stake responsibly.
"""

    html_link = (
        f'<p><a href="https://logies-edges-site.vercel.app{fixture_path}">'
        f"View game on CSB</a></p>"
        if fixture_path
        else ""
    )

    html = f"""
<p>New pick from <strong>{tipster_name}</strong> (@{tipster_username}):</p>

<p><strong>{fixture_label}</strong><br />
{market} @ {price:.2f}</p>

{html_link}

<p style="font-size: 0.9em; color: #555;">
You’re receiving this because you follow this tipster on CSB.
To unfollow, open their profile and tap “Unfollow”.<br /><br />
Always stake responsibly.
</p>
"""

    _send_raw_email(to_email, subject, text, html)
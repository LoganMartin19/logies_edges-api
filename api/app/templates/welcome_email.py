# api/app/templates/welcome_email.py
from datetime import datetime
import html


def welcome_email_html(name: str) -> str:
    safe_name = html.escape(name or "there")
    year = datetime.utcnow().year

    return f"""
<!DOCTYPE html>
<html>
  <body style="margin:0;padding:0;background:#020817;">
    <div style="font-family: Inter, Arial, sans-serif; padding:24px; background:#020817; color:#ffffff;">
      <div style="max-width:560px; margin:0 auto; background:#122818; padding:32px; border-radius:16px; border:1px solid #1f4d31;">

        <div style="text-align:center; margin-bottom:24px;">
          <img src="https://charteredsportsbetting.com/logo.png"
               alt="Chartered Sports Betting"
               width="72" height="72"
               style="border-radius:12px; display:block; margin:0 auto;" />
          <h2 style="margin-top:12px; font-size:24px; font-weight:600; color:#6ee7b7; margin-bottom:0;">
            Welcome to Chartered Sports Betting
          </h2>
        </div>

        <p style="font-size:16px; line-height:1.6; margin-top:20px;">
          Hi <strong>{safe_name}</strong>,<br/><br/>
          Thanks for joining <strong>CSB</strong> – the home of sharp value betting,
          expert picks, and real data-driven insight.
        </p>

        <p style="font-size:16px; line-height:1.6;">
          You now have access to your personalised dashboard, fixtures,
          shortlist edges, and tipster insights.
        </p>

        <p style="text-align:center; margin:24px 0 16px;">
          <a href="https://charteredsportsbetting.com"
             style="display:inline-block; padding:12px 20px;
                    background:#22c55e; color:#000000; text-decoration:none;
                    border-radius:8px; font-weight:600;">
            Go to your dashboard →
          </a>
        </p>

        <p style="margin-top:12px; font-size:14px; opacity:0.8;">
          You'll receive updates on featured picks, shortlist edges,
          and important account notices from this email address.
        </p>

        <hr style="border:0; border-top:1px solid rgba(255,255,255,0.08); margin:24px 0;" />

        <p style="font-size:12px; opacity:0.6; text-align:center; line-height:1.5;">
          Chartered Sports Betting • © {year}<br/>
          You’re receiving this because you created an account at charteredsportsbetting.com.
        </p>
      </div>
    </div>
  </body>
</html>
"""
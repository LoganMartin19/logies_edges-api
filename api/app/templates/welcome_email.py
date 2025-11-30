# api/app/templates/welcome_email.py

def welcome_email_html(name: str) -> str:
    safe_name = name or "there"

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
               style="border-radius:12px;" />
          <h2 style="margin-top:12px; font-size:24px; font-weight:600; color:#6ee7b7;">
            Welcome to Chartered Sports Betting
          </h2>
        </div>

        <p style="font-size:16px; line-height:1.6;">
          Hi <strong>{safe_name}</strong>,<br/><br/>
          Thanks for joining <strong>CSB</strong> – the home of sharp value betting,
          expert picks, and real data-driven insight.
        </p>

        <p style="font-size:16px; line-height:1.6;">
          You now have access to your personalised dashboard, fixtures,
          shortlist edges, and tipster insights.
        </p>

        <p style="text-align:center; margin:24px 0;">
          <a href="https://charteredsportsbetting.com"
             style="display:inline-block; margin-top:20px; padding:12px 20px;
                    background:#22c55e; color:#000000; text-decoration:none;
                    border-radius:8px; font-weight:600;">
            Go to your dashboard →
          </a>
        </p>

        <p style="margin-top:28px; font-size:14px; opacity:0.8;">
          You'll receive updates on featured picks, shortlist edges,
          and important account notices from this email address.
        </p>

        <hr style="border:0; border-top:1px solid rgba(255,255,255,0.08); margin:24px 0;" />

        <p style="font-size:13px; opacity:0.6; text-align:center;">
          Chartered Sports Betting • © {2025}
        </p>
      </div>
    </div>
  </body>
</html>
"""
# api/app/templates/featured_picks_email.py
from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Mapping


def _fmt_day(d: date) -> str:
    return d.strftime("%A %d %B %Y")


def _fmt_time(dt: datetime | None) -> str:
    if not dt:
        return ""
    # keep it simple – UTC label, consistent with site
    return dt.strftime("%H:%M UTC")


def featured_picks_email_html(
    day: date,
    picks: Iterable[Mapping],
    *,
    recipient_name: str | None = None,
    is_premium_user: bool = False,
    free_count: int = 0,
    premium_count: int = 0,
    unsubscribe_url: str = "https://charteredsportsbetting.com/account",
) -> str:
    """
    Build a HTML digest for a set of Featured Picks, personalised per user.

    `picks` is an iterable of dict-like objects with keys:
      comp, home_team, away_team, kickoff_utc, market, bookmaker, price, edge

    - `is_premium_user`  : whether this recipient is a paying CSB Premium user
    - `free_count`       : how many free (non-premium) picks exist for this card
    - `premium_count`    : how many premium-only picks exist for this card
    - `unsubscribe_url`  : where users can manage / turn off email picks
    """

    safe_name = recipient_name or "there"
    day_str = _fmt_day(day)

    # ---------- Heading + intro copy ----------
    if is_premium_user:
        # Premium members see everything; tweak title based on mix
        if free_count and premium_count:
            title = "Today's Featured & Premium Picks"
            intro_line = (
                "Here’s today’s full card – featured & premium – "
                "from <strong>Chartered Sports Betting</strong>."
            )
        elif premium_count:
            title = "Today's Premium Picks"
            intro_line = (
                "Here’s your premium edge card from "
                "<strong>Chartered Sports Betting</strong>."
            )
        else:
            title = "Today's Featured Picks"
            intro_line = (
                "Here are today’s featured best edges from "
                "<strong>Chartered Sports Betting</strong>."
            )
        teaser_line = ""
    else:
        # Free user – only sees free picks but we can tease premium count
        title = "Today's Free Featured Picks"
        if premium_count:
            intro_line = (
                "Here are today’s free featured picks from "
                "<strong>Chartered Sports Betting</strong>."
            )
            teaser_line = (
                f"There are also <strong>{premium_count}</strong> "
                "extra <strong>premium picks</strong> live on the CSB dashboard."
            )
        else:
            intro_line = (
                "Here are today’s featured picks from "
                "<strong>Chartered Sports Betting</strong>."
            )
            teaser_line = ""

    # ---------- Table rows ----------
    rows_html = ""
    for p in picks:
        comp = p.get("comp") or ""
        home = p.get("home_team") or ""
        away = p.get("away_team") or ""
        ko = _fmt_time(p.get("kickoff_utc"))
        market = p.get("market") or ""
        book = (p.get("bookmaker") or "").upper()
        price = p.get("price")
        edge = p.get("edge")

        price_str = f"{float(price):.2f}" if price is not None else "—"
        edge_str = f"{float(edge) * 100:.1f}%" if edge is not None else "—"

        rows_html += f"""
          <tr>
            <td style="padding:8px 6px;font-size:13px;white-space:nowrap;">{ko}</td>
            <td style="padding:8px 6px;font-size:13px;">
              <strong>{home}</strong> vs <strong>{away}</strong><br/>
              <span style="opacity:0.8;font-size:12px;">{comp}</span>
            </td>
            <td style="padding:8px 6px;font-size:13px;white-space:nowrap;">
              {market}<br/>
              <span style="opacity:0.8;font-size:12px;">{book}</span>
            </td>
            <td style="padding:8px 6px;font-size:13px;white-space:nowrap;text-align:right;">
              {price_str}
            </td>
            <td style="padding:8px 6px;font-size:13px;white-space:nowrap;text-align:right;">
              {edge_str}
            </td>
          </tr>
        """

    if not rows_html:
        rows_html = """
          <tr>
            <td colspan="5" style="padding:16px 8px;font-size:14px;opacity:0.8;text-align:center;">
              No featured picks were found for this day.
            </td>
          </tr>
        """

    teaser_block = ""
    if teaser_line:
        teaser_block = f"""
        <p style="margin-top:8px;font-size:13px;line-height:1.5;opacity:0.9;">
          {teaser_line}
        </p>
        """

    return f"""
<!DOCTYPE html>
<html>
  <body style="margin:0;padding:0;background:#020817;">
    <div style="font-family:Inter,Arial,sans-serif;padding:24px;background:#020817;color:#ffffff;">
      <div style="max-width:560px;margin:0 auto;background:#122818;padding:32px;border-radius:16px;border:1px solid #1f4d31;">

        <div style="text-align:center;margin-bottom:24px;">
          <img src="https://charteredsportsbetting.com/logo.png"
               alt="Chartered Sports Betting"
               width="72" height="72"
               style="border-radius:12px;" />
          <h2 style="margin-top:12px;font-size:22px;font-weight:600;color:#6ee7b7;">
            {title}
          </h2>
          <div style="font-size:13px;opacity:0.8;">{day_str}</div>
        </div>

        <p style="font-size:15px;line-height:1.6;">
          Hi <strong>{safe_name}</strong>,<br/><br/>
          {intro_line}
        </p>

        {teaser_block}

        <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;margin-top:12px;background:#020817;border-radius:12px;overflow:hidden;">
          <thead>
            <tr style="background:#052e16;">
              <th align="left" style="padding:8px 6px;font-size:12px;text-transform:uppercase;letter-spacing:0.04em;">KO</th>
              <th align="left" style="padding:8px 6px;font-size:12px;text-transform:uppercase;letter-spacing:0.04em;">Match</th>
              <th align="left" style="padding:8px 6px;font-size:12px;text-transform:uppercase;letter-spacing:0.04em;">Market</th>
              <th align="right" style="padding:8px 6px;font-size:12px;text-transform:uppercase;letter-spacing:0.04em;">Price</th>
              <th align="right" style="padding:8px 6px;font-size:12px;text-transform:uppercase;letter-spacing:0.04em;">Edge</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>

        <p style="margin-top:24px;font-size:14px;line-height:1.6;opacity:0.9;">
          You can view live odds, model probabilities, and bet tracking for these games
          on the CSB dashboard.
        </p>

        <p style="text-align:center;margin-top:12px;">
          <a href="https://charteredsportsbetting.com"
             style="display:inline-block;padding:10px 18px;background:#22c55e;color:#000000;
                    text-decoration:none;border-radius:999px;font-weight:600;font-size:14px;">
            Open CSB dashboard →
          </a>
        </p>

        <hr style="border:0;border-top:1px solid rgba(255,255,255,0.08);margin:24px 0;" />

        <p style="font-size:12px;opacity:0.6;text-align:center;">
          Chartered Sports Betting • © 2025<br/>
          You’re receiving this because you have a CSB account.
        </p>
        <p style="font-size:11px;opacity:0.6;text-align:center;margin-top:4px;">
          Don’t want these emails? You can{" "}
          <a href="{unsubscribe_url}" style="color:#6ee7b7;text-decoration:underline;">
            update your email preferences here
          </a>.
        </p>
      </div>
    </div>
  </body>
</html>
"""
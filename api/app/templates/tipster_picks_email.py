from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Mapping


def _fmt_day(d: date) -> str:
    return d.strftime("%A %d %B %Y")


def _fmt_time(dt: datetime | None) -> str:
    if not dt:
        return ""
    return dt.strftime("%H:%M UTC")


def tipster_picks_email_html(
    day: date,
    tipster_name: str,
    picks: Iterable[Mapping],
    *,
    recipient_name: str | None = None,
    tipster_handle: str | None = None,
    tipster_bio: str | None = None,
    roi_30d: float | None = None,
    record_30d: str | None = None,       # e.g. "23W-12L-1P"
    unsubscribe_url: str = "https://charteredsportsbetting.com/account",

    # ⭐ NEW: gating-aware teaser info
    is_premium_user: bool = False,
    is_tipster_subscriber: bool = False,
    premium_only_total: int = 0,
    subscriber_only_total: int = 0,
) -> str:
    """
    Email sent when a tipster posts today's picks and clicks
    'Send Picks Email' on their dashboard.

    Works for both FOLLOWERS and SUBSCRIBERS.

    We also show a small teaser if there are premium/sub-only picks
    that the recipient is not seeing in this email.
    """

    safe_name = recipient_name or "there"
    picks = list(picks)
    day_str = _fmt_day(day)

    handle_str = f" (@{tipster_handle})" if tipster_handle else ""
    title = f"Picks from {tipster_name}{handle_str}"

    # 🎯 Deep link to tipster page if handle is known
    if tipster_handle:
        cta_href = f"https://charteredsportsbetting.com/tipsters/{tipster_handle}"
    else:
        cta_href = "https://charteredsportsbetting.com"

    stat_line = ""
    if roi_30d is not None or record_30d:
        bits = []
        if roi_30d is not None:
            bits.append(f"{roi_30d:+.1f}% ROI (30d)")
        if record_30d:
            bits.append(record_30d)
        stat_line = " • ".join(bits)

    bio_block = (
        f'<p style="font-size:13px;opacity:0.85;margin-top:6px;">{tipster_bio}</p>'
        if tipster_bio
        else ""
    )

    # ---------- Teaser about hidden premium / sub-only picks ----------
    extra_line = ""

    # Plain follower (no premium, no sub)
    if not is_premium_user and not is_tipster_subscriber:
        if premium_only_total or subscriber_only_total:
            parts: list[str] = []
            if premium_only_total:
                parts.append(
                    f"<strong>{premium_only_total}</strong> premium-only pick"
                    f"{'s' if premium_only_total != 1 else ''}"
                )
            if subscriber_only_total:
                parts.append(
                    f"<strong>{subscriber_only_total}</strong> subscriber-only pick"
                    f"{'s' if subscriber_only_total != 1 else ''}"
                )
            joined = " and ".join(parts) if len(parts) == 2 else parts[0]
            extra_line = (
                f"This tipster also has {joined} live on CSB "
                "for paying members."
            )

    # CSB Premium, but not a subscriber of this tipster
    elif is_premium_user and not is_tipster_subscriber and subscriber_only_total:
        extra_line = (
            "Your CSB Premium unlocks all premium picks. "
            f"This tipster also has <strong>{subscriber_only_total}</strong> "
            f"subscriber-only pick{'s' if subscriber_only_total != 1 else ''} "
            "for their paid subs."
        )

    # Subscriber of this tipster, but not CSB Premium
    elif is_tipster_subscriber and not is_premium_user and premium_only_total:
        extra_line = (
            "Your subscription unlocks all of this tipster’s subscriber-only picks. "
            f"There are also <strong>{premium_only_total}</strong> "
            f"CSB Premium-only pick{'s' if premium_only_total != 1 else ''} today "
            "for site-wide premium members."
        )

    # Premium + subscriber → they already see the full card → no teaser

    extra_block = ""
    if extra_line:
        extra_block = f"""
        <p style="font-size:13px;line-height:1.5;opacity:0.9;margin-top:8px;">
          {extra_line}
        </p>
        """

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
        stake = p.get("stake_units") or p.get("stake") or None
        note = p.get("note") or ""

        price_str = f"{float(price):.2f}" if price is not None else "—"
        stake_str = f"{stake:.1f}u" if stake is not None else "—"

        note_block = (
            f'<div style="font-size:11px;opacity:0.7;margin-top:2px;">{note}</div>'
            if note
            else ""
        )

        rows_html += f"""
        <tr>
          <td style="padding:8px 6px;font-size:13px;white-space:nowrap;">{ko}</td>
          <td style="padding:8px 6px;font-size:13px;">
            <strong>{home}</strong> vs <strong>{away}</strong><br/>
            <span style="opacity:0.8;font-size:12px;">{comp}</span>
          </td>
          <td style="padding:8px 6px;font-size:13px;">
            {market}<br/>
            <span style="opacity:0.7;font-size:12px;">{book}</span>
            {note_block}
          </td>
          <td style="padding:8px 6px;font-size:13px;text-align:right;">
            {price_str}<br/>
            <span style="opacity:0.8;font-size:12px;">{stake_str}</span>
          </td>
        </tr>
        """

    if not rows_html:
        rows_html = """
        <tr>
          <td colspan="4" style="padding:16px;text-align:center;opacity:0.8;">
            No picks were submitted for today.
          </td>
        </tr>
        """

    return f"""
<!DOCTYPE html>
<html>
  <body style="margin:0;padding:0;background:#020817;">
    <div style="font-family:Inter,Arial,sans-serif;padding:24px;background:#020817;color:#ffffff;">
      <div style="max-width:560px;margin:0 auto;background:#122818;
                  padding:32px;border-radius:16px;border:1px solid #1f4d31;">

        <div style="text-align:center;margin-bottom:24px;">
          <img src="https://charteredsportsbetting.com/logo.png"
               width="72" height="72" alt="CSB"
               style="border-radius:12px;" />
          <h2 style="margin-top:12px;font-size:22px;font-weight:600;color:#6ee7b7;">
            {title}
          </h2>
          <div style="font-size:13px;opacity:0.8;">{day_str}</div>
          {bio_block}
          {f'<div style="font-size:12px;opacity:0.8;margin-top:6px;">{stat_line}</div>' if stat_line else ""}
        </div>

        <p style="font-size:15px;line-height:1.6;">
          Hi <strong>{safe_name}</strong>,<br/><br/>
          Here are today's picks from <strong>{tipster_name}{handle_str}</strong>.
        </p>

        {extra_block}

        <table width="100%" cellspacing="0" cellpadding="0"
               style="border-collapse:collapse;margin-top:12px;background:#020817;
                      border-radius:12px;overflow:hidden;">
          <thead>
            <tr style="background:#052e16;">
              <th align="left"  style="padding:8px 6px;font-size:12px;">KO</th>
              <th align="left"  style="padding:8px 6px;font-size:12px;">Match</th>
              <th align="left"  style="padding:8px 6px;font-size:12px;">Market</th>
              <th align="right" style="padding:8px 6px;font-size:12px;">Price / Stake</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>

        <p style="margin-top:24px;font-size:14px;opacity:0.9;">
          You can view more details and track results on the CSB dashboard.
        </p>

        <p style="text-align:center;margin-top:12px;">
          <a href="{cta_href}"
             style="padding:10px 18px;background:#22c55e;color:#000;
                    text-decoration:none;border-radius:999px;font-weight:600;">
            View Tipster →
          </a>
        </p>

        <hr style="border:0;border-top:1px solid rgba(255,255,255,0.08);margin:24px 0;" />

        <p style="font-size:12px;opacity:0.6;text-align:center;">
          You're receiving this because you follow or subscribe to {tipster_name}.
        </p>
        <p style="font-size:11px;opacity:0.6;text-align:center;margin-top:4px;">
          Don’t want these emails? You can
          <a href="{unsubscribe_url}" style="color:#6ee7b7;text-decoration:underline;">
            update your preferences here
          </a>.
        </p>

      </div>
    </div>
  </body>
</html>
"""
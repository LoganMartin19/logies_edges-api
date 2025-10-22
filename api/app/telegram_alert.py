# api/app/telegram_alert.py
from __future__ import annotations
import html
import requests
from api.app.settings import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

def _b(x: str) -> str:
    return f"<b>{html.escape(str(x))}</b>"

def _i(x: str) -> str:
    return f"<i>{html.escape(str(x))}</i>"

def _t(x: str) -> str:
    return html.escape(str(x))

def send_telegram_alert(
    match: str,
    market: str,
    odds: float,
    edge: float,            # already in %
    kickoff: str,           # preformatted, e.g. "Sat 06 Sep, 14:00"
    league: str,
    bookmaker: str,
    model_source: str,
    link: str | None = None,
    bet_id: str | None = None,
):
    # Build a robust, scannable HTML message (all dynamic parts escaped)
    parts = [
        _b("📢 Value Edge Alert!"),
        f"🏆 { _i(league) }",
        f"⚽ {_b(match)}",
        f"🕒 {_t(kickoff)} UTC",
        "",
        f"🎯 Market: {_b(market)}",
        f"🏦 Bookmaker: {_t(bookmaker)}",
        f"💰 Odds: {_b(f'{float(odds):.2f}')}",
        f"📊 Edge: {_b(f'+{edge:.1f}%')}",
        f"🧠 Model: {_t(model_source)}",
    ]
    if link:
        # link is not escaped inside href; visible text is escaped
        parts.append(f"\n🔗 <a href=\"{link}\">{_t('View odds')}</a>")
    if bet_id:
        parts.append(f"\n🔎 {_i('Ref')}: {_t(bet_id)}")

    message = "\n".join(parts)

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",         # HTML is more robust than Markdown
        "disable_web_page_preview": True,
    }
    resp = requests.post(url, data=payload, timeout=10)
    if resp.status_code != 200:
        print(f"[Telegram] ❌ Error: {resp.status_code} - {resp.text}")
    else:
        print(f"[Telegram] ✅ Alert sent: {match}")
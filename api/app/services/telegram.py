import requests
import os

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_alert_message(alert: dict):
    message = (
        f"📢 {alert['home_team']} vs {alert['away_team']} | "
        f"{alert['market']} @ {alert['bookmaker']} {alert['price']} "
        f"({alert['edge']*100:.1f}%)"
    )

    if not BOT_TOKEN or not CHAT_ID:
        return {"ok": False, "error": "Missing TELEGRAM_BOT_TOKEN or CHAT_ID"}

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    resp = requests.post(url, json={"chat_id": CHAT_ID, "text": message})
    
    if resp.status_code == 200:
        return {"ok": True}
    else:
        return {"ok": False, "error": resp.text}
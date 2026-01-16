import os
import re
import requests
import unicodedata

from api.app.services.apifootball import LEAGUE_MAP  # ✅ absolute import

LOGO_DIR = "frontend/public/logos"
os.makedirs(LOGO_DIR, exist_ok=True)

API_HOST = "v3.football.api-sports.io"
API_KEY = "ac315f4445f13022c2d3e2c8e4bbe604"  # your key

# ✅ Direct API-Football (NOT RapidAPI)
headers = {
    "x-apisports-key": API_KEY,
}

def slugify(name: str) -> str:
    """Convert team name into a safe slug for filenames."""
    slug = (
        unicodedata.normalize("NFKD", name)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", slug)
    return slug.strip("_").lower()

def fetch_teams_for_league(league_id: int, season: int = 2024):
    url = f"https://{API_HOST}/teams?league={league_id}&season={season}"
    print(f"📡 Fetching: {url}")
    res = requests.get(url, headers=headers)
    print(f"➡️ Status: {res.status_code}")
    if res.status_code != 200:
        print("Body:", res.text[:400])
        res.raise_for_status()
    data = res.json()
    return data.get("response", [])

def save_logo(team_name: str, logo_url: str):
    slug = slugify(team_name)
    path = os.path.join(LOGO_DIR, f"{slug}.png")

    if os.path.exists(path):
        print(f"✅ Already exists: {team_name}")
        return

    try:
        img = requests.get(logo_url, timeout=10)
        img.raise_for_status()
        with open(path, "wb") as f:
            f.write(img.content)
        print(f"✔ Saved {team_name} → {path}")
    except Exception as e:
        print(f"❌ Failed to save {team_name}: {e}")

if __name__ == "__main__":
    # pick league here
    league_id = LEAGUE_MAP["LA_LIGA2"]  # you can swap to EPL/UCL/etc

    print(f"🔍 Fetching teams for league {league_id}...")
    teams = fetch_teams_for_league(league_id)

    print(f"Found {len(teams)} teams")
    for t in teams:
        team = t["team"]
        save_logo(team["name"], team["logo"])

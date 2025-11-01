import os
import re
import time
import requests
import unicodedata

from api.app.services.apifootball import LEAGUE_MAP  # ✅ absolute import

LOGO_DIR = "frontend/public/logos"
os.makedirs(LOGO_DIR, exist_ok=True)

API_HOST = "api-football-v1.p.rapidapi.com"
API_KEY = "864e09cc4dmsh09a1fd80628528bp1171a7jsn39805d312e97" # set in your .env

headers = {
    "x-rapidapi-host": API_HOST,
    "x-rapidapi-key": API_KEY,
}

def slugify(name: str) -> str:
    """
    Convert team name into a safe slug for filenames.
    Handles accents, spaces, and special characters.
    """
    slug = (
        unicodedata.normalize("NFKD", name)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", slug)
    return slug.strip("_").lower()

def fetch_teams_for_league(league_id: int):
    url = f"https://{API_HOST}/v3/teams?league={league_id}&season=2025"
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    return res.json()["response"]

def save_logo(team_name: str, logo_url: str):
    slug = slugify(team_name)
    path = os.path.join(LOGO_DIR, f"{slug}.png")
    if os.path.exists(path):
        print(f"✅ Already have {team_name}")
        return
    try:
        img = requests.get(logo_url, timeout=10)
        img.raise_for_status()
        with open(path, "wb") as f:
            f.write(img.content)
        print(f"✔ Saved {team_name} → {path}")
    except Exception as e:
        print(f"❌ Failed {team_name}: {e}")

if __name__ == "__main__":
    # Just fetch EPL today
    league_id = LEAGUE_MAP["LA_LIGA2"]   # 👈 change code here
    teams = fetch_teams_for_league(league_id)
    for t in teams:
        save_logo(t["team"]["name"], t["team"]["logo"])

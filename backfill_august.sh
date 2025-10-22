#!/usr/bin/env bash
set -euo pipefail

BASE="http://localhost:8000"
LEAGUES="EPL,CHAMP,LG1,LG2,SCO_PREM,SCO_CHAMP,SCO1,SCO2,LA_LIGA,BUNDES,BUNDES2,SERIE_A,SERIE_B,LIGUE1,UCL,UEL,UECL,WCQ_EUR"

# simple helper: call an endpoint and retry once on rate-limit text
call_api () {
  local method="$1"; shift
  local url="$1"; shift

  echo "→ $method $url"
  RESP="$(curl -s -X "$method" "$url")" || true
  echo "$RESP"

  # crude 429 detection (RapidAPI’s text)
  if echo "$RESP" | grep -qi "exceeded the rate limit"; then
    echo "⚠️  Rate limit hit. Cooling down 90s..."
    sleep 90
    RESP="$(curl -s -X "$method" "$url")" || true
    echo "$RESP"
  fi
}

# Generate dates for August with Python (portable on macOS)
python3 - <<'PY' > /tmp/aug_dates.txt
from datetime import date, timedelta
d = date(2025,8,1)
while d.month == 8:
    print(d.isoformat())
    d += timedelta(days=1)
PY

i=0
while read -r DAY; do
  i=$((i+1))
  echo ""
  echo "===== $DAY ====="

  # 1) Fixtures backfill (server-side ID filter; safer & lighter)
  call_api POST "$BASE/admin/backfill/top-leagues-by-day?day=$DAY"

  # 2) Odds ingest for the same day (soccer only here)
  call_api POST "$BASE/admin/run-dates?days=$DAY&leagues=$LEAGUES&prefer_book=bet365&compute_after=1"

  # light pacing
  sleep 3

  # deeper cooldown every 10 days
  if (( i % 10 == 0 )); then
    echo "⏸ cooldown 60s…"
    sleep 60
  fi
done < /tmp/aug_dates.txt

echo "✅ August backfill + odds pass complete."

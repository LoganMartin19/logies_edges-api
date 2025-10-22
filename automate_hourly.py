# automate_hourly.py
from __future__ import annotations
import os, sys, time
from datetime import datetime, timedelta, timezone, date
import requests

# ------------------------------------------------------------
# Config via ENV (sane defaults)
# ------------------------------------------------------------
BASE_URL           = os.getenv("VE_BASE_URL", "http://localhost:8000")
LEAGUES            = os.getenv("VE_LEAGUES", None)              # CSV of league keys; omit to use backend defaults

# Ingestion
DAYS               = int(os.getenv("VE_DAYS", "1"))            # 1 => today only; 2 => today & tomorrow
PREFER_BOOK        = os.getenv("VE_PREFER_BOOK", "bet365")
ODDS_DELAY         = float(os.getenv("VE_ODDS_DELAY", "0.35")) # seconds between odds hits (used in gap fills)
MAX_FIX            = int(os.getenv("VE_MAX_FIX", "500"))       # not used in run-dates (kept for parity)

# Housekeeping
RUN_CLEANUP        = int(os.getenv("VE_RUN_CLEANUP", "0"))     # 0=off, 1=on
CLEANUP_AGE_DAYS   = int(os.getenv("VE_CLEANUP_AGE_DAYS", "14"))

# Compute / shortlist
RECOMPUTE_MIN_EDGE = float(os.getenv("VE_MIN_EDGE", "0.00"))
SL_HOURS_AHEAD     = int(os.getenv("VE_SL_HOURS", "96"))
SL_MIN_EDGE        = float(os.getenv("VE_SL_MIN_EDGE", "0.00"))
SL_SEND_ALERTS     = int(os.getenv("VE_SL_SEND_ALERTS", "1"))  # 1 = send alerts

# Optional: odds gap backfill (refresh comps with missing odds)
BACKFILL_GAPS      = int(os.getenv("VE_BACKFILL_GAPS", "1"))   # 0 to disable
GAPS_LOOKAHEAD     = int(os.getenv("VE_GAPS_HOURS", "36"))
GAPS_BATCH_CAP     = int(os.getenv("VE_GAPS_BATCH_CAP", "20")) # safety cap per run

# Optional: apply calibration
APPLY_CALIB        = int(os.getenv("VE_APPLY_CALIB", "0"))
CALIB_SOURCE_IN    = os.getenv("VE_CALIB_SOURCE_IN", "consensus_v2")
CALIB_SOURCE_OUT   = os.getenv("VE_CALIB_SOURCE_OUT", "consensus_calib")

# ------------------------------------------------------------
# HTTP helper
# ------------------------------------------------------------
def srequest(s: requests.Session, method: str, path: str, *, params=None, json=None, timeout=90):
    url = f"{BASE_URL}{path}"
    try:
        r = s.request(method, url, params=params, json=json, timeout=timeout)
        status = r.status_code
        # soft-handle 404s for optional endpoints
        if status == 404:
            print(f"[{method}] {url} -> 404 (optional/missing endpoint)")
            return None
        r.raise_for_status()
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}

        # print a compact line
        print(f"[{method}] {url} -> {status}")
        return body
    except requests.RequestException as e:
        print(f"[HTTP-ERR] {method} {url}: {e}")
        return None

def _iso_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _dates_csv_from_days(n: int) -> str:
    if n <= 1:
        return date.today().isoformat()
    return ",".join((date.today() + timedelta(days=i)).isoformat() for i in range(n))

def _rate_limited(last_http: dict | None) -> bool:
    if not isinstance(last_http, dict):
        return False
    if last_http.get("status") == 429:
        return True
    err = str(last_http.get("error") or "").lower()
    return "rate_limited" in err or "429" in err

def _sleep_for_reset(last_http: dict | None, fallback_sec=120):
    if not isinstance(last_http, dict):
        time.sleep(fallback_sec)
        return
    reset = last_http.get("ratelimit_reset")
    try:
        # Some providers give seconds-to-reset; if absent fall back
        sec = int(reset)
        if sec <= 0 or sec > 3600:
            sec = fallback_sec
        print(f"[ratelimit] sleeping {sec}s…")
        time.sleep(sec)
    except Exception:
        print(f"[ratelimit] sleeping {fallback_sec}s (fallback)…")
        time.sleep(fallback_sec)

# ------------------------------------------------------------
# Main automation
# ------------------------------------------------------------
def automate_value_alerts():
    print(f"=== automate_value_alerts @ {_iso_now()} ===")
    with requests.Session() as s:
        # 0) Optional: provider pre-warm (if you use it) — safe to ignore 404
        srequest(s, "POST", "/admin/telegram/test")

        # 1) Cleanup (optional + age-gated)
        if RUN_CLEANUP:
            srequest(s, "POST", "/admin/cleanup-past", params={"min_age_days": CLEANUP_AGE_DAYS})

        # 2) Ingest fixtures/odds for selected dates (compute_after=0 for speed)
        dates_csv = _dates_csv_from_days(DAYS)
        run_dates_params = {
            "days": dates_csv,
            "prefer_book": PREFER_BOOK,
            "compute_after": 0,          # we'll compute once at end
            "cfb_uk_only": 1,            # safe default for gridiron
        }
        if LEAGUES:
            run_dates_params["leagues"] = LEAGUES

        ing = srequest(s, "POST", "/admin/run-dates", params=run_dates_params)
        if isinstance(ing, dict) and "ingested" in ing:
            fx = sum(x.get("fixtures_upserted", 0) for x in ing["ingested"])
            od = sum(x.get("odds_rows_written", 0) for x in ing["ingested"])
            print(f"[ingested] fixtures={fx} odds_rows={od}")

        # brief pause so DB catches up
        time.sleep(2)

        # 3) First recompute pass (consensus_v2)
        rec = srequest(s, "POST", "/admin/recompute", params={"min_edge": RECOMPUTE_MIN_EDGE, "source": "team_form"})
        if rec:
            print(f"[recompute v2] edges={rec.get('edges')} model_probs={rec.get('model_probs')}")

        # 4) Optional: backfill odds gaps (if endpoint exists)
        if BACKFILL_GAPS:
            gaps = srequest(s, "GET", "/admin/odds-gaps", params={"hours_ahead": GAPS_LOOKAHEAD})
            if isinstance(gaps, dict) and gaps.get("items"):
                # group by competition and cap for safety
                comps_count = {}
                for it in gaps["items"]:
                    comp = it.get("comp")
                    if not comp:
                        continue
                    comps_count[comp] = comps_count.get(comp, 0) + 1

                print(f"[gaps] {gaps.get('missing_count')} fixtures missing odds across {len(comps_count)} comps")
                comps_sorted = sorted(comps_count.items(), key=lambda kv: kv[1], reverse=True)[:GAPS_BATCH_CAP]

                for comp, ct in comps_sorted:
                    print(f"[refresh-comp] {comp} ({ct} missing)")
                    resp = srequest(
                        s, "POST", "/admin/refresh-odds-for-comp",
                        params={
                            "comp": comp,
                            "hours_ahead": GAPS_LOOKAHEAD,
                            "prefer_book": PREFER_BOOK,
                            "allow_fallback": 1,
                            "compute_after": 0,          # no inner recompute
                            "odds_delay_sec": ODDS_DELAY # your route may already default; pass anyway
                        }
                    )

                    # read provider's last-http to respect rate limits
                    last = srequest(s, "GET", "/admin/last-http")
                    if _rate_limited(last):
                        _sleep_for_reset(last, fallback_sec=120)

                # recompute after gap-filling
                rec2 = srequest(s, "POST", "/admin/recompute", params={"min_edge": RECOMPUTE_MIN_EDGE, "source": "team_form"})
                if rec2:
                    print(f"[recompute v2 after gaps] edges={rec2.get('edges')} model_probs={rec2.get('model_probs')}")
            else:
                print("[gaps] none or endpoint missing")

        # 5) Optional: apply calibration and recompute calibrated edges
        if APPLY_CALIB:
            app = srequest(
                s, "POST", "/admin/apply-calibration",
                params={
                    "source_in": CALIB_SOURCE_IN,
                    "source_out": CALIB_SOURCE_OUT,
                    "book": PREFER_BOOK,
                    "hours_ahead": SL_HOURS_AHEAD
                }
            )
            if app:
                print(f"[apply-calib] wrote={app.get('written')}")
            rec3 = srequest(s, "POST", "/admin/recompute", params={"min_edge": RECOMPUTE_MIN_EDGE, "source": CALIB_SOURCE_OUT})
            if rec3:
                print(f"[recompute calib] edges={rec3.get('edges')} model_probs={rec3.get('model_probs')}")

        # 6) Pull shortlist (this can also send alerts)
        sl = srequest(
            s, "GET", "/shortlist/today",
            params={
                "hours_ahead": SL_HOURS_AHEAD,
                "min_edge": SL_MIN_EDGE,
                "send_alerts": 0,
                # "prefer_book": PREFER_BOOK,   # uncomment to limit shortlist to a book
                # "leagues": "EPL,CHAMP,LG1",   # optional filter
            }
        )
        top_edges = []
        if isinstance(sl, list):
            print(f"[shortlist] count={len(sl)}")
            for it in sl:
                try:
                    comp = it.get('comp')
                    home = it.get('home_team'); away = it.get('away_team')
                    mkt  = it.get('market');    bk  = it.get('bookmaker')
                    price = float(it.get('price', 0.0))
                    p     = float(it.get('prob', 0.0))
                    edge  = float(it.get('edge', 0.0)) * 100.0

                    if edge >= 4.0:  # 👈 filter noise, tweak this as needed
                        print(f"[CANDIDATE] {home} vs {away} | {mkt} @ {bk} {price:.2f} (p={p:.3f}, edge={edge:.1f}%)")
                        top_edges.append(it)
                except Exception:
                    pass
        if top_edges:
            print(f"\nSelect edges to send to Telegram (max 5). Enter indexes separated by comma (e.g. 0,2,3):\n")
            for idx, it in enumerate(top_edges):
                print(f"{idx}) {it['home_team']} vs {it['away_team']} | {it['market']} @ {it['bookmaker']} | {it['price']} ({it['edge']*100:.1f}%)")

            selected = input("\nEnter selection: ")
            try:
                indexes = [int(x.strip()) for x in selected.split(",") if x.strip().isdigit()]
                to_send = [top_edges[i] for i in indexes if 0 <= i < len(top_edges)]

                for edge in to_send:
                    msg = f"📢 {edge['home_team']} vs {edge['away_team']} | {edge['market']} @ {edge['bookmaker']} {edge['price']} ({edge['edge']*100:.1f}%)"
                    print(f"\n[PREVIEW] {msg}")
                    confirm = input("Send this alert? (y/n): ").strip().lower()
                    if confirm != "y":
                        print("[SKIPPED]")
                        continue

                    resp = srequest(s, "POST", "/telegram/send-alert", json=edge)
                    if resp and resp.get("ok"):
                        print(f"[ALERT SENT] {edge['home_team']} vs {edge['away_team']} | {edge['market']}")
                    else:
                        print(f"[ERROR] Failed to send alert for: {edge}")
            except Exception as e:
                print(f"[SELECTION ERROR] {e}")

        # 7) Diagnostics: last HTTPs (soccer & gridiron). Safe if missing.
        srequest(s, "GET", "/admin/last-http")
        srequest(s, "GET", "/admin/gridiron-last-http")

# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        automate_value_alerts()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
"""Fetch every free data source for the EPL DFS model.

Usage:  python dfs/pipeline/fetch.py --gw 12

Sources (all free, no key):
  - FPL API: players, xG/xA, minutes, injuries, set-piece orders, fixtures,
    per-match history (element-summary).
  - ESPN API: per-player shots on target / saves per match (season),
    DraftKings match odds (1X2 + total goals) for the gameweek.

Raw responses are cached in dfs/data/raw/ (git-ignored). Processed tables go
to dfs/data/*.csv.
"""
import argparse, csv, datetime as dt, json, os, sys, time, urllib.request, gzip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
DATA = os.path.join(ROOT, "data")
FPL = "https://fantasy.premierleague.com/api"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1"
UA = {"User-Agent": "Mozilla/5.0 (dfs-model)", "Accept-Encoding": "gzip"}


def get(url, cache=None, max_age_h=6, tries=4):
    path = os.path.join(RAW, cache) if cache else None
    if path and os.path.exists(path) and (time.time() - os.path.getmtime(path)) < max_age_h * 3600:
        return json.load(open(path))
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                b = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    b = gzip.decompress(b)
            d = json.loads(b)
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                json.dump(d, open(path, "w"))
            return d
        except Exception as e:  # noqa
            if i == tries - 1:
                print(f"  FAIL {url}: {e}", file=sys.stderr)
                return None
            time.sleep(2 ** i)


def write_csv(name, rows):
    if not rows:
        return
    p = os.path.join(DATA, name)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {p} ({len(rows)} rows)")


def fetch_fpl(gw):
    print("FPL bootstrap + fixtures")
    bs = get(f"{FPL}/bootstrap-static/", "fpl_bootstrap.json", max_age_h=2)
    fx = get(f"{FPL}/fixtures/", "fpl_fixtures.json", max_age_h=2)
    teams = {t["id"]: t for t in bs["teams"]}
    pos = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    gw_fx = [f for f in fx if f["event"] == gw]
    slate_teams = {f["team_h"] for f in gw_fx} | {f["team_a"] for f in gw_fx}

    write_csv("fpl_fixtures.csv", [{
        "gw": f["event"], "kickoff_utc": f["kickoff_time"],
        "home": teams[f["team_h"]]["short_name"], "away": teams[f["team_a"]]["short_name"],
        "home_score": f["team_h_score"], "away_score": f["team_a_score"],
        "finished": f["finished"]} for f in fx if f["event"]])

    players, hist = [], []
    els = [e for e in bs["elements"] if e["team"] in slate_teams]
    print(f"FPL element-summary for {sum(e['minutes'] > 0 for e in els)} players")
    for e in els:
        players.append({
            "fpl_id": e["id"], "name": e["web_name"],
            "full_name": f"{e['first_name']} {e['second_name']}",
            "team": teams[e["team"]]["short_name"], "pos": pos[e["element_type"]],
            "fpl_price": e["now_cost"] / 10, "status": e["status"],
            "chance_next": e["chance_of_playing_next_round"], "news": e["news"].replace("\n", " "),
            "minutes": e["minutes"], "starts": e["starts"], "goals": e["goals_scored"],
            "assists": e["assists"], "xg": e["expected_goals"], "xa": e["expected_assists"],
            "saves": e["saves"], "yellow": e["yellow_cards"], "red": e["red_cards"],
            "pen_order": e["penalties_order"], "corner_order": e["corners_and_indirect_freekicks_order"],
            "fk_order": e["direct_freekicks_order"], "selected_pct": e["selected_by_percent"],
        })
        if e["minutes"] == 0:
            continue
        s = get(f"{FPL}/element-summary/{e['id']}/", f"fpl_es/{e['id']}.json", max_age_h=12)
        for h in (s or {}).get("history", []):
            hist.append({
                "fpl_id": e["id"], "gw": h["round"], "opp": teams[h["opponent_team"]]["short_name"],
                "home": h["was_home"], "minutes": h["minutes"], "starts": h.get("starts", int(h["minutes"] >= 60)),
                "goals": h["goals_scored"], "assists": h["assists"], "xg": h["expected_goals"],
                "xa": h["expected_assists"], "cs": h["clean_sheets"], "gc": h["goals_conceded"],
                "saves": h["saves"], "yellow": h["yellow_cards"], "red": h["red_cards"],
                "pen_missed": h["penalties_missed"], "og": h["own_goals"],
            })
        time.sleep(0.05)
    write_csv("fpl_players.csv", players)
    write_csv("fpl_history.csv", hist)
    return bs, fx, gw_fx, teams


def american_to_prob(o):
    o = float(str(o).replace("+", ""))
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


def fetch_espn(fx, gw_fx):
    dates = sorted({f["kickoff_time"][:10] for f in fx if f["kickoff_time"]})
    today = dt.date.today().isoformat()
    gw_dates = {f["kickoff_time"][:10] for f in gw_fx}
    past = [d for d in dates if d < today]
    print(f"ESPN scoreboards: {len(past)} past dates + GW dates")
    rows, odds, past_odds = [], [], []
    for d in past + sorted(gw_dates):
        is_gw = d in gw_dates
        sb = get(f"{ESPN}/scoreboard?dates={d.replace('-', '')}", f"espn_sb/{d}.json",
                 max_age_h=1 if is_gw else 24 * 365)
        for ev in (sb or {}).get("events", []):
            c = ev["competitions"][0]
            done = ev["status"]["type"]["completed"]
            if is_gw and not done:
                o = (c.get("odds") or [None])[0]
                home = next(t for t in c["competitors"] if t["homeAway"] == "home")["team"]["abbreviation"]
                away = next(t for t in c["competitors"] if t["homeAway"] == "away")["team"]["abbreviation"]
                if o and o.get("moneyline"):
                    ml = o["moneyline"]
                    pick = lambda side: (ml[side].get("close") or ml[side].get("open"))["odds"]
                    tot = o.get("total", {})
                    ov = (tot.get("over", {}).get("close") or tot.get("over", {}).get("open") or {})
                    un = (tot.get("under", {}).get("close") or tot.get("under", {}).get("open") or {})
                    odds.append({"espn_id": ev["id"], "kickoff_utc": ev["date"], "home": home, "away": away,
                                 "ml_home": pick("home"), "ml_draw": pick("draw"), "ml_away": pick("away"),
                                 "total_line": o.get("overUnder"), "over_odds": ov.get("odds"),
                                 "under_odds": un.get("odds"), "source": "DraftKings via ESPN"})
                continue
            if not done:
                continue
            sm = get(f"{ESPN}/summary?event={ev['id']}", f"espn_sum/{ev['id']}.json", max_age_h=24 * 365)
            if not sm:
                continue
            po = (sm.get("pickcenter") or [None])[0]
            if po and po.get("moneyline"):
                ml = po["moneyline"]
                pick = lambda side: (ml[side].get("close") or ml[side].get("open"))["odds"]
                tot = po.get("total", {})
                ov = (tot.get("over", {}).get("close") or tot.get("over", {}).get("open") or {})
                un = (tot.get("under", {}).get("close") or tot.get("under", {}).get("open") or {})
                home = next(t for t in c["competitors"] if t["homeAway"] == "home")
                away = next(t for t in c["competitors"] if t["homeAway"] == "away")
                past_odds.append({"espn_id": ev["id"], "date": d, "home": home["team"]["abbreviation"],
                                  "away": away["team"]["abbreviation"], "home_goals": home.get("score"),
                                  "away_goals": away.get("score"), "ml_home": pick("home"),
                                  "ml_draw": pick("draw"), "ml_away": pick("away"),
                                  "total_line": po.get("overUnder"), "over_odds": ov.get("odds"),
                                  "under_odds": un.get("odds")})
            for r in sm.get("rosters", []):
                team = r["team"]["abbreviation"]
                for p in r.get("roster", []):
                    st = {s["name"]: s.get("value", 0) for s in p.get("stats", [])}
                    if not st.get("appearances"):
                        continue
                    rows.append({"espn_event": ev["id"], "date": d, "team": team,
                                 "player": p["athlete"]["displayName"], "starter": p.get("starter"),
                                 "pos": p.get("position", {}).get("abbreviation"),
                                 "shots": st.get("totalShots", 0), "sot": st.get("shotsOnTarget", 0),
                                 "goals": st.get("totalGoals", 0), "saves": st.get("saves", 0),
                                 "yellow": st.get("yellowCards", 0), "red": st.get("redCards", 0)})
    write_csv("espn_player_matches.csv", rows)
    write_csv("odds_gw.csv", odds)
    write_csv("odds_past.csv", past_odds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, required=True)
    a = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    bs, fx, gw_fx, teams = fetch_fpl(a.gw)
    fetch_espn(fx, gw_fx)
    print("done")

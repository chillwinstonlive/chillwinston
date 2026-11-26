"""Project FanTeam points for every slate player.

Usage:  python dfs/pipeline/project.py --gw 12 [--exclude NFO,CHE]

Inputs (from fetch.py): dfs/data/fpl_players.csv, fpl_history.csv,
espn_player_matches.csv, odds_gw.csv, fpl_fixtures.csv.
Optional: dfs/epl/GW<gw>/players.csv  (FanTeam market export / template CSV)
  -> FanTeam position overrides FPL position (mis-classification edge)
  -> FanTeam price + id used by optimiser.
Output: dfs/epl/GW<gw>/projections.csv, team_lambdas.csv

The same functions run "as of" any past gameweek for backtest.py.
"""
import argparse, math, os, re, unicodedata, difflib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ESPN_TEAM = {"MAN": "MUN", "MNC": "MCI"}

# FanTeam scoring (confirmed 24 Nov 2026)
GOAL = {"GK": 8, "DEF": 6, "MID": 5, "FWD": 4}
SOT = {"GK": 1, "DEF": 0.6, "MID": 0.4, "FWD": 0.4}
CS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
CONC = {"GK": -1, "DEF": -1, "MID": 0, "FWD": 0}   # per 2 conceded
FULL90 = {"GK": 0, "DEF": 0, "MID": 1, "FWD": 1}
ASSIST, SAVE, YEL, RED, PENMISS, OG = 3, 0.5, -1, -3, -2, -2
FANTASY_ASSIST_MULT = 1.1   # GUESS: fantasy assists add ~10% on top of real assists

# Shrinkage priors (per 90) and pseudo-minutes
PRIOR = {"GK": dict(xg=0.0, xa=0.01, sot=0.0, yc=0.05),
         "DEF": dict(xg=0.04, xa=0.05, sot=0.15, yc=0.17),
         "MID": dict(xg=0.12, xa=0.12, sot=0.35, yc=0.17),
         "FWD": dict(xg=0.30, xa=0.10, sot=0.80, yc=0.12)}
PSEUDO_MIN = 720   # tuned on GW4-11 backtest
ODDS_ALPHA = 0.75  # how hard odds rescale player rates (1 = fully)
RECENT_W = 1.0     # extra weight on last 6 GWs


# ---------- helpers ----------
def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", s).strip()


def pois(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def am(o):
    o = float(str(o).replace("+", ""))
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


def devig(p):
    s = sum(p)
    return [x / s for x in p]


def fit_lambdas(ph, pd_, pa, line, pover):
    """Grid-fit independent Poisson means to de-vigged 1X2 + total-goals odds."""
    def score(lh, la):
        P = [[pois(i, lh) * pois(j, la) for j in range(11)] for i in range(11)]
        h = sum(P[i][j] for i in range(11) for j in range(i))
        d = sum(P[i][i] for i in range(11))
        ov = sum(P[i][j] for i in range(11) for j in range(11) if i + j > line)
        return (h - ph) ** 2 + (d - pd_) ** 2 + ((1 - h - d) - pa) ** 2 + (ov - pover) ** 2
    best = None
    grid = [(0.2 + 0.1 * i, 0.2 + 0.1 * j) for i in range(39) for j in range(39)]
    for lh, la in grid:
        s = score(lh, la)
        if best is None or s < best[0]:
            best = (s, lh, la)
    _, bh, ba = best
    for i in range(-6, 7):
        for j in range(-6, 7):
            lh, la = bh + 0.02 * i, ba + 0.02 * j
            if lh > 0.05 and la > 0.05:
                s = score(lh, la)
                if s < best[0]:
                    best = (s, lh, la)
    return best[1], best[2]


def e_floor_half(lam):
    return sum((k // 2) * pois(k, lam) for k in range(15))


def lambdas_from_odds(odds, exclude=()):
    out = {}
    for _, r in odds.iterrows():
        h, a = ESPN_TEAM.get(r.home, r.home), ESPN_TEAM.get(r.away, r.away)
        if h in exclude or a in exclude or pd.isna(r.over_odds) or pd.isna(r.under_odds):
            continue
        ph, pdr, pa = devig([am(r.ml_home), am(r.ml_draw), am(r.ml_away)])
        po, _ = devig([am(r.over_odds), am(r.under_odds)])
        lh, la = fit_lambdas(ph, pdr, pa, r.total_line, po)
        out[h] = dict(opp=a, home=True, lam_for=lh, lam_against=la, p_win=ph)
        out[a] = dict(opp=h, home=False, lam_for=la, lam_against=lh, p_win=pa)
    return out


# ---------- data joins ----------
def espn_with_gw(espn, fixtures):
    """Tag each ESPN player-match row with FPL gameweek and FPL team code."""
    espn = espn.copy()
    espn["team"] = espn.team.map(lambda t: ESPN_TEAM.get(t, t))
    fx = fixtures.dropna(subset=["kickoff_utc"]).copy()
    fx["date"] = fx.kickoff_utc.str[:10]
    key = {}
    for _, f in fx.iterrows():
        key[(f.home, f.date)] = f.gw
        key[(f.away, f.date)] = f.gw
    espn["gw"] = [key.get((t, d)) for t, d in zip(espn.team, espn.date)]
    return espn


def map_espn_names(players, espn):
    """fpl_id -> ESPN display name (within club)."""
    names_by_team = espn.groupby("team").player.unique().to_dict()
    out = {}
    for _, p in players.iterrows():
        cands = list(names_by_team.get(p.team, []))
        if not cands:
            continue
        nmap = {norm(c): c for c in cands}
        full, web = norm(p.full_name), norm(p["name"]).replace(".", "")
        hit = None
        for n, c in nmap.items():
            if n == full or (len(web) > 2 and (n.endswith(" " + web) or n == web)):
                hit = c; break
        if not hit:
            m = difflib.get_close_matches(full, list(nmap), n=1, cutoff=0.72)
            hit = nmap[m[0]] if m else None
        if hit:
            out[p.fpl_id] = hit
    return out


def load_fanteam(path):
    """Read a FanTeam market/template CSV with flexible headers."""
    if not os.path.exists(path):
        return None
    ft = pd.read_csv(path)
    ft.columns = [c.strip().lower() for c in ft.columns]
    col = lambda *opts: next((c for c in ft.columns for o in opts if o == c or o in c), None)
    m = {"name": col("name", "player"), "team": col("team", "club"), "pos": col("position", "pos"),
         "price": col("price", "value", "cost"), "ft_id": col("id"), "ft_status": col("status")}
    ft = ft.rename(columns={v: k for k, v in m.items() if v})
    posmap = {"goalkeeper": "GK", "gk": "GK", "defender": "DEF", "def": "DEF", "d": "DEF",
              "midfielder": "MID", "mid": "MID", "m": "MID", "forward": "FWD", "fwd": "FWD",
              "for": "FWD", "f": "FWD", "attacker": "FWD", "striker": "FWD", "g": "GK"}
    ft["pos"] = ft.pos.astype(str).str.strip().str.lower().map(posmap).fillna(ft.pos)
    ft["price"] = pd.to_numeric(ft.price.astype(str).str.replace("M", "").str.replace("m", ""), errors="coerce")
    return ft


TEAM_ALIASES = {"ARS": ["arsenal"], "AVL": ["aston villa", "villa"], "BHA": ["brighton"],
                "BOU": ["bournemouth"], "BRE": ["brentford"], "CHE": ["chelsea"], "COV": ["coventry"],
                "CRY": ["crystal palace", "palace"], "EVE": ["everton"], "FUL": ["fulham"],
                "HUL": ["hull"], "IPS": ["ipswich"], "LEE": ["leeds"], "LIV": ["liverpool"],
                "MCI": ["manchester city", "man city"], "MUN": ["manchester united", "man utd", "man united"],
                "NEW": ["newcastle"], "NFO": ["nottingham", "forest"], "SUN": ["sunderland"],
                "TOT": ["tottenham", "spurs"]}


def ft_team_code(t):
    t = str(t).strip()
    if t.upper() in TEAM_ALIASES:
        return t.upper()
    tl = t.lower()
    for code, al in TEAM_ALIASES.items():
        if any(a in tl for a in al):
            return code
    return t


def match_fanteam(players, ft):
    ft = ft.copy()
    ft["code"] = ft.team.map(ft_team_code)
    ft["n"] = ft.name.map(norm)
    out = {}
    for _, p in players.iterrows():
        c = ft[ft.code == p.team]
        if c.empty:
            continue
        full, web = norm(p.full_name), norm(p["name"]).replace(".", "")
        hit = c[(c.n == full) | (c.n == web)]
        if hit.empty:
            hit = c[c.n.map(lambda n: len(web) > 2 and (n.endswith(" " + web) or full.endswith(" " + n)))]
        if hit.empty:
            m = difflib.get_close_matches(full, list(c.n), n=1, cutoff=0.75)
            hit = c[c.n == m[0]] if m else hit
        if not hit.empty:
            out[p.fpl_id] = hit.iloc[0]
    return out


# ---------- model ----------
def minutes_model(h, status="a", chance=None, live=True):
    h = h.sort_values("gw")
    last = h.tail(6)
    if last.empty:
        return dict(p_start=0, p_sub=0, p60s=0, p90s=0, mins_s=0)
    w = list(range(1, len(last) + 1))
    started = (last.minutes >= 60) | ((last.starts.astype(int) == 1) & (last.minutes > 0))
    start_rate = sum(wi * s for wi, s in zip(w, started)) / sum(w)
    non = last[~started]
    sub_rate = (non.minutes > 0).mean() if len(non) else 0.3
    st = h[(h.minutes >= 45)].tail(8)
    mins_s = st.minutes.mean() if len(st) else 70
    p60s = (st.minutes >= 60).mean() if len(st) else 0.75
    p90s = (st.minutes >= 90).mean() if len(st) else 0.4
    avail = 1.0
    if live:
        if chance is not None and not pd.isna(chance):
            avail = chance / 100
        elif status != "a":
            avail = 0.0
        if avail > 0 and last.minutes.iloc[-1] == 0 and start_rate > 0.5:
            start_rate *= 0.8
    return dict(p_start=avail * start_rate, p_sub=avail * (1 - start_rate) * sub_rate,
                p60s=p60s, p90s=p90s, mins_s=mins_s)


def project_players(players, hist, espn_tagged, lam, emap, ft_map=None, live=True, overrides=None):
    """hist/espn_tagged must already be filtered to gameweeks BEFORE the target."""
    sot_by_name = espn_tagged.groupby(["team", "player"]).sot.sum().to_dict()
    tm = espn_tagged.groupby(["espn_event", "team"]).agg(sot=("sot", "sum"), g=("goals", "sum"))
    sot_per_goal = tm.sot.sum() / max(tm.g.sum(), 1)
    team_games = espn_tagged.groupby("team").espn_event.nunique()
    agg = hist.groupby("fpl_id").agg(min=("minutes", "sum"), xg=("xg", "sum"), xa=("xa", "sum"),
                                     yc=("yellow", "sum"), rc=("red", "sum"))
    pl_team = players.set_index("fpl_id").team
    team_xg = agg.join(pl_team).groupby("team").xg.sum()
    team_xg_pm = (team_xg / team_games).to_dict()

    rows = []
    for _, p in players[players.team.isin(lam.keys())].iterrows():
        if p.fpl_id not in agg.index:
            continue
        h = hist[hist.fpl_id == p.fpl_id]
        mm = minutes_model(h, p.status, p.chance_next, live)
        if overrides and p.fpl_id in overrides:          # team-news override of P(start)
            ov = overrides[p.fpl_id]
            mm["p_sub"] = 0.0 if ov >= 0.95 else (1 - ov) * 0.6
            mm["p_start"] = ov
        if mm["p_start"] + mm["p_sub"] < 0.05:
            continue
        t = lam[p.team]
        s = agg.loc[p.fpl_id]
        rec = h.sort_values("gw").tail(6)
        pos = p.pos
        ftr = ft_map.get(p.fpl_id) if ft_map else None
        if ftr is not None and ftr.pos in GOAL:
            pos = ftr.pos
        pr = PRIOR[p.pos]
        mins_tot = s["min"] + RECENT_W * rec.minutes.sum()
        xg90 = (s.xg + RECENT_W * rec.xg.sum() + pr["xg"] * PSEUDO_MIN / 90) / (mins_tot + PSEUDO_MIN) * 90
        xa90 = (s.xa + RECENT_W * rec.xa.sum() + pr["xa"] * PSEUDO_MIN / 90) / (mins_tot + PSEUDO_MIN) * 90
        sot_s = sot_by_name.get((p.team, emap.get(p.fpl_id)), 0)
        sot90 = (sot_s + pr["sot"] * PSEUDO_MIN / 90) / (s["min"] + PSEUDO_MIN) * 90
        yc90 = (s.yc + pr["yc"] * PSEUDO_MIN / 90) / (s["min"] + PSEUDO_MIN) * 90
        rc90 = (s.rc + 0.005 * PSEUDO_MIN / 90) / (s["min"] + PSEUDO_MIN) * 90

        factor = min(max(t["lam_for"] / max(team_xg_pm.get(p.team, 1.4), 0.5), 0.5), 2.0) ** ODDS_ALPHA
        m90 = (mm["p_start"] * mm["mins_s"] + mm["p_sub"] * 20) / 90
        goals = xg90 * m90 * factor
        assists = xa90 * m90 * factor * FANTASY_ASSIST_MULT
        sot = sot90 * m90 * factor
        p_app = mm["p_start"] + mm["p_sub"]
        p60 = mm["p_start"] * mm["p60s"]
        p90 = mm["p_start"] * mm["p90s"]
        lam_ag = t["lam_against"] * mm["mins_s"] / 90
        p_cs = p60 * math.exp(-lam_ag)
        conc = mm["p_start"] * e_floor_half(lam_ag)
        saves = 0.0
        if pos == "GK":
            saves = mm["p_start"] * t["lam_against"] * (sot_per_goal - 0.95) * mm["mins_s"] / 90
        pen_miss = 0.02 * m90 if p.pen_order == 1 else 0

        parts = dict(app=p_app + p60, goal=goals * GOAL[pos], ast=assists * ASSIST, sot=sot * SOT[pos],
                     cs=p_cs * CS[pos], conc=conc * CONC[pos], full=p90 * FULL90[pos], saves=saves * SAVE,
                     cards=m90 * (yc90 * YEL + rc90 * RED), pen=pen_miss * PENMISS)
        pts = sum(parts.values())
        var = goals * GOAL[pos] ** 2 + assists * ASSIST ** 2 + p_cs * (1 - p_cs) * CS[pos] ** 2 + saves * SAVE ** 2
        rows.append(dict(
            fpl_id=p.fpl_id, name=p["name"], team=p.team, opp=t["opp"], home=t["home"], pos=pos,
            fpl_pos=p.pos, ft_id=None if ftr is None else ftr.get("ft_id"),
            ft_price=None if ftr is None else ftr.price, ft_status=None if ftr is None else ftr.get("ft_status"),
            fpl_price=p.fpl_price, status=p.status, chance=p.chance_next,
            p_start=round(mm["p_start"], 2), p60=round(p60, 2), p90=round(p90, 2),
            xg=round(goals, 3), xa=round(assists, 3), sot=round(sot, 2), p_cs=round(p_cs, 2),
            saves=round(saves, 2), team_lam=round(t["lam_for"], 2), opp_lam=round(t["lam_against"], 2),
            pen=p.pen_order, pts=round(pts, 2), sd=round(math.sqrt(var), 2),
            **{f"pt_{k}": round(v, 2) for k, v in parts.items()},
            sel_pct=p.selected_pct, news=p.news if isinstance(p.news, str) else ""))
    return pd.DataFrame(rows).sort_values("pts", ascending=False)


def actual_points(players, hist_gw, espn_gw, emap):
    """APPROX FanTeam points from FPL + ESPN match data (no impact/fantasy-assist/caused-pen)."""
    sot = espn_gw.groupby(["team", "player"]).sot.sum().to_dict()
    pos = players.set_index("fpl_id").pos.to_dict()
    team = players.set_index("fpl_id").team.to_dict()
    out = {}
    for _, h in hist_gw.iterrows():
        if h.minutes == 0:
            continue
        ps = pos.get(h.fpl_id)
        if ps is None:
            continue
        s = sot.get((team[h.fpl_id], emap.get(h.fpl_id)), 0)
        pts = (1 + (h.minutes >= 60) + h.goals * GOAL[ps] + h.assists * ASSIST + s * SOT[ps]
               + (h.cs if h.minutes >= 60 else 0) * CS[ps] + (h.gc // 2) * CONC[ps]
               + (h.minutes >= 90) * FULL90[ps] + h.saves * SAVE + h.yellow * YEL + h.red * RED
               + h.pen_missed * PENMISS + h.og * OG)
        out[h.fpl_id] = out.get(h.fpl_id, 0) + pts
    return pd.Series(out, name="actual")


def load_all():
    players = pd.read_csv(os.path.join(DATA, "fpl_players.csv"))
    hist = pd.read_csv(os.path.join(DATA, "fpl_history.csv"))
    fixtures = pd.read_csv(os.path.join(DATA, "fpl_fixtures.csv"))
    espn = espn_with_gw(pd.read_csv(os.path.join(DATA, "espn_player_matches.csv")), fixtures)
    emap = map_espn_names(players, espn)
    return players, hist, fixtures, espn, emap


def main(gw, exclude):
    players, hist, fixtures, espn, emap = load_all()
    lam = lambdas_from_odds(pd.read_csv(os.path.join(DATA, "odds_gw.csv")), exclude)
    out_dir = os.path.join(ROOT, "epl", f"GW{gw}")
    os.makedirs(out_dir, exist_ok=True)
    ft = load_fanteam(os.path.join(out_dir, "players.csv"))
    ft_map = match_fanteam(players[players.team.isin(lam)], ft) if ft is not None else None
    overrides = {}
    ov_path = os.path.join(out_dir, "overrides.csv")
    if os.path.exists(ov_path):                      # columns: name,team,p_start[,note]
        ov = pd.read_csv(ov_path)
        hits = match_fanteam(players, ov)
        overrides = {fid: float(r.p_start) for fid, r in hits.items()}
        missed = set(ov.index) - {r.name for r in hits.values()}
        for i in missed:
            print(f"  override not matched: {ov.loc[i, 'name']} {ov.loc[i, 'team']}")
        print(f"Applied {len(overrides)} team-news overrides")
    df = project_players(players, hist[hist.gw < gw], espn[espn.gw < gw], lam, emap, ft_map,
                         live=True, overrides=overrides)
    df.to_csv(os.path.join(out_dir, "projections.csv"), index=False)
    lam_df = pd.DataFrame([dict(team=k, **v) for k, v in lam.items()]).sort_values("lam_for", ascending=False)
    lam_df.round(3).to_csv(os.path.join(out_dir, "team_lambdas.csv"), index=False)
    print(f"ESPN name-matched {len(emap)} players")
    if ft is not None:
        unmatched = ft[~ft.index.isin([r.name for r in ft_map.values()])]
        print(f"FanTeam matched {len(ft_map)}/{len(ft)}; unmatched rows -> {out_dir}/ft_unmatched.csv")
        unmatched.to_csv(os.path.join(out_dir, "ft_unmatched.csv"), index=False)
        mis = df[df.ft_price.notna() & (df.pos != df.fpl_pos)]
        if len(mis):
            print("\nPOSITION MIS-CLASSIFICATIONS (FanTeam vs FPL):")
            print(mis[["name", "team", "fpl_pos", "pos", "ft_price", "pts"]].to_string(index=False))
    print(lam_df[["team", "opp", "lam_for", "lam_against", "p_win"]].round(2).to_string(index=False))
    for ps in ["GK", "DEF", "MID", "FWD"]:
        print(f"\nTop {ps}")
        cols = ["name", "team", "opp", "ft_price" if ft is not None else "fpl_price", "p_start", "xg", "xa", "p_cs", "pts"]
        print(df[df.pos == ps].head(8)[cols].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, required=True)
    ap.add_argument("--exclude", default="", help="teams not on the slate, e.g. NFO,CHE")
    a = ap.parse_args()
    main(a.gw, {x for x in a.exclude.split(",") if x})

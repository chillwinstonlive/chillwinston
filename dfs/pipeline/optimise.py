"""Pick the max-expected-points FanTeam lineup (integer program, PuLP).

Usage:
  python dfs/pipeline/optimise.py --gw 12 --budget 113 [--max-club 3]
         [--stacking] [--min-start 0.7] [--tournament 1174816]

Price: FanTeam price from projections.csv (ft_price). If missing, falls back
to a PROXY (FPL price x 1.3) -> output is labelled TEST ONLY.
Builds three lineups: Main (max mean), Ceiling (mean + 0.6*sd),
Contrarian (penalise popular picks via FPL ownership as proxy).
"""
import argparse, os
import pandas as pd
import pulp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# FanTeam default 11-a-side limits (GUESS until contest Details confirm)
LIMITS = {"GK": (1, 1), "DEF": (3, 5), "MID": (3, 5), "FWD": (1, 3)}
STACK = [0, 1, 3, 6, 9, 12, 15]   # cumulative CS stacking penalty for n defensive players


def solve(df, budget, max_club, stacking, mode, exclude_ids=(), lock_ids=()):
    df = df.reset_index(drop=True)
    prob = pulp.LpProblem("ft", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x{i}", cat="Binary") for i in df.index}
    c = {i: pulp.LpVariable(f"c{i}", cat="Binary") for i in df.index}
    if mode == "ceiling":
        val = df.pts + 0.6 * df.sd
    elif mode == "contrarian":
        val = df.pts - 0.04 * df.sel_pct.clip(upper=60)
    else:
        val = df.pts
    obj = pulp.lpSum((x[i] + c[i]) * val[i] for i in df.index)

    prob += pulp.lpSum(x.values()) == 11
    prob += pulp.lpSum(c.values()) == 1
    for i in df.index:
        prob += c[i] <= x[i]
    prob += pulp.lpSum(x[i] * df.price[i] for i in df.index) <= budget
    for pos, (lo, hi) in LIMITS.items():
        s = pulp.lpSum(x[i] for i in df.index if df.pos[i] == pos)
        prob += s >= lo
        prob += s <= hi
    for team in df.team.unique():
        idx = [i for i in df.index if df.team[i] == team]
        if max_club:
            prob += pulp.lpSum(x[i] for i in idx) <= max_club
        if stacking:
            d_idx = [i for i in idx if df.pos[i] in ("GK", "DEF")]
            if len(d_idx) > 1:
                pcs = float(df.loc[d_idx, "p_cs"].max())
                n = pulp.lpSum(x[i] for i in d_idx)
                pen = pulp.LpVariable(f"pen_{team}", lowBound=0)
                for k in range(1, len(STACK)):       # convex piecewise lower bounds
                    slope = STACK[k] - STACK[k - 1]
                    prob += pen >= STACK[k - 1] + slope * (n - (k - 1))
                obj -= pen * pcs
    for i in df.index:
        if df.fpl_id[i] in exclude_ids:
            prob += x[i] == 0
        if df.fpl_id[i] in lock_ids:
            prob += x[i] == 1
    prob += obj
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    pick = df[[x[i].value() > 0.5 for i in df.index]].copy()
    cap = df.loc[[i for i in df.index if c[i].value() > 0.5][0]]
    rest = pick[pick.fpl_id != cap.fpl_id]
    vice = rest.assign(v=rest.pts * rest.p_start).sort_values("v", ascending=False).iloc[0]
    pick["role"] = ""
    pick.loc[pick.fpl_id == cap.fpl_id, "role"] = "C"
    pick.loc[pick.fpl_id == vice.fpl_id, "role"] = "VC"
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    pick = pick.sort_values(["pos", "pts"], key=lambda s: s.map(order) if s.name == "pos" else -s)
    total = pick.pts.sum() + cap.pts * cap.p_start + vice.pts * (1 - cap.p_start)
    return pick, total


def show(name, pick, total, budget):
    print(f"\n== {name}: {total:.1f} proj pts | {pick.price.sum():.1f}/{budget}M")
    print(pick[["role", "name", "team", "opp", "pos", "price", "p_start", "pts"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, required=True)
    ap.add_argument("--budget", type=float, default=113)
    ap.add_argument("--max-club", type=int, default=0)
    ap.add_argument("--stacking", action="store_true")
    ap.add_argument("--min-start", type=float, default=0.7)
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(ROOT, "epl", f"GW{a.gw}", "projections.csv"))
    test = df.ft_price.isna().all()
    if test:
        print("!! No FanTeam prices: PROXY prices (FPL x 1.3). TEST ONLY.")
        df["price"] = (df.fpl_price * 1.3).round(1)
    else:
        df = df[df.ft_price.notna()].copy()
        df["price"] = df.ft_price
    df = df[df.p_start >= a.min_start].copy()
    out = []
    main, t1 = solve(df, a.budget, a.max_club, a.stacking, "main")
    show("MAIN", main, t1, a.budget)
    ceil, t2 = solve(df, a.budget, a.max_club, a.stacking, "ceiling")
    show("CEILING", ceil, t2, a.budget)
    con, t3 = solve(df, a.budget, a.max_club, a.stacking, "contrarian")
    show("CONTRARIAN", con, t3, a.budget)
    d = os.path.join(ROOT, "epl", f"GW{a.gw}")
    for n, p in [("main", main), ("ceiling", ceil), ("contrarian", con)]:
        p.assign(test_only=test).to_csv(os.path.join(d, f"lineup_{n}.csv"), index=False)

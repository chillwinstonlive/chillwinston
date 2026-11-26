"""Backtest the projection model on past gameweeks.

Usage:  python dfs/pipeline/backtest.py --from 5 --to 11

For each GW g: project using only data from GWs < g and ESPN closing odds,
then score with APPROX actual FanTeam points. Reports bias by position,
correlation, and how the model's best-11 (by points, no prices) compares with
the hindsight best-11.
"""
import argparse, os
import pandas as pd
import project as P


def run(g_from, g_to):
    players, hist, fixtures, espn, emap = P.load_all()
    odds = pd.read_csv(os.path.join(P.DATA, "odds_past.csv"))
    fx = fixtures.copy()
    fx["date"] = fx.kickoff_utc.str[:10]
    res = []
    for g in range(g_from, g_to + 1):
        dates = set(fx[fx.gw == g].date)
        lam = P.lambdas_from_odds(odds[odds.date.isin(dates)])
        proj = P.project_players(players, hist[hist.gw < g], espn[espn.gw < g], lam, emap, live=False)
        act = P.actual_points(players, hist[hist.gw == g], espn[espn.gw == g], emap)
        m = proj.set_index("fpl_id").join(act).fillna({"actual": 0})
        m["gw"] = g
        res.append(m)
        print(f"GW{g}: {len(lam)//2} games, {len(m)} players, corr {m.pts.corr(m.actual):.2f}")
    r = pd.concat(res)
    r.to_csv(os.path.join(P.ROOT, "data", "backtest.csv"))
    print("\nBias by position (proj - actual, mean per player):")
    print(r.groupby("pos").apply(lambda d: pd.Series({
        "n": len(d), "proj": d.pts.mean(), "actual": d.actual.mean(),
        "bias": (d.pts - d.actual).mean(), "corr": d.pts.corr(d.actual)})).round(2).to_string())
    r["bucket"] = pd.cut(r.pts, [-5, 2, 3, 4, 5, 6, 20])
    print("\nCalibration (projected bucket -> actual):")
    print(r.groupby("bucket", observed=True).agg(n=("pts", "size"), proj=("pts", "mean"),
                                                 actual=("actual", "mean")).round(2).to_string())
    print("\nComponent check (per GW totals):")
    print(r.groupby("gw").agg(proj=("pts", "sum"), actual=("actual", "sum")).round(0).to_string())
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="f", type=int, default=5)
    ap.add_argument("--to", dest="t", type=int, default=11)
    a = ap.parse_args()
    run(a.f, a.t)

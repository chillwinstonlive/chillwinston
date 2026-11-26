"""Lock Claude's team: write claude_team.md with UK timestamp + SHA-256.

Usage:  python dfs/pipeline/lock.py --gw 12 --lock "2026-11-28 15:00" [--lineup main] [--tournament 1174816]

Refuses to overwrite an existing claude_team.md (locked teams never change).
"""
import argparse, datetime as dt, hashlib, os, sys
from zoneinfo import ZoneInfo
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--gw", type=int, required=True)
ap.add_argument("--lock", required=True, help="UK lock time, e.g. '2026-11-28 15:00'")
ap.add_argument("--lineup", default="main")
ap.add_argument("--budget", type=float, default=113)
ap.add_argument("--tournament", default="")
a = ap.parse_args()

d = os.path.join(ROOT, "epl", f"GW{a.gw}")
out = os.path.join(d, "claude_team.md")
if os.path.exists(out):
    sys.exit(f"{out} already exists: locked teams are never changed.")
p = pd.read_csv(os.path.join(d, f"lineup_{a.lineup}.csv"))
if p.get("test_only", pd.Series([False])).any():
    sys.exit("Lineup uses PROXY prices (test only). Add FanTeam players.csv and re-run before locking.")

now = dt.datetime.now(ZoneInfo("Europe/London")).strftime("%Y-%m-%d %H:%M %Z")
cap = p[p.role == "C"].iloc[0]
vc = p[p.role == "VC"].iloc[0]
total = p.pts.sum() + cap.pts * cap.p_start + vc.pts * (1 - cap.p_start)
lines = [f"# Claude's team - GW{a.gw}", "",
         f"Locked: {now}  ", f"Contest lock: {a.lock} UK  ",
         f"Price: {p.price.sum():.1f} / {a.budget}M  ", f"Projected: {total:.1f} pts (captain x2 included)", "",
         "| Player | Pos | £M | Proj |", "|---|---|---|---|"]
for _, r in p.iterrows():
    tag = f" ({r.role})" if isinstance(r.role, str) and r.role else ""
    lines.append(f"| {r['name']}{tag} {r.team} v {r.opp} | {r.pos} | {r.price} | {r.pts:.1f} |")
body = "\n".join(lines) + "\n"
h = hashlib.sha256(body.encode()).hexdigest()
upload = ""
if a.tournament and p.ft_id.notna().all():
    ids = [str(int(i)) for i in p.ft_id]
    upload = f"{a.tournament},{','.join(ids)},{int(cap.ft_id)},{int(vc.ft_id)}"
with open(out, "w") as f:
    f.write(body)
    f.write(f"\nSHA-256 of the text above (everything before this line):\n`{h}`\n")
    if upload:
        f.write(f"\nFanTeam upload line:\n`{upload}`\n")
print(open(out).read())

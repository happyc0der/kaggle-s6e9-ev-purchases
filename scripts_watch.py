"""Headless competition watch: new/improved public notebooks (Kaggle CLI) diffed against a stored snapshot.

Usage:  zsh -ic '.venv/bin/python scripts_watch.py'        (interactive zsh so KAGGLE_API_TOKEN is loaded)
Writes experiments/watch/report.md and updates experiments/watch/seen_kernels.json."""
import json, subprocess, re, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
W = ROOT / "experiments" / "watch"; W.mkdir(parents=True, exist_ok=True)
SEEN = W / "seen_kernels.json"
COMP = "playground-series-s6e9"
OUR_PUBLIC = 0.94649  # current best public score of this pipeline (update after each submission)

def kernels():
    r = subprocess.run([str(ROOT / ".venv/bin/kaggle"), "kernels", "list", "--competition", COMP, "--sort-by", "scoreDescending",
                        "--page-size", "60", "-v"], capture_output=True, text=True)
    out = r.stdout
    # fail loudly: an API/network error used to look exactly like "0 new notebooks"
    if r.returncode != 0 or "ref," not in out:
        raise SystemExit("kaggle kernels list failed (rc=%s). stderr tail:\n%s" % (r.returncode, r.stderr[-500:]))
    rows = []
    for line in out.splitlines():
        if line.startswith("ref,") or not line.strip() or "," not in line:
            continue
        parts = line.split(",")
        rows.append({"ref": parts[0].strip(), "title": ",".join(parts[1:-3]).strip().strip('"'), "lastRun": parts[-2].strip(), "votes": parts[-1].strip()})
    return rows

def lb_top():
    r = subprocess.run([str(ROOT / ".venv/bin/kaggle"), "competitions", "leaderboard", "-c", COMP, "-s"], capture_output=True, text=True)
    if r.returncode != 0:
        print("WARNING: leaderboard fetch failed (rc=%s)" % r.returncode)
        return []
    scores = [float(m) for m in re.findall(r"\b0\.9\d{4}\b", r.stdout)]
    return scores[:60]

seen = json.loads(SEEN.read_text()) if SEEN.exists() else {}
rows = kernels()
new, changed = [], []
for r in rows:
    key = r["ref"]
    if key not in seen:
        new.append(r)
    elif seen[key].get("lastRun") != r["lastRun"]:
        changed.append(r)
    seen[key] = {"title": r["title"], "lastRun": r["lastRun"]}
SEEN.write_text(json.dumps(seen, indent=1))
scores = lb_top()
lines = [f"# Watch report {time.strftime('%Y-%m-%d %H:%M')}", ""]
if scores:
    lines += [f"Leaderboard top: #1 {scores[0]:.5f} | #10 {scores[min(9,len(scores)-1)]:.5f} | #50 {scores[min(49,len(scores)-1)]:.5f} | ours {OUR_PUBLIC:.5f}", ""]
def fmt(r):
    m = re.search(r"0[.,]\s?9\d{3,4}", r["title"].replace(" ", ""))
    return f"- {r['ref']}  |  {r['title']}  |  votes {r['votes']}  |  last run {r['lastRun'][:16]}"
lines += [f"## New public notebooks ({len(new)})"] + [fmt(r) for r in new] + ["", f"## Updated notebooks ({len(changed)})"] + [fmt(r) for r in changed]
# flag titles that advertise a score above ours
flag = [r for r in new + changed if any(float(x.replace(',', '.').replace(' ', '')) > OUR_PUBLIC for x in re.findall(r"0[.,]\s?9\d{3,4}", r["title"]))]
lines += ["", f"## Titles advertising a score above ours ({len(flag)})"] + [fmt(r) for r in flag]
(W / "report.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

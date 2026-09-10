#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
while pgrep -f "scripts_phase1d.py|scripts_phase1e.py|run_chain3.sh|run_chain4.sh" >/dev/null; do sleep 15; done
.venv/bin/python scripts_phase1f.py > experiments/log_p1f.txt 2>&1

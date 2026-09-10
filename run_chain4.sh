#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
while pgrep -f "scripts_phase1c.py|scripts_phase1d.py|run_chain3.sh" >/dev/null; do sleep 15; done
.venv/bin/python scripts_phase1e.py > experiments/log_p1e.txt 2>&1

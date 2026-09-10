#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
while pgrep -f "scripts_phase1f.py|run_chain5.sh" >/dev/null; do sleep 15; done
.venv/bin/python scripts_phase1g.py > experiments/log_p1g.txt 2>&1

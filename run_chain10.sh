#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
while pgrep -f "scripts_phase1i.py|run_chain9.sh" >/dev/null; do sleep 15; done
.venv/bin/python scripts_phase2b.py > experiments/log_p2b.txt 2>&1

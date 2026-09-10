#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
while pgrep -f "scripts_phase1g.py|run_chain6.sh" >/dev/null; do sleep 15; done
.venv/bin/python scripts_xgb_bag.py > experiments/log_xgbbag.txt 2>&1

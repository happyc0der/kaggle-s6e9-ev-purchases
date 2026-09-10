#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
while pgrep -f run_chain.sh >/dev/null; do sleep 15; done
.venv/bin/python scripts_phase1c.py > experiments/log_p1c.txt 2>&1

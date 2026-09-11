#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
while pgrep -f "scripts_phase2b.py|run_chain10.sh" >/dev/null; do sleep 15; done
.venv/bin/python scripts_phase2c.py > experiments/log_p2c.txt 2>&1

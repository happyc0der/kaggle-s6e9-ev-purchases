#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
while pgrep -f "scripts_phase2c.py|scripts_phase3a.py|run_chain11.sh|run_chain12.sh" >/dev/null; do sleep 15; done
.venv/bin/python scripts_phase2d.py > experiments/log_p2d.txt 2>&1

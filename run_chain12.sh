#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
while pgrep -f "scripts_phase2b.py|scripts_phase2c.py|run_chain10.sh|run_chain11.sh" >/dev/null; do sleep 15; done
.venv/bin/python scripts_phase3a.py > experiments/log_p3a.txt 2>&1

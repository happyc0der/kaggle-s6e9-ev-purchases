#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
while pgrep -f "scripts_phase1h.py" >/dev/null; do sleep 10; done
.venv/bin/python scripts_phase1i.py > experiments/log_p1i.txt 2>&1

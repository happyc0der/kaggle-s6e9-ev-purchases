#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
while pgrep -f "scripts_phase1c.py|scripts_phase2.py" >/dev/null; do sleep 15; done
.venv/bin/python scripts_phase1d.py > experiments/log_p1d.txt 2>&1

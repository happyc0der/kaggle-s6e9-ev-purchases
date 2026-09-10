#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
while pgrep -f "scripts_phase0.py base" >/dev/null; do sleep 10; done
.venv/bin/python scripts_phase0.py fe > experiments/log_fe.txt 2>&1
.venv/bin/python scripts_phase1.py > experiments/log_p1.txt 2>&1

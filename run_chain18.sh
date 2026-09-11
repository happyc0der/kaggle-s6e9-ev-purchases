#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
while pgrep -f "scripts_pairte.py" >/dev/null; do sleep 15; done
.venv/bin/python scripts_naji_extras.py > experiments/log_naji_extras.txt 2>&1

#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
while pgrep -f "scripts_nte.py" >/dev/null; do sleep 15; done
.venv/bin/python scripts_foldseed.py 1 > experiments/log_foldseed1.txt 2>&1

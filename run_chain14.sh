#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
while pgrep -f "scripts_honest.py gbdt" >/dev/null; do sleep 15; done
.venv/bin/python scripts_allte.py all > experiments/log_allte.txt 2>&1

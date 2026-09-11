#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
while pgrep -f "scripts_final_more.py|run_chain20.sh" >/dev/null; do sleep 15; done
.venv/bin/python scripts_nte.py > experiments/log_nte.txt 2>&1

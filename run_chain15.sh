#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
while pgrep -f "scripts_allte.py|run_chain14.sh" >/dev/null; do sleep 15; done
.venv/bin/python scripts_rowkey.py > experiments/log_rowkey.txt 2>&1

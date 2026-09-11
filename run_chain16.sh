#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
while pgrep -f "scripts_rowkey.py|run_chain15.sh" >/dev/null; do sleep 15; done
.venv/bin/python scripts_shallow_sweep.py > experiments/log_shsweep.txt 2>&1

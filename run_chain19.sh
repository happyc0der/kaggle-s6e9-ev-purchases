#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
while pgrep -f "scripts_pairte.py|scripts_naji_extras.py|run_chain18.sh" >/dev/null; do sleep 15; done
.venv/bin/python scripts_cb_more.py sweep > experiments/log_cb_sweep.txt 2>&1
.venv/bin/python scripts_cb_more.py bag > experiments/log_cb_bag.txt 2>&1

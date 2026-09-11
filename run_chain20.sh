#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
while pgrep -f "scripts_cb_more.py|run_chain19.sh" >/dev/null; do sleep 15; done
.venv/bin/python scripts_final_more.py cb05 > experiments/log_cb05.txt 2>&1
.venv/bin/python scripts_final_more.py lgb01 > experiments/log_lgb01.txt 2>&1

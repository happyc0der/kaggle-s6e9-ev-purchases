#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
while pgrep -f "scripts_shallow_sweep.py|run_chain16.sh" >/dev/null; do sleep 15; done
.venv/bin/python scripts_final_gbdt.py check > experiments/log_final_check.txt 2>&1
.venv/bin/python scripts_final_gbdt.py lgb20 > experiments/log_final_lgb20.txt 2>&1
.venv/bin/python scripts_final_gbdt.py xgb20 > experiments/log_final_xgb20.txt 2>&1
.venv/bin/python scripts_final_gbdt.py cb20 > experiments/log_final_cb20.txt 2>&1

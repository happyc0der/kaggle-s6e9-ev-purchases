#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
while pgrep -f "scripts_nte.py|run_chain21.sh|scripts_final_more.py|run_chain20.sh" >/dev/null; do sleep 15; done
.venv/bin/python scripts_quant.py > experiments/log_quant.txt 2>&1

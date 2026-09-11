#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
.venv/bin/python scripts_phase1h.py > experiments/log_p1h.txt 2>&1
.venv/bin/python scripts_phase1i.py > experiments/log_p1i.txt 2>&1

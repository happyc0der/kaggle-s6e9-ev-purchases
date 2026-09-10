#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
.venv/bin/python scripts_phase1.py > experiments/log_p1.txt 2>&1
.venv/bin/python scripts_phase2.py > experiments/log_p2.txt 2>&1

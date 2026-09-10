#!/bin/zsh
cd /Users/happyc0der/Projects/kaggle-s6e9
for s in 1 2; do .venv/bin/python scripts_nn_full.py $s > experiments/log_nnE9_s$s.txt 2>&1; done

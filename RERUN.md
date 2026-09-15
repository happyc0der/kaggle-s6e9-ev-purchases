# Rerun playbook (for a fresh session)

Project: `~/Projects/kaggle-s6e9` — Kaggle playground-series-s6e9 (Will_Buy_EV, ROC AUC). Deadline 2026-09-30 23:59 UTC.
Environment: `.venv/bin/python` (Python 3.12; LightGBM/XGBoost/CatBoost/torch-MPS). Kaggle CLI needs the user's token from
`~/.zshrc`, so run Kaggle commands through `zsh -ic '.venv/bin/kaggle ...'`. Never print or copy the token.

## Current best (2026-09-11)
- Members (all 20-fold, shallow all-column-TE recipe): `lgb_final_bag10` (0.946361), `cb_final_bag8` (0.946383), `nn_final_bag15` (0.945916).
- Blend: `scripts_final.py final6 lgb_final_bag10 cb_final_bag8 nn_final_bag15` → OOF 0.946410 (hard edges) → public 0.94636.
- Public LB tracks our OOF within ±0.0001. Top-50 line ≈ 0.94644, leader 0.94672.
- Recipe: `scripts_final_gbdt.py` (feature set `full` + `temulti(3)` + `te_all((10,100))`, params `S_LGB`/`S_XGB`/`S_CB`), NN in `scripts_nn_final.py`.

## How to evaluate a new idea (paired, honest)
1. Implement as static features (`src/features/static.py`, or an `extra_fn(tr, te, static) -> DataFrame`) or as `TESpec`/`NTESpec` specs.
2. Run 10-fold with the shallow recipe and a paired reference:
   `run("sh_<idea>", full, te_specs=TE, static_version="v2", params=S, extra_fn=..., ref="lgb_allte_shallow")`
   (copy the header of `scripts_tebag.py`). Pooled OOF, per-fold AUC and paired DeLong z vs the reference are printed and appended to `experiments/results.csv`.
3. Accept only if the paired gain is ≥ +0.00008 with z ≥ 3 (the noise floor is ~0.00003). Everything in `experiments/results.csv` below that was rejected.
4. If accepted: add it to `scripts_final_gbdt.py` feature set, rerun stages `check`, `lgb20`, `xgb20`, `cb20` (≈2 h CPU), rerun `scripts_nn_final.py` seeds 0-9 (GPU, ≈70 min), rebuild bags (see the bag() helper in `scripts_final_more.py`), then `scripts_final.py final<N> <members>`.
5. Submit only if OOF improves by ≥ +0.00005 over the last submitted OOF (see `experiments/lb_log.csv`):
   `zsh -ic '.venv/bin/kaggle competitions submit -c playground-series-s6e9 -f submissions/<file> -m "<msg>"'`, then append to `experiments/lb_log.csv`.

## Watching for new public insight
- `zsh -ic '.venv/bin/python scripts_watch.py'` writes `experiments/watch/report.md` (new/updated public notebooks, LB top).
- To read a public notebook: `zsh -ic '.venv/bin/kaggle kernels pull <ref> -p notebooks/public/<name>'` then print code cells (see how notebooks/public/ were produced).
- The discussion tab is JS-rendered: open https://www.kaggle.com/competitions/playground-series-s6e9/discussion?sort=recent in the browser tool and compare against `experiments/watch/seen_discussion.json`.

## Already tested and dead (do not repeat)
Interaction TEs (exact, binned, pairs, triples), original-row similarity, density windows, digit residues/mod TEs, residual TE, neighbourhood TE (NTESpec),
quantized columns, orig-mean & frequency for all columns, EB smoothing, pseudo-labelling, native categoricals (LGBM/CatBoost), extra_trees, deeper trees,
extra hard-edge cells beyond the two we force (commute >= 83 km: +0.0000006; income==30000 & no-subsidy & (env==1 | anxiety M/H): +0.000000, measured 2026-09-15),
fold-partition bagging (already in the members via scripts_foldseed.py),
NN value embeddings / PLE / EMA / residue embeddings, bagged training-side TE, segment isotonic, segment blend weights, synthetic-ness feature, row-key TEs.

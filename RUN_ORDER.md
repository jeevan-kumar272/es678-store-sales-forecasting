# Run order for the final stage (do these in order, once the tuning has finished)

1. `python src/run_cv.py --models ridge,lasso,lightgbm --params artifacts/best_params.json --extra-seeds 2,3,4,5`
   Seeded comparison with tuned settings on all stores. Overwrites data/processed/cv_oof.pkl (needed by later steps).
2. `python src/select_model.py`            -> artifacts/model_choice.json (the pre-declared rule decides, not you)
3. FREEZE. Commit everything, then tag:  `git add -A` (check `git status`), `git commit -m "Freeze model choice and settings before the test"`,
   `git tag freeze-before-test`, `git push --tags`.
4. `python src/final.py`                   -> scores the test period ONCE (writes artifacts/test_lock.json, model.pkl, metrics.json)
5. `python src/intervals.py`               -> prediction intervals and their real coverage
6. `python src/failure_audit.py`           -> slices, worst stores, cold start, retransformation bias
7. `python src/ablation.py`                -> feature-group ablation, horizon sweep, Customers leakage demo (development period only)

Never delete test_lock.json to run step 4 again. If something is genuinely wrong, make a new commit, write the reason in decision_log.md, and tell your instructor.

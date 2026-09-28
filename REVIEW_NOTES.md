# Review notes – what changed and why

Status legend: **TESTED-LOCAL** = executed here (data → preprocess → train → evaluate → inference → drift, plus 5 pytest tests).
**UNTESTED-AWS** = written against verified v3 SDK signatures / boto3 APIs but never run against a real account.

| File | Status | Key changes and reasoning |
|---|---|---|
| data/generate_data.py | TESTED-LOCAL | CLI args; `default_rng`; stronger click/purchase signal (old 0.015/click ≈ noise → AUC gate meaningless); `--drift` mode for the drift lab. Local AUC ≈ 0.67, conversion ≈ 16 %. |
| src/preprocess.py | TESTED-LOCAL | Fixed categories (deterministic columns), schema validation, label last, removed commented-out duplicate. |
| src/train.py | TESTED-LOCAL | Real hyperparameters, SM_* env defaults, writes `feature_columns.json`, class weighting, metrics file. |
| src/evaluate.py | TESTED-LOCAL | **Bug fix:** model input is a tar.gz – now extracted. `binary_classification_metrics` schema for Registry UI + ConditionStep. |
| src/inference.py | TESTED-LOCAL | Column contract from artifact, multi-row CSV, JSON reorder, configurable threshold. **Response schema changed** to lists. |
| pipeline/pipeline.py | UNTESTED-AWS | Single copy (your file had two); + Evaluate + AUC gate + metrics on package; inference image (**bug fix**: was training image); inference.py bundled; per-execution S3 paths; CreateModel step removed; imports from `sagemaker.core.training.configs` (`sagemaker.train.configs` is a deprecated shim). |
| deployment/deploy.py | UNTESTED-AWS | boto3, from the newest Approved package; Data Capture; optional canary + alarm rollback on updates. |
| deployment/invoke_traffic.py | UNTESTED-AWS | Generates normal or drifted traffic. |
| monitoring/*.py | TESTED-LOCAL (PSI/baseline), UNTESTED-AWS (S3 capture parsing, CloudWatch, Lambda) | Baseline stores bin edges (old one stored mean/std – cannot compute PSI); PSI bins from baseline only; reads real captures; min-sample guard; Lambda skips if a run is already executing. |
| scripts/bootstrap.ps1, teardown.ps1 | UNTESTED-AWS | Idempotent rebuild / dependency-ordered teardown with a verification report (includes EFS check). |
| .github/workflows | UNTESTED | CI (tests) and ml-pipeline (OIDC → upsert + start). |
| requirements.txt | – | Moved to repo root. **Do not put a requirements.txt in `src/`** – the sklearn container pip-installs it and can replace its pinned scikit-learn. |

## First-run expectations (UNTESTED-AWS items)
1. `python pipeline/pipeline.py` may need one iteration (SDK is young). If `register()` complains the model is not built, re-add `ModelStep(step_args=model_builder.build())` before it.
2. After deploying, if invocation fails with a missing handler, inspect the package container env:
   `aws sagemaker describe-model-package --model-package-name <arn> --query "InferenceSpecification.Containers[0].Environment"` – you want `SAGEMAKER_PROGRAM=inference.py`.
3. Set `MinAuc` to 0.70 in one run to watch the gate's *else* branch (no registration) – a good UI exercise.

## GitHub OIDC (one-time, for ml-pipeline.yml)
IAM → Identity providers → add `token.actions.githubusercontent.com` (audience `sts.amazonaws.com`); create a role trusted by it, condition `sub` = `repo:<you>/<repo>:ref:refs/heads/main`; permissions: SageMaker + `iam:PassRole` on the execution role + S3 on the lab bucket; store its ARN as repo secret `AWS_ROLE_ARN`.

## Not yet built (later stages, by design)
shadow tests, serverless endpoint, multi-model endpoint, spot + distributed training, EventBridge/alarm wiring commands (in README_DETAILED §14–20), Evidently/MLflow monitoring layer.

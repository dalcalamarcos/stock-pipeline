# DECISIONS.md

A record of architectural and design decisions made across the pipeline — what was built, what was deliberately left out, and why. Intended as context for anyone reading the code, and as honest documentation of known limitations.

---

## Phase 1 — Ingestion

**DuckDB over Supabase/Postgres**
Zero ops overhead for local dev. Columnar storage is appropriate for time-series analytical workloads. The feature SQL is standard enough to be portable to Postgres later without rewriting queries. The tradeoff is the single-writer file lock, which constrains parallelism — see the Phase 2 note on sequential DAG tasks.

**LocalExecutor over CeleryExecutor**
A single-machine dev setup does not need distributed workers. LocalExecutor is simpler to start, easier to debug, and sufficient for weekly batch workloads at this data volume. The cost is that the scheduler and task executor share the same process — acceptable here, not acceptable in production at scale.

**Incremental loads with INSERT OR IGNORE**
Idempotency is guaranteed at two layers: the delta query fetches only new rows by date, and the PRIMARY KEY constraint silently discards duplicates at the database layer. Safe to re-run without side effects.

**Per-ticker error isolation**
One bad ticker (rate limit, missing symbol, API outage) does not abort the full pipeline run. Errors are collected and re-raised as a single aggregate exception at the end of the task. This surfaces all failures at once rather than hiding them behind an early exit.

---

## Phase 2 — Feature Engineering

**ASOF JOIN over pandas merge_asof for macro data**
ASOF JOIN runs inside DuckDB, keeping the macro join logic in the same SQL layer as the rest of the feature definitions. It avoids a round-trip to Python and is more readable at this query complexity. It is also more efficient for the data volume in question.

**RSI computed in Python, not SQL**
The gain/loss decomposition requires splitting `return_1d` into two conditional columns and applying separate rolling means. This is correct but verbose in SQL. The Python implementation is more readable and easier to validate. Mixing SQL and Python for feature computation is a documented tradeoff, not an oversight — the boundary is: pure window functions stay in SQL, conditional rolling logic goes to Python.

**NULL warmup rows retained in the features table**
Filtering warmup NULLs during feature build would silently discard rows and misrepresent the true NULL rate. Keeping them means the features table accurately reflects the warmup cost of each signal (14 rows for RSI, up to 50 rows for `ma_50`). The training query filters them explicitly. The feature table does not.

**Sequential DAG tasks**
DuckDB has a single-writer file lock. Parallel tasks writing to the same `.duckdb` file cause lock contention failures. Sequential execution is not a performance limitation at this data volume — it is the correct architecture for a single-file embedded database. If the pipeline scales to warrant parallelism, the database layer needs to change first (Postgres, or sharded DuckDB files).

---

## Phase 3 — Modelling

**Single cross-ticker model over per-ticker models**
Training one model on all tickers pooled. Per-ticker models with approximately 1,200 rows each are too small for walk-forward validation with a meaningful test set. A pooled model treats ticker identity as implicit context in the shared feature space rather than an explicit input. Revisit if the feature set expands to include ticker-specific signals, or if the dataset grows substantially.

**`rsi_14` null handling — drop, not impute**
Approximately 70 rows with NULL RSI (14 warmup rows × 5 tickers) are dropped at training time. Imputing with 50 (the neutral RSI midpoint) is defensible but introduces a signal where none exists. Dropping is cleaner for a first pass and the row count is negligible relative to the training set.

**Most recent year as test set**
The test set is the most recent calendar year of data, respecting temporal order and avoiding lookahead bias. The consequence is that the KS drift test is almost guaranteed to flag macro features — the test year is by definition the furthest from the centre of the training distribution. This is a known limitation of the current drift detection setup. See Phase 4 notes.

**LR accuracy near baseline (0.541 vs 0.543)**
The logistic regression model barely beats the majority-class baseline. This is surfaced honestly via the `/health` endpoint and not hidden. Short-term price direction prediction is a hard problem; a near-baseline result on a simple feature set is expected. The pipeline is built to serve and monitor what exists correctly. Model improvement is Phase 5 scope.

**Isolation forest anomaly rate (0.366 on test set)**
The high anomaly rate is driven by the April–May 2025 volatility window dominating the test period. Isolation Forest correctly identifies this period as anomalous relative to the training distribution. The `contamination` parameter was not tuned in Phase 3 or 4 — doing so would require a principled estimate of the true anomaly rate in the training data, which is not available. This is a Phase 5 concern.

---

## Phase 4 — API and Monitoring

**Model loading via `lifespan`, not per-request**
Models are loaded once at server startup into a module-level dict and held in memory for the process lifetime. Loading a `.pkl` file on every request would add unnecessary disk I/O and latency. The `lifespan` context manager (FastAPI's modern pattern) keeps startup side effects isolated from import time, which matters for testing and tooling.

**Per-request read-only DuckDB connections, no pool**
The API opens a fresh read-only DuckDB connection on each request and closes it immediately. A persistent connection pool would risk holding the write lock against Airflow's ingestion tasks. DuckDB's in-process architecture makes per-request connection overhead negligible. Read-only mode is enforced at the OS level — not just a convention.

**Graceful degradation over crash-on-missing-artifact**
If an artifact file is missing at startup, the relevant model key is set to `None` and the server boots. Endpoints return a `503` with a descriptive message rather than raising an unhandled exception. This means a partial deployment (e.g. scaler present but model missing) is diagnosable via `/health` without restarting the container.

**Covariate drift detection only — concept drift is not detected**
Evidently's `DataDriftPreset` uses the Kolmogorov-Smirnov test to compare feature distributions between the training reference window and the current window. This detects when the inputs have shifted — not when the relationship between inputs and outputs has changed. The latter (concept drift) requires a prediction outcomes feedback loop: storing predictions, joining against realized returns after the fact, and tracking rolling accuracy. This infrastructure is planned for Phase 5.

**Drift detection almost certainly flags macro features**
`fedfunds`, `vix`, and `t10y2y` are likely to show drift in any deployment period more than a few months after the training cutoff. This is a structural consequence of using the most recent year as the test set and running Evidently against current data. The drift signal is real but operationally noisy. Phase 5 will add segmented reference windows (per-year KS comparisons) to distinguish genuine regime shift from expected cyclical variation.

**Evidently pinned at `0.4.16`**
Evidently has a transitive dependency on scikit-learn with an upper bound of `<1.6.0` in versions prior to the patch for the `squared` argument removal. The pipeline uses `scikit-learn==1.8.0`, which sits above that boundary. Upgrading Evidently to `>=0.4.30` without verifying the upper bound was lifted would silently break the drift task. The pin stays until this is verified.

**`requirements.api.txt` split from `requirements.txt`**
The original monolithic `requirements.txt` caused a protobuf version conflict between `streamlit==1.33.0` and Airflow's `opentelemetry-proto` dependency. The API image does not need Streamlit, Airflow, or any of their transitive dependencies. Splitting into a lean `requirements.api.txt` resolved the conflict and produced a smaller image. The Streamlit dashboard will be containerised separately in Phase 5.

**DuckDB version aligned to `1.4.4` across all environments**
DuckDB's storage format is versioned and not guaranteed to be backward compatible across minor versions. The database file was created by host DuckDB `1.4.4`. Container environments were initially pinned to `0.10.1`, causing file format rejection at runtime. All environments (host, Airflow containers, API container) are now aligned to `1.4.4`.

---

## What was deliberately not built

**MLflow model registry**
Models are stored as joblib files in `models/artifacts/`. A proper registry (MLflow or similar) would provide versioning, experiment tracking, artifact lineage, and promotion workflows. This is the highest-priority infrastructure gap for a production system. Planned for a Phase 6 extension.

**Automated retraining trigger on drift**
The current setup detects drift and logs it. It does not automatically trigger a retraining run. Wiring Evidently's output into an Airflow sensor that triggers `stock_pipeline_train` when drift is detected requires a persistent drift state store and a threshold policy. Planned alongside the Phase 5 Bonferroni correction work.

**CI/CD for model pipeline**
No automated testing of model training, feature engineering, or API endpoints exists. A minimal CI setup would run the ingestion and feature tasks against a fixture DuckDB, assert feature shapes and NULL rates, and smoke-test the API against the resulting artifacts. GitHub Actions is the natural host. Not built in Phase 4 because the feedback loop infrastructure (predictions table, outcomes join) it would need to validate is Phase 5 scope.

**Deployment beyond Docker Compose**
The pipeline runs locally via Docker Compose. A production deployment would target ECS (Fargate) for the API and MWAA for the Airflow scheduler, with infrastructure managed by Terraform. This would involve a separate `terraform-stock-pipeline` repository with a VPC, ECS cluster, MWAA environment, S3 bucket for DAG storage, and ECR registry for the API image. The architecture is designed to be portable — DuckDB can be replaced with RDS or a mounted EFS volume, and the FastAPI container is already containerised and stateless. Not built in Phase 4; the Docker Compose setup is sufficient for demo and interview purposes.

**Bonferroni correction in drift detection**
With 11 features at p < 0.05, fewer than one false positive is expected by chance. Bonferroni would set the threshold to p < 0.0045. The correction is appropriate for a surveillance context (high precision on alerts preferred over high recall) but is not implemented in Phase 4 to keep the drift output consistent with Evidently's documented defaults. Planned as a configurable flag in `run_drift_report()` in Phase 5.

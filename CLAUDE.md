# CLAUDE.md — Predictive Evaluation Competition (extending `torch_measure`)

This file is the persistent project context. Claude Code reads it every session.
It is intentionally self-contained: it carries the **complete competition
specification** so no external handbook PDF is needed. Treat the rules in
"Hard constraints" and "Record-keeping mandate" as non-negotiable.

---

## 0. First session: EXPLORE before writing model code

Before proposing or implementing anything, read and summarize back to me:

- `README.md`, `CONTRIBUTING.md`, `pyproject.toml`, `.pre-commit-config.yaml`
- `src/torch_measure/`, focusing on:
  - `models` — the **Amortized IRT / amortized model class** (this is the
    centerpiece for this competition), plus Rasch, 2PL, 3PL, Many-Facet,
    Beta IRT, factor models
  - `data` — HuggingFace/HELM loaders, `ResponseMatrix`, masking strategies
  - `fitting` — MLE, EM, JML, Bayesian SVI estimation
  - `metrics` — `expected_calibration_error`, infit/outfit, tetrachoric
    correlation, Mokken, DIF
  - `cat` — `AdaptiveTester`, Fisher-information item selection
  - `viz` — heatmaps, ICCs, information plots
- `tests/` and `tutorials/` — learn the package's conventions and abstractions

Then report: (a) the exact public API of the amortized model and the data
loaders, (b) a proposed implementation plan, (c) open questions. **Wait for my
approval before implementing.** Verify two things explicitly: how the amortized
model ingests per-item embeddings, and what columns the data loader exposes so
we can build an item-disjoint cross-validation split.

---

## 1. Competition specification (complete)

### 1.1 Problem and terminology

- **Subject**: an AI system under evaluation (e.g., GPT-4, Claude, Llama-3);
  the "test-taker."
- **Item**: a single benchmark question/task; the "test question."
- **Response**: binary in `{0, 1}` — `1` if the subject answered the item
  correctly (pass), `0` otherwise (fail).
- Each response also carries a **benchmark** name and a **test condition**
  (e.g., zero-shot, chain-of-thought).
- **Goal**: predict the probability that a subject answers an item correctly
  under a given (benchmark, condition), **without ever running the subject on
  the item**. The four fields (subject, item, benchmark, condition) are exactly
  the inputs the submission receives at test time.

### 1.2 Cold-start structure (the crux)

- Every **test item is brand new**: it has zero observed responses in the
  training matrix, often from benchmarks that did not exist when training data
  was collected.
- Every **test subject already appears in the training matrix**. The cold-start
  dimension is the **item side only**.
- Consequence: the subject side is essentially a lookup; the **item-side
  text→parameter map is the only component that must generalize at test time.**

### 1.3 Metric

- Primary: **mean log-likelihood of the true labels under the predicted
  probabilities — higher is better** (the handbook calls this "negative
  log-loss, higher-is-better"; equivalently, minimize binary cross-entropy).
- This rewards **calibrated** probabilities, not just correct ranking.
- Confidently-wrong predictions are catastrophic: a prediction of exactly `0`
  or `1` that is wrong sends the log-likelihood to `-inf`. **Always clip the
  returned probability to roughly `[1e-6, 1 - 1e-6]`.**
- Secondary (debug only, also higher-is-better): **AUC-ROC**.

### 1.4 Data

- Public training data (HuggingFace): `aims-foundations/measurement-db`.
  Long format, fields: item description, subject description, benchmark name,
  test condition (if any), response (binary scalar; `1`=pass, `0`=fail).
- Curating additional training data is explicitly allowed and encouraged.
- The test set has the same format, is strictly held out, and is never
  released. At test time the program receives, for one hidden model-item pair
  at a time: test item text, benchmark name, condition, and subject
  description. Only the ground-truth label is hidden.
- **Data categories** group related benchmarks. They are used by the platform
  for sampling and adaptive-label budgets and are **not** passed as an input
  field.
- Each round scores predictions on a **random subset** of held-out entries
  (the subset changes every round, so repeated submissions cannot
  reverse-engineer test labels).
- The platform treats the same upstream item under different normalized test
  conditions as **different item variants**. Null/missing/blank conditions
  normalize to the literal string `"none"` in `input["condition"]`. At runtime
  the submission gets `(model_id, item_id)` pairs where `item_id` is already
  condition-specific; the condition is exposed separately via
  `input["condition"]`. Prediction CSV contract (platform-side):
  `model_id, item_id, predicted_probability`.
- Each submission is scored on the model-item pairs induced by **N = 5,000
  item variants** sampled from the hidden pool, stratified so each data
  category contributes an approximately equal share.
- **Submission cap: 50 scored submissions per team per calendar day (UTC).**

### 1.5 Adaptive labeling (optional but high-value)

- `labeling.py` may define `acquisition_function(input: dict) -> float`.
- `input` carries the same four string fields. The returned scalar indicates
  how desirable it is for this input's ground-truth label to be revealed —
  **only the ranking matters**, not the absolute value.
- The platform calls the function **once per candidate, with no view of the
  other candidates**. Any cross-candidate strategy (diversity, k-center) must
  accumulate state in **module-level variables**; the platform iterates the
  candidates in a single pass.
- It reveals the **top K labels per data category** (default **K = 5**; ties
  broken uniformly at random) and passes them to `predict()` as `labeled`: a
  `list[dict]` with the four fields plus a `"label"` key in `{0, 1}`.
- With `m` data categories in a round's sampled slice, the submission receives
  **at most `K · m`** labeled inputs before any `predict()` call. The list may
  be shorter (small categories) or empty (offline smoke tests). The same
  `labeled` list is passed on every `predict()` call within a round.
- Labels are resampled every round; they cannot be accumulated across rounds.
- If no `labeling.py` is shipped, K inputs per data category are chosen
  uniformly at random (a well-defined default).
- **Fallback:** if `acquisition_function()` raises, times out, or returns a
  non-finite value (NaN, ±inf) for any candidate, the platform discards that
  round's acquisition scores and falls back to random selection; `predict()`
  is still called, just with a random `labeled` sample.

### 1.6 Submission format and sandbox

- Upload a ZIP to Codabench containing `model.py` (required), and optionally
  `labeling.py`, `requirements.txt`, `models.txt`. Runtime package
  installation is organizer-controlled and **disabled by default** — keep
  dependencies minimal.
- `models.txt` declares HuggingFace model repositories the platform pre-fetches
  into the container before it starts.
- **Input dict — exactly these four string keys:** `"benchmark"`,
  `"condition"` (literal `"none"` when none applies), `"subject_content"`,
  `"item_content"`. A labeled input adds `"label"` (int in `{0,1}`).
- Required entry points:

  ```python
  # model.py (required)
  def predict(input: dict, labeled: list[dict] | None = None) -> float:
      # returns a native Python float in [0, 1]
      ...

  # labeling.py (optional)
  def acquisition_function(input: dict) -> float:
      # returns a finite, non-NaN native Python float; higher = more desired
      ...
  ```

- **Sandbox is network-isolated at test time.** No outbound calls from
  `predict()` or `acquisition_function()` to *any* third-party endpoint
  (hosted LLM APIs, remote embeddings, S3/GCS/Azure, remote DBs, webhooks,
  your own cloud functions). Every model used at test time must be baked into
  the ZIP or declared in `models.txt`.
- Each submission runs in a **fresh container destroyed when the round ends**:
  module-level state does **not** persist across rounds.
- Hosted APIs *may* be used **offline** during data curation and training.

### 1.7 Grading and timeline

- Teams of **1–3** students. Roster locks at the **end of Week 2**; no new
  teams in the final two weeks. Code/predictions must be independent per team.
- **Technical report — 50%** (Gradescope, NeurIPS 2025 LaTeX template, **4
  pages** main content; refs/appendix excluded but appendix not guaranteed to
  be read). Required sections:
  1. Problem formulation and method description
  2. Description of the training dataset used
  3. Experimental results and ablation studies on the validation set
  4. Analysis of failure modes / interesting patterns in the data
  Graded on clarity, depth of analysis, and insight.
- **Leaderboard — 50%.** Score = best (max over all rounds) negative log-loss
  on the private test set. **Beating the organizer baseline guarantees ≥ 80%
  of this component**; extra points by ranking among teams that beat it.
  Not beating it → partial credit by closeness. The baseline's private score
  is released **three weeks after the start**.
- Final code goes to **Gradescope** (`model.py`, `labeling.py`, `models.txt`,
  small checkpoints, `README.md`) and is the authoritative grading record;
  it must match the best Codabench submission.
- Submitted code undergoes **manual review** for rule compliance.

### 1.8 Reference baselines (from the handbook's worked examples)

- **Content-based NCF**: embed `subject_content` and `item_content` with a
  sentence transformer (e.g., `all-mpnet-base-v2`, 768-dim), concatenate, feed
  a small MLP head trained offline. Stronger variant: Platt-scale the score on
  the revealed `labeled` set at the first call of each round.
- **Local LLM-as-judge**: a local open-weights model (e.g.,
  `Qwen/Qwen2.5-7B-Instruct`) declared in `models.txt`; read the yes/no
  next-token log-probabilities rather than free-form text. Expensive over
  5,000 items — needs caching/batching/quantization. The handbook is explicit
  that the leaderboard rewards efficient content-based predictors over
  expensive inference; treat this as an ablation/ensemble member, not the
  primary model.
- **k-means diversity acquisition**: cluster embeddings of a fixed
  concatenation of the four fields offline (e.g., 64 clusters); at test time
  assign each candidate to its nearest centroid and prefer less-visited
  clusters. Online farthest-point sampling is a centroid-free variant.

---

## 2. The `torch_measure` package (reference)

Repo layout: `.github/workflows`, `data`, `docs`, `scripts`,
`src/torch_measure`, `tests`, `tutorials`, plus `pyproject.toml`,
`.pre-commit-config.yaml`, `.readthedocs.yaml`.

Modules:

| Module | Contents |
| --- | --- |
| `torch_measure.models` | IRT: Rasch, 2PL, 3PL, **Amortized**, Many-Facet; Beta IRT (BetaRasch, Beta2PL); factor models; rotation |
| `torch_measure.cat` | Computerized Adaptive Testing, Fisher-information selection |
| `torch_measure.fitting` | MLE, EM, JML, Bayesian SVI |
| `torch_measure.metrics` | tetrachoric correlation, Mokken, infit/outfit, `expected_calibration_error`, DIF |
| `torch_measure.data` | `ResponseMatrix`, masking strategies, HuggingFace/HELM loaders |
| `torch_measure.viz` | response heatmaps, ICCs, information plots |

Known API shape (verify against the actual source during exploration):

```python
from torch_measure.models import Rasch
from torch_measure.data import ResponseMatrix
from torch_measure.cat import AdaptiveTester
from torch_measure.metrics import expected_calibration_error

rm = ResponseMatrix(responses)                       # models x items
model = Rasch(n_subjects=rm.n_rows, n_items=rm.n_cols)
model.fit(rm.data, method="mle")
abilities    = model.ability                          # per-subject
difficulties = model.difficulty                       # per-item
probs        = model.predict()                        # P(correct)
tester = AdaptiveTester(model, strategy="fisher")
est_ability = tester.run(responses=new_resp, budget=50)
```

The **Amortized** model "predicts item parameters from embeddings without
per-item calibration" — this is exactly Stage 2a below. Confirm its constructor
and forward signature before building on it.

Optional extras: `pip install -e ".[all]"` (data loaders + Bayesian + viz).

---

## 3. Required architecture — the PGE pipeline

Build from `torch_measure` components, not from scratch:

1. **Stage 1** — fit an IRT/factor model on the training response matrix to get
   per-subject ability and per-item parameters. Prefer a model whose logistic
   link yields naturally calibrated probabilities.
2. **Stage 2b** — for known test subjects, **look up** the fitted ability. Do
   not learn a metadata→ability map first; the lookup is a strong baseline.
3. **Stage 2a (the core lever)** — use the amortized model to map item text
   (sentence embeddings of `item_content` + encoded `benchmark` +
   `condition`) → item parameters, so new items get parameters with no
   per-item calibration.
4. **Calibration** — at the first `predict()` call each round, fit a
   Platt/temperature scaler (per data category or per benchmark) on the
   `labeled` dicts, cache it at module scope, and pass every prediction
   through it before returning.
5. **`labeling.py`** — start from the package's Fisher-information / CAT
   selection; also implement a k-means diversity sampler so the two can be
   ablated. Remember the acquisition function sees one candidate at a time.

---

## 4. Validation discipline (do not get this wrong)

Local cross-validation **must** simulate item cold-start:

- **Split by item** — hold out *entire items / entire benchmarks*. A random
  row split leaks items across train/val and massively overestimates the
  score, then collapses on the real leaderboard.
- **Stratify** the held-out set by data category to mirror the platform's
  stratified sampling.
- The leaderboard resamples every round and is noisy. **Trust local
  item-disjoint CV for model selection**; use the 50/day submission budget to
  confirm direction, not to tune.

---

## 5. Hard constraints (non-negotiable, enforce every session)

1. No network calls of any kind inside `predict()` or
   `acquisition_function()`.
2. Load all models/encoders at **module scope** (top of `model.py`), never
   inside `predict()`. `predict()` is called once per item, thousands of
   times per round.
3. `predict()` returns a **native Python `float`** in `[0, 1]` — wrap with
   `float(...)`; numpy/torch scalars fail serialization. Clip to
   `[1e-6, 1 - 1e-6]`.
4. `acquisition_function()` returns a **finite, non-NaN** native float.
5. Large encoders → `models.txt`; small fitted state (adapters, lookup
   tables, centroids) → baked into the ZIP.
6. Keep dependencies minimal; runtime pip installs are disabled by default.
7. Public fork: never commit secrets (HF tokens, Modal coupon, `.env`).

---

## 6. Repo workflow and conventions (instructor's requirements)

- Work on the `my-project` branch.
- New models/algorithms go **inside the appropriate `src/torch_measure`
  module**, following existing abstractions and coding style.
- Analysis/empirical studies go in `tutorials/` as **clean, reproducible
  notebooks**.
- Every substantial contribution gets **`pytest` tests in `tests/`** —
  verify output shapes, numerical stability, edge cases, and expected
  statistical behavior. Tests must pass via
  `pytest tests/ -m "not slow and not gpu"` before any commit.
- Keep commits small and well-described. Maintain documentation and
  reproducible run instructions.
- Mirror the final submission entry points under `submission/`
  (`model.py`, `labeling.py`, `requirements.txt`, `models.txt`).

---

## 7. RECORD-KEEPING MANDATE (so the report writes itself)

The technical report is 50% of the grade. **Maintaining the records below is
a required part of every task — not optional, not deferred to the end.** Keep
everything under `experiments/`. After *any* of the following, append to the
relevant file **before moving on**: training/fitting a model, changing the CV
split, running an ablation, a data-curation step, a leaderboard submission,
or discovering a bug, failure mode, or surprising pattern.

Create and maintain these files:

### `experiments/LOG.md` — append-only experiment log → feeds Report §3
One entry per experiment. Never edit or delete past entries; the history of
what was tried is itself report material. Template:

```
## [YYYY-MM-DD HH:MM] <short title>
- Commit: <git short hash>  | Branch: my-project
- Goal / hypothesis:
- Setup: model + key hyperparameters; CV split (item-disjoint? strata?);
  dataset slice used
- Result: local mean log-likelihood = <...>, AUC = <...>
  (and Codabench round score if submitted = <...>)
- Delta vs previous best: <what changed and the effect>
- Decision / next step:
```

### `experiments/DECISIONS.md` — methodological choices → feeds Report §1
For each significant choice (IRT variant, embedding model, calibration scheme,
acquisition strategy): the decision, the rationale, the alternatives
considered, and what evidence settled it.

### `experiments/DATASET_NOTES.md` — data provenance → feeds Report §2
EDA findings (subject/item/benchmark/condition counts, class balance, matrix
density), any external data curated and exactly how, preprocessing steps, and
the train/val split definition with seeds.

### `experiments/FINDINGS.md` — failure modes & patterns → feeds Report §4
Where the model breaks (which benchmarks/conditions/subject families), error
analysis, calibration diagnostics (ECE per category), and any surprising or
counter-intuitive patterns, each with a one-line takeaway.

### `experiments/ABLATIONS.md` — running results table → feeds Report §3
A maintained table: configuration × {local log-likelihood, AUC, beats
baseline?}. Add a row whenever a component is toggled (with/without adaptive
labels; amortized vs. ability-lookup; embedding choice; IRT variant;
calibration on/off; acquisition strategy).

### `experiments/SUBMISSIONS.md` — leaderboard ledger
Every Codabench submission: date, commit hash, what was submitted, round
score, whether it beat the baseline, and which `experiments/LOG.md` entry it
corresponds to. The Gradescope final must match the best row here.

### `experiments/REPORT_OUTLINE.md` — living draft of the NeurIPS paper
Maintain a 4-section skeleton (matching Report §1–§4). After each meaningful
result, add 1–3 sentences and reference the figure/table that supports it.
Generate report figures (ICCs, calibration plots, ablation bars) into
`experiments/figures/` as they become available, using `torch_measure.viz`
where possible. By the deadline this file should be ~80% of the paper.

**Rule of thumb:** if an experiment was run but not logged, it did not happen.
Every result that could appear in the report must be reproducible from a
logged commit + config.

---

## 8. Definition of done (per deliverable)

- Model + acquisition implemented inside `torch_measure` modules, with tests
  passing under `pytest tests/ -m "not slow and not gpu"`.
- `submission/` ZIP contents satisfy every Hard Constraint and beat the
  baseline on item-disjoint local CV.
- A reproducible tutorial notebook with the cold-start CV results and the
  full ablation set.
- `experiments/` records complete and current; `REPORT_OUTLINE.md` reflects
  the latest state.

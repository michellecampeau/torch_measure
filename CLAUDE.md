# CLAUDE.md — Predictive Evaluation Competition (extending `torch_measure`)

This file is the persistent project context. Claude Code reads it every session.
It is intentionally self-contained: it carries the **complete competition
specification** so no external handbook PDF is needed. Treat the rules in
"Hard constraints" and "Record-keeping mandate" as non-negotiable.

> Several specifics below were verified against the live HuggingFace dataset
> viewer and the Codabench data/evaluation pages, and **override the handbook's
> idealized description and worked-example code**. Where they conflict, this
> file wins. The "Known gotchas" subsection lists the traps explicitly.

---

## 0. First session: EXPLORE and VERIFY before writing model code

Before proposing or implementing anything, do this and report back:

1. Read `README.md`, `CONTRIBUTING.md`, `pyproject.toml`,
   `.pre-commit-config.yaml`.
2. Read `src/torch_measure/`, focusing on:
   - `models` — the **Amortized IRT / amortized model class** (centerpiece),
     plus Rasch, 2PL, 3PL, Many-Facet, Beta IRT, factor models
   - `data` and any `datasets` submodule — loaders, `ResponseMatrix`,
     masking strategies, `load()` / `list_datasets()`
   - `fitting` — MLE, EM, JML, Bayesian SVI
   - `metrics` — `expected_calibration_error`, infit/outfit, tetrachoric,
     Mokken, DIF
   - `cat` — `AdaptiveTester`, Fisher-information selection
   - `viz` — heatmaps, ICCs, information plots
3. Read `tests/` and `tutorials/` for conventions/abstractions.
4. **Day-one data verification (critical — the handbook code is wrong here):**
   - Inspect `aims-foundations/measurement-db` on HuggingFace: confirm the
     actual split name(s) and the exact column list. (As observed: a single
     split named `test`, ~1.54M rows, ~254 subjects, columns
     `subject_id, item_id, benchmark_id, trial, test_condition, response,
     correct_answer, trace` — i.e. IDs + a numeric `response`, **not**
     `subject_content`/`item_content`/`label`.)
   - Determine the **authoritative way to obtain `(subject_content,
     item_content, benchmark, condition, binary_label)` tuples** for training.
     Most likely: join IDs to content via `torch_measure.datasets.load(...)`
     / `list_datasets()` (returns a `ResponseMatrix` with `item_contents`,
     `subject_ids`, `subject_metadata`), and/or per-benchmark
     `item_content.csv`. Verify exact repo/org names (the curation pipeline
     references `aims-foundation/torch-measure-data` — note possible org
     spelling difference vs `aims-foundations`); do not hard-code until
     confirmed.
   - Record the resolved loading path, the label/response semantics per
     benchmark, and the split definition in `experiments/DATASET_NOTES.md`.

Then report: (a) exact public API of the amortized model and the data
loaders, (b) the verified data-loading path, (c) a proposed implementation
plan, (d) open questions. **Wait for my approval before implementing.**

---

## 1. Competition specification (complete)

### 1.1 Problem and terminology

- **Subject**: an AI system under evaluation; the "test-taker."
- **Item**: a single benchmark question/task; the "test question."
- **Response / label**: hidden scoring labels are strictly binary in `{0,1}`
  — `1` = subject answered correctly, `0` = otherwise. (Public training
  `response` values may NOT be binary; see 1.4.)
- Each response also carries a **benchmark** and a **test condition**.
- **Goal**: predict P(subject answers item correctly) under a given
  (benchmark, condition), **without running the subject on the item**.

### 1.2 Cold-start structure (the crux — benchmark-level)

- **Test items come from benchmarks that are entirely absent from the public
  training data** — the training matrix has *no rows at all* for those
  benchmarks (specific identities undisclosed).
- **Test subjects are the same subjects present in the training matrix.**
- So: the subject side is a lookup; the **item-side text→parameter map is the
  only component that must generalize, and it must generalize to whole unseen
  benchmarks**, not merely unseen items within seen benchmarks.

### 1.3 Metric

- Primary: **mean log-likelihood of the true labels under predicted
  probabilities — higher is better**, a negative number bounded above by 0
  ("negative log-loss, higher-is-better"; equivalently minimize BCE).
- Rewards **calibration**; punishes overconfident wrong answers.
- A wrong prediction of exactly `0` or `1` sends the score to `-inf`.
  **Always clip the returned probability to ~`[1e-6, 1 - 1e-6]`.**
- Secondary (debug only): **AUC-ROC** (higher better).

### 1.4 Data — actual schema and known gotchas

Public training data: HuggingFace `aims-foundations/measurement-db`, long
format. Curate additional public data/metadata if useful.

**Observed real schema (verify on day one):** a single split named **`test`**
(~1.54M rows; ~254 unique subjects). Columns: `subject_id`, `item_id`,
`benchmark_id`, `trial`, `test_condition`, `response` (float, can be null),
`correct_answer`, `trace`. The full dataset spans many benchmarks (the
curation pipeline registers ~146; ~92 with per-item matrices).

**Known gotchas (these break the handbook's sample code):**

- The handbook's `load_dataset("aims-foundations/measurement-db",
  split="train")` is wrong: the public split is named **`test`**, not
  `train`. There is no `train` split.
- The label-bearing column is **`response`** (numeric), not `label`.
- There are **no `subject_content` / `item_content` text columns** in the
  parquet view — only `subject_id` / `item_id`. **Item and subject text must
  be joined in** (via `torch_measure.datasets` loaders / `item_content.csv`).
  Resolve the canonical path on day one.
- `response` is binary for hidden scoring, but public training `response`
  may be **binary, Likert-style, fractional, or other**. Inspect and
  **binarize per benchmark** before training; document the rule used.
- The public export keeps only the **smallest `trial`** within each
  `(subject_id, item_id, test_condition)` group; `test_condition` is
  normalized and different conditions are **separate item variants**.
  **Preserve `test_condition`** when building training/validation data.
- Runtime exposes **no stable IDs**: `predict()` receives content text only.
  `runtime model_id` corresponds to public `subject_id`, but you must match
  by visible content. `subject_content` **begins with a `Name:` line** — use
  it to map a runtime subject back to a trained `subject_id`. Match `labeled`
  examples by visible content fields too.

### 1.5 Test protocol

- Test set: same field structure, strictly held out, never released. The
  platform passes one curated input dict per hidden subject-item pair to
  `predict()`. Only the label is hidden.
- Each submission gets a fresh deterministic sample: **5,000 hidden items**,
  stratified across **data categories** so each contributes ~equally.
  Data categories drive sampling and adaptive-label budgets and are **not**
  an input field. The subset changes per round (no label reverse-engineering).
- **Cap: 50 scored submissions per team per calendar day (UTC).**

### 1.6 Adaptive labeling

- Optional `labeling.py`: `acquisition_function(input: dict) -> float`.
- Called **once per candidate, no view of other candidates** — cross-candidate
  strategies must accumulate **module-level** state across the single pass.
- Only the **ranking** of returned scores matters. Platform reveals the
  **top K = 5 labels per data category** (ties random), passed to `predict()`
  as `labeled: list[dict]` (four fields + `"label"` in `{0,1}`).
- With `m` categories in the slice, ≤ `K·m` labeled inputs arrive before any
  `predict()` call; the same list is reused for every `predict()` in the
  round. May be shorter, or empty in offline smoke tests. Resampled each
  round; not accumulable across rounds.
- No `labeling.py` → K random labels per category.
- **Fallback:** any exception, timeout, or non-finite return for *any*
  candidate → all acquisition scores for that whole run discarded, random
  selection used; `predict()` still runs with the random `labeled`.

### 1.7 Submission format, sandbox, and infrastructure limits

- ZIP at archive root: `model.py` (required); optional `labeling.py`,
  `models.txt`, `requirements.txt`, plus any other files. Build it from
  inside the submission dir: `cd my_submission && zip -r ../my_submission.zip .`
- `requirements.txt` is accepted but runtime installs are organizer-gated and
  **disabled by default** — keep deps minimal.
- `models.txt`: HF repos pre-fetched before code runs. **At most 5 repos.**
  **≤ 300B params per single repo** (per-repo, not summed; >300B rejected).
  Routing by the largest declared model: ≤70B→B200, ≤140B→B200:2,
  ≤300B→B200:4; **8-hour** tier wall-clock that **includes module-import
  setup time**. Per-repo download ≤1000 GB; combined ≤1024 GB.
  **`trust_remote_code` is disabled by default.**
- **Input dict — exactly four string keys:** `"benchmark"`, `"condition"`
  (`"none"` if N/A), `"subject_content"` (starts with `Name:`),
  `"item_content"`. Labeled input adds `"label"` (int `{0,1}`).
- Entry points:

  ```python
  def predict(input: dict, labeled: list[dict] | None = None) -> float: ...
  def acquisition_function(input: dict) -> float: ...   # optional
  ```

- `predict()` must return a **finite native Python float in [0,1]**. NaN,
  inf, tensors, strings, out-of-range, or any raised exception **fail the
  submission**. Module-level import/setup failure fails the submission
  **before** `predict()` is ever called.
- **No runtime training hook.** Train offline; load small fitted state /
  templates / pre-fetched models at module import.
- **Network-isolated at test time.** No outbound calls from `predict()` or
  `acquisition_function()` to any third-party endpoint. Everything must be in
  the ZIP or declared in `models.txt`.
- Fresh container per round; **module-level state does not persist across
  rounds**. Hosted APIs may be used **offline** only.
- Scoring matches predictions by `(model_id, item_id)`. Missing, empty, or
  duplicate rows fail. Organizer per-subject scoring weights may apply
  (default 1.0).
- **Hosted logs hide stdout/stderr and the pass/fail reason.** Local smoke
  tests are the only debugging path.
- Starter-kit local checks (run before every upload):
  `python tools/check_submission_zip.py my_submission.zip` and
  `python tools/run_smoke_test.py my_submission/`.

### 1.8 Grading and timeline

- Teams of **1–3**. Roster locks **end of Week 2**; no new teams in the final
  two weeks. Code/predictions independent per team.
- **Technical report — 50%** (Gradescope, NeurIPS 2025 LaTeX, **4 pages**
  main; refs/appendix excluded, appendix not guaranteed read). Sections:
  (1) problem formulation & method, (2) training dataset description,
  (3) results & ablations on validation, (4) failure modes / patterns.
  Graded on clarity, depth, insight.
- **Leaderboard — 50%.** Best (max over rounds) negative log-loss on the
  private set. **Beating the organizer baseline guarantees ≥ 80%** of this
  component; extra points by rank among teams that beat it. Baseline private
  score released **three weeks after start**.
- Final code to **Gradescope** is authoritative and must match the best
  Codabench run. Code undergoes **manual review**.

### 1.9 Reference baselines (handbook worked examples)

- **Content-based NCF**: sentence-transformer embeddings of subject + item
  → small offline-trained MLP head; optional Platt scaling on `labeled`.
- **Local LLM-as-judge** (e.g. `Qwen/Qwen2.5-7B-Instruct` via `models.txt`):
  read yes/no next-token log-probs. Expensive over 5,000 items; treat as
  ablation/ensemble, not the primary model.
- **k-means diversity acquisition**: cluster fixed concatenation of the four
  fields offline; prefer less-visited clusters; farthest-point variant is
  centroid-free.

---

## 2. The `torch_measure` package (reference)

Repo layout: `.github/workflows`, `data`, `docs`, `scripts`,
`src/torch_measure`, `tests`, `tutorials`, plus `pyproject.toml`,
`.pre-commit-config.yaml`, `.readthedocs.yaml`.

| Module | Contents |
| --- | --- |
| `torch_measure.models` | Rasch, 2PL, 3PL, **Amortized**, Many-Facet; Beta IRT; factor models; rotation |
| `torch_measure.cat` | Adaptive testing, Fisher-information selection |
| `torch_measure.fitting` | MLE, EM, JML, Bayesian SVI |
| `torch_measure.metrics` | tetrachoric, Mokken, infit/outfit, `expected_calibration_error`, DIF |
| `torch_measure.data` / `.datasets` | `ResponseMatrix`, masking; `load()`, `list_datasets()` (per-benchmark matrices with `item_contents`, `subject_ids`, `subject_metadata`) |
| `torch_measure.viz` | heatmaps, ICCs, information plots |

Indicative API (verify against source):

```python
from torch_measure.models import Rasch
from torch_measure.data import ResponseMatrix
from torch_measure.cat import AdaptiveTester
from torch_measure.metrics import expected_calibration_error
# likely: from torch_measure.datasets import load, list_datasets
rm = ResponseMatrix(responses)                  # models x items
model = Rasch(n_subjects=rm.n_rows, n_items=rm.n_cols)
model.fit(rm.data, method="mle")
model.ability; model.difficulty; model.predict()
AdaptiveTester(model, strategy="fisher").run(responses=r, budget=50)
```

The **Amortized** model "predicts item parameters from embeddings without
per-item calibration" — this is Stage 2a below. Optional extras:
`pip install -e ".[all]"`.

---

## 3. Required architecture — the PGE pipeline

1. **Stage 1** — fit an IRT/factor model on the training response matrix for
   per-subject ability and per-item parameters; prefer a logistic-link model
   for calibrated probabilities.
2. **Stage 2b** — for known test subjects, **look up** the fitted ability.
   Because runtime has no IDs, build the lookup keyed on the **model name
   parsed from the `Name:` line** of `subject_content` (normalize; have a
   fuzzy/content fallback). Do not learn a metadata→ability map first.
3. **Stage 2a (core lever)** — use the amortized model to map item text
   (embeddings of `item_content` + encoded `benchmark` + `condition`) → item
   parameters, so brand-new items/benchmarks get parameters with no per-item
   calibration.
4. **Calibration** — at the first `predict()` of each round, fit a
   Platt/temperature scaler on `labeled` (match by content fields), cache at
   module scope, pass all predictions through it. Clip to `[1e-6, 1-1e-6]`.
5. **`labeling.py`** — start from Fisher-information/CAT selection; also a
   k-means diversity sampler for ablation. Accumulate cross-candidate state at
   module scope.

---

## 4. Validation discipline (do not get this wrong)

The real regime is **whole-benchmark cold-start**, so local CV must be
**leave-whole-benchmarks-out**: hold out *every row of entire benchmarks*,
never a random row split and never item-level holdout inside a seen
benchmark — either leaks the regime and badly overestimates the score.

- Stratify the held-out benchmarks to mimic the platform's category-balanced
  sampling. Preserve `test_condition` (separate variants).
- The leaderboard resamples and is noisy. **Trust leave-benchmarks-out local
  CV for model selection**; use the 50/day budget to confirm direction only.

---

## 5. Hard constraints (non-negotiable, enforce every session)

1. No network calls anywhere inside `predict()` / `acquisition_function()`.
2. Load all models/encoders at **module scope**, never inside `predict()`.
3. `predict()` returns a **finite native Python float**, clipped to
   `[1e-6, 1-1e-6]`; wrap with `float(...)`.
4. `acquisition_function()` returns a **finite, non-NaN** native float.
5. Module import must not fail and must stay within the wall-clock budget
   (setup time counts).
6. `models.txt`: ≤5 repos, ≤300B/repo, no `trust_remote_code` reliance.
7. Keep deps minimal (runtime installs disabled by default).
8. Public fork: never commit secrets (HF tokens, Modal coupon, `.env`).
9. Always run `tools/check_submission_zip.py` + `tools/run_smoke_test.py`
   before any Codabench upload (hosted logs are blind).

---

## 6. Repo workflow and conventions (instructor's requirements)

- Work on the `my-project` branch.
- New models/algorithms go **inside the appropriate `src/torch_measure`
  module**, following existing abstractions/style.
- Analysis/empirical studies go in `tutorials/` as clean, reproducible
  notebooks.
- Every substantial contribution gets **`pytest` tests in `tests/`**
  (shapes, numerical stability, edge cases, expected statistical behavior);
  `pytest tests/ -m "not slow and not gpu"` must pass before any commit.
- Small, well-described commits; maintained docs and reproducible run
  instructions.
- Mirror final entry points under `submission/` (`model.py`, `labeling.py`,
  `requirements.txt`, `models.txt`).

---

## 7. RECORD-KEEPING MANDATE (so the report writes itself)

The report is 50% of the grade. **Maintaining these records is a required
part of every task — not deferred to the end.** Keep everything under
`experiments/`. After *any* of: training/fitting a model, changing the CV
split, an ablation, a data-curation/binarization step, a leaderboard
submission, or discovering a bug/failure/surprising pattern — append to the
relevant file **before moving on**.

### `experiments/LOG.md` — append-only experiment log → Report §3
Never edit/delete past entries. Template:

```
## [YYYY-MM-DD HH:MM] <title>
- Commit: <hash> | Branch: my-project
- Goal / hypothesis:
- Setup: model + hyperparameters; CV split (benchmark-disjoint? strata?);
  data slice + response-binarization rule used
- Result: local mean log-likelihood = <...>, AUC = <...>
  (Codabench round score if submitted = <...>)
- Delta vs previous best:
- Decision / next step:
```

### `experiments/DECISIONS.md` — methodological choices → Report §1
Each significant choice: decision, rationale, alternatives, settling evidence.

### `experiments/DATASET_NOTES.md` — data provenance → Report §2
Resolved data-loading path (split name, ID→content join, exact loader);
per-benchmark response semantics + binarization rule; trial/`test_condition`
handling; EDA (subject/item/benchmark counts, class balance, density); any
curated external data and how; the leave-benchmarks-out split + seeds.

### `experiments/FINDINGS.md` — failure modes & patterns → Report §4
Where it breaks (which benchmarks/conditions/subject families), error and
calibration analysis (ECE per category), surprising patterns, each with a
one-line takeaway.

### `experiments/ABLATIONS.md` — running results table → Report §3
Config × {local log-likelihood, AUC, beats baseline?}. Add a row whenever a
component is toggled (adaptive labels on/off; amortized vs ability-lookup;
embedding choice; IRT variant; calibration on/off; acquisition strategy).

### `experiments/SUBMISSIONS.md` — leaderboard ledger
Every Codabench submission: date, commit, what was submitted, round score,
beat baseline?, and the corresponding `LOG.md` entry. Gradescope final must
match the best row.

### `experiments/REPORT_OUTLINE.md` — living NeurIPS draft
A 4-section skeleton (Report §1–§4). After each meaningful result, add 1–3
sentences referencing the supporting figure/table. Generate figures into
`experiments/figures/` (use `torch_measure.viz` where possible). Target ~80%
of the paper assembled by the deadline.

**Rule of thumb:** an unlogged experiment did not happen. Every result that
could appear in the report must be reproducible from a logged commit + config.

---

## 8. Definition of done (per deliverable)

- Model + acquisition implemented inside `torch_measure` modules; tests pass
  under `pytest tests/ -m "not slow and not gpu"`.
- `submission/` satisfies every Hard Constraint and **passes
  `tools/check_submission_zip.py` and `tools/run_smoke_test.py` locally**,
  and beats the baseline on leave-benchmarks-out local CV.
- A reproducible tutorial notebook with the cold-start CV results and the
  full ablation set.
- `experiments/` records complete and current; `REPORT_OUTLINE.md` reflects
  the latest state.

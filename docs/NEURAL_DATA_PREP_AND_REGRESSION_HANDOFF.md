# Neural Data Prep And Regression Handoff

This document is for an LLM or engineer helping with ShakeyBot neural-eval data preparation. It describes the current data contracts, preprocessing, training, export, engine integration, evaluation flow, and the known regressions we need to explain.

The main goal now is not "make more data" blindly. The goal is to identify why larger/newer datasets and larger networks sometimes produce worse tournament strength despite apparently reasonable validation or benchmark behavior.

## Current Baseline

Current strongest practical line:

- Feature family: `Simple768`
- Model shape: usually `768 -> 192 -> 1`
- Engine backend: `neural_quant_accum`
- Champion model family: `simple768_h192_e20_66m_aug2_quant`
- Engine eval contract: board in, white-POV centipawn score out
- Search consumes this through the normal static-eval path, then converts to side-to-move POV and adds tempo.

Current default in engine config:

- `EvalBackend=neural_quant_accum`
- `NeuralModelPath=models/simple768_model_h192_e20_aug2_quant.txt`
- `NeuralEndgameFallback=true`
- `NeuralEndgameMaterialLimit=10`
- `NeuralPawnOnlyFallback=false`
- `TempoBonus=6`

Important: the neural backend is not only used at leaf eval. It feeds static eval, qsearch stand-pat, pruning gates, TT static eval storage, move selection side effects, and search stability.

## Golden Contracts

These must stay true unless doing a deliberate controlled experiment.

1. Raw CSV rows are `board,cp`.
2. Parse by splitting on the final comma. FEN contains spaces and must not be split as CSV columns naively.
3. Raw `cp` is actually pawn units, despite the column name.
4. Public model output is white-POV.
5. Engine converts white-POV to side-to-move POV at `evaluate_for_side_to_move_with_config()`.
6. Tempo is added outside the neural model.
7. KPK exact probe runs before backend dispatch.
8. Low-material HCE fallback may override neural eval in engine play.
9. Training validation MAE alone does not decide promotion. Tournament strength decides.
10. All comparisons must control for model, feature set, label scale, data mix, fallback settings, and time control.

## Data Sources

Primary public training data Drive folder:

```text
https://drive.google.com/drive/folders/1lXw-fRzKVZysfv4tsGJr0yK_RK-TGtha
```

Model upload Drive folder:

```text
https://drive.google.com/drive/folders/1FfJkqKZQvP46fT6U_IGkkEClATRkcuoB
```

Common source file patterns:

- Original CSVs: `stockfish_evals_batch*.csv`
- Large generated label archives: `stockfish_labels_depth1*`
- Split shard archives: `shards_simple768_*_split.zip.001`, `.002`, etc.
- HalfKP split shard archives: `shards_halfkp_*_split.zip.001`, `.002`, etc.

Known large datasets:

- 18M source rows:
  - original plus extra CSVs
  - color flip and horizontal flip augmentation produced about 56.6M Simple768 rows in one run
- 66M source rows:
  - produced about 207.3M augmented rows
  - Simple768 shard shape: `indices (200000, 32)`, `mask (200000, 32)`, `target (200000,)`
  - HalfKP shard shape: `indices (200000, 2, 30)`, `mask (200000, 2, 30)`, `stm (200000,)`, `target (200000,)`
- "200M" current source mix:
  - actual source rows from latest Colab run: `141115942`
  - augmented rows: `499903724`
  - train rows: `475093342`
  - validation rows: `24810382` before latest cleanup, `21540522` in final clip30 training run
  - includes old mixed data plus newly generated lower-material/endgame-skewed data

## Raw Label Scale

The raw label is in pawns.

Current label modes in `training/simple768_train.py` and `training/halfkp_train.py`:

- `cp_clip`
  - `target_cp = clamp(raw_cp * 100, -clip_cp, +clip_cp)`
  - `target_norm = target_cp / clip_cp`
  - exported `output_scale_cp = clip_cp`
- `pawn_linear`
  - clamps raw pawn eval to `[-label_pawn_scale, +label_pawn_scale]`
  - then divides by `label_pawn_scale`
  - previously tested as `pawn_linear30`; tournament result was worse

Critical recent bug/trap:

- New generated data was already clipped to `[-30, +30]` pawns.
- Training it with old `cp_clip` default `--clip-cp 1200` means anything beyond `+12` pawns saturates to `+1.0`.
- Correct clip30 setup is:

```text
--label-mode cp_clip --clip-cp 3000
```

Then:

- `+30 pawns -> +3000 cp -> +1.0`
- `-30 pawns -> -3000 cp -> -1.0`
- exported `output_scale_cp = 3000`

Do not compare validation MAE across datasets with different target distributions as if it were the same task.

## Simple768 Feature Encoding

Script: `training/simple768_train.py`

Feature count:

```text
64 squares * 6 piece types * 2 colors = 768
```

Max active features:

```text
32
```

Feature index:

```text
((color * 6 + piece_type) * 64 + square)
```

Where:

- `WHITE = 0`
- `BLACK = 1`
- `PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0..5`
- `A1 = 0`
- `H8 = 63`

Shard format:

```text
indices uint16[N,32]
mask bool[N,32]
target_norm float32[N]
```

Current Simple768 training supports:

- one layer: `768 -> H -> 1`
- two layer: `768 -> H -> L2 -> 1`
- three layer: `768 -> H -> L2 -> L3 -> 1`

Model headers:

- `SHAKEYBOT_SIMPLE768_V1`
- `SHAKEYBOT_SIMPLE768_2L_V1`
- `SHAKEYBOT_SIMPLE768_3L_V1`
- `SHAKEYBOT_SIMPLE768_QUANT_V1`
- `SHAKEYBOT_SIMPLE768_2L_QUANT_V1`
- `SHAKEYBOT_SIMPLE768_3L_QUANT_V1`

## HalfKP Feature Encoding

Script: `training/halfkp_train.py`

HalfKP was added to follow Stockfish-style NNUE ideas, but current HalfKP models have not clearly beaten Simple768 in tournaments.

Feature count:

```text
64 king squares * 64 piece squares * 5 non-king piece types * 2 relative colors = 40960
```

Max active features:

```text
30 per perspective
```

Perspectives:

- white perspective
- black perspective

Current corrected contract:

- labels are white-POV
- perspective order is fixed `[white, black]`
- no STM-first ordering
- engine returns white-POV directly

Shard format:

```text
indices uint16[N,2,30]
mask bool[N,2,30]
stm int/bool[N]
target_norm float32[N]
```

The `stm` field exists in shards, but current corrected HalfKP training does not use STM-first output ordering.

Model headers:

- `SHAKEYBOT_HALFKP_WP_V1`
- `SHAKEYBOT_HALFKP_WP_2L_V1`
- `SHAKEYBOT_HALFKP_WP_3L_V1`
- `SHAKEYBOT_HALFKP_WP_QUANT_V1`
- `SHAKEYBOT_HALFKP_WP_2L_QUANT_V1`
- `SHAKEYBOT_HALFKP_WP_3L_QUANT_V1`

Old STM-trained HalfKP models should be treated as failed experiments.

## Augmentation

Supported flags:

```text
--augment-color-flip
--augment-horizontal-flip
```

Color flip:

- vertically flips the board
- swaps white and black pieces
- swaps side to move
- negates eval

Horizontal flip:

- mirrors files `a <-> h`
- preserves eval sign
- only allowed when no castling rights exist
- otherwise castling semantics change and position is skipped

Best-case source row expansion:

- original
- color flip
- horizontal flip
- horizontal plus color flip

Important split rule:

- split train/validation by canonicalized piece placement, not by row order.
- augmented siblings must not leak across train and validation.

## Material Bucket Diagnostics

Material metric:

- combined non-pawn material for both sides
- kings excluded
- pawns excluded
- N/B = 3
- R = 5
- Q = 9

Buckets:

```text
pawn_only = 0
very_low = 1..6
low = 7..14
medium = 15..24
late_mid = 25..34
high = 35+
```

Preprocessing manifest records:

- `material_buckets`
- `material_bucket_clipped_rows`
- `train_material_buckets`
- `val_material_buckets`

Training prints and logs:

```text
val_bucket_mae_cp pawn_only=...(rows) very_low=...(rows) ...
```

This matters because global MAE hid failures. The new 200M set is heavily different by phase/material, so global MAE can move for distribution reasons even when engine strength drops.

## Preprocessing Commands

Smoke Simple768 preprocessing:

```powershell
python training\simple768_train.py preprocess-parallel `
  --input "C:\path\to\csv_or_extracted_data" `
  --output-dir "training\shards_simple768_smoke" `
  --label-mode cp_clip `
  --clip-cp 3000 `
  --augment-color-flip `
  --augment-horizontal-flip `
  --workers 4 `
  --chunk-rows 100000 `
  --max-rows 1000000 `
  --compression none
```

Production Simple768 clip30 preprocessing:

```powershell
python training\simple768_train.py preprocess-parallel `
  --input "C:\path\to\csv_or_extracted_data" `
  --output-dir "training\shards_simple768_200m_clip30_aug2" `
  --label-mode cp_clip `
  --clip-cp 3000 `
  --augment-color-flip `
  --augment-horizontal-flip `
  --workers 4 `
  --chunk-rows 100000 `
  --compression none
```

Notes:

- `--workers` controls actual worker processes.
- `--max-in-flight` only controls how many chunks can be queued at once. It does not increase CPU parallelism.
- `--max-in-flight 0` means `workers * 2`.
- On Windows, antivirus may lock new `.npz` files. The script has retry logic for atomic replace failures.
- Local preprocessing can be faster to zip/upload than recomputing in Colab, but always validate archive extraction.

Validation after preprocessing:

- check `manifest.json`
- check total rows, train rows, val rows
- check clip rate
- check material bucket counts
- inspect one shard shape
- confirm `output_scale_cp`

Expected Simple768 shard:

```text
indices (N,32)
mask (N,32)
target (N,)
```

Expected HalfKP shard:

```text
indices (N,2,30)
mask (N,2,30)
stm (N,)
target (N,)
```

## Training Commands

Use normal Colab notebook cells with `!python ...`. Avoid helper wrappers that hide output.

Simple768 one-layer h192:

```python
!python "{SCRIPT}" train \
  --shard-dir "{SHARD_DIR}" \
  --output-dir "{MODEL_DIR}" \
  --hidden-size 192 \
  --epochs 40 \
  --batch-size 65536 \
  --learning-rate 0.0002 \
  --upload-drive-folder-id "{UPLOAD_FOLDER_ID}" \
  --upload-drive-name-prefix "{MODEL_TAG}_"
```

Simple768 one-layer h256:

```python
!python "{SCRIPT}" train \
  --shard-dir "{SHARD_DIR}" \
  --output-dir "{MODEL_DIR}" \
  --hidden-size 256 \
  --epochs 30 \
  --batch-size 65536 \
  --learning-rate 0.0002 \
  --upload-drive-folder-id "{UPLOAD_FOLDER_ID}" \
  --upload-drive-name-prefix "{MODEL_TAG}_"
```

Simple768 2L example:

```python
!python "{SCRIPT}" train \
  --shard-dir "{SHARD_DIR}" \
  --output-dir "{MODEL_DIR}" \
  --hidden-size 192 \
  --layer2-size 32 \
  --epochs 40 \
  --batch-size 65536 \
  --learning-rate 0.0002
```

Simple768 3L example:

```python
!python "{SCRIPT}" train \
  --shard-dir "{SHARD_DIR}" \
  --output-dir "{MODEL_DIR}" \
  --hidden-size 192 \
  --layer2-size 32 \
  --layer3-size 32 \
  --epochs 40 \
  --batch-size 65536 \
  --learning-rate 0.0002
```

Defaults:

- `--weight-decay 1e-5`
- this is AdamW-style weight decay, not dropout
- `--huber-delta 0.1`
- `--seed 1`

Batch size and LR:

- larger batch reduces gradient noise
- bigger batch often tolerates same or slightly larger LR, but with this project we have mostly lowered LR for huge datasets to reduce instability
- do not change LR, batch size, data mix, and architecture in the same experiment if trying to diagnose regression

## Export And Quantization

Check float export:

```python
!python "{SCRIPT}" check-export \
  --model "{MODEL_DIR / 'simple768_model.txt'}"
```

Quantize:

```python
!python "{SCRIPT}" quantize \
  --input "{MODEL_DIR / 'simple768_model.txt'}" \
  --output "{MODEL_DIR / f'{MODEL_TAG}_quant.txt'}" \
  --random-checks 4096 \
  --upload-drive-folder-id "{UPLOAD_FOLDER_ID}" \
  --upload-drive-name-prefix "{MODEL_TAG}_"
```

Good quantization signs:

- MAE under about `1cp`
- max absolute difference usually `1cp` to `2cp`
- activation/output scales printed and finite

Recent examples:

- h192 200M clip30:
  - quant MAE `0.113cp`
  - max abs `1cp`
- h256 200M clip30:
  - quant MAE `0.275cp`
  - max abs `2cp`

## Python Validation

Evaluate float model against validation shards:

```python
!python "{SCRIPT}" eval-shards \
  --model "{MODEL_FLOAT}" \
  --shard-dir "{SHARD_DIR}" \
  --split val \
  --batch-size 8192 \
  --by-material
```

Use this for diagnosis:

- compare multiple models on the same validation shards
- compare by material bucket
- do not compare a 66M validation MAE directly to a 200M validation MAE without checking distribution

## Engine Evaluation Flow

Main public flow:

1. `evaluate_white_pov_with_config(board, cfg)`
2. KPK exact probe first
3. neural low-material fallback may route to HCE
4. selected backend computes white-POV score
5. eval cache stores white-POV result
6. `evaluate_for_side_to_move_with_config()` converts to STM POV
7. tempo bonus is applied

Backends:

```text
hce
neural_dummy
neural_simple
neural_accum
neural_quant
neural_quant_accum
neural_halfkp
neural_halfkp_quant
neural_halfkp_quant_accum
```

Important backend meanings:

- `neural_simple`: float stateless Simple768
- `neural_accum`: float accumulator Simple768
- `neural_quant`: quant stateless Simple768
- `neural_quant_accum`: quant accumulator Simple768, current practical path
- `neural_halfkp_quant_accum`: quant accumulator HalfKP

Accumulator stats in benchmark output:

- `nnAccRefresh`
- `nnAccInvalid`
- `nnAccDelta`
- `nnAccCheckFail`

Good signs:

- `nnAccInvalid=0` or near zero in normal stable benchmark
- `nnAccCheckFail=0`

If `NeuralAccumulatorCheck=true`, the engine compares accumulator output against stateless quant output. This is diagnostic only and slows things down.

## Search And Move Selection Consequences

Neural eval influences more than final position scoring.

Search uses static eval for:

- qsearch stand-pat
- null move pruning gate
- futility pruning
- move-count pruning
- ProbCut decisions indirectly
- LMR improving signal
- TT static-eval reuse
- correction history residuals if enabled

Therefore a neural model can be "reasonable" as an evaluator but still bad in play if its score scale, phase bias, or tactical smoothness changes pruning behavior.

This is one reason tournaments can disagree with validation MAE.

## Engine Benchmark Command

Use:

```powershell
python docs\uci_benchmark_runner.py `
  --output "Benchmarks NAME.txt" `
  --label "NAME" `
  --setoption EvalBackend=neural_quant_accum `
  --setoption NeuralModelPath=models\MODEL.txt
```

Useful extra diagnostics:

```powershell
--setoption NeuralAccumulatorCheck=true
--setoption NeuralEndgameFallback=false
--setoption NeuralEndgameMaterialLimit=5
--setoption TempoBonus=0
```

Only use these one at a time when diagnosing.

## Known Results Snapshot

These numbers are approximate but useful context.

### Simple768 h192 66M Aug2 Quant Accum

Benchmark:

- positions: `9`
- nodes: `3364379`
- time: `6.71s`
- NPS: about `501k`
- `nnAccInvalid=0`
- `nnAccCheckFail=0`

Tournament:

- 66M h192 beat 18M h192 by about `+34 Elo` at standard time control.
- This is the current strongest practical direction.

### Simple768 h192 200M Clip30 Quant Accum

Validation:

- global MAE: `167.42cp`
- buckets:
  - pawn_only `136.63`
  - very_low `122.76`
  - low `172.71`
  - medium `249.94`
  - late_mid `243.61`
  - high `107.19`

Benchmark:

- positions: `9`
- nodes: `3279227`
- time: `9.74s`
- NPS: about `337k`
- `nnAccInvalid=0`
- `nnAccCheckFail=0`

Interpretation:

- engine plumbing looks fine
- validation distribution is much harder/different than 66M
- tournament result must decide

### Simple768 h256 200M Clip30 Quant Accum

Validation:

- global MAE: `165.55cp`
- buckets:
  - pawn_only `133.81`
  - very_low `119.98`
  - low `170.93`
  - medium `248.14`
  - late_mid `241.72`
  - high `105.37`

Benchmark:

- positions: `9`
- nodes: `3439062`
- time: `13.57s`
- NPS: about `253k`
- `nnAccInvalid=0`
- `nnAccCheckFail=0`

Interpretation:

- slightly better validation than h192 200M clip30 in every bucket
- about 25 percent slower than h192 200M clip30 in the benchmark
- tournament needed

### HalfKP h192 66M Quant Accum

Benchmark:

- positions: `9`
- nodes: `3739689`
- time: `13.13s`
- NPS: about `285k`
- accumulator clean

Tournament:

- mixed
- not clearly superior to Simple768 champion

### HalfKP 3L h192x32x32 66M Quant Accum

Validation:

- reached about `87.9cp`, much better than earlier HalfKP validation

Benchmark:

- positions: `9`
- nodes: `3182792`
- time: `30.52s`
- NPS: about `104k`
- accumulator clean

Tournament:

- clearly bad versus Simple768 champion
- likely suffers from speed tax and/or phase/endgame behavior despite validation improvement

## Regression Pattern To Explain

Observed pattern:

- More data and larger feature sets can improve validation but lose tournaments.
- HalfKP/deeper networks often look good in validation and early/midgame but collapse in endgames or under time control.
- The 200M mixed dataset was intended to fix endgame weakness but initially regressed badly.
- Label scale bug explained one failure, but clip30 fixed scaling still has high bucket MAE and uncertain tournament strength.

Likely causes to investigate:

1. Dataset distribution shift
   - new generated data is not the same distribution as engine games
   - lower-material/endgame skew can harm opening/middlegame calibration

2. Label heterogeneity
   - old and new labels may come from different Stockfish versions, depths, clipping rules, or FEN generation processes
   - depth-1 labels can be noisy, especially in tactical/endgame positions

3. Clip behavior
   - old data had raw values beyond `+/-30`
   - generated data was clipped to `+/-30`
   - mixed label tails may be inconsistent

4. Duplicates and oversampling
   - augmentation can multiply already common positions
   - opening/start-like positions and generated endgames may have very uneven coverage

5. Validation set is not a stable product metric
   - if the validation distribution changes, MAE is not comparable
   - current material buckets help but are not enough alone

6. Engine fallback confounds endgame tests
   - HCE fallback can hide some neural failures
   - fallback settings must be logged for every tournament

7. Static-eval pruning sensitivity
   - wrong scale or bias affects search selectivity, not just eval ordering

8. Missing input information
   - Simple768 ignores side to move, castling, en passant, move counters
   - this was intentional, but may limit endgame and tactical accuracy
   - previous raw STM experiment (`Simple770STM`) improved validation but lost tournaments, so any STM retry must be controlled

## Data-Prep Investigation Checklist

Before another full training run, produce these reports for old 66M and new 200M data.

Per source file:

- source row count
- parsed row count
- label min/max
- label mean/std
- clip rate under `clip_cp=1200`
- clip rate under `clip_cp=3000`
- side-to-move distribution
- material bucket distribution
- duplicate canonical placement count
- unique canonical placement count

Across full dataset:

- same stats globally
- same stats by material bucket
- same stats by source family: old CSV vs generated CSV
- same stats before augmentation and after augmentation

Validation split:

- verify each material bucket has enough validation rows
- verify augmented symmetries do not cross train/validation
- build a fixed diagnostic validation suite that does not change between experiments

Model comparison:

- evaluate h192 66M, h192 200M, h256 200M on the same validation shards if possible
- do not compare only each model's own validation set
- report bucket MAE and source-family MAE

Engine comparison:

- run benchmark with same backend and fallback settings
- run tournament with same openings, time control, hash, and fallback settings
- run at least one diagnostic tournament with `NeuralEndgameFallback=false` if endgame behavior is suspected

## Suggested Next Controlled Experiments

Do these one at a time.

1. Diagnose the 200M data mix
   - measure source-family and bucket distributions
   - compare old vs new labels
   - check whether medium/late_mid buckets dominate loss

2. Train with stratified sampling instead of full raw mix
   - keep a sane high/mid/endgame ratio
   - avoid letting generated low-material data dominate
   - keep a fixed validation set

3. Train Simple768 h192 with source weights
   - old data weight 1.0
   - generated endgame data weight lower or bucket-balanced

4. Try adding side-to-move as input, but keep white-POV output
   - do not change engine eval boundary
   - do not use STM-first output ordering
   - compare against Simple768 h192 champion

5. Add castling/en-passant features only after data distribution is understood

6. Revisit HalfKP only after Simple768 data issues are understood
   - HalfKP is slower and more sparse
   - it needs more careful architecture/data balancing

## Things Not To Repeat Blindly

- Do not use `pawn_linear30` as the main path. It lost in tournaments.
- Do not use old STM-trained HalfKP models.
- Do not judge models by global validation MAE only.
- Do not train bigger networks before checking distribution and label-scale issues.
- Do not mix new endgame data into the full dataset without source/bucket reporting.
- Do not change engine eval boundary from white-POV to STM-POV.
- Do not compare models trained on different validation distributions and call the lower MAE "better."

## Files To Know

Training:

- `training/simple768_train.py`
- `training/halfkp_train.py`
- `training/nn_simple768_200m_aug2_experiment.ipynb`
- `training/nn_simple768_66m_aug2_experiment.ipynb`
- `training/nn_halfkp_66m_aug2_experiment.ipynb`
- `training/nn_halfkp_3l_66m_aug2_experiment.ipynb`

Engine:

- `include/fast_engine/config.hpp`
- `include/fast_engine/evaluation.hpp`
- `src/evaluation.cpp`
- `src/eval/20_neural_simple.inc`
- `src/eval/21_neural_quant.inc`
- `src/eval/22_neural_halfkp.inc`
- `src/eval/99_public.inc`
- `src/search/search_context.inc`
- `src/search/search_ab.inc`
- `src/search/search_qsearch.inc`
- `apps/fast_engine_uci.cpp`

Benchmark:

- `docs/uci_benchmark_runner.py`

Architecture reference:

- `docs/ENGINE_ARCHITECTURE.md`

## Minimum Handoff Summary

If only one thing is remembered:

The engine pipeline is probably working. The current regression is more likely data distribution, label scale, label heterogeneity, or search sensitivity to eval scale. Keep white-POV output, use `cp_clip --clip-cp 3000` for `[-30,+30]` pawn-clipped data, diagnose by material bucket and source family, and promote only by tournament results against the Simple768 h192 66M quant-accum champion.

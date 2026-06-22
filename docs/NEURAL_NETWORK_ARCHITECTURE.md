# ShakeyBot Neural Network Architecture And Training Guide

This document is a focused handoff for the neural-evaluation work in ShakeyBot. It covers the C++ neural evaluation files, the training scripts and notebooks, the supported model formats, the accumulator and quantization optimizations, and the current experiment workflow.

The public evaluation contract is:

- neural models produce a score from White's point of view
- positive means White is better
- the search-facing eval converts White POV to side-to-move POV later
- tempo remains outside the neural network
- KPK probing and low-material HCE fallback remain active unless disabled by UCI options

## Current Neural State

Default engine config is in `include/fast_engine/config.hpp`:

- `EvalBackend=neural_quant_accum`
- `NeuralModelPath=models/simple768_model_h192_e20_aug2_quant.txt`
- `NeuralEndgameFallback=true`
- `NeuralEndgameMaterialLimit=10`
- `NeuralPawnOnlyFallback=false`
- `NeuralAccumulatorCheck=false`

Current best tested Simple768 model family:

- old champion: `models/simple768_h192_e20_66m_aug2_quant.txt`
- stronger 200M h192: `models/simple768_h192_e30_200m_clip30_aug2_quant.txt`
- stronger 200M h256 candidate: `models/simple768_h256_e25_200m_clip30_aug2_quant.txt`

Current HalfKP work:

- HalfKP white-POV models now train correctly and benchmark cleanly.
- HalfKP validation MAE can be lower than Simple768, but tournament strength has not reliably beaten Simple768 yet.
- Recent h512 HalfKP quant model: `models/halfkp_wp_h512_e21_200m_clip30_aug2_quant.txt`.
- HalfKP should not become default until direct tournaments prove it stronger.

## C++ Neural File Map

### `include/fast_engine/config.hpp`

Defines `EvalBackend`:

- `Hce`
- `NeuralDummy`
- `NeuralSimple`
- `NeuralAccum`
- `NeuralQuant`
- `NeuralQuantAccum`
- `NeuralHalfkp`
- `NeuralHalfkpQuant`
- `NeuralHalfkpQuantAccum`

Also stores the UCI-controlled neural options listed above.

### `include/fast_engine/evaluation.hpp`

Public evaluation API and accumulator structures:

- `evaluate_white_pov_with_config(board, cfg)`
- `evaluate_for_side_to_move_with_config(board, cfg)`
- `evaluate_white_pov_with_accumulator(board, cfg, accum, stats)`
- neural model load/unload functions
- accumulator refresh/update helpers

Important structs:

- `NeuralAccumulator`
  - `pre_activation`: float Simple768 accumulator
  - `quant_pre_activation`: quantized Simple768 accumulator
  - `halfkp_quant_pre_activation`: two-perspective quantized HalfKP accumulator
  - `hidden_size`
  - `board_hash`
  - `quantized`
  - `halfkp`
  - `valid`
- `NeuralAccumulatorStats`
  - `refreshes`
  - `invalid_fallbacks`
  - `delta_updates`
  - `check_failures`

The maximum supported first hidden layer size is currently `512`.

### `apps/fast_engine_uci.cpp`

Owns UCI-facing neural setup.

Important behavior:

- exposes `EvalBackend`, `NeuralModelPath`, `NeuralEndgameFallback`, `NeuralEndgameMaterialLimit`, `NeuralPawnOnlyFallback`, and `NeuralAccumulatorCheck`
- loads the right model loader based on backend and model family
- rejects wrong model headers through loader error messages
- prints neural accumulator stats in normal search output:
  - `nnAccRefresh`
  - `nnAccInvalid`
  - `nnAccDelta`
  - `nnAccCheckFail`
- includes an accumulator self-test path through the `testaccum` UCI command

Backend-to-loader mapping:

- `neural_simple`, `neural_accum` -> Simple768 float loader
- `neural_quant`, `neural_quant_accum` -> Simple768 quant loader
- `neural_halfkp` -> HalfKP float loader
- `neural_halfkp_quant`, `neural_halfkp_quant_accum` -> HalfKP quant loader
- `neural_dummy` ignores `NeuralModelPath`

### `src/evaluation.cpp`

Unity compilation unit for evaluation. It includes all `src/eval/*.inc` files.

### `src/eval/00_cache.inc`

Defines the full eval cache. The cache key includes:

- board hash
- HCE scale options
- selected eval backend
- neural model path
- neural fallback options

This matters because two neural models with the same position must not share cached evals. If a new neural option changes eval output, it must be added to `eval_config_signature()`.

### `src/eval/19_neural_dummy.inc`

Dummy neural backend used for plumbing tests.

It:

- reuses the Simple768 feature encoder
- uses deterministic pseudo-random weights
- has no model file
- returns a bounded White-POV score

Use this only to prove that neural backend selection changes engine behavior. It is not a trained evaluator.

### `src/eval/20_neural_simple.inc`

Float Simple768 model loader and inference.

Feature set:

- `SIMPLE_768_FEATURE_COUNT = 64 * 6 * 2 = 768`
- `SIMPLE_768_MAX_ACTIVE = 32`
- feature index:

```text
feature = ((color * 6 + piece_type) * 64) + square
```

Supported float headers:

- `SHAKEYBOT_SIMPLE768_V1`
- `SHAKEYBOT_SIMPLE768_2L_V1`
- `SHAKEYBOT_SIMPLE768_3L_V1`

Supported shapes:

- `768 -> H -> 1`
- `768 -> H -> L2 -> 1`
- `768 -> H -> L2 -> L3 -> 1`

Float activation:

- Simple768 uses ReLU: `max(0, x)`
- first-layer accumulator stores pre-activation: `b1 + sum(w1[active_feature])`
- later dense layers are computed at eval time

The loader reads `output_scale_cp` from the model. This must match label scaling from preprocessing. For `cp_clip --clip-cp 3000`, the exported model must contain `output_scale_cp 3000`.

### `src/eval/21_neural_quant.inc`

Quantized Simple768 model loader, stateless inference, accumulator inference, and SSE2 row-add optimization.

Supported quant headers:

- `SHAKEYBOT_SIMPLE768_QUANT_V1`
- `SHAKEYBOT_SIMPLE768_2L_QUANT_V1`
- `SHAKEYBOT_SIMPLE768_3L_QUANT_V1`

Quantized math:

- first layer weights are `int16`
- first layer biases/pre-activations are `int32`
- dense hidden/output intermediates use `int64`
- final output rescales by:

```text
cp = output_q * output_scale_cp / (activation_scale * layer2_weight_scale * layer3_weight_scale * output_weight_scale)
```

Only scales that exist for the current architecture are used.

Accumulator path:

- refresh initializes accumulator from `b1` then adds active feature rows
- delta update adds/subtracts affected feature rows for moves
- supports quiet moves, captures, promotions, en passant, and castling
- null moves copy accumulator state and update board hash after the move
- if the accumulator is missing or invalid, the engine refreshes from board and increments `nnAccInvalid`

Optimization:

- `neural_quant_add_i16_row_to_i32()` and subtraction use SSE2 when available
- fallback scalar loops remain available
- arrays are cache-line aligned where practical

### `src/eval/22_neural_halfkp.inc`

HalfKP feature encoding, float inference, quant inference, and quantized two-perspective accumulator.

Feature set:

- `HALFKP_FEATURE_COUNT = 64 * 64 * 5 * 2 = 40960`
- `HALFKP_MAX_ACTIVE = 30`
- `HALFKP_PERSPECTIVES = 2`
- kings are excluded from active non-king piece features

Feature index:

```text
feature = piece_square + ((piece_type * 2 + relative_color) + king_square * 10) * 64
```

Perspective convention:

- perspective `0` is White
- perspective `1` is Black
- White perspective uses board squares as-is and the White king square
- Black perspective mirrors squares vertically with `sq ^ 56` and uses the Black king square after mirroring
- `relative_color=0` means our piece for that perspective
- `relative_color=1` means opponent piece
- output order is fixed `[white_accum, black_accum]`
- model output is already White POV

Old STM-first HalfKP models were rejected. The current loaders accept only white-POV headers.

Supported float headers:

- `SHAKEYBOT_HALFKP_WP_V1`
- `SHAKEYBOT_HALFKP_WP_2L_V1`
- `SHAKEYBOT_HALFKP_WP_3L_V1`

Supported quant headers:

- `SHAKEYBOT_HALFKP_WP_QUANT_V1`
- `SHAKEYBOT_HALFKP_WP_2L_QUANT_V1`
- `SHAKEYBOT_HALFKP_WP_3L_QUANT_V1`

Supported shapes:

- `HalfKP -> Hx2 -> 1`
- `HalfKP -> Hx2 -> L2 -> 1`
- `HalfKP -> Hx2 -> L2 -> L3 -> 1`

Activation:

- HalfKP uses clipped ReLU
- float path clamps activations to `[0, 1]`
- quant path clamps first-layer activations to `[0, activation_scale]`
- dense hidden layers are also clipped before the next layer

Accumulator behavior:

- stores two first-layer pre-activation vectors, one per perspective
- normal moves delta-update both perspectives
- king moves refresh the affected perspective because all HalfKP features for that side depend on the king square
- castling handles both king and rook feature changes
- promotions, captures, and en passant are handled explicitly
- checker compares accumulator output against stateless quant output when `NeuralAccumulatorCheck=true`

### `src/eval/99_public.inc`

Dispatches between HCE and neural backends.

Important order:

1. Probe exact KPK first.
2. If selected backend is neural and low-material HCE fallback applies, use HCE.
3. Otherwise call selected neural or HCE implementation.
4. Store White-POV result in eval cache.
5. `evaluate_for_side_to_move_with_config()` converts White POV to side-to-move POV and adds tempo.

Fallback rules:

- `NeuralEndgameFallback=true` enables fallback.
- `NeuralEndgameMaterialLimit` uses pawns plus non-pawn material units.
- if a position is pawn-only and `NeuralPawnOnlyFallback=false`, fallback does not apply.

Accumulator evaluation follows the same fallback/KPK/cache rules.

### Search Integration

Search accumulator state is in `src/search/search_context.inc`.

Key behavior:

- root initializes accumulator stack when accumulator backend is active
- each ply stores a `NeuralAccumulator`
- before eval, search ensures the accumulator matches the current board hash
- after make-move, search tries delta update from parent accumulator to child accumulator
- if delta update fails, it refreshes from board
- stats are accumulated into `SearchStats` and printed by UCI

`src/engine.cpp` aggregates accumulator stats across iterations.

## Model Families

### Simple768

Simple768 is the lower-cost feature family.

Input:

- every piece including kings
- color, piece type, and square
- 768 total possible features
- at most 32 active features

Shard arrays:

- `indices`: shape `(N, 32)`
- `mask`: shape `(N, 32)`
- `target_norm`: shape `(N,)`

Strength notes:

- Simple768 learned well from 18M, 66M, and 200M datasets.
- More clean static data improved tournament strength when label scale was correct.
- 200M static/depth0 `h256 -> 1` has tested better than 200M `h192 -> 1`.
- Simple768 remains practical because it is fast under `neural_quant_accum`.

### HalfKP

HalfKP is Stockfish-style and much larger.

Input:

- non-king pieces only
- indexed relative to each side's king square
- two perspectives
- 40960 total possible features
- at most 30 active features per perspective

Shard arrays:

- `indices`: shape `(N, 2, 30)`
- `mask`: shape `(N, 2, 30)`
- `stm`: shape `(N,)`
- `target_norm`: shape `(N,)`

The `stm` field is retained for metadata and possible future experiments, but current model output remains White POV with fixed `[white, black]` perspective order.

Strength notes:

- HalfKP needs much more data than Simple768 because the input space is sparse.
- 66M HalfKP models were weak in tournaments.
- 200M HalfKP h256/h512 models have much better validation MAE and clean benchmarks, but still require tournament proof.
- HalfKP is slower, so lower MAE must beat the speed cost.

## Training Directory

### `training/simple768_train.py`

Main Simple768 preprocessing, training, export, eval, and quantization script.

Commands:

- `download`
- `download-folder`
- `preprocess`
- `preprocess-parallel`
- `train`
- `make-smoke-model`
- `check-export`
- `eval-shards`
- `quantize`
- `test-encoder`
- `test-labels`
- `test-augmentation`

Important preprocessing args:

- `--input`
- `--output-dir`
- `--shard-rows`
- `--validation-permille`
- `--clip-cp`
- `--label-mode cp_clip|pawn_linear`
- `--label-pawn-scale`
- `--clip-pawns`
- `--augment-color-flip`
- `--augment-horizontal-flip`
- `--workers`
- `--chunk-rows`
- `--max-rows`
- `--max-in-flight`
- `--compression compressed|none`

Important training args:

- `--hidden-size`
- `--layer2-size`
- `--layer3-size`
- `--epochs`
- `--batch-size`
- `--learning-rate`
- `--weight-decay`
- `--huber-delta`
- `--max-train-batches`
- `--upload-drive-folder-id`
- `--upload-drive-name-prefix`
- `--no-upload-replace-existing`

Important eval/quant args:

- `eval-shards --by-material`
- `eval-shards --max-rows`
- `quantize --random-checks`
- quantize scale overrides for first layer, layer 2, layer 3, and output
- Drive upload options for quantized output

Outputs:

- `best_params.npz`: Python/JAX checkpoint weights
- `simple768_model.txt`: exported float engine model
- model-specific exported float model, for example `simple768_h256_e25_200m_clip30_aug2.txt`
- quantized text model, for example `simple768_h256_e25_200m_clip30_aug2_quant.txt`
- `training_log.jsonl`
- `best_metrics.json`
- shard `manifest.json`

### `training/halfkp_train.py`

Main HalfKP preprocessing, training, export, eval, and quantization script.

It mirrors `simple768_train.py` but uses HalfKP encoding and supports two-perspective shards.

Extra important training args:

- `--full-validation-interval`
- `--quick-validation-rows`

These allow long HalfKP runs to print quick validation most epochs and full material-bucket validation only every N epochs. Best-model save/upload happens on full-validation epochs.

Outputs:

- `best_params.npz`
- `halfkp_model.txt`
- model-specific float model, for example `halfkp_wp_h512_e21_200m_clip30_aug2.txt`
- quantized model, for example `halfkp_wp_h512_e21_200m_clip30_aug2_quant.txt`
- `training_log.jsonl`
- `best_metrics.json`
- shard `manifest.json`

### `training/nn_simple768_200m_aug2_experiment.ipynb`

Latest Simple768 200M Colab runbook.

Purpose:

- authenticate Google Drive
- download raw static/depth0 label archives by Drive API pattern
- validate archives with `7z t`
- extract raw CSV files
- preprocess to Simple768 shards
- smoke train/export/quantize
- production train/export/quantize
- upload best weights and models to Drive

This is the best reference notebook for new large Simple768 runs.

### `training/nn_halfkp_200m_aug2_experiment.ipynb`

Latest HalfKP 200M Colab runbook.

Purpose:

- same high-level flow as the Simple768 200M notebook
- preprocesses HalfKP shards
- uses `FULL_VALIDATION_INTERVAL` and `QUICK_VALIDATION_ROWS`
- uses normal Colab `!python "{SCRIPT}" ...` cells so output streams visibly

This is the best reference notebook for new HalfKP runs.

### `training/nn_simple768_66m_aug2_experiment.ipynb`

Older Simple768 66M runbook.

Purpose:

- download/extract cached 66M Simple768 shards
- train `768 -> 256 -> 1`
- train `768 -> 192 -> 32 -> 1`

Keep as reference for cached-shard workflows, not as the main current runbook.

### `training/nn_halfkp_66m_aug2_experiment.ipynb`

Older HalfKP 66M runbook.

Purpose:

- train first large-data HalfKP white-POV models
- upload best params/model files on validation improvement

Historical reference. The 66M HalfKP models did not beat Simple768 in tournaments.

### `training/nn_halfkp_3l_66m_aug2_experiment.ipynb`

Older HalfKP 3-layer 66M runbook.

Purpose:

- trains `HalfKP -> 192x2 -> 32 -> 32 -> 1`
- exports quant model for `neural_halfkp_quant_accum`

Historical reference for 3-layer cells and naming.

### `training/nn_halfkp_experiment.ipynb`

Earlier experimental notebook. Despite the name, it contains early Simple768/HalfKP transition material and should not be treated as current.

### `training/nn_768_experiment.ipynb`

Initial Simple768 training notebook. Useful only for early context.

### `training/nn_768_clipped_30.ipynb`

Clipping/normalization experiment notebook. The tournament result was worse than the non-clipped baseline, but some label-scale lessons came from this work.

### `training/models/`

Local generated model directory. Treat contents as artifacts, not source. Do not commit large downloaded or generated models unless explicitly intended.

### `training/shards/`

Local generated shard directory. Treat contents as artifacts, not source.

### `training/__pycache__/`

Python cache directory. Ignore.

## Label Normalization

Raw CSV evals are treated as pawn units.

Supported modes:

- `cp_clip`
- `pawn_linear`

Current recommended mode for static/depth0 labels:

```text
--label-mode cp_clip --clip-cp 3000
```

This maps:

- `+30.0` pawns -> `+3000cp` -> `+1.0`
- `-30.0` pawns -> `-3000cp` -> `-1.0`

Important lesson:

- training data clipped to `[-30,+30]` pawns must not use old `output_scale_cp=1200`
- old scale saturated everything beyond `+-12` pawns and caused a regression
- model exports must carry `output_scale_cp=3000`
- engine loaders honor `output_scale_cp`

`pawn_linear` was tested and did not improve tournament strength.

## Data Augmentation And Splitting

Augmentations:

- color flip:
  - vertically mirror board
  - swap piece colors
  - flip side to move
  - negate label
- horizontal flip:
  - mirror files
  - preserve label
  - only allowed when castling rights are `-`
- color plus horizontal:
  - apply horizontal variant then color flip
  - negate label

The split key is a symmetry-canonical piece placement. This prevents original and augmented versions of the same board from leaking across train/validation.

Material bucket metric:

- non-pawn material excluding kings, both sides combined
- N/B = 3
- R = 5
- Q = 9

Buckets:

- `pawn_only`: 0
- `very_low`: 1..6
- `low`: 7..14
- `medium`: 15..24
- `late_mid`: 25..34
- `high`: 35+

Preprocessing manifest records:

- source rows
- augmented rows
- train rows
- validation rows
- label mode
- output scale
- bucket counts
- clipped row counts
- rows per second
- worker timing

## Preprocessing Optimizations

Both training scripts support `preprocess-parallel`.

Optimizations added:

- process-pool chunk workers
- `--max-rows` for smoke tests
- `--max-in-flight` to limit queued chunks and RAM pressure
- `--compression none` for faster shard writing
- timing fields in manifest
- `rows_per_second`
- `source_rows_per_second`
- retry wrapper around file replacement to survive Windows antivirus file locks

The Windows retry path prints lock retries and can wait up to several minutes before failing.

Recommended defaults are hardware-dependent:

- local Windows Simple768: `--workers 4` was reasonable
- Colab HalfKP preprocessing tested well with:

```text
--workers 20 --max-in-flight 20 --chunk-rows 10000 --compression none
```

For very large datasets, smoke preprocess first with `--max-rows`.

## Training Cycle

Training happens on Google Colab GPUs. The local machine is used for engine builds, benchmarks, tournaments, and sometimes local preprocessing/compression.

Standard cycle:

1. Upload current `training/*.py` script and notebook to Colab.
2. Authenticate Google Drive near the top of the notebook when Drive upload/download is needed.
3. Download raw CSV archives or cached shard archives.
4. Validate archives with `7z t`.
5. Extract archives.
6. Smoke preprocess a small row count.
7. Run script tests:
   - `test-encoder`
   - `test-labels`
   - `test-augmentation`
8. Smoke train with `--max-train-batches`.
9. `check-export` the float model.
10. `quantize` the float model and compare quantized inference to float.
11. Full production preprocessing.
12. Production training.
13. Upload `best_params.npz`, exported float model, and `best_metrics.json` on new best.
14. Quantize final/best float model.
15. Upload/download quantized model.
16. Place quantized model in local `models/`.
17. Run UCI benchmark.
18. Run tournament against current champion.

Drive upload behavior:

- optional through `--upload-drive-folder-id`
- `--upload-drive-name-prefix` makes filenames clear
- existing remote file with the same name is deleted by default before upload
- use `--no-upload-replace-existing` only when intentionally keeping every upload

## Example Commands

Simple768 preprocessing:

```powershell
python training\simple768_train.py preprocess-parallel `
  --input "C:\path\to\csvs" `
  --output-dir training\shards_simple768_200m_clip30_aug2 `
  --label-mode cp_clip `
  --clip-cp 3000 `
  --augment-color-flip `
  --augment-horizontal-flip `
  --workers 4 `
  --chunk-rows 100000 `
  --compression none
```

HalfKP preprocessing:

```bash
python /content/training/halfkp_train.py preprocess-parallel \
  --input /content/shakeybot_halfkp_200m/data_stockfish_labels_static \
  --output-dir /content/shakeybot_halfkp_200m/shards_halfkp_200m_clip30_aug2 \
  --label-mode cp_clip \
  --clip-cp 3000 \
  --augment-color-flip \
  --augment-horizontal-flip \
  --workers 20 \
  --max-in-flight 20 \
  --chunk-rows 10000 \
  --compression none
```

Simple768 h256 training:

```bash
python /content/training/simple768_train.py train \
  --shard-dir /content/shakeybot_nn_200m/shards_simple768_200m_clip30_aug2 \
  --output-dir /content/shakeybot_nn_200m/models_simple768_h256_e30_200m_clip30_aug2 \
  --hidden-size 256 \
  --epochs 30 \
  --batch-size 65536 \
  --learning-rate 0.0002 \
  --weight-decay 1e-05 \
  --upload-drive-folder-id 1FfJkqKZQvP46fT6U_IGkkEClATRkcuoB \
  --upload-drive-name-prefix simple768_h256_e30_200m_clip30_aug2_
```

HalfKP h512 training:

```bash
python /content/training/halfkp_train.py train \
  --shard-dir /content/shakeybot_halfkp_200m/shards_halfkp_200m_clip30_aug2 \
  --output-dir /content/shakeybot_halfkp_200m/models_halfkp_wp_h512_e30_200m_clip30_aug2 \
  --hidden-size 512 \
  --epochs 30 \
  --batch-size 4096 \
  --learning-rate 0.0002 \
  --weight-decay 1e-05 \
  --full-validation-interval 5 \
  --quick-validation-rows 5000000 \
  --upload-drive-folder-id 1FfJkqKZQvP46fT6U_IGkkEClATRkcuoB \
  --upload-drive-name-prefix halfkp_wp_h512_e30_200m_clip30_aug2_
```

Quantize:

```bash
python /content/training/simple768_train.py quantize \
  --input /content/shakeybot_nn_200m/models_simple768_h256_e30_200m_clip30_aug2/simple768_h256_e30_200m_clip30_aug2.txt \
  --output /content/shakeybot_nn_200m/models_simple768_h256_e30_200m_clip30_aug2/simple768_h256_e30_200m_clip30_aug2_quant.txt \
  --random-checks 4096 \
  --upload-drive-folder-id 1FfJkqKZQvP46fT6U_IGkkEClATRkcuoB \
  --upload-drive-name-prefix simple768_h256_e30_200m_clip30_aug2_
```

Validate exported float model:

```bash
python /content/training/simple768_train.py eval-shards \
  --model /content/shakeybot_nn_200m/models_simple768_h256_e30_200m_clip30_aug2/simple768_h256_e30_200m_clip30_aug2.txt \
  --shard-dir /content/shakeybot_nn_200m/shards_simple768_200m_clip30_aug2 \
  --split val \
  --batch-size 8192 \
  --by-material
```

Local benchmark:

```powershell
python docs/uci_benchmark_runner.py `
  --output "Benchmarks simple768 h256 200m quant.txt" `
  --label "Simple768 h256 200m quant_accum" `
  --setoption EvalBackend=neural_quant_accum `
  --setoption NeuralModelPath=models/simple768_h256_e25_200m_clip30_aug2_quant.txt
```

HalfKP benchmark:

```powershell
python docs/uci_benchmark_runner.py `
  --output "Benchmarks halfkp h512 200m quant.txt" `
  --label "HalfKP h512 200m quant_accum" `
  --setoption EvalBackend=neural_halfkp_quant_accum `
  --setoption NeuralModelPath=models/halfkp_wp_h512_e21_200m_clip30_aug2_quant.txt
```

Accumulator checker benchmark:

```powershell
python docs/uci_benchmark_runner.py `
  --output "Benchmarks halfkp h512 checker.txt" `
  --label "HalfKP h512 checker" `
  --setoption EvalBackend=neural_halfkp_quant_accum `
  --setoption NeuralModelPath=models/halfkp_wp_h512_e21_200m_clip30_aug2_quant.txt `
  --setoption NeuralAccumulatorCheck=true
```

Acceptance for accumulator correctness:

- `nnAccInvalid=0` is ideal in normal benchmark once root state is established
- `nnAccCheckFail=0` is required when checker is enabled

## Benchmarks And Tournaments

Benchmarks are safety and speed checks. Tournaments decide promotion.

Benchmark outputs to watch:

- total time
- nodes
- NPS
- best moves
- `nnAccRefresh`
- `nnAccInvalid`
- `nnAccDelta`
- `nnAccCheckFail`

Tournament rules learned so far:

- validation MAE alone is not enough
- direct match versus current champion is mandatory
- also test against external engines such as FabChess when useful
- use the same opening suite and time control for fair comparison
- if a model is clearly negative after a few hundred games, stop early

## Important Lessons

### White POV Is The Stable Contract

STM-first HalfKP was tried and failed in tournaments despite validation looking plausible. Current models use White-POV labels and fixed perspective order.

Do not switch the engine boundary to side-to-move model output unless a whole new controlled experiment is planned.

### Static Labels Fit This Engine Better Than Search Labels

The engine search sees hanging pieces and shallow tactics. The eval function should mostly learn static structure. Depth0/static labels have produced much lower validation MAE on the 200M dataset than depth1/qsearch-style labels.

### Label Scale Must Match Data

If labels are clipped to `[-30,+30]` pawns, use:

```text
--label-mode cp_clip --clip-cp 3000
```

Using `output_scale_cp=1200` on that data caused a serious regression.

### HalfKP Needs More Than Low MAE

HalfKP has a larger, sparser input space and is slower. Even when validation improves, the model can lose tournaments if the speed cost or error distribution is bad.

### Bigger Batches Can Be Slower

For HalfKP, `4096` and `8192` batch sizes tested faster than `16384` and `32768`. The likely cause is large sparse-gather and dense intermediate pressure on GPU memory/cache. Use smoke timing before committing to a production batch size.

### Do Not Track Notebooks And Artifacts

Notebooks are local/Colab working documents and should remain ignored unless explicitly requested. Large model files, shard directories, split archives, and `__pycache__` are artifacts.

## Current Follow-Up Ideas

Highest-value next checks:

- continue Simple768 h256 200M tournament validation
- train/benchmark HalfKP h512 or h512x32 only if tournament results justify more HalfKP time
- keep per-material bucket validation in all large runs
- consider data balancing by material and eval bucket before more architecture expansion
- keep `NeuralAccumulatorCheck` disabled in tournaments

Do not promote a model because it has lower validation MAE. Promote only after benchmark correctness and direct tournament strength.

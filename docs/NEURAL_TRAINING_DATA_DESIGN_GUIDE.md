# Neural Training Data Design Guide

This document is for the LLM or engineer helping generate ShakeyBot neural-eval training data.

Current lesson: more data is not automatically better. ShakeyBot has already seen regressions where bigger datasets or bigger networks improved some validation numbers but lost tournaments. Best next step is a cleaner, balanced, auditable dataset that matches how the engine actually uses static evaluation.

## Goal

Train a neural evaluator that helps search pick better moves.

This is not exactly the same as minimizing global validation MAE on a random dataset. The model must produce stable, well-scaled static evals for:

- root move selection
- internal alpha-beta static eval
- qsearch stand-pat
- pruning gates
- null-move decisions
- futility decisions
- TT static eval reuse
- endgame decisions

Bad eval scale or phase bias can hurt search even when validation MAE looks reasonable.

## Best Label POV

Use **white-POV labels always**.

Model contract:

```text
board -> white-POV centipawn score
```

If Stockfish output is side-to-move POV, convert it before writing the row:

```text
if side_to_move == white:
    label_white_cp = stockfish_cp
else:
    label_white_cp = -stockfish_cp
```

Do not train the main engine model as:

```text
side-to-move POV output
```

Reason:

- current engine eval boundary expects white-POV
- engine converts to side-to-move POV later
- tempo is added outside the model
- changing output POV risks breaking search assumptions

Good future experiment:

```text
input includes side-to-move feature
output still white-POV
```

Bad experiment to avoid:

```text
if black to move, output black POV
if white to move, output white POV
```

That changes the eval contract and already caused trouble in earlier HalfKP/STM experiments.

## Label Scale

Raw CSV labels are in **pawn units**, even if the column is named `cp`.

Correct conversion:

```text
target_cp = raw_pawn_eval * 100
```

If source data is clipped to `[-30,+30]` pawns, use:

```text
label_mode = cp_clip
clip_cp = 3000
output_scale_cp = 3000
```

This gives:

```text
+30 pawns -> +3000cp -> +1.0 target
-30 pawns -> -3000cp -> -1.0 target
+5 pawns  -> +500cp  -> +0.1667 target
```

Do **not** use `clip_cp=1200` with `[-30,+30]` pawn-clipped source data.

That would make:

```text
+12 pawns and +30 pawns both become +1.0
```

This destroys useful label resolution for decisive positions.

## Material-Axis Distribution

Natural chess-position distribution is not ideal. Generated data can also become badly skewed. Use bucket-balanced sampling.

Material metric:

- combined non-pawn material for both sides
- kings excluded
- pawns excluded
- knight/bishop = 3
- rook = 5
- queen = 9

Recommended buckets:

```text
pawn_only = 0
very_low = 1..6
low      = 7..14
medium   = 15..24
late_mid = 25..34
high     = 35+
```

Recommended training mix:

```text
high       25%
late_mid   20%
medium     20%
low        15%
very_low   10%
pawn_only  10%
```

If HCE/tablebase fallback covers many pure pawn endings, `pawn_only` can be reduced to about `5%`, but do not remove it entirely.

Why this matters:

- too much high material: endgames weak
- too much generated endgame data: opening/middlegame calibration can degrade
- too much low material with noisy labels: model may learn strange simplification behavior

## Evaluation-Axis Distribution

Balance positions by eval size inside each material bucket.

Recommended bins by absolute white-POV eval:

```text
0-50cp        25%
50-150cp      25%
150-400cp     20%
400-1000cp    15%
1000-3000cp   10%
mate/huge       5%
```

Why:

- near-equal positions are most move-sensitive
- moderate advantages are common in real games
- decisive positions teach conversion and avoiding blunders
- too many huge winning positions teach only "winner signal"
- too many equal positions make model timid

Keep huge/mate labels separate if possible. They are useful but should not dominate.

## Source-Family Distribution

Best dataset should mix several source families.

Recommended source mix:

```text
40% real games / normal game positions
25% ShakeyBot search-distribution positions
20% generated late-midgame and endgame positions
10% tactical / blunder / sharp positions
 5% curated hard endings or tablebase positions
```

Most important new source: **ShakeyBot search-distribution positions**.

Collect positions that ShakeyBot actually evaluates:

- root positions from games
- internal static-eval positions
- qsearch stand-pat positions
- positions before/after tactical swings
- positions from lost tournament games
- positions from known blunder FENs

Reason:

```text
training distribution should match engine usage distribution
```

Random chess positions are not enough. Engine sees a biased subset of positions created by its own search.

## Amount Of Data

More rows help only when rows add real information.

For Simple768:

```text
50M-150M unique, balanced source-equivalent positions likely enough
```

For HalfKP:

```text
needs more data because input is much sparser
```

But HalfKP also costs more speed, so do not scale HalfKP blindly before data balance is fixed.

Priority order:

```text
unique balanced positions > huge augmented biased rows
```

Bad:

```text
500M rows from skewed generated source
```

Better:

```text
100M rows, balanced by source, material, eval bin, and duplicates
```

## Deduplication

Dedup or cap repeated feature-equivalent positions.

For Simple768 without STM/castling features:

```text
dedup key = piece placement
```

For Simple768 plus side-to-move:

```text
dedup key = piece placement + side_to_move
```

For Simple768 plus side-to-move and castling:

```text
dedup key = piece placement + side_to_move + castling rights
```

For HalfKP:

```text
dedup key = piece placement, with king squares naturally important
```

Recommended duplicate cap:

```text
max 1-4 copies per canonical key before augmentation
```

Augmented sibling rule:

```text
original and all augmented versions must go to same train/validation side
```

No augmented leakage across train and validation.

## Augmentation

Allowed augmentation:

1. Color flip
2. Horizontal flip

Color flip:

- flip board vertically
- swap white and black pieces
- swap side to move
- negate eval

Horizontal flip:

- mirror files `a <-> h`
- keep eval sign
- only use when no castling rights exist

Do not horizontally flip positions with castling rights unless castling rights are transformed correctly. Safest rule:

```text
horizontal flip only when castling rights == "-"
```

Maximum expansion for one source row:

```text
original
color-flipped
horizontal-flipped
horizontal + color-flipped
```

Augmentation is useful, but not equal to new independent data.

## Stockfish Label Depth

Best practical labeling setup:

```text
bulk data: depth 0 using modified stockfish binary
quality subset: depth 4-8 or fixed nodes
tablebase endings: Syzygy WDL/DTZ if possible
```


Avoid making all labels deep-search labels too.

Reason:

- deep search includes tactical/strategic foresight that engine search should find itself
- static eval may become less smooth
- labels become expensive and harder to reproduce

Best compromise:

- most data: Stockfish depth 1 with qsearch
- calibration subset: depth 4-8
- special endgames: tablebase if available

Record depth/source for every row.

## Row Metadata

Ideal raw row should include more than just FEN and label.

Recommended schema:

```text
fen
label_white_cp
raw_stockfish_score
stockfish_score_pov
side_to_move
stockfish_depth
stockfish_nodes_or_movetime
stockfish_version
source_family
material_bucket
eval_bin
canonical_key
augmentation_type
is_duplicate_count
was_clipped
```

Minimum acceptable schema:

```text
fen
label_white_cp
source_family
stockfish_depth
stockfish_version
```

Without source metadata, regression debugging becomes guesswork.

## Validation Design

Do not rely only on random split from current dataset.

Create fixed validation suites:

```text
fixed_global_val
fixed_material_bucket_val
fixed_source_family_val
fixed_endgame_val
fixed_opening_midgame_val
fixed_tactical_val
fixed_tournament_blunder_val
```

Every model should report:

- global MAE
- MAE by material bucket
- MAE by eval bin
- MAE by source family
- MAE by Stockfish depth/source
- calibration slope/intercept if possible

Important:

```text
do not compare models only on each model's own validation set
```

Compare all models on the same fixed validation suites.

## Train/Validation Split

Use deterministic hash split.

Split key should match active feature contract.

Simple768 without STM:

```text
canonical piece placement
```

Simple768 with STM:

```text
canonical piece placement + side_to_move
```

Simple768 with castling:

```text
canonical piece placement + side_to_move + castling rights
```

HalfKP:

```text
canonical piece placement with king positions
```

All augmented siblings must share split key.

## Best Next Dataset Shape

Recommended next controlled dataset:

```text
feature target: Simple768 or Simple768+STM
output target: white-POV
label mode: cp_clip
clip_cp: 3000
source rows: 80M-150M unique balanced
augmentation: color flip + safe horizontal flip
duplicate cap: 1-4 per canonical key
validation: fixed, bucketed, source-aware
```

Balanced source mix:

```text
real/self-play normal:       40%
ShakeyBot search positions:  25%
generated endgames:          20%
tactics/blunders:            10%
curated tablebase/hard EG:    5%
```

Balanced material mix:

```text
high       25%
late_mid   20%
medium     20%
low        15%
very_low   10%
pawn_only  10%
```

Balanced eval mix per material bucket:

```text
0-50cp        25%
50-150cp      25%
150-400cp     20%
400-1000cp    15%
1000-3000cp   10%
mate/huge       5%
```

## Side-To-Move Feature Recommendation

Current Simple768 ignores:

- side to move
- castling
- en passant
- move counters

This was intentional for first model. But side-to-move likely matters.

Best next model input:

```text
Simple768 + STM feature(s)
```

But output remains:

```text
white-POV
```

Possible STM encoding:

```text
feature 768 = white_to_move
feature 769 = black_to_move
```

or one scalar/binary feature if trainer/engine support dense extras.

Do not use STM-first perspective ordering for Simple768.

Do not make output side-to-move POV.

## Castling And En Passant

Castling rights can matter in openings and king safety.

Possible future features:

```text
white can castle kingside
white can castle queenside
black can castle kingside
black can castle queenside
```

En passant probably lower priority but can affect tactics. Add later if needed.

Do not add many metadata features until basic data distribution is fixed.

Recommended order:

1. fix data balance and label scale
2. add STM input
3. add castling rights
4. consider en passant

## Evaluation Of New Data

Before training, produce a data report:

Per source family:

- rows
- unique canonical keys
- duplicate rate
- side-to-move distribution
- material bucket distribution
- eval bin distribution
- label min/max/mean/std
- clip rate under `clip_cp=1200`
- clip rate under `clip_cp=3000`

Per material bucket:

- rows
- unique keys
- eval bin distribution
- source-family mix
- clip rate

After preprocessing:

- source rows
- augmented rows
- train rows
- val rows
- material bucket train/val rows
- source-family train/val rows
- eval-bin train/val rows
- shard count
- shard shape
- `output_scale_cp`

## What Went Wrong Recently

Observed regression:

- 200M mixed dataset produced worse tournament results than 66M champion.
- More data did not help.
- h192 200M with old scale was bad.
- h192/h256 200M clip30 fixed scale, validation still high and tournament uncertain.

Likely causes:

1. Label scale mismatch
   - fixed by `clip_cp=3000`, but not whole story

2. Dataset distribution shift
   - generated lower-material data may dominate
   - model may lose high/medium calibration

3. Label heterogeneity
   - old and new data may differ in Stockfish version, depth, qsearch, clipping

4. Duplicate/augmentation overload
   - many augmented rows may not add independent information

5. Validation mismatch
   - global MAE changed because validation distribution changed

6. Search sensitivity
   - neural eval feeds pruning gates, not only final eval

## Things To Avoid

Avoid:

- random giant data dumps without metadata
- comparing only global MAE
- mixing label scales
- mixing Stockfish depth/version without metadata
- outputting side-to-move POV for current engine
- training bigger network before fixing data distribution
- assuming augmentation equals new data
- letting generated endgames dominate full dataset
- judging by benchmark speed only
- judging by validation only

## Promotion Criteria

A dataset/model is promising only if:

1. preprocessing report looks sane
2. fixed validation suites improve or expose clear tradeoff
3. quantization error is small
4. engine benchmark completes cleanly
5. accumulator stats are clean
6. direct tournament beats current champion or is at least clearly not worse

Current champion to beat:

```text
Simple768 h192 66M Aug2 quant accumulator
EvalBackend=neural_quant_accum
```

Tournament decides promotion.

## Short Final Recommendation

Best training data now:

```text
white-POV labels
cp_clip with clip_cp=3000
balanced by material bucket
balanced by eval bin
source-family metadata preserved
duplicate capped
augmented safely
includes real engine-search positions
mostly Stockfish depth 1 with qsearch
small higher-depth/tablebase calibration subset
fixed validation suites
```

Best near-term model target:

```text
Simple768 + side-to-move input -> h192 or h256 -> white-POV output
```

But only after data audit confirms the new dataset is balanced and label-consistent.

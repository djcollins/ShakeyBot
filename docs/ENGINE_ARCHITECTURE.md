# ShakeyBot Engine Architecture

This is a map of how the engine is put together so future cleanup or search work can start from the current structure.

## High-Level Layout

- `apps/fast_engine_uci.cpp`
  - UCI protocol front end
  - parses `uci`, `isready`, `setoption`, `position`, `go`, `stop`, `quit`
  - prints `info` and `bestmove`
- `src/engine.cpp`
  - search orchestration
  - iterative deepening
  - aspiration windows
  - time management integration
  - iteration callback reporting
- `src/search.cpp`
  - single compilation unit for search
  - includes the `.inc` search modules
- `src/evaluation.cpp`
  - single compilation unit for evaluation
  - includes the `.inc` eval modules
- `src/transposition.cpp`
  - TT implementation

## Public Headers

- `include/fast_engine/config.hpp`
  - search and eval tuning knobs exposed to the engine and UCI
- `include/fast_engine/engine.hpp`
  - `Engine`, `SearchResult`, `IterationInfo`, time-control structures
- `include/fast_engine/search.hpp`
  - `SearchStats`, `SearchControl`, `RootMove`, `negamax`, `qsearch`, `find_best_move`
- `include/fast_engine/transposition.hpp`
  - TT entry and table types
- `include/fast_engine/evaluation.hpp`
  - public evaluation entry points

## UCI Layer

File: `apps/fast_engine_uci.cpp`

Responsibilities:

- construct and own `Engine`
- expose config knobs through `setoption`
- translate UCI `go` limits into `SearchLimits`
- stream iteration info lines
- output final `bestmove`

Important note:

- Many tuning knobs live in UCI already, especially around move ordering, history multipliers, LMR knobs, correction history, razoring, and eval scales.
- Evaluation backend selection is also controlled here through `EvalBackend`, `NeuralModelPath`, and neural endgame-fallback options.

## Engine Orchestration

File: `src/engine.cpp`

Responsibilities:

- drive iterative deepening
- manage aspiration windows
- preserve last completed iteration when interrupted
- coordinate TT search generation boundaries
- produce cumulative search statistics

Root search state:

- single-PV design
- root move persistence support exists through `RootMove`
- aspiration and iterative deepening live here, not inside `negamax`

## Search Compilation Unit

File: `src/search.cpp`

The search code is split into include fragments:

- `search_context.inc`
  - global search constants
  - history tables
  - search stack entries
  - helper math for LMR, futility, correction history, history updates
- `search_ordering_see.inc`
  - static exchange evaluation helpers
  - move scoring for ordering
- `move_picker.inc`
  - staged move picker
- `search_qsearch.inc`
  - quiescence search
- `search_ab.inc`
  - main negamax alpha-beta with pruning, extensions, reductions, TT logic
- `99_public.inc`
  - public search entry points and heuristic reset code

## Move Ordering

Core node ordering is staged by `MovePicker` in `src/search/move_picker.inc`.

Stage order:

1. TT move
2. promotions
3. good captures
4. killers
5. counter-move
6. quiets
7. bad captures

Important details:

- captures are split by SEE
- checking captures can be promoted into the good-capture path
- quiets are not fully scored until needed
- bad captures are deliberately searched last, not discarded globally

Quiet ordering signals currently include:

- main history
- pawn history
- piece-aware continuation history
- killers
- counter-move / refutation move

Capture ordering signals currently include:

- SEE
- MVV-LVA style base
- capture history

## Search Stack And State

`search_context.inc` maintains per-ply state used by search and ordering.

Important stack/state concepts:

- previous move
- moved piece type
- whether previous move was a capture
- static eval cache per ply
- in-check state
- parent move count
- TT-hit state

This stack feeds:

- continuation history lookup and updates
- countermove lookup
- improving signal
- parent-sensitive history learning

## Search Heuristics In The Main Tree

Main implementation: `src/search/search_ab.inc`

Important mechanisms already present:

- fail-soft negamax alpha-beta
- PVS
- LMR
- null-move pruning
- razoring
- ProbCut
- internal iterative deepening
- singular extension
- one-reply extension
- selective check extension
- move-count pruning
- extended futility pruning
- correction-history adjusted static eval for pruning gates
- repetition and draw handling
- mate-distance pruning

When tuning this file, remember:

- many heuristics already interact through shared signals
- move count, history, and TT signals can easily double-count
- a patch that looks theoretically right can still be Elo-negative if it over-stacks existing selectivity

## Histories And Learning Tables

Search history tables live in `search_context.inc`.

Important tables:

- main history
  - quiet move success/failure by side and from/to square
- continuation history
  - piece-aware responses keyed from prior move context
- pawn history
  - quiet move history keyed by pawn structure bucket
- capture history
  - capture success/failure by attacker, target square, and victim class
- counter-moves
  - quiet refutation move keyed by previous move
- killer moves
  - quiet beta-cutoff moves stored per ply
- correction history
  - learned static-eval correction keyed by side and pawn-key bucket

These tables influence both:

- ordering
- selectivity and reductions

That is why "small" history patches can have larger-than-expected interactions.

## Transposition Table

Public type: `include/fast_engine/transposition.hpp`
Implementation: `src/transposition.cpp`

Current design:

- 4-way clustered table
- 16-byte packed entries
- power-of-two bucket count
- generation tagging
- `clear()` is effectively O(1) via generation bump
- `new_search()` bumps generation while keeping old entries probeable
- replacement prefers:
  - empty slots
  - stale-generation entries
  - lower-quality entries

Stored data:

- truncated key signature
- depth
- bound type
- score
- static eval
- best move
- generation

TT quality matters a lot in this engine. Recent successful work came from TT design and usage, not from another tiny quiet-ordering tweak.

## Root Search

Public API in `include/fast_engine/search.hpp`

Key points:

- root search is separate from inner-node negamax
- `RootMove` holds:
  - move
  - last score
  - previous score
  - subtree nodes
  - searched flag
- engine is single-PV
- root ordering persistence exists and matters because it affects aspiration stability and early PV quality

## Quiescence Search

Implementation: `src/search/search_qsearch.inc`

Behavior:

- if not in check:
  - stand-pat first
  - then tactical moves
- if in check:
  - no stand-pat
  - search evasions

Qsearch tuning is high leverage but also easy to destabilize tactically.

## Evaluation

Compilation unit: `src/evaluation.cpp`

Evaluation is modular and mostly white-POV internally, then converted at the public boundary.

Major modules:

- cache
- material and phase
- Stockfish-style material imbalance
- endgame scaling and king crowding
- shared eval helpers
- mobility
- outposts
- pawn structure
- passed pawns and pawn hash
- king safety
- king cover cache
- king-zone pressure
- bishop pair / bad bishop
- rook activity
- x-ray pins
- PSQT / PST
- threats and space
- closedness
- complexity
- queen vulnerability

Top-level weights are exposed in `EngineConfig::EvalScales`.

### Neural Evaluation Backend

The engine has a backend switch at the public eval boundary:

- `EvalBackend::Hce`
  - handcrafted evaluation
- `EvalBackend::NeuralDummy`
  - deterministic plumbing/test backend
- `EvalBackend::NeuralSimple`
  - stateless Simple768 float model
- `EvalBackend::NeuralAccum`
  - Simple768 float model with accumulator support
- `EvalBackend::NeuralQuant`
  - stateless Simple768 quantized model
- `EvalBackend::NeuralQuantAccum`
  - Simple768 quantized accumulator model
- `EvalBackend::NeuralHalfkp`
  - stateless HalfKP float model
- `EvalBackend::NeuralHalfkpQuant`
  - stateless HalfKP quantized model
- `EvalBackend::NeuralHalfkpQuantAccum`
  - HalfKP quantized accumulator model
  - current default backend in `EngineConfig`

Default v2.0.0 config:

- backend: `neural_halfkp_quant_accum`
- model: `models/halfkp_wp_h512_e15_500m_clip30_quant.txt`
- neural endgame fallback: disabled
- neural pawn-only fallback: disabled
- accumulator checker: disabled by default for speed

UCI options:

- `EvalBackend`
  - combo: `hce`, `neural_dummy`, `neural_simple`, `neural_accum`, `neural_quant`, `neural_quant_accum`, `neural_halfkp`, `neural_halfkp_quant`, `neural_halfkp_quant_accum`
- `NeuralModelPath`
  - text path to exported float or quantized model
- `NeuralEndgameFallback`
  - enables HCE fallback for low-material neural positions
- `NeuralEndgameMaterialLimit`
  - material-unit threshold for fallback
- `NeuralPawnOnlyFallback`
  - controls whether pure pawn endings also fall back to HCE
- `NeuralAccumulatorCheck`
  - debug checker comparing accumulator path against stateless neural path

Model loading:

- public declarations live in `include/fast_engine/evaluation.hpp`
- Simple768 float: `load_neural_simple_model(path, error)`
- Simple768 quantized: `load_neural_quant_model(path, error)`
- HalfKP float: `load_neural_halfkp_model(path, error)`
- HalfKP quantized: `load_neural_halfkp_quant_model(path, error)`
- UCI loading is routed by `apps/fast_engine_uci.cpp` according to selected backend
- model path resolution tries the literal path first, then walks upward from the process working directory and executable directory to find `models/`, then loads the requested filename from that directory

Simple768 input convention:

- feature count: `64 * 6 * 2 = 768`
- at most 32 active piece features
- index formula: `((color * 6 + piece_type) * 64 + square)`
- square convention: `A1 = 0`, `H8 = 63`

HalfKP input convention:

- feature count: `64 * 5 * 2 * 64 = 40960`
- kings are anchors, not normal piece features
- non-king piece type count is 5: pawn, knight, bishop, rook, queen
- model uses white/black perspectives and side-to-move information
- accumulator backends maintain and update transformed features across make/unmake instead of rebuilding every eval

Inference files:

- `src/eval/19_neural_dummy.inc`
  - deterministic backend for testing
- `src/eval/20_neural_simple.inc`
  - Simple768 float/stateless inference and model format
- `src/eval/21_neural_quant.inc`
  - Simple768 quantized inference and accumulator path
- `src/eval/22_neural_halfkp.inc`
  - HalfKP float/quantized inference, HalfKP accumulator, and model loading

Public eval flow:

1. Build an eval-cache key from board and eval-affecting config.
2. Probe exact KPK first, before backend dispatch.
3. Apply optional low-material neural fallback if configured.
4. Dispatch to selected HCE/neural backend.
5. Store white-POV result in eval cache.
6. Convert to side-to-move POV in `evaluate_for_side_to_move_with_config()`.
7. Apply normal tempo bonus at side-to-move boundary.

Current tablebase status:

- exact KPK handling exists in the eval path
- root-only Syzygy probing is active through vendored Fathom when `SyzygyRootProbe=true`
- default local path is the directory name `Zyzygy_EGTB_345`; runtime resolution walks upward from the process working directory and executable directory to find that folder
- root probing filters legal root moves to the best tablebase WDL class and orders those moves by DTZ/progress, then normal search still chooses inside that filtered set
- a post-v2.0.0 Syzygy/Fathom experiment was rolled back because root tablebase cutoffs could stop search too early and still allow poor practical conversion
- future Syzygy work should use tablebases to filter/order and score eligible nodes, not blindly replace root search with the first safe tablebase move

Cache and search integration:

- eval-cache signature includes backend/model/fallback state so HCE and neural values do not alias
- eval-affecting UCI option changes clear TT/eval state as needed
- search static eval and pruning gates call `evaluate_white_pov_with_config()` through normal search context
- neural backends feed static eval, qsearch stand-pat, and pruning decisions
- a `go` command requiring a neural model but lacking a valid loaded model reports a UCI-visible error and avoids searching with uninitialized weights

Recent neural training lessons:

- v2.0.0 release model is `halfkp_wp_h512_e15_500m_clip30_quant.txt`, a HalfKP white-POV `40960 -> 512 -> 1` quantized accumulator model trained from natural Lichess game positions.
- Post-release local candidate `halfkp_wp_h512_e6_all_data_clip30_quant.txt` uses the same shape trained on about 3.7 billion natural positions and is currently the strongest local result.
- Early local result: `HalfKP h512 3.7b e6` beat `HalfKP h512 500m e15` by `89 - 47 - 115`, score `0.584`, about `+58.7 +/- 31.7 Elo`.
- Early local result: `HalfKP h512 3.7b e6` beat `Ceibo v1.0 2985` by `310 - 17 - 23`, score `0.919`, about `+420.9 +/- 60.5 Elo`.
- Natural game-position data transferred much better to tournaments than hand-shaped/generated one-ply position clouds, even when synthetic validation MAE looked good.
- More natural data improved tournament strength strongly; data distribution quality mattered more than adding hand-crafted endgame heuristics.
- Keep future data experiments anchored to real game/search-position distributions; use targeted synthetic data only as a small diagnostic supplement.

## Diagnostics You Will See In Benchmarks

Search result and benchmark output may include:

- `nodes`
- `time`
- `nps`
- `tt_hits` / `tt_misses`
- `q10`
- `q10r`
- root PV first-move change counters
- bad-capture stage diagnostics
- razoring diagnostics

These are not all equally valuable.

High-value diagnostics:

- broad node shifts across many positions
- score drift on tactical positions
- `q10` / `q10r` movement when tuning quiet selectivity
- root wobble / PV stability when tuning root or TT behavior

Low-value diagnostics:

- raw log volume without a clear decision path

## Practical Tuning Guidance

Areas that have recently shown real value:

- TT design and TT usage
- root iteration behavior
- infrastructure that improves signal quality before adding another consumer

Areas that have often disappointed in this repo when changed narrowly:

- small quiet-ordering consumers
- extra selective pruning of late quiets
- minor history-split ideas without a stronger supporting signal

So before changing search:

- identify which existing signal you are improving
- identify which consumer uses it
- check whether that consumer area has already failed recent A/B tests

## Update Log

Impact labels use this scale: `very bad`, `bad`, `neutral`, `good`, `great`.

| Area | Concise technical change | Impact on performance | Status | Superseded by |
| --- | --- | --- | --- | --- |
| Build system | Added dual-platform Makefile flow, standardized binary target naming, and Windows runtime-DLL staging in build output. | neutral | kept | - |
| Source structure | Kept thin orchestrators (`search.cpp`, `evaluation.cpp`) with module fragments under `src/search` and `src/eval`. | neutral | kept | - |
| IDE/tooling | Applied workspace IntelliSense/squiggle tuning for `.inc`-based architecture and single-language-server setup. | neutral | partially_rolled_back | later workspace setting changes |
| Code hygiene | Removed stale/misleading comments and cleaned warning-producing unused paths/locals in search/eval/UCI glue. | neutral | kept | - |
| Time management | Tightened early-game time allocation policy and conservative budget behavior in iterative search orchestration. | good | kept | - |
| PV semantics | Switched pruning/extension PV gating from null-window proxies to explicit PV-state semantics. | bad | rolled_back | Regression recovery |
| Regression recovery | Rolled back unstable PV-gating behavior and restored stable search behavior on `main`. | good | kept | - |
| History learning | Added multi-ply continuation-history signal integration in move ordering/learning path. | good | kept | - |
| LMR gating | Changed late-move reduction trigger logic to use quiet-move count instead of aggregate move count. | good | kept | - |
| Qsearch micro-optimizations | Cached check-generation signals and tightened pawn-push helper usage in tactical leaf logic. | neutral | kept | - |
| SEE/selectivity | Introduced earlier SEE-based filtering in capture/selectivity path. | neutral | kept | - |
| Endgame heuristics | Added recognizer/eval adjustments for king safety, pawn push handling, and endgame-specific signals. | neutral | kept | - |
| Search state infra | Added explicit search-stack state plumbing used by ordering and pruning consumers. | good | kept | - |
| TT integration | Improved transposition-table usage across search flow and bound/lookup interaction points. | great | kept | - |
| SEE reuse | Cached SEE in staged move ordering and qsearch so the same capture does not recompute SEE across multiple consumers. | neutral | kept | - |
| Cheap gating | Moved cheap filtering and `givesCheck`/capture bookkeeping ahead of more expensive selectivity work in main search and qsearch. | neutral | kept | - |
| Movegen diagnostics | Added legal-move-generation counters to benchmark output to measure staged picker and full-movegen pressure. | neutral | kept | - |
| Lazy staged movegen | Refactored `MovePicker`/root flow to generate TT, captures, and quiets lazily instead of prebuilding full legal move lists at every node. | neutral | kept | - |
| MovePicker bug fix | Fixed a staged-picker TT suppression bug introduced during lazy move generation that could drop a legal TT move from later stages. | good | kept | - |
| Qsearch leaf cleanup | Cached qsearch check-state data and cleaned up pawn-push helper usage in tactical leaf code. | good | kept | - |
| Qsearch evasion ordering | Replaced in-check qsearch evasion ordering with a lighter custom scorer. | bad | rolled_back | Qsearch evasion rollback |
| Qsearch evasion rollback | Restored the safer generic qsearch-in-check evasion ordering after tactical benchmark regressions. | good | kept | - |
| Quiet promotion generation | Switched qsearch quiet-promotion generation to a pawn-only quiet-move path. | bad | rolled_back | Quiet promotion rollback |
| Quiet promotion rollback | Restored direct quiet-promotion construction after repeated speed regressions. | good | kept | - |
| Eval shared infra | Added shared `EvalInfo`/attack-map/file-summary infrastructure so overlapping eval terms reuse derived board data instead of rebuilding it. | good | kept | - |
| Closedness eval | Added a closedness/openness evaluation term with a tuning knob; adopted a small nonzero default after modest positive testing. | neutral | kept | - |
| Complexity eval | Added a complexity-scaling evaluation term behind a tuning knob; testing left the default at zero. | neutral | kept | - |
| Queen vulnerability eval | Added a queen-vulnerability evaluation term behind a tuning knob; testing left the default at zero. | neutral | kept | - |
| History knob cleanup | Removed dead killer/countermove UCI knobs and fixed TT quiet-learning so continuation updates still work when main history is disabled. | neutral | kept | - |
| Continuation cleanup | Reduced continuation-history influence, unplugged the plain non-piece continuation table, and kept the piece-aware path as the active continuation signal. | neutral | kept | - |
| LMR core formula | Decoupled LMR from root aspiration width, switched it to local-window behavior, and corrected PV-node reduction direction. | good | kept | - |
| LMR history influence | Narrowed and symmetrically capped the history-based LMR adjustment so it nudges reductions instead of dominating them. | neutral | kept | - |
| Null-move conservatism | Tightened null-move entry guards and simplified-endgame handling while reshaping the reduction. | bad | rolled_back | Null-move rollback |
| Null-move rollback | Restored the prior null-move pruning behavior after a clear regression in tournament play. | good | kept | - |
| Root iteration ordering | Added persistent root-move ordering with iteration feedback across iterative deepening and aspiration re-searches. | neutral | kept | - |
| Pawn history | Added pawn-history as an active quiet ordering and learning signal. | good | kept | - |
| Low-ply history | Added a low-ply history consumer on top of the existing history stack. | bad | rolled_back | - |
| Threat-aware quiet shaping | Added threat-aware quiet bonus/penalty shaping in the quiet-ordering path. | neutral | rolled_back | - |
| Quiet and checking ordering consumer | Added a search-stack consumer to bias quiet and checking move ordering from the newer state plumbing. | bad | rolled_back | - |
| Capture ordering consumer | Added a new capture-ordering consumer on top of the same enabling infrastructure. | bad | rolled_back | - |
| Good vs bad quiet split | Split quiet handling into preferred and poor-history buckets inside the search path. | bad | rolled_back | - |
| Cut-node quiet selectivity bundle | Retuned cut-node quiet handling across LMR, history, and late-quiet move-count interplay in one combined patch. | very bad | rolled_back | LMR local-history cleanup |
| LMR local-history cleanup | Kept only the narrow local-history/stat-score cleanup from the failed selectivity bundle and dropped the more aggressive late-quiet behavior. | neutral | kept | - |
| Late quiet vs bad history split | Separated shallow bad-history quiet handling from mere late-move status in cut-node selectivity. | bad | rolled_back | - |
| Root telemetry | Added root aspiration, score-jump, best-vs-second, subtree-share, and root-TT-quality telemetry to iteration/result reporting. | neutral | kept | - |
| Root TT/PV hygiene | Tried cleaner root TT seeding, completed-root TT writeback, and stricter root PV trust/extraction handling. | bad | rolled_back | - |
| Dynamic aspiration | Replaced fixed root aspiration startup with telemetry-driven initial sizing and directional fail-low/fail-high widening. | neutral | kept | - |
| Root signal time allocation | Fed PV stability, aspiration chaos, score jump, root gap, and subtree dominance into soft-deadline adjustment. | good | kept | - |
| Easy-move early stop | Tried stopping slightly before soft time when shallow root signals looked stable and dominant. | bad | rolled_back | - |
| Panic extra depth | Tried allowing one extra iterative-deepening depth past soft time on unstable roots. | bad | rolled_back | - |
| Razoring retune | Raised razoring margins to `900/1800` and restored them as accepted baseline after tournament gain. | great | kept | - |
| TT key hardening | Increased packed TT signature from 16 bits to 32 bits while preserving 16-byte entries and TT capacity. | neutral | kept | - |
| Singular verification safety | Disabled local forward-pruning at the excluded-move verification ply so singular-extension tests search alternatives more honestly. | neutral | kept | - |
| ProbCut telemetry | Added ProbCut counters for nodes, candidates, SEE rejects, qsearch passes, reduced searches, and cutoffs. | neutral | kept | - |
| ProbCut safety retune | Removed qsearch-only ProbCut cutoffs, required reduced-search verification, tightened candidate SEE/try limits, added stronger contradictory-TT gate, and stored only verified reduced-depth bounds. | neutral | kept | - |
| Correction-history quality upgrade | Tried material-bucketed correction history plus depth-weighted exact-node updates and extra update guards. Tournament result was negative, especially for Black. | bad | rolled_back | Patch 4 rollback |
| Patch 4 rollback | Restored previous correction-history key and update policy after benchmark/tournament regression; current baseline returns to ProbCut 3B behavior. | good | kept | - |
| TT generation lifecycle | Fixed TT generation clear/new-search wrap handling so generation reuse and clearing stay coherent across searches. | neutral | kept | - |
| Lazy legal-count restoration | Restored lazy main-search legal-move handling by avoiding eager full move generation and using bounded legal-count probes where needed. | good | kept | - |
| Continuation tuning controls | Fixed continuation-history tuning controls so fractional UCI multiplier values are interpreted as intended. | neutral | kept | - |
| Root passer/liquidation verification | Tried root-level endgame passer-race and liquidation verification that disabled selective pruning in candidate child searches. Tournament result was clearly negative. | very bad | rolled_back | Root passer/liquidation rollback |
| Root passer/liquidation rollback | Restored the accepted continuation-tuning baseline after the root passer/liquidation verification patch regressed badly in A/B play. | good | kept | - |
| Blunder-FEN diagnostics | Verified tournament blunder FENs from PGN and used UCI option matrices to separate pruning, qsearch, eval-scale, and correction-history signals before the next patch. | neutral | kept | - |
| History default cleanup | Disabled correction history and continuation history by default, with their default scalar knobs set to zero after isolated A/B regressions. | good | kept | - |
| History inactive guards | Centralized correction/continuation active checks so disabled flags or zero scale knobs fully remove those signals from eval correction, ordering, pruning, and learning paths. | neutral | kept | - |
| Qsearch recapture protection | Kept only the immediate-recapture protection from the qsearch tactical-pruning batch; recaptures are protected from delta and SEE pruning, while TT-only qsearch protection was not kept. | good | kept | - |
| Release strength docs | Updated public project wording around the FabChess local match result, including the 10s/50 moves time control and rating-list caveat. | neutral | kept | - |
| Release cleanup pass | Cleaned comments, spacing, stale tuning notes, and unclear local names across root/docs/include/apps/src/search/eval without changing benchmark behavior. | neutral | kept | - |
| Neural eval plumbing | Added eval backend selection, Simple768 stateless model loading/inference, neural UCI options, eval-cache backend signatures, and low-material HCE fallback for neural play. | great | kept | - |
| Natural HalfKP training data | Replaced hand-shaped/generated training distributions with natural Lichess game-position data for the HalfKP h512 model. The resulting model became the new tournament champion by a large margin. | great | kept | - |
| Syzygy/Fathom experiment | Tried 3-5 piece Syzygy probing after v2.0.0. The first approach let root TB hits short-circuit search too aggressively and still produced poor practical conversion in pawn endings. | bad | rolled_back | Future tablebase design |
| All-data HalfKP h512 candidate | Trained the same HalfKP h512 shape on about 3.7 billion natural positions, producing `halfkp_wp_h512_e6_all_data_clip30_quant.txt`. Early local tournaments beat the 500m HalfKP champion by about `+58.7 Elo` and Ceibo v1.0 2985 by about `+420.9 Elo`. | great | local_candidate | - |

## Update Log Interpretation

- `Impact on performance` is an A/B-oriented outcome label, not a code quality label.
- `Status` tracks current lifecycle:
  - `kept`: active in current mainline behavior.
  - `rolled_back`: intentionally reverted.
  - `partially_rolled_back`: only part of the original change remains.
  - `superseded`: replaced by a later approach.
  - `local_candidate`: validated locally but not part of the current release package.
- `Superseded by` links an entry to the follow-up entry that replaced or reverted it.
- Use one line per technically coherent patch group; if a patch group has mixed outcomes, split it into smaller lines.
- Do not edit old lines to hide regressions; add a new follow-up line and update `Status`/`Superseded by`.
- Keep descriptions concise and architectural; avoid constants, margins, or line-level details in this log.

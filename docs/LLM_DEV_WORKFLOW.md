# LLM Development Workflow For ShakeyBot

Read [ENGINE_ARCHITECTURE.md](ENGINE_ARCHITECTURE.md) first. Do not start proposing patches before you understand the engine layout, current heuristics, and recent search work.

## Agent Mission

Every agent working on ShakeyBot should behave like a chess engine developer, not a generic coding helper.

- First read the Markdown files in `docs/`, especially [ENGINE_ARCHITECTURE.md](ENGINE_ARCHITECTURE.md), then inspect the relevant code paths before recommending changes.
- If unsure about a chess-engine concept, search credible engine-development resources, strong open-source engines, Chess Programming Wiki, forums, or papers. Do not guess when outside knowledge is needed.
- Goal is stronger tournament Elo under the user's match conditions, not prettier code, higher NPS, or faster benchmark depth.
- Start with a quick code review and sanity check of the subsystem under discussion. Do not attempt a full codebase audit unless asked.
- Propose measurable patches with a clear Elo hypothesis, expected search-shape impact, and rollback criteria.
- If a patch is too small to justify a 10-20 hour A/B tournament to validate, suggest including it with another patch. 
- Prefer correctness, safety, and search accuracy over aggressive pruning or speed.

## Mandatory Working Style

- Use the `caveman` module for user-facing communication. Be terse, direct, and technical.
- Be honest. If you have not done a real code review, do not claim you did one.
- Do real code reading before recommending patches. This codebase has many interacting search heuristics; shallow guesses waste time.
- Prefer one narrow patch at a time unless the user explicitly asks for a grouped patch family.
- Do not add noisy instrumentation or informational logs that do not lead to an actionable tuning decision.
- Do not use `stockfish` in any variable name, identifier, or newly added code symbol.
- Respect that the user cares about Elo at fixed time, not just prettier code or deeper benchmark depth.
- Optimize search correctness, safety, and accuracy first. Raw NPS, raw depth, and lower node count are secondary.

## User Preferences And Process

- Do not assume a classic idea is automatically good here.
- The user strongly prefers practical, testable changes over speculative micro-tuning.
- If a patch starts clearly negative in A/B, recommend stopping early and reverting.
- If a patch is almost benchmark-identical, treat it as enabling infrastructure unless there is a strong reason to isolate it in A/B.
- Neutral tournament patches can still be worth keeping when they harden search safety without clear Elo loss.
- If a patch harms one color badly, especially Black, treat that as a serious warning even if aggregate CI is wide.

## Development Loop

1. Read the relevant code first.
2. Explain the exact subsystem you are touching.
3. Make the smallest coherent patch that tests the idea.
4. Build release.
5. Run a smoke test.
6. Run the standard benchmark script and save the output to a root-level benchmark text file.
7. Compare sanity metrics against the current accepted baseline.
8. Only then recommend an A/B tournament if the patch has a coherent Elo hypothesis and benchmark sanity does not show breakage.

## Build And Smoke Expectations

Use a release build, not debug.

Windows release build:

```powershell
mingw32-make.exe clean
mingw32-make.exe MODE=release
```

If `mingw32-make.exe` is not on `PATH`, use the known WinLibs toolchain path from this machine:

```powershell
$env:PATH = "C:\Users\collinsd\Documents\Development_work\CPP_Stuff\winlibs-x86_64-posix-seh-gcc-15.2.0-mingw-w64ucrt-13.0.0-r4\mingw64\bin;$env:PATH"
& "C:\Users\collinsd\Documents\Development_work\CPP_Stuff\winlibs-x86_64-posix-seh-gcc-15.2.0-mingw-w64ucrt-13.0.0-r4\mingw64\bin\mingw32-make.exe" MODE=release CXX=g++
```

Expected binary:

```text
build/bin/ShakeyBot.exe
```

Minimum smoke expectation after search changes:

- engine starts under UCI
- returns a legal `bestmove` from `startpos`
- returns a legal `bestmove` from at least one tactical FEN

If a build or smoke test fails, stop and fix that before continuing.

## Benchmark Sanity-Check Workflow

After every patch, the agent must run the deterministic benchmark suite with the repo script and save the output in a root-level text file such as `Benchmarks after patch 3B.txt`.

Standard command from repo root:

```powershell
python docs/uci_benchmark_runner.py --output "Benchmarks after <patch name>.txt"
```

Optional explicit release binary and depth:

```powershell
python docs/uci_benchmark_runner.py --engine build/bin/ShakeyBot.exe --depth 15 --output "Benchmarks after <patch name>.txt"
```

The script runs the standard 9-position UCI sanity suite, uses `ucinewgame` before each position, and records aggregate search metrics.

Important benchmark interpretation:

- Benchmarks are guardrails for build/search sanity, not an Elo oracle.
- Do not treat node count, NPS, depth, final score drift, or one-position PV changes as direct evidence of tournament strength.
- A benchmark can look worse because the engine is exploring harder defensive resources or unstable positions, not because the patch is Elo-negative.
- A benchmark can also look faster because the engine is pruning too confidently or missing defensive resources. Faster is not automatically better.
- If total nodes increase, that is acceptable when the patch has a plausible safety/accuracy reason. Do not reject accuracy patches for spending more nodes.
- Use benchmarks to catch catastrophic breakage, illegal moves, tactical collapses, runaway node explosions, or obviously incoherent broad behavior.
- If build and smoke pass and the patch has a coherent tournament-Elo hypothesis, do not reject it solely because the benchmark shape is unattractive.

Your job before suggesting A/B:

- compare new benchmark file against the current accepted baseline benchmark file
- look at:
  - total nodes
  - time
  - NPS
  - final score on each test
  - `q10`
  - `q10r`
  - PV stability or wobble counters when present
- check whether the change is broad or just one-position noise
- call out when the effect is concentrated in a single outlier position

Interpretation rules:

- If nearly every position is unchanged and only one case moves, do not push for a long standalone A/B. That is usually infrastructure or noise.
- If benchmark behavior changes materially across many positions, and the change is coherent rather than random, then an A/B is justified.
- Benchmarks are a sanity screen only; tournament results decide Elo.

## A/B Tournament Workflow

Common user tournament formats:

- `10 seconds per side / 200 moves`
- `10 seconds per side / 50 moves`
- equal opening book up to `ply 10`
- each opening played twice, once per side
- no termination rules set
- typical early/midgame search depth is around `8-11`

When reading results, focus on:

- Elo difference
- confidence interval
- LOS
- draw ratio
- whether the direction is stabilizing or collapsing as games accumulate

Practical policy:

- A long A/B is expensive: roughly `5000 games` may be needed for about `+/-10 Elo` confidence, which can take around `16 hours` in the user's setup.
- Reserve that cost for patches with a plausible `~10 Elo` upside, broad strategic value, or a need to validate important search behavior.
- clearly negative early result: recommend stop and revert
- flat result with wide CI: usually not worth spending many more games unless the patch is strategically important infrastructure
- clearly positive result with decent game count: recommend keeping it


## What To Optimize For

- safer pruning decisions
- more accurate search, especially in tactical/defensive positions
- better cutoff quality
- better root stability
- fewer wasted re-searches
- stronger practical Elo under the user's actual match conditions

Do not optimize for:

- benchmark depth alone
- NPS alone
- lower node count alone
- speculative eval/signal changes that feed many pruning consumers without a narrow safety reason

## Repo Hygiene

- Benchmark logs are local artifacts. They must not be tracked in Git.
- New benchmark files should stay ignored by `.gitignore`.
- Do not commit tournament dumps, benchmark notes, or scratch analysis files unless the user explicitly asks.

## Useful Repo Entry Points

- UCI front end: `apps/fast_engine_uci.cpp`
- Engine orchestration: `src/engine.cpp`
- Search compilation unit: `src/search.cpp`
- Search internals:
  - `src/search/search_context.inc`
  - `src/search/search_ordering_see.inc`
  - `src/search/move_picker.inc`
  - `src/search/search_qsearch.inc`
  - `src/search/search_ab.inc`
  - `src/search/99_public.inc`
- Evaluation compilation unit: `src/evaluation.cpp`
- Evaluation modules: `src/eval/*.inc`
- TT: `include/fast_engine/transposition.hpp`, `src/transposition.cpp`
- Config knobs: `include/fast_engine/config.hpp`

## Patch Selection Advice

Before proposing a patch, ask:

- is this infrastructure, or a consumer change?
- has this exact search area already failed multiple A/Bs?
- does this patch create a new signal, or just reweight an already noisy one?
- can the effect be measured cleanly in existing benchmark outputs?

When in doubt, prefer:

- infrastructure that enables future stronger patches
- corrections to signal quality over raw extra pruning
- hardening existing search assumptions before adding more selectivity

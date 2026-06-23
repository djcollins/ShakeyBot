# ShakeyBot

ShakeyBot is a UCI chess engine written in C++.

ShakeyBot v2.0 uses a HalfKP NNUE-style neural evaluator by default, with a quantized accumulator backend for tournament play.

The default release model is:

```text
models/halfkp_wp_h512_e15_500m_clip30_quant.txt
```

That model is distributed in the GitHub Release package, not tracked directly in the source repository.

## Project Layout

- `apps/fast_engine_uci.cpp` - UCI frontend executable entry point
- `src/` - engine/search/eval/transposition implementation
- `include/fast_engine/` - public engine headers
- `external/chess.hpp` - chess move generation/types dependency
- `models/` - release-model notes; model files are shipped in release assets
- `docs/` - architecture notes and benchmark helper
- `Makefile` - cross-platform build (Windows + Linux)

## Build

### Windows (WinLibs / MinGW)

From the project root:

```powershell
mingw32-make.exe
```

Release build:

```powershell
mingw32-make.exe MODE=release
```

Output binary:

- `build/bin/ShakeyBot.exe`

Note: the Makefile copies MinGW runtime DLLs (`libstdc++-6.dll`, `libgcc_s_seh-1.dll`, `libwinpthread-1.dll`) into `build/bin` after link on Windows.

### Linux

```bash
make
```

Release build:

```bash
make MODE=release
```

Output binary:

- `build/bin/ShakeyBot`

### Clean

```bash
make clean
```

or on Windows:

```powershell
mingw32-make.exe clean
```

## Run

### Windows

```powershell
.\build\bin\ShakeyBot.exe
```

### Linux

```bash
./build/bin/ShakeyBot
```

## UCI Usage

ShakeyBot speaks standard UCI. Example manual session:

```text
uci
isready
ucinewgame
position startpos
go depth 12
```

You can also search from a FEN:

```text
position fen 5kb1/8/8/1K6/1P6/P7/8/8 w - - 3 59
go depth 15
```

Stop / exit:

```text
stop
quit
```

## Estimated Strength

ShakeyBot v2.0.0 is provisionally estimated around 3000 Elo based on a local match against Ceibo v1.0, which is listed around 2985 Elo.

```text
Scope    W-D-L        Score        Score %   Elo diff       95% CI
Overall  651-266-202  784.0/1119   70.06%   +147.7         +128.9 to +166.5
```

This is a provisional local estimate, not an official rating-list result. Broader testing against more engines and time controls is still needed.

### Post-Release Training Result

After v2.0.0, a larger HalfKP model training run used the same `40960 -> 512 -> 1` shape but trained on about 3.7 billion natural Lichess-derived, Stockfish depth-0-labelled positions. The downloaded local model name was:

```text
halfkp_wp_h512_e6_all_data_clip30_quant.txt
```

Early local tournament results at `10 seconds / 50 moves + 1 second increment` were strong:

```text
HalfKP h512 3.7b e6 vs HalfKP h512 500m e15:
Score 89 - 47 - 115 [0.584], Elo +58.7 +/- 31.7, 251 games

HalfKP h512 3.7b e6 vs Ceibo v1.0 2985:
Score 310 - 17 - 23 [0.919], Elo +420.9 +/- 60.5, 350 games
```

Treat this as a provisional development result until it is tested against a broader engine pool. The current release package still ships the 500m-position model listed above.

## Notes

- Search and evaluation are currently organized as thin orchestrators (`src/search.cpp`, `src/evaluation.cpp`) that include internal module fragments under `src/search/*.inc` and `src/eval/*.inc`.
- Engine options are exposed via UCI `setoption` in `apps/fast_engine_uci.cpp`.

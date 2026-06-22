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

## Notes

- Search and evaluation are currently organized as thin orchestrators (`src/search.cpp`, `src/evaluation.cpp`) that include internal module fragments under `src/search/*.inc` and `src/eval/*.inc`.
- Engine options are exposed via UCI `setoption` in `apps/fast_engine_uci.cpp`.

# ShakeyBot

ShakeyBot is a UCI chess engine written in C++.

Not sure of it's ELO just yet, but once I manage to enter it in tournaments we will find out! 

## Project Layout

- `apps/fast_engine_uci.cpp` - UCI frontend executable entry point
- `src/` - engine/search/eval/transposition implementation
- `include/fast_engine/` - public engine headers
- `external/chess.hpp` - chess move generation/types dependency
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

## Version released on 11 Feb 2026 
Score of ShakeyBot_New vs ShakeyBot_Lichess: 1431 - 1114 - 1099 [0.543]

...      ShakeyBot_New playing White: 754 - 520 - 548  [0.564] 1822

...      ShakeyBot_New playing Black: 677 - 594 - 551  [0.523] 1822

...      White vs Black: 1348 - 1197 - 1099  [0.521] 3644

Elo difference: 30.3 +/- 9.4, LOS: 100.0 %, DrawRatio: 30.2 %

SPRT: llr 0 (0.0%), lbound -inf, ubound inf

3644 of 10000 games finished.

## Notes

- Search and evaluation are currently organized as thin orchestrators (`src/search.cpp`, `src/evaluation.cpp`) that include internal module fragments under `src/search/*.inc` and `src/eval/*.inc`.
- Engine options are exposed via UCI `setoption` in `apps/fast_engine_uci.cpp`.


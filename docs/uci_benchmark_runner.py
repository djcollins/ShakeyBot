#!/usr/bin/env python3
"""
Run ShakeyBot's standard 9-position UCI benchmark.

Usage from repo root:
    python docs/uci_benchmark_runner.py --output "Benchmarks after my patch.txt"

Optional:
    python docs/uci_benchmark_runner.py --engine build/bin/ShakeyBot.exe --depth 15

This is a benchmark sanity tool only. Tournament Elo decides strength.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable


STANDARD_FENS = [
    "r1b1kb1r/p1pn1ppp/2p1P3/3p4/5B2/2N5/PqP1QPPP/R3KB1R w KQkq - 0 11",
    "r1bq1rk1/ppppnpp1/7p/n7/2B1PN2/4PN2/PP4PP/R2Q1RK1 w - - 3 12",
    "r1b1r1k1/pp3ppp/5b2/3qn3/3pB1PP/5Q2/PPPN1P2/R1B1K2R w KQ - 5 15",
    "6k1/1p3rpp/p1p3q1/P1Pr4/3PbP2/1Q2R1BP/6P1/3R2K1 b - - 1 27",
    "k3rr2/p1p5/1qbb2pp/2N5/PP5Q/2P1B3/6PP/R3R1K1 b - - 0 24",
    "5k2/pprq1ppp/4pb1B/3n4/P5QP/1P3B2/5PP1/3R2K1 w - - 1 26",
    "8/pp3p2/2p3p1/3k4/5PPP/P7/5K2/8 w - - 1 37",
    "5kb1/8/8/1K6/1P6/P7/8/8 w - - 3 59",
    "8/1pp5/3p4/pP1Pp1k1/P1P2pPp/3P1P2/8/5K2 w - - 3 39",
]


AGGREGATE_KEYS = [
    "nodes",
    "q10",
    "q10r",
    "tt_hits",
    "tt_misses",
    "razorAttempts",
    "razorCutoffs",
    "pvchg10",
    "probcutNodes",
    "probcutCandidates",
    "probcutSeeRejects",
    "probcutQsPasses",
    "probcutSearches",
    "probcutCutoffs",
]


def metric(line: str, key: str, cast: Callable[[str], int | float] = int) -> int | float | None:
    match = re.search(rf"{re.escape(key)}=([-0-9.]+)", line)
    if not match:
        return None
    return cast(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ShakeyBot standard UCI benchmark.")
    parser.add_argument("--engine", default="build/bin/ShakeyBot.exe", help="Path to UCI engine binary.")
    parser.add_argument("--depth", type=int, default=15, help="Fixed depth for each position.")
    parser.add_argument("--output", help="Optional output text file.")
    parser.add_argument("--label", default="ShakeyBot benchmark sanity run", help="Header label.")
    args = parser.parse_args()

    engine = Path(args.engine)
    if not engine.exists():
        print(f"Engine not found: {engine}", file=sys.stderr)
        return 2

    out_file = open(args.output, "w", encoding="utf-8") if args.output else None

    def emit(text: str = "") -> None:
        print(text, flush=True)
        if out_file:
            out_file.write(text + "\n")
            out_file.flush()

    try:
        emit(args.label)
        emit("")
        emit("Run method:")
        emit("- scripted UCI runner")
        emit(f"- {engine}")
        emit(f"- go depth {args.depth}")
        emit("- ucinewgame before every position")
        emit("")
        emit("Note:")
        emit("- Benchmark is sanity data only; tournament Elo decides.")
        emit("")

        process = subprocess.Popen(
            [str(engine)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def send(command: str, echo: bool = False) -> None:
            if echo:
                emit(command)
            assert process.stdin is not None
            process.stdin.write(command + "\n")
            process.stdin.flush()

        def read_until(predicate: Callable[[str], bool], collect: bool = False) -> list[str]:
            lines: list[str] = []
            assert process.stdout is not None
            while True:
                raw = process.stdout.readline()
                if raw == "":
                    raise RuntimeError("engine exited before expected response")
                line = raw.rstrip("\r\n")
                if collect:
                    emit(line)
                    lines.append(line)
                if predicate(line):
                    return lines

        send("uci")
        read_until(lambda line: line == "uciok")
        send("isready")
        read_until(lambda line: line == "readyok")

        totals: dict[str, int | float] = {"positions": 0, "time": 0.0}
        totals.update({key: 0 for key in AGGREGATE_KEYS})

        for fen in STANDARD_FENS:
            emit("")
            send("ucinewgame", echo=True)
            send("isready")
            read_until(lambda line: line == "readyok")
            send("position fen " + fen, echo=True)
            send(f"go depth {args.depth}", echo=True)
            lines = read_until(lambda line: line.startswith("bestmove"), collect=True)

            for line in lines:
                if not line.startswith("[GO]"):
                    continue
                totals["positions"] = int(totals["positions"]) + 1
                for key in AGGREGATE_KEYS:
                    value = metric(line, key)
                    if value is not None:
                        totals[key] = int(totals[key]) + int(value)
                time_value = metric(line, "time", float)
                if time_value is not None:
                    totals["time"] = float(totals["time"]) + float(time_value)

        send("quit")
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

        hits = int(totals["tt_hits"])
        misses = int(totals["tt_misses"])
        tt_rate = (100.0 * hits / (hits + misses)) if (hits + misses) else 0.0

        emit("")
        emit("Aggregate:")
        emit(
            f"positions={totals['positions']} "
            f"nodes={totals['nodes']} "
            f"time={float(totals['time']):.2f}s "
            f"q10={totals['q10']} "
            f"q10r={totals['q10r']} "
            f"tt_hits={hits} "
            f"tt_misses={misses} "
            f"tt_hit_rate={tt_rate:.1f}% "
            f"razorAttempts={totals['razorAttempts']} "
            f"razorCutoffs={totals['razorCutoffs']} "
            f"pvchg10={totals['pvchg10']} "
            f"probcutNodes={totals['probcutNodes']} "
            f"probcutCandidates={totals['probcutCandidates']} "
            f"probcutSeeRejects={totals['probcutSeeRejects']} "
            f"probcutQsPasses={totals['probcutQsPasses']} "
            f"probcutSearches={totals['probcutSearches']} "
            f"probcutCutoffs={totals['probcutCutoffs']}"
        )
    finally:
        if out_file:
            out_file.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

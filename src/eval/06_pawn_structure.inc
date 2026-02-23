// Evaluation module: pawn structure helpers
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

// ---------------------- Pawn structure (permanent) ----------------------
//
// Intentionally simple and stable:
//   - isolated pawns
//   - doubled pawns (extra pawns on a file)
//   - pawn islands (extra islands beyond the first)
//
// NOTE: All passed-pawn / race / blockade logic is intentionally removed here to
//       keep this term predictable and low-risk.

constexpr double ISOLATED_PAWN_PENALTY = 0.10;
constexpr double DOUBLED_PAWN_PENALTY = 0.08;
constexpr double PAWN_ISLAND_PENALTY = 0.06;

static inline int pawn_island_count(Bitboard pawns) noexcept
{
    int islands = 0;
    bool prev_has = false;
    for (int file = 0; file < 8; ++file)
    {
        const bool has = static_cast<bool>(pawns & chess::attacks::MASK_FILE[file]);
        if (has && !prev_has)
            ++islands;
        prev_has = has;
    }
    return islands;
}

[[maybe_unused]] double pawn_structure_term_for_color(const Board &board, Color color)
{
    const Bitboard pawns = board.pieces(PieceType::PAWN, color);
    if (!pawns)
        return 0.0;

    int isolated = 0;
    int doubled = 0;

    for (int file = 0; file < 8; ++file)
    {
        const Bitboard file_pawns = pawns & chess::attacks::MASK_FILE[file];
        if (!file_pawns)
            continue;

        const int cnt = static_cast<int>(file_pawns.count());

        // Doubled pawns: count extra pawns beyond the first on the file.
        if (cnt > 1)
            doubled += (cnt - 1);

        // Isolated pawns: no friendly pawn on adjacent files.
        Bitboard adj = 0;
        if (file > 0)
            adj |= (pawns & chess::attacks::MASK_FILE[file - 1]);
        if (file < 7)
            adj |= (pawns & chess::attacks::MASK_FILE[file + 1]);
        if (!adj)
            isolated += cnt;
    }

    const int islands = pawn_island_count(pawns);
    const int extra_islands = std::max(0, islands - 1);

    double score = 0.0;
    score -= ISOLATED_PAWN_PENALTY * static_cast<double>(isolated);
    score -= DOUBLED_PAWN_PENALTY * static_cast<double>(doubled);
    score -= PAWN_ISLAND_PENALTY * static_cast<double>(extra_islands);
    return score;
}

double pawn_structure_term_white_minus_black(const Board &board, double midgame_weight, double endgame_weight)
{
    const auto &pe = pawn_hash_probe(board);
    // We store pawn-structure in MG/EG convention; blend happens here.
    return midgame_weight * pe.pawn_structure_mid_wmb + endgame_weight * pe.pawn_structure_end_wmb;
}

constexpr double PASSED_PAWN_BONUS_BY_ADVANCE[8] = {
    0.0,  // advance 0 (shouldn't occur for legal pawns)
    0.0,  // advance 1 (starting rank)
    0.05, // advance 2
    0.10, // advance 3
    0.20, // advance 4
    0.35, // advance 5
    0.55, // advance 6 (7th rank)
    0.0   // advance 7 (promotion rank, shouldn't occur as pawn)
};

static inline const Bitboard &passed_mask_for(Color c, int sq)
{
    // masks[color_index][sq]
    static Bitboard masks[2][64];
    static bool inited = false;
    if (!inited)
    {
        for (int s = 0; s < 64; ++s)
        {
            const int r = s >> 3;
            const int f = s & 7;

            for (int df = -1; df <= 1; ++df)
            {
                const int ff = f + df;
                if (ff < 0 || ff > 7)
                    continue;

                // White: squares with higher rank on files f-1,f,f+1
                for (int rr = r + 1; rr <= 7; ++rr)
                {
                    masks[0][s] |= Bitboard::fromSquare(rr * 8 + ff);
                }
                // Black: squares with lower rank on files f-1,f,f+1
                for (int rr = r - 1; rr >= 0; --rr)
                {
                    masks[1][s] |= Bitboard::fromSquare(rr * 8 + ff);
                }
            }
        }
        inited = true;
    }
    return (c == Color::WHITE) ? masks[0][sq] : masks[1][sq];
}

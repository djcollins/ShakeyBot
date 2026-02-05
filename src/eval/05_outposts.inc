// Evaluation module: outposts
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

// ---------------------- Minor-piece outposts (middlegame-weighted) ----------------------
//
// Outpost definition:
//  - Must be supported by >=1 pawn (bonus if supported by 2 pawns)
//  - Must NOT be pawn-chaseable (Stockfish-style pawnAttacksSpan gate)
//
// Notes:
//  - We intentionally do NOT try to predict multi-move pawn chases here; we’re aiming
//    for a strict, low-false-positive signal that won’t cause shallow-search optimism.
double outpost_term_for_color(const Board &board, Color us)
{
    using namespace chess;

    // Tunable constants (kept inside the function so you can tweak easily).
    constexpr double KNIGHT_OUTPOST_BONUS = 0.18;        // ~18cp
    constexpr double BISHOP_OUTPOST_BONUS = 0.10;        // ~10cp
    constexpr double OUTPOST_CENTER_FILE_BONUS = 0.03;   // extra on c..f
    constexpr double OUTPOST_2PAWN_SUPPORT_BONUS = 0.03; // extra if supported by 2 pawns

    const Bitboard our_pawns = board.pieces(PieceType::PAWN, us);

    // Use Stockfish-style pawnAttacksSpan (cached) as the gate: an outpost square
    // must not be controllable by an enemy pawn push/capture.
    const auto &pe = pawn_hash_probe(board);
    const std::uint64_t enemy_pawn_span_bits = (us == Color::WHITE) ? pe.pawn_attacks_span_b
                                                                    : pe.pawn_attacks_span_w;

    double score = 0.0;

    auto score_piece = [&](PieceType pt, double base_bonus)
    {
        Bitboard bb = board.pieces(pt, us);
        while (bb)
        {
            const int s = static_cast<int>(bb.pop());
            const int r = s >> 3;
            const int f = s & 7;

            // Only count outposts in enemy-ish territory (avoid shallow “home rank outposts”).
            const int advance = (us == Color::WHITE) ? r : (7 - r);
            if (advance < 3 || advance > 6)
                continue;

            // 1) Pawn support required (and bonus if 2 pawns support).
            const int support_cnt = pawn_support_count_for_square(our_pawns, us, s);
            if (support_cnt == 0)
                continue;

            // 2) No enemy pawn can capture/chase it (pawnAttacksSpan).
            if (enemy_pawn_span_bits & (1ULL << s))
                continue;

            double bonus = base_bonus;

            // Center files (c..f) get a small extra.
            if (f >= 2 && f <= 5)
                bonus += OUTPOST_CENTER_FILE_BONUS;

            // Extra if supported by two pawns.
            if (support_cnt == 2)
                bonus += OUTPOST_2PAWN_SUPPORT_BONUS;

            score += bonus;
        }
    };

    score_piece(PieceType::KNIGHT, KNIGHT_OUTPOST_BONUS);
    score_piece(PieceType::BISHOP, BISHOP_OUTPOST_BONUS);

    return score;
}

double outpost_term_white_minus_black(const Board &board)
{
    const double w = outpost_term_for_color(board, Color::WHITE);
    const double b = outpost_term_for_color(board, Color::BLACK);
    return w - b;
}

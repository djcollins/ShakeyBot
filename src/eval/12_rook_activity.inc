// Evaluation module: rook activity
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

// ---------------------- Rook activity (open files + 7th rank) ----------------------
//
// Stable, low-risk rook heuristics:
//  - Open file: no pawns of either color on the file.
//  - Semi-open file: no own pawns on the file, but enemy pawn(s) exist.
//  - Rook on 7th (or 2nd for Black): only if it hits enemy pawns on that rank OR traps king on back rank.
//
// Values are in pawns; we blend MG/EG using mg_weight/eg_weight from evaluate().
constexpr double ROOK_OPEN_FILE_BONUS_MG = 0.06;
constexpr double ROOK_OPEN_FILE_BONUS_EG = 0.10;
constexpr double ROOK_SEMI_OPEN_FILE_BONUS_MG = 0.04;
constexpr double ROOK_SEMI_OPEN_FILE_BONUS_EG = 0.07;
constexpr double ROOK_ON_7TH_BONUS_MG = 0.10;
constexpr double ROOK_ON_7TH_BONUS_EG = 0.16;

double rook_activity_term_for_color(const Board &board, Color us, double mg_weight, double eg_weight)
{
    using namespace chess;

    const Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;

    const Bitboard our_rooks = board.pieces(PieceType::ROOK, us);
    if (!our_rooks)
        return 0.0;

    const Bitboard our_pawns = board.pieces(PieceType::PAWN, us);
    const Bitboard enemy_pawns = board.pieces(PieceType::PAWN, them);

    const Square enemy_king = board.kingSq(them);
    const int enemy_king_rank = enemy_king.is_valid() ? (enemy_king.index() >> 3) : -1;
    const bool enemy_king_back_rank = (enemy_king_rank == ((us == Color::WHITE) ? 7 : 0));

    const Bitboard enemy_pawns_on_7th =
        enemy_pawns & attacks::MASK_RANK[(us == Color::WHITE) ? 6 : 1];

    // Precompute blended bonuses once per call.
    const double open_bonus = ROOK_OPEN_FILE_BONUS_MG * mg_weight + ROOK_OPEN_FILE_BONUS_EG * eg_weight;
    const double semi_bonus = ROOK_SEMI_OPEN_FILE_BONUS_MG * mg_weight + ROOK_SEMI_OPEN_FILE_BONUS_EG * eg_weight;
    const double rook7_bonus = ROOK_ON_7TH_BONUS_MG * mg_weight + ROOK_ON_7TH_BONUS_EG * eg_weight;

    double score = 0.0;

    Bitboard bb = our_rooks;
    while (bb)
    {
        const int s = static_cast<int>(bb.pop());
        const int r = s >> 3;
        const int f = s & 7;

        const Bitboard file_mask = attacks::MASK_FILE[f];
        const bool has_our_pawn = static_cast<bool>(our_pawns & file_mask);
        const bool has_enemy_pawn = static_cast<bool>(enemy_pawns & file_mask);

        if (!has_our_pawn && !has_enemy_pawn)
            score += open_bonus;
        else if (!has_our_pawn && has_enemy_pawn)
            score += semi_bonus;

        const bool on_7th = (us == Color::WHITE) ? (r == 6) : (r == 1);
        if (on_7th && (enemy_king_back_rank || enemy_pawns_on_7th))
            score += rook7_bonus;
    }

    return score;
}

double rook_activity_term_white_minus_black(const Board &board, double mg_weight, double eg_weight)
{
    const double w = rook_activity_term_for_color(board, Color::WHITE, mg_weight, eg_weight);
    const double b = rook_activity_term_for_color(board, Color::BLACK, mg_weight, eg_weight);
    return w - b;
}

// ---------------------- Rook file activity (permanent) ----------------------
//
// Open / semi-open file bonuses for rooks.
//
// NOTE: "Rook on 7th" is intentionally DISABLED in this build to isolate the
// impact of open/semi-open file heuristics in A/B testing.

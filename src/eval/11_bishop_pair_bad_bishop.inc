// Evaluation module: bishop pair + bad bishop
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

// ---------------------- Bishop pair + bad bishop (permanent) ----------------------
//
// Bishop pair: handled by the Stockfish 12 material-imbalance term.
// Bad bishop: penalize bishops hemmed in by own pawns on the same color complex.
// Taper both terms by phase; the bad-bishop penalty is mainly a middlegame effect.
//
// NOTE: Re-enabled with conservative weights (open/semi-open files + rook on 7th).
// If it regresses, flip ENABLE_ROOK_FILE_ACTIVITY back to false.

constexpr bool ENABLE_ROOK_FILE_ACTIVITY = true;

constexpr double BAD_BISHOP_PAWN_PENALTY_MG = 0.030;
constexpr double BAD_BISHOP_PAWN_PENALTY_EG = 0.010;

// A1 is dark in this bitboard mapping (bit 0), so dark squares start with 0x55 on rank 1.
constexpr Bitboard DARK_SQUARES(0xAA55AA55AA55AA55ULL);
constexpr Bitboard LIGHT_SQUARES(0x55AA55AA55AA55AAULL);

double bishop_pair_bad_bishop_term_for_color(const Board &board, Color color, double mg_weight, double eg_weight)
{
    Bitboard bishops = board.pieces(PieceType::BISHOP, color);
    if (!bishops)
        return 0.0;

    const Bitboard pawns = board.pieces(PieceType::PAWN, color);

    const int pawns_dark = static_cast<int>((pawns & DARK_SQUARES).count());
    const int pawns_light = static_cast<int>((pawns & LIGHT_SQUARES).count());

    // Bishop pair is already handled by the Stockfish 12 material-imbalance term
    // (via the synthetic "pair" piece), so we only keep the bad-bishop component here.
    const double pawn_pen = BAD_BISHOP_PAWN_PENALTY_MG * mg_weight + BAD_BISHOP_PAWN_PENALTY_EG * eg_weight;

    double score = 0.0;

    // Bad bishop: count own pawns on the same color squares as each bishop.
    // Intentionally simple and fast.
    Bitboard bb = bishops;
    while (bb)
    {
        const std::uint8_t s = bb.pop();
        const Bitboard sq_bb = Bitboard::fromSquare(static_cast<int>(s));
        const bool is_dark = static_cast<bool>(sq_bb & DARK_SQUARES);

        const int same_color_pawns = is_dark ? pawns_dark : pawns_light;
        score -= pawn_pen * static_cast<double>(same_color_pawns);
    }

    return score;
}

double bishop_pair_bad_bishop_term_white_minus_black(const Board &board, double mg_weight, double eg_weight)
{
    const double w = bishop_pair_bad_bishop_term_for_color(board, Color::WHITE, mg_weight, eg_weight);
    const double b = bishop_pair_bad_bishop_term_for_color(board, Color::BLACK, mg_weight, eg_weight);
    return w - b;
}

// Evaluation module: complexity scaling
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

static inline bool pawn_files_on_one_flank(std::uint8_t file_mask) noexcept
{
    if (file_mask == 0)
        return false;
    return (file_mask & 0x0F) == 0 || (file_mask & 0xF0) == 0;
}

static inline double complexity_scale_factor_white_pov(const Board &board,
                                                       double score_white_pov_pawns,
                                                       const MaterialEntry &mat,
                                                       const EvalInfo &info) noexcept
{
    using namespace chess;

    const double score_mag = std::abs(score_white_pov_pawns);
    if (score_mag < 0.30)
        return 1.0;

    const Color strong = (score_white_pov_pawns > 0.0) ? Color::WHITE : Color::BLACK;
    const Color weak = (strong == Color::WHITE) ? Color::BLACK : Color::WHITE;

    const int total_pawns = static_cast<int>(mat.wP) + static_cast<int>(mat.bP);
    const int total_minors = static_cast<int>(mat.w_minors) + static_cast<int>(mat.b_minors);
    const int total_rooks = static_cast<int>(mat.wR) + static_cast<int>(mat.bR);
    const int total_queens = static_cast<int>(mat.wQ) + static_cast<int>(mat.bQ);
    const int strong_pawns = (strong == Color::WHITE) ? static_cast<int>(mat.wP) : static_cast<int>(mat.bP);
    const int weak_pawns = (weak == Color::WHITE) ? static_cast<int>(mat.wP) : static_cast<int>(mat.bP);

    int drawish = 0;

    if (total_queens == 0)
        drawish += 1;
    if (total_rooks == 0)
        drawish += 1;
    if (total_pawns <= 6)
        drawish += 1;
    if (total_pawns <= 4)
        drawish += 1;
    if (total_minors <= 2)
        drawish += 1;
    if (strong_pawns == 0 || weak_pawns == 0)
        drawish += 1;

    const std::uint8_t all_pawn_files = static_cast<std::uint8_t>(info.pawn_files[cidx(Color::WHITE)] |
                                                                  info.pawn_files[cidx(Color::BLACK)]);
    if (pawn_files_on_one_flank(all_pawn_files))
        drawish += 1;

    if (total_queens == 0 &&
        total_rooks == 0 &&
        mat.wB == 1 && mat.bB == 1)
    {
        Bitboard wb = board.pieces(PieceType::BISHOP, Color::WHITE);
        Bitboard bb = board.pieces(PieceType::BISHOP, Color::BLACK);
        if (!wb.empty() && !bb.empty() && single_bishop_is_dark(wb) != single_bishop_is_dark(bb))
            drawish += 2;
    }

    double sf = 1.0 - 0.04 * static_cast<double>(drawish);

    if (score_mag >= 3.0)
        sf += 0.08;
    else if (score_mag >= 2.0)
        sf += 0.04;

    if (sf < 0.65)
        sf = 0.65;
    if (sf > 1.0)
        sf = 1.0;
    return sf;
}

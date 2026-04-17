// Evaluation module: closedness / openness
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

static inline int closedness_index_0_8(const Board &board, const EvalInfo &info) noexcept
{
    using namespace chess;

    const Bitboard white_pawns = board.pieces(PieceType::PAWN, Color::WHITE);
    const Bitboard black_pawns = board.pieces(PieceType::PAWN, Color::BLACK);
    const int pawns_total = static_cast<int>((white_pawns | black_pawns).count());

    const std::uint8_t occupied_files = static_cast<std::uint8_t>(info.pawn_files[cidx(Color::WHITE)] |
                                                                  info.pawn_files[cidx(Color::BLACK)]);
    const int open_files = std::popcount(static_cast<unsigned>(static_cast<std::uint8_t>(~occupied_files)));

    // Rammed pawns are opposing pawns blocked directly on the same file.
    const int rammed = static_cast<int>((white_pawns & attacks::shift<Direction::SOUTH>(black_pawns)).count());

    // 0 = very open, 8 = very closed. Start position lands near the neutral middle.
    return std::clamp((pawns_total + 2 * rammed - 2 * open_files) / 4, 0, 8);
}

static inline double closedness_term_white_minus_black(const Board &board,
                                                       const MaterialEntry &mat,
                                                       double mg_weight,
                                                       const EvalInfo &info)
{
    static constexpr int KNIGHT_CP[9] = {-10, -8, -6, -3, 0, 3, 6, 9, 12};
    static constexpr int ROOK_CP[9] = {12, 9, 6, 3, 0, -3, -6, -9, -12};

    const int idx = closedness_index_0_8(board, info);
    const int knight_delta = static_cast<int>(mat.wN) - static_cast<int>(mat.bN);
    const int rook_delta = static_cast<int>(mat.wR) - static_cast<int>(mat.bR);
    const int score_cp = KNIGHT_CP[idx] * knight_delta + ROOK_CP[idx] * rook_delta;

    return mg_weight * (static_cast<double>(score_cp) / 100.0);
}

// Evaluation module: material + phase computation
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

constexpr double MATE_SCORE = 1e6; // Checkmate score is 1 million
/*
using chess::Board;
using chess::Color;
using chess::PieceType;
using chess::Bitboard;
using chess::Square;
*/
constexpr int STOCK_OPENING_PHASE_SCORE = 15258;
constexpr int STOCK_ENDGAME_PHASE_SCORE = 3915;

// Piece types we care about.
constexpr PieceType PIECE_TYPES[] = {
    PieceType::PAWN,
    PieceType::KNIGHT,
    PieceType::BISHOP,
    PieceType::ROOK,
    PieceType::QUEEN,
    PieceType::KING,
};

// Sum signed material for a given color, in pawns.
// Uses piece_value(PieceType, Color) from config.hpp.
double material_for_color(const Board &board, Color color)
{
    double total = 0.0;

    for (PieceType pt : PIECE_TYPES)
    {
        auto bb = board.pieces(pt, color); // (PieceType, Color)
        int count = static_cast<int>(bb.count());
        if (!count)
            continue;

        total += count * piece_value(pt, color);
    }

    return total;
}

// Net material from White's point of view, in pawn units.
// Positive means White is ahead.
double material_white_pov_pawns(const Board &board)
{
    const Color white(Color::WHITE);
    const Color black(Color::BLACK);

    const double white_sum = material_for_color(board, white); // positive
    const double black_sum = material_for_color(board, black); // negative
    return white_sum + black_sum;
}

// ---------------------- Phase computation ----------------------

// Mirrors your Python _PHASE_* logic.
constexpr int PHASE_PAWN = 0;
constexpr int PHASE_KNIGHT = 1;
constexpr int PHASE_BISHOP = 1;
constexpr int PHASE_ROOK = 2;
constexpr int PHASE_QUEEN = 4;

constexpr int TOTAL_PHASE =
    4 * PHASE_KNIGHT +
    4 * PHASE_BISHOP +
    4 * PHASE_ROOK +
    2 * PHASE_QUEEN;

// Returns phase in [0, 256], same semantics as Python _compute_phase_0_256.
int compute_phase_0_256(const Board &board)
{
    int phase = TOTAL_PHASE;

    auto sub_piece = [&](PieceType pt, int weight)
    {
        auto w = board.pieces(pt, Color::WHITE);
        auto b = board.pieces(pt, Color::BLACK);
        int count = static_cast<int>(w.count() + b.count());
        phase -= weight * count;
    };

    sub_piece(PieceType::KNIGHT, PHASE_KNIGHT);
    sub_piece(PieceType::BISHOP, PHASE_BISHOP);
    sub_piece(PieceType::ROOK, PHASE_ROOK);
    sub_piece(PieceType::QUEEN, PHASE_QUEEN);

    if (phase < 0)
        phase = 0;

    // int((phase * 256 + TOTAL_PHASE / 2) // TOTAL_PHASE)
    int num = phase * 256 + TOTAL_PHASE / 2;
    return num / TOTAL_PHASE;
}

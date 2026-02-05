#include "fast_engine/config.hpp"

namespace fast_engine
{

    const EngineConfig DEFAULT_CONFIG{};

    namespace
    {
        using chess::PieceType;
        using PTU = PieceType::underlying;

        // Base piece values in pawns (no sign), index = PieceType underlying value.
        // Order: PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NONE
        constexpr double BASE_PIECE_VALUES[] = {
            1.0, // PAWN
            3.2, // KNIGHT
            3.3, // BISHOP
            5.0, // ROOK
            9.0, // QUEEN
            2.0, // KING (matches your Python config)
            0.0  // NONE
        };
    } // anonymous namespace

    double piece_value(PieceType pt, chess::Color color)
    {
        int idx = static_cast<int>(pt);
        if (idx < 0 || idx >= static_cast<int>(std::size(BASE_PIECE_VALUES)))
            return 0.0;

        double v = BASE_PIECE_VALUES[idx];

        // Apply sign: white positive, black negative, none = 0
        if (color == chess::Color(chess::Color::underlying::WHITE))
            return v;
        if (color == chess::Color(chess::Color::underlying::BLACK))
            return -v;
        return 0.0;
    }

    double piece_value_signed(chess::Piece piece)
    {
        if (piece == chess::Piece::NONE)
            return 0.0;
        return piece_value(piece.type(), piece.color());
    }

} // namespace fast_engine
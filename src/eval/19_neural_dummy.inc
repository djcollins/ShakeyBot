// Evaluation module: stateless neural plumbing dummy
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

constexpr int SIMPLE_768_FEATURE_COUNT = 64 * 6 * 2;
constexpr int SIMPLE_768_MAX_ACTIVE = 32;
constexpr int DUMMY_HIDDEN_UNITS = 16;

struct Simple768Features
{
    std::array<int, SIMPLE_768_MAX_ACTIVE> active{};
    int count = 0;
};

static_assert(SIMPLE_768_FEATURE_COUNT == 768);

static inline int simple_768_feature_index(Color color, PieceType type, Square square) noexcept
{
    const int color_index = static_cast<int>(color);
    const int piece_index = static_cast<int>(type);
    return ((color_index * 6 + piece_index) * 64) + square.index();
}

static inline Simple768Features encode_simple_768_features(const Board &board) noexcept
{
    Simple768Features features{};

    for (Color color : {Color::WHITE, Color::BLACK})
    {
        for (int piece_index = 0; piece_index < 6; ++piece_index)
        {
            const PieceType type(static_cast<PieceType::underlying>(piece_index));
            Bitboard pieces = board.pieces(type, color);
            while (pieces)
            {
                const Square square(pieces.pop());
                if (features.count < SIMPLE_768_MAX_ACTIVE)
                    features.active[features.count++] = simple_768_feature_index(color, type, square);
            }
        }
    }

    return features;
}

static inline std::uint32_t dummy_mix32(std::uint32_t x) noexcept
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static inline int dummy_weight(std::uint32_t key, int spread) noexcept
{
    const std::uint32_t span = static_cast<std::uint32_t>(2 * spread + 1);
    return static_cast<int>(dummy_mix32(key) % span) - spread;
}

static inline Score evaluate_neural_dummy_white_pov(const Board &board)
{
    const Simple768Features features = encode_simple_768_features(board);
    std::array<int, DUMMY_HIDDEN_UNITS> hidden{};

    for (int h = 0; h < DUMMY_HIDDEN_UNITS; ++h)
    {
        int acc = dummy_weight(0xB5297A4Du ^ static_cast<std::uint32_t>(h * 0x68E31DA4u), 64);
        for (int i = 0; i < features.count; ++i)
        {
            const std::uint32_t key = 0x1B56C4E9u ^
                                      static_cast<std::uint32_t>(features.active[i] * 0x45D9F3Bu) ^
                                      static_cast<std::uint32_t>(h * 0x9E3779B9u);
            acc += dummy_weight(key, 31);
        }
        hidden[h] = std::clamp(acc, 0, 512);
    }

    int output = dummy_weight(0xA511E9B3u, 512);
    for (int h = 0; h < DUMMY_HIDDEN_UNITS; ++h)
    {
        const std::uint32_t key = 0x63D83595u ^ static_cast<std::uint32_t>(h * 0x27D4EB2Du);
        output += hidden[h] * dummy_weight(key, 16);
    }

    return static_cast<Score>(std::clamp(output / 64, -2000, 2000));
}

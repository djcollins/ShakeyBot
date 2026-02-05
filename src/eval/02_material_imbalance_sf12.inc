// Evaluation module: Stockfish 12 material imbalance + material cache
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

// ---------------------- Material imbalance + material flags (Stockfish 12) ----------------------
//
// High-ROI evaluation component:
//   - Stockfish 12 quadratic material-imbalance polynomial (Tord Romstad).
//   - Cached by material key (piece counts only) to avoid recomputation.
//   - Exposes material-derived flags used to gate noisy terms (Donna-style usage),
//     e.g. king safety is disabled when the attacker lacks sufficient material.
//
// Notes on units:
//   Stockfish computes:
//     value = (imbalance<WHITE>(pc) - imbalance<BLACK>(pc)) / 16
//   in Stockfish Value units where PawnValueEg == 208.
//   We convert to this engine's pawn units (1.0 = one pawn) by dividing by 208.0.

constexpr int SF12_PAWN_VALUE_EG = 208;

enum Sf12MatPt : int
{
    SF12_PAIR = 0, // synthetic piece: bishop pair presence (0/1)
    SF12_PAWN = 1,
    SF12_KNIGHT = 2,
    SF12_BISHOP = 3,
    SF12_ROOK = 4,
    SF12_QUEEN = 5,
    SF12_PT_NB = 6
};

// Stockfish 12 material-imbalance parameters (src/material.cpp)
constexpr int SF12_QuadraticOurs[SF12_PT_NB][SF12_PT_NB] = {
    // pair, pawn, knight, bishop, rook, queen
    {1438, 0, 0, 0, 0, 0},          // Pair
    {40, 38, 0, 0, 0, 0},           // Pawn
    {32, 255, -62, 0, 0, 0},        // Knight
    {0, 104, 4, 0, 0, 0},           // Bishop
    {-26, -2, 47, 105, -208, 0},    // Rook
    {-189, 24, 117, 133, -134, -6}, // Queen
};

constexpr int SF12_QuadraticTheirs[SF12_PT_NB][SF12_PT_NB] = {
    // pair, pawn, knight, bishop, rook, queen
    {0, 0, 0, 0, 0, 0},          // Pair
    {36, 0, 0, 0, 0, 0},         // Pawn
    {9, 63, 0, 0, 0, 0},         // Knight
    {59, 65, 42, 0, 0, 0},       // Bishop
    {46, 39, 24, -24, 0, 0},     // Rook
    {97, 100, -42, 137, 268, 0}, // Queen
};

static inline int sf12_imbalance_for_side(const int pc[2][SF12_PT_NB], int us) noexcept
{
    const int them = 1 - us;
    int bonus = 0;

    // Second-degree polynomial material imbalance, by Tord Romstad.
    for (int pt1 = 0; pt1 < SF12_PT_NB; ++pt1)
    {
        if (!pc[us][pt1])
            continue;

        int v = SF12_QuadraticOurs[pt1][pt1] * pc[us][pt1];

        for (int pt2 = 0; pt2 < pt1; ++pt2)
            v += SF12_QuadraticOurs[pt1][pt2] * pc[us][pt2] + SF12_QuadraticTheirs[pt1][pt2] * pc[them][pt2];

        bonus += pc[us][pt1] * v;
    }

    return bonus;
}

static inline std::uint64_t material_key(const Board &board) noexcept
{
    // Pack piece counts into 10 nibbles (40 bits):
    //   WP WN WB WR WQ BP BN BB BR BQ
    auto nib = [](std::uint64_t k, int idx, int v) noexcept
    {
        return k | (std::uint64_t(v & 0xF) << (idx * 4));
    };

    const int wp = static_cast<int>(board.pieces(PieceType::PAWN, Color::WHITE).count());
    const int wn = static_cast<int>(board.pieces(PieceType::KNIGHT, Color::WHITE).count());
    const int wb = static_cast<int>(board.pieces(PieceType::BISHOP, Color::WHITE).count());
    const int wr = static_cast<int>(board.pieces(PieceType::ROOK, Color::WHITE).count());
    const int wq = static_cast<int>(board.pieces(PieceType::QUEEN, Color::WHITE).count());

    const int bp = static_cast<int>(board.pieces(PieceType::PAWN, Color::BLACK).count());
    const int bn = static_cast<int>(board.pieces(PieceType::KNIGHT, Color::BLACK).count());
    const int bb = static_cast<int>(board.pieces(PieceType::BISHOP, Color::BLACK).count());
    const int br = static_cast<int>(board.pieces(PieceType::ROOK, Color::BLACK).count());
    const int bq = static_cast<int>(board.pieces(PieceType::QUEEN, Color::BLACK).count());

    std::uint64_t k = 0;
    k = nib(k, 0, wp);
    k = nib(k, 1, wn);
    k = nib(k, 2, wb);
    k = nib(k, 3, wr);
    k = nib(k, 4, wq);
    k = nib(k, 5, bp);
    k = nib(k, 6, bn);
    k = nib(k, 7, bb);
    k = nib(k, 8, br);
    k = nib(k, 9, bq);

    // Avoid 0 as a sentinel (also diffuses extremely low-material cases slightly).
    return k | 1ULL;
}

struct MaterialEntry
{
    std::uint64_t key = 0ULL;

    // White POV, in pawn units.
    double imbalance_pawns_wmb = 0.0;

    // Piece counts (for gates).
    std::uint8_t wP = 0, wN = 0, wB = 0, wR = 0, wQ = 0;
    std::uint8_t bP = 0, bN = 0, bB = 0, bR = 0, bQ = 0;

    std::uint8_t w_minors = 0, b_minors = 0;
    std::uint8_t w_heavies = 0, b_heavies = 0;
};

constexpr int MATERIAL_CACHE_CLUSTER = 4;
constexpr std::size_t MATERIAL_CACHE_BUCKETS = (1u << 16); // 65,536 buckets => ~262k entries

struct MaterialBucket
{
    MaterialEntry e[MATERIAL_CACHE_CLUSTER];
};

class MaterialTable
{
public:
    MaterialTable()
        : mask_(MATERIAL_CACHE_BUCKETS - 1)
    {
        table_.assign(MATERIAL_CACHE_BUCKETS, MaterialBucket{});
    }

    const MaterialEntry &probe(const Board &board) noexcept
    {
        const std::uint64_t key = material_key(board);
        MaterialBucket &b = table_[static_cast<std::size_t>(key) & mask_];

        for (int i = 0; i < MATERIAL_CACHE_CLUSTER; ++i)
        {
            if (b.e[i].key == key)
                return b.e[i];
        }

        const int victim = static_cast<int>((key >> 1) & (MATERIAL_CACHE_CLUSTER - 1));
        MaterialEntry &m = b.e[victim];
        m = compute_entry(board, key);
        return m;
    }

private:
    static MaterialEntry compute_entry(const Board &board, std::uint64_t key) noexcept
    {
        MaterialEntry m{};
        m.key = key;

        m.wP = static_cast<std::uint8_t>(board.pieces(PieceType::PAWN, Color::WHITE).count());
        m.wN = static_cast<std::uint8_t>(board.pieces(PieceType::KNIGHT, Color::WHITE).count());
        m.wB = static_cast<std::uint8_t>(board.pieces(PieceType::BISHOP, Color::WHITE).count());
        m.wR = static_cast<std::uint8_t>(board.pieces(PieceType::ROOK, Color::WHITE).count());
        m.wQ = static_cast<std::uint8_t>(board.pieces(PieceType::QUEEN, Color::WHITE).count());

        m.bP = static_cast<std::uint8_t>(board.pieces(PieceType::PAWN, Color::BLACK).count());
        m.bN = static_cast<std::uint8_t>(board.pieces(PieceType::KNIGHT, Color::BLACK).count());
        m.bB = static_cast<std::uint8_t>(board.pieces(PieceType::BISHOP, Color::BLACK).count());
        m.bR = static_cast<std::uint8_t>(board.pieces(PieceType::ROOK, Color::BLACK).count());
        m.bQ = static_cast<std::uint8_t>(board.pieces(PieceType::QUEEN, Color::BLACK).count());

        m.w_minors = static_cast<std::uint8_t>(m.wN + m.wB);
        m.b_minors = static_cast<std::uint8_t>(m.bN + m.bB);
        m.w_heavies = static_cast<std::uint8_t>(m.wR + m.wQ);
        m.b_heavies = static_cast<std::uint8_t>(m.bR + m.bQ);

        const int pc[2][SF12_PT_NB] = {
            {(m.wB >= 2) ? 1 : 0, m.wP, m.wN, m.wB, m.wR, m.wQ},
            {(m.bB >= 2) ? 1 : 0, m.bP, m.bN, m.bB, m.bR, m.bQ},
        };

        const int imbW = sf12_imbalance_for_side(pc, 0);
        const int imbB = sf12_imbalance_for_side(pc, 1);

        const int sf_value = (imbW - imbB) / 16; // Stockfish Value units (PawnValueEg==208)
        m.imbalance_pawns_wmb = static_cast<double>(sf_value) / static_cast<double>(SF12_PAWN_VALUE_EG);
        return m;
    }

    std::vector<MaterialBucket> table_;
    std::size_t mask_ = 0;
};

static inline MaterialTable &material_table()
{
    static MaterialTable t;
    return t;
}

static inline const MaterialEntry &material_probe(const Board &board) noexcept
{
    return material_table().probe(board);
}

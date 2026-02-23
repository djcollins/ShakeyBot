// Evaluation module: eval cache + pawn hash entry decl
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

struct PawnHashEntry
{
    std::uint64_t pawns_w = 0ULL;
    std::uint64_t pawns_b = 0ULL;

    std::uint64_t passed_w = 0ULL;
    std::uint64_t passed_b = 0ULL;

    // Pawn attack maps (pawn-only, independent of other pieces).
    std::uint64_t pawn_attacks_w = 0ULL;
    std::uint64_t pawn_attacks_b = 0ULL;

    // Squares potentially attacked by pawns as they advance.
    std::uint64_t pawn_attacks_span_w = 0ULL;
    std::uint64_t pawn_attacks_span_b = 0ULL;

    // Pawn structure score from White's POV, split into midgame/endgame components (in pawn units).
    // Midgame/Endgame, the final blend is done in evaluate().
    double pawn_structure_mid_wmb = 0.0;
    double pawn_structure_end_wmb = 0.0;

    std::uint8_t used = 0;
    std::uint8_t _pad[7] = {0};
};

const PawnHashEntry &pawn_hash_probe(const Board &board);

// ---------------------- Full evaluation cache ----------------------
// Caches the full (non-tempo) evaluation from White's POV, keyed by the
// position hash + evaluation-relevant EngineConfig bits.

struct EvalCacheEntry
{
    std::uint64_t key = 0ULL;
    Score eval_white_pov_cp = 0;
    std::uint8_t gen = 0;
    std::uint8_t _pad[3] = {0, 0, 0};
};

constexpr int EVAL_CACHE_CLUSTER_SIZE = 4;
constexpr std::size_t EVAL_CACHE_BUCKETS = (1u << 20); // 1,048,576 buckets => ~4.2M entries

struct EvalCacheBucket
{
    EvalCacheEntry e[EVAL_CACHE_CLUSTER_SIZE];
};

static inline std::uint64_t rotl64(std::uint64_t x, int r)
{
    return (x << r) | (x >> (64 - r));
}

static inline std::uint64_t eval_config_signature(const EngineConfig &cfg)
{
    // Hash only fields that affect evaluate_white_pov_with_config().
    // Use the raw IEEE-754 bits for doubles to avoid accidental aliasing.
    std::uint64_t h = 0x9E3779B97F4A7C15ULL;
    auto mix = [&](std::uint64_t v)
    {
        h ^= v + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
    };

    auto mix_double = [&](double x)
    {
        std::uint64_t u = 0;
        std::memcpy(&u, &x, sizeof(u));
        mix(u);
    };

    mix_double(cfg.king_crowding_scale);
    mix_double(cfg.mobility_scale);
    mix_double(cfg.pst_scale);
    mix_double(cfg.threat_term);
    mix(cfg.use_stock_pst ? 0xF0F0F0F0F0F0F0F0ULL : 0x0ULL);

    // Final avalanche.
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

static inline std::uint64_t eval_cache_key(const Board &board, const EngineConfig &cfg)
{
    // NOTE: evaluate_white_pov_with_config() is independent of side-to-move (tempo
    // is applied later in evaluate_for_side_to_move_with_config). The underlying
    // library hash includes side-to-move and other state; we keep it as-is for
    // correctness and simplicity.
    std::uint64_t k = board.hash();

    const std::uint64_t sig = eval_config_signature(cfg);

    // Combine and diffuse.
    k ^= rotl64(sig, 17);
    k ^= 0xC0FFEE1234ABCDEFULL;
    k |= 1ULL; // avoid 0 as a sentinel (extremely unlikely but free).
    return k;
}

class EvalCacheTable
{
public:
    EvalCacheTable()
        : mask_(EVAL_CACHE_BUCKETS - 1)
    {
        table_.assign(EVAL_CACHE_BUCKETS, EvalCacheBucket{});
        gen_ = 1; // generation 0 is reserved for "unused"
    }

    void clear()
    {
        // Bump generation instead of memset'ing ~64MB per game.
        ++gen_;
        if (gen_ == 0)
        {
            // Wrapped: hard clear and restart generations.
            std::fill(table_.begin(), table_.end(), EvalCacheBucket{});
            gen_ = 1;
        }
    }

    bool probe(std::uint64_t key, Score &out) noexcept
    {
        EvalCacheBucket &b = table_[static_cast<std::size_t>(key) & mask_];
        for (int i = 0; i < EVAL_CACHE_CLUSTER_SIZE; ++i)
        {
            const EvalCacheEntry &e = b.e[i];
            if (e.gen == gen_ && e.key == key)
            {
                out = e.eval_white_pov_cp;
                return true;
            }
        }
        return false;
    }

    void store(std::uint64_t key, Score v) noexcept
    {
        EvalCacheBucket &b = table_[static_cast<std::size_t>(key) & mask_];

        // Pseudo-random replacement to avoid hotspots.
        const int victim = static_cast<int>((key >> 1) & (EVAL_CACHE_CLUSTER_SIZE - 1));
        EvalCacheEntry &e = b.e[victim];
        e.key = key;
        e.eval_white_pov_cp = v;
        e.gen = gen_;
    }

private:
    std::vector<EvalCacheBucket> table_;
    std::size_t mask_ = 0;
    std::uint8_t gen_ = 1;
};

static inline EvalCacheTable &eval_cache_table()
{
    static EvalCacheTable t;
    return t;
}

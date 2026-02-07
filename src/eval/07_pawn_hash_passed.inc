// Evaluation module: pawn hash + passed pawns
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

// ---------------------- Pawn hash (pawn structure cache) ----------------------
//
// Cache pawn-only features keyed by the pawn bitboards (white pawns, black pawns).
// This reduces repeated work across evaluation calls during search.
//
// Cached:
//   - pawn structure (isolated, doubled, islands) from White's POV (centipawns)
//   - passed pawn bitboards for each side (pawn-only definition)
//
// Not cached here (depends on other pieces/king squares/occupancy):
//   - blockade, rook-behind, king-distance adjustments for passed pawns
//
// NOTE: Your current testing uses 1 thread. If you later add multi-threading,
//       make this table thread-safe (e.g., per-thread pawn hash).

static inline int popcount64(std::uint64_t x) noexcept
{
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    // Portable fallback (used only on pawn-hash misses).
    int c = 0;
    while (x)
    {
        x &= (x - 1);
        ++c;
    }
    return c;
#endif
}

static inline std::uint64_t splitmix64(std::uint64_t x) noexcept
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static inline std::uint64_t pawn_key(std::uint64_t pw, std::uint64_t pb) noexcept
{
    // Strong, cheap mix; collisions are extremely unlikely.
    return splitmix64(pw ^ splitmix64(pb + 0xD1B54A32D192ED03ULL));
}

// --- Stockfish-style pawn feature helpers (pawn-only) ---

// Connected pawn bonus by relative rank (0..7) from Stockfish 12 (pawns.cpp).
// Index 0 and 7 are unused for legal pawns and left as 0.
static constexpr int CONNECTED_BONUS_BY_REL_RANK[8] = {0, 7, 8, 11, 24, 45, 85, 0};

// Stockfish 12 pawn-structure penalties/bonuses (centipawns), used inside the pawn hash.
// These are pawn-only terms (no piece/king dependency), matching SF12 pawns.cpp.
static constexpr int PAWN_BACKWARD_MG_CP = 8;
static constexpr int PAWN_BACKWARD_EG_CP = 27;
static constexpr int PAWN_DOUBLED_MG_CP = 11;
static constexpr int PAWN_DOUBLED_EG_CP = 55;
static constexpr int PAWN_ISOLATED_MG_CP = 5;
static constexpr int PAWN_ISOLATED_EG_CP = 17;
static constexpr int PAWN_WEAK_LEVER_MG_CP = 2;
static constexpr int PAWN_WEAK_LEVER_EG_CP = 54;
static constexpr int PAWN_WEAK_UNO_MG_CP = 15;
static constexpr int PAWN_WEAK_UNO_EG_CP = 25;

// Bonus/penalty for blocked pawns on 5th/6th rank from pawn POV (SF12 BlockedPawn[]).
static constexpr int PAWN_BLOCKED_5_MG_CP = -13;
static constexpr int PAWN_BLOCKED_5_EG_CP = -4;
static constexpr int PAWN_BLOCKED_6_MG_CP = -4;
static constexpr int PAWN_BLOCKED_6_EG_CP = 3;

static inline int relative_rank_0_7(Color us, int sq) noexcept
{
    const int r = (sq >> 3) & 7;
    return (static_cast<int>(us) == static_cast<int>(Color::WHITE)) ? r : (7 - r);
}

static inline std::uint64_t forward_ranks_bits(Color us, int sq) noexcept
{
    const int r = (sq >> 3) & 7;
    if (static_cast<int>(us) == static_cast<int>(Color::WHITE))
    {
        if (r >= 7)
            return 0ULL;
        const int sh = (r + 1) * 8;
        return (sh >= 64) ? 0ULL : (~0ULL << sh);
    }
    else
    {
        if (r <= 0)
            return 0ULL;
        const int sh = r * 8;
        return (sh >= 64) ? ~0ULL : ((1ULL << sh) - 1ULL);
    }
}

static inline std::uint64_t forward_file_bits(Color us, int sq) noexcept
{
    const int f = sq & 7;
    return forward_ranks_bits(us, sq) & chess::attacks::MASK_FILE[f].getBits();
}

static inline std::uint64_t adjacent_files_bits(int sq) noexcept
{
    const int f = sq & 7;
    std::uint64_t m = 0ULL;
    if (f > 0)
        m |= chess::attacks::MASK_FILE[f - 1].getBits();
    if (f < 7)
        m |= chess::attacks::MASK_FILE[f + 1].getBits();
    return m;
}

// Stockfish's pawn_attack_span(): squares on adjacent files in front of the pawn.
static inline std::uint64_t pawn_attack_span_bits(Color us, int sq) noexcept
{
    return forward_ranks_bits(us, sq) & adjacent_files_bits(sq);
}

// Compute the same pawn-structure term as pawn_structure_term_for_color(),
// but directly from a pawn bitboard (no Board dependency).
[[maybe_unused]] static inline double pawn_structure_term_from_bits(std::uint64_t pawns_bits) noexcept
{
    const Bitboard pawns(pawns_bits);
    if (!pawns)
        return 0.0;

    int isolated = 0;
    int doubled = 0;

    for (int file = 0; file < 8; ++file)
    {
        const std::uint64_t file_mask = chess::attacks::MASK_FILE[file].getBits();
        const std::uint64_t file_pawns_bits = pawns_bits & file_mask;
        if (!file_pawns_bits)
            continue;

        const int cnt = popcount64(file_pawns_bits);

        // Doubled pawns: count extra pawns beyond the first on the file.
        if (cnt > 1)
            doubled += (cnt - 1);

        // Isolated pawns: no friendly pawn on adjacent files.
        std::uint64_t adj = 0ULL;
        if (file > 0)
            adj |= (pawns_bits & chess::attacks::MASK_FILE[file - 1].getBits());
        if (file < 7)
            adj |= (pawns_bits & chess::attacks::MASK_FILE[file + 1].getBits());
        if (!adj)
            isolated += cnt;
    }

    const int islands = pawn_island_count(pawns);
    const int extra_islands = std::max(0, islands - 1);

    double score = 0.0;
    score -= ISOLATED_PAWN_PENALTY * static_cast<double>(isolated);
    score -= DOUBLED_PAWN_PENALTY * static_cast<double>(doubled);
    score -= PAWN_ISLAND_PENALTY * static_cast<double>(extra_islands);
    return score;
}

static inline std::uint64_t passed_pawns_from_bits(Color c,
                                                   std::uint64_t our_pawns_bits,
                                                   std::uint64_t enemy_pawns_bits) noexcept
{
    std::uint64_t passed = 0ULL;

    Bitboard tmp(our_pawns_bits);
    while (tmp)
    {
        const int s = static_cast<int>(tmp.pop());
        const std::uint64_t mask = passed_mask_for(c, s).getBits();
        if ((enemy_pawns_bits & mask) == 0ULL)
            passed |= (1ULL << s);
    }

    return passed;
}

static constexpr std::size_t PAWN_HASH_CLUSTER_SIZE = 2;

struct PawnHashBucket
{
    PawnHashEntry e[PAWN_HASH_CLUSTER_SIZE];
};

class PawnHashTable
{
public:
    PawnHashTable()
    {
        // 131072 buckets * 2-way = 262144 entries (~10MB with current entry layout).
        const std::size_t buckets = 1u << 17;
        table_.assign(buckets, PawnHashBucket{});
        mask_ = buckets - 1;
    }

    const PawnHashEntry &probe(const Board &board)
    {
        const std::uint64_t pw = board.pieces(PieceType::PAWN, Color::WHITE).getBits();
        const std::uint64_t pb = board.pieces(PieceType::PAWN, Color::BLACK).getBits();

        const std::uint64_t k = pawn_key(pw, pb);
        PawnHashBucket &b = table_[static_cast<std::size_t>(k) & mask_];

        for (auto &ent : b.e)
        {
            if (ent.used && ent.pawns_w == pw && ent.pawns_b == pb)
                return ent;
        }

        // Miss: compute and store.
        PawnHashEntry ne;
        ne.used = 1;
        ne.pawns_w = pw;
        ne.pawns_b = pb;

        // 1) Pawn attacks (fast bitboard generation).
        {
            const Bitboard wp(pw);
            const Bitboard bp(pb);
            const Bitboard w_att = chess::attacks::pawnLeftAttacks<Color::WHITE>(wp) | chess::attacks::pawnRightAttacks<Color::WHITE>(wp);
            const Bitboard b_att = chess::attacks::pawnLeftAttacks<Color::BLACK>(bp) | chess::attacks::pawnRightAttacks<Color::BLACK>(bp);
            ne.pawn_attacks_w = w_att.getBits();
            ne.pawn_attacks_b = b_att.getBits();

            // Stockfish-style pawnAttacksSpan starts as direct pawn attacks and is expanded per-pawn below.
            ne.pawn_attacks_span_w = ne.pawn_attacks_w;
            ne.pawn_attacks_span_b = ne.pawn_attacks_b;
        }

        // 2) Pawn structure (Stockfish 12 pawns.cpp), split into MG/EG (Stockfish convention).
        //    This is pawn-only and safe to cache in the pawn hash.
        double w_mid = 0.0, w_end = 0.0;
        double b_mid = 0.0, b_end = 0.0;

        auto apply_pawn_structure_and_span = [&](Color us, std::uint64_t our_pawns_bits, std::uint64_t their_pawns_bits,
                                                 std::uint64_t &pawn_attacks_span_bits, double &mid, double &end)
        {
            const Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;
            const int up = (us == Color::WHITE) ? 8 : -8;

            Bitboard tmp(our_pawns_bits);
            while (tmp)
            {
                const int s = static_cast<int>(tmp.pop());
                const int abs_r = (s >> 3) & 7;
                const int rel_r = relative_rank_0_7(us, s);

                const std::uint64_t neighbours = our_pawns_bits & adjacent_files_bits(s);
                const std::uint64_t phalanx = neighbours & chess::attacks::MASK_RANK[abs_r].getBits();
                const std::uint64_t opposed = their_pawns_bits & forward_file_bits(us, s);

                // support = adjacent-file pawns one rank behind the pawn (Stockfish rank_bb(s - Up)).
                std::uint64_t support = 0ULL;
                const int s_down = s - up;
                if (s_down >= 0 && s_down < 64)
                    support = neighbours & chess::attacks::MASK_RANK[(s_down >> 3) & 7].getBits();

                // lever = enemy pawns attacking this pawn now
                const std::uint64_t lever = their_pawns_bits & chess::attacks::pawn(us, Square(static_cast<std::uint8_t>(s))).getBits();

                // blocked/leverPush/backward classification (SF12)
                std::uint64_t blocked = 0ULL;
                std::uint64_t lever_push = 0ULL;
                bool backward = false;
                const int s_up = s + up;
                if (s_up >= 0 && s_up < 64)
                {
                    blocked = their_pawns_bits & (1ULL << s_up);
                    lever_push = their_pawns_bits & chess::attacks::pawn(us, Square(static_cast<std::uint8_t>(s_up))).getBits();

                    // A pawn is backward when it is behind all pawns of the same color on
                    // the adjacent files and cannot safely advance.
                    backward = ((neighbours & forward_ranks_bits(them, s_up)) == 0ULL) && ((lever_push | blocked) != 0ULL);

                    // Compute additional span if pawn is not backward nor blocked.
                    if (!backward && blocked == 0ULL)
                        pawn_attacks_span_bits |= pawn_attack_span_bits(us, s);
                }
                else
                {
                    // On the 7th rank, s_up is off-board; pawn cannot be blocked by an enemy pawn.
                    // pawn_attack_span_bits() is still correct here (it will only reach the 8th rank).
                    pawn_attacks_span_bits |= pawn_attack_span_bits(us, s);
                }

                // --- Stockfish 12 pawn-structure scoring ---
                if (support || phalanx)
                {
                    const int v = CONNECTED_BONUS_BY_REL_RANK[rel_r] * (2 + (phalanx != 0ULL) - (opposed != 0ULL)) + 21 * popcount64(support);

                    // Stockfish: make_score(v, v * (rel_r - 2) / 4). Values are centipawns.
                    mid += static_cast<double>(v) / 100.0;
                    end += (static_cast<double>(v) * (static_cast<double>(rel_r) - 2.0) / 4.0) / 100.0;
                }
                else if (!neighbours)
                {
                    // Isolated pawn (no neighbours on adjacent files).
                    // Special case (SF12): if opposed, doubled behind, and enemy has no adjacent-file pawns,
                    // treat as Doubled instead of Isolated.
                    const bool is_opposed = (opposed != 0ULL);
                    const bool has_our_pawn_behind_on_file = (our_pawns_bits & forward_file_bits(them, s)) != 0ULL;
                    const bool enemy_has_adjacent_file_pawn = (their_pawns_bits & adjacent_files_bits(s)) != 0ULL;

                    if (is_opposed && has_our_pawn_behind_on_file && !enemy_has_adjacent_file_pawn)
                    {
                        mid -= static_cast<double>(PAWN_DOUBLED_MG_CP) / 100.0;
                        end -= static_cast<double>(PAWN_DOUBLED_EG_CP) / 100.0;
                    }
                    else
                    {
                        mid -= static_cast<double>(PAWN_ISOLATED_MG_CP) / 100.0;
                        end -= static_cast<double>(PAWN_ISOLATED_EG_CP) / 100.0;

                        if (!is_opposed)
                        {
                            mid -= static_cast<double>(PAWN_WEAK_UNO_MG_CP) / 100.0;
                            end -= static_cast<double>(PAWN_WEAK_UNO_EG_CP) / 100.0;
                        }
                    }
                }
                else if (backward)
                {
                    mid -= static_cast<double>(PAWN_BACKWARD_MG_CP) / 100.0;
                    end -= static_cast<double>(PAWN_BACKWARD_EG_CP) / 100.0;

                    if (opposed == 0ULL)
                    {
                        mid -= static_cast<double>(PAWN_WEAK_UNO_MG_CP) / 100.0;
                        end -= static_cast<double>(PAWN_WEAK_UNO_EG_CP) / 100.0;
                    }
                }

                // Doubled and weak-lever penalties apply when the pawn is not supported.
                if (support == 0ULL)
                {
                    const bool doubled = (s_down >= 0 && s_down < 64) && ((our_pawns_bits & (1ULL << s_down)) != 0ULL);
                    if (doubled)
                    {
                        mid -= static_cast<double>(PAWN_DOUBLED_MG_CP) / 100.0;
                        end -= static_cast<double>(PAWN_DOUBLED_EG_CP) / 100.0;
                    }

                    if (popcount64(lever) > 1)
                    {
                        mid -= static_cast<double>(PAWN_WEAK_LEVER_MG_CP) / 100.0;
                        end -= static_cast<double>(PAWN_WEAK_LEVER_EG_CP) / 100.0;
                    }
                }

                // Bonus/penalty for blocked pawns at 5th/6th rank (SF12 BlockedPawn[]).
                if (blocked != 0ULL && rel_r > 3)
                {
                    if (rel_r == 4)
                    {
                        mid += static_cast<double>(PAWN_BLOCKED_5_MG_CP) / 100.0;
                        end += static_cast<double>(PAWN_BLOCKED_5_EG_CP) / 100.0;
                    }
                    else if (rel_r == 5)
                    {
                        mid += static_cast<double>(PAWN_BLOCKED_6_MG_CP) / 100.0;
                        end += static_cast<double>(PAWN_BLOCKED_6_EG_CP) / 100.0;
                    }
                }
            }
        };

        apply_pawn_structure_and_span(Color::WHITE, pw, pb, ne.pawn_attacks_span_w, w_mid, w_end);
        apply_pawn_structure_and_span(Color::BLACK, pb, pw, ne.pawn_attacks_span_b, b_mid, b_end);

        ne.pawn_structure_mid_wmb = (w_mid - b_mid);
        ne.pawn_structure_end_wmb = (w_end - b_end);

        // Passed pawns are pawn-only; keep your current definition for now.
        ne.passed_w = passed_pawns_from_bits(Color::WHITE, pw, pb);
        ne.passed_b = passed_pawns_from_bits(Color::BLACK, pb, pw);

        // Replacement policy: fill empty slot first, else overwrite slot 0.
        for (auto &ent : b.e)
        {
            if (!ent.used)
            {
                ent = ne;
                return ent;
            }
        }
        b.e[0] = ne;
        return b.e[0];
    }

private:
    std::vector<PawnHashBucket> table_;
    std::size_t mask_ = 0;
};

static inline PawnHashTable &pawn_hash_table()
{
    static PawnHashTable t;
    return t;
}

const PawnHashEntry &pawn_hash_probe(const Board &board)
{
    return pawn_hash_table().probe(board);
}

double passed_pawn_term_for_color(const Board &board, Color c)
{
    const Color them = (c == Color::WHITE) ? Color::BLACK : Color::WHITE;

    const Bitboard our_pawns = board.pieces(PieceType::PAWN, c);

    // Extras use only cheap information: occupancy + (pawns, rooks, kings).
    const Bitboard occ = board.occ();
    const Bitboard our_rooks = board.pieces(PieceType::ROOK, c);
    const Bitboard enemy_rooks = board.pieces(PieceType::ROOK, them);

    const int our_king_sq = board.kingSq(c).index();
    const int enemy_king_sq = board.kingSq(them).index();

    auto chebyshev_dist = [](int a, int b) noexcept -> int
    {
        const int ar = a >> 3, af = a & 7;
        const int br = b >> 3, bf = b & 7;
        int dr = ar - br;
        if (dr < 0)
            dr = -dr;
        int df = af - bf;
        if (df < 0)
            df = -df;
        return (dr > df) ? dr : df;
    };

    auto is_protected_by_pawn = [&](int s) noexcept -> bool
    {
        const int r = s >> 3;
        const int f = s & 7;

        if (c == Color::WHITE)
        {
            if (r == 0)
                return false;
            if (f > 0 && (our_pawns & Bitboard::fromSquare(s - 9)))
                return true; // down-left
            if (f < 7 && (our_pawns & Bitboard::fromSquare(s - 7)))
                return true; // down-right
        }
        else
        {
            if (r == 7)
                return false;
            if (f > 0 && (our_pawns & Bitboard::fromSquare(s + 7)))
                return true; // up-left
            if (f < 7 && (our_pawns & Bitboard::fromSquare(s + 9)))
                return true; // up-right
        }
        return false;
    };

    // Step 2.6 helper: enemy rook in front (first blocker on file ahead is an enemy rook).
    auto has_enemy_rook_in_front = [&](int s, Bitboard enemy_rooks_local) noexcept -> bool
    {
        const int r = s >> 3;
        const int f = s & 7;

        if (c == Color::WHITE)
        {
            for (int rr = r + 1; rr <= 7; ++rr)
            {
                const int sq = rr * 8 + f;
                const Bitboard bb = Bitboard::fromSquare(sq);
                if (occ & bb)
                    return static_cast<bool>(enemy_rooks_local & bb);
            }
        }
        else
        {
            for (int rr = r - 1; rr >= 0; --rr)
            {
                const int sq = rr * 8 + f;
                const Bitboard bb = Bitboard::fromSquare(sq);
                if (occ & bb)
                    return static_cast<bool>(enemy_rooks_local & bb);
            }
        }

        return false;
    };

    auto is_blocked_in_front = [&](int s) noexcept -> bool
    {
        const int front = (c == Color::WHITE) ? (s + 8) : (s - 8);
        if (front < 0 || front >= 64)
            return false;
        return static_cast<bool>(occ & Bitboard::fromSquare(front));
    };

    // "Rook behind passed pawn" along the pawn's direction:
    // For White pawn: rook on same file with lower rank, with no pieces in between.
    // For Black pawn: rook on same file with higher rank, with no pieces in between.
    auto has_rook_behind = [&](int s, Bitboard rooks) noexcept -> bool
    {
        const int r = s >> 3;
        const int f = s & 7;

        if (c == Color::WHITE)
        {
            for (int rr = r - 1; rr >= 0; --rr)
            {
                const int sq = rr * 8 + f;
                const Bitboard bb = Bitboard::fromSquare(sq);
                if (rooks & bb)
                    return true;
                if (occ & bb)
                    break;
            }
        }
        else
        {
            for (int rr = r + 1; rr <= 7; ++rr)
            {
                const int sq = rr * 8 + f;
                const Bitboard bb = Bitboard::fromSquare(sq);
                if (rooks & bb)
                    return true;
                if (occ & bb)
                    break;
            }
        }
        return false;
    };

    const auto &pe = pawn_hash_probe(board);
    const std::uint64_t passed_bits = (c == Color::WHITE) ? pe.passed_w : pe.passed_b;
    if (!passed_bits)
        return 0.0;

    // Step 2.3 (path safety tiers) uses only enemy pawn attacks (cheap first).
    const std::uint64_t enemy_pawn_attacks_bits = (them == Color::WHITE) ? pe.pawn_attacks_w : pe.pawn_attacks_b;

    // Step 2.4/2.5 helpers: targeted attacked-square queries (cheap; used only for passer block squares).
    auto is_sq_attacked_by = [&](Color attacker, int target_sq) noexcept -> bool
    {
        if (target_sq < 0 || target_sq >= 64)
            return false;

        const std::uint64_t bit = 1ULL << target_sq;

        // Pawn attacks (from pawn hash).
        if (attacker == Color::WHITE)
        {
            if (pe.pawn_attacks_w & bit)
                return true;
        }
        else
        {
            if (pe.pawn_attacks_b & bit)
                return true;
        }

        // Knights.
        {
            Bitboard bb = board.pieces(PieceType::KNIGHT, attacker);
            while (bb)
            {
                const Square sq(bb.pop());
                if (chess::attacks::knight(sq).getBits() & bit)
                    return true;
            }
        }

        // Bishops.
        {
            Bitboard bb = board.pieces(PieceType::BISHOP, attacker);
            while (bb)
            {
                const Square sq(bb.pop());
                if (chess::attacks::bishop(sq, occ).getBits() & bit)
                    return true;
            }
        }

        // Rooks.
        {
            Bitboard bb = board.pieces(PieceType::ROOK, attacker);
            while (bb)
            {
                const Square sq(bb.pop());
                if (chess::attacks::rook(sq, occ).getBits() & bit)
                    return true;
            }
        }

        // Queens.
        {
            Bitboard bb = board.pieces(PieceType::QUEEN, attacker);
            while (bb)
            {
                const Square sq(bb.pop());
                if (chess::attacks::queen(sq, occ).getBits() & bit)
                    return true;
            }
        }

        // King (pseudo-attacks).
        {
            const Square ksq = board.kingSq(attacker);
            if (ksq.is_valid() && (chess::attacks::king(ksq).getBits() & bit))
                return true;
        }

        return false;
    };

    double bonus = 0.0;

    Bitboard tmp(passed_bits);
    while (tmp)
    {
        const std::uint8_t s_u8 = tmp.pop();
        const int s = static_cast<int>(s_u8);

        const int r = s >> 3;
        const int f = s & 7;

        const int advance = (c == Color::WHITE) ? r : (7 - r);
        const int adv_clamped = std::clamp(advance, 0, 7);

        // Base bonus from rank advance (already fairly strong).
        double b = PASSED_PAWN_BONUS_BY_ADVANCE[adv_clamped];

        // Scale add-ons by how advanced the pawn is (keeps things stable).
        const double adv_scale = std::clamp(static_cast<double>(adv_clamped) / 6.0, 0.0, 1.0);

        // Block square is the square directly in front of the passer.
        const int block_sq = (c == Color::WHITE) ? (s + 8) : (s - 8);

        // Step 2.2 (file penalty): edge-file passed pawns are typically less valuable.
        // Apply a small penalty for a/h and a smaller penalty for b/g, scaled by advance.
        if (f == 0 || f == 7)
            b -= 0.03 * adv_scale;
        else if (f == 1 || f == 6)
            b -= 0.015 * adv_scale;

        // (1) Blocked passed pawn: significantly less valuable if immediately blocked.
        if (is_blocked_in_front(s))
        {
            b *= 0.45;
        }
        else
        {
            // Step 2.3 (path safety tiers): if the pawn is free to advance one square,
            // add a bonus depending on whether enemy pawns attack the pawn's span/path.
            const int front = block_sq;
            if (front >= 0 && front < 64 && !(occ & Bitboard::fromSquare(front)))
            {
                // Squares in front on same file up to promotion (inclusive).
                std::uint64_t squares_to_queen_bits = 0ULL;
                if (c == Color::WHITE)
                {
                    for (int rr = r + 1; rr <= 7; ++rr)
                        squares_to_queen_bits |= (1ULL << (rr * 8 + f));
                }
                else
                {
                    for (int rr = r - 1; rr >= 0; --rr)
                        squares_to_queen_bits |= (1ULL << (rr * 8 + f));
                }

                // Passed-pawn span: forward squares on file f-1,f,f+1 (pawn-only).
                const std::uint64_t span_bits = passed_mask_for(c, s).getBits();
                const std::uint64_t unsafe_bits = span_bits & enemy_pawn_attacks_bits;

                double path_bonus = 0.0;
                if (unsafe_bits == 0ULL)
                {
                    // Best: no enemy pawn attacks on the span.
                    path_bonus = 0.08 * adv_scale;
                }
                else if ((unsafe_bits & squares_to_queen_bits) == 0ULL)
                {
                    // Medium: span attacked, but the direct file path is safe.
                    path_bonus = 0.05 * adv_scale;
                }
                else if ((unsafe_bits & (1ULL << front)) == 0ULL)
                {
                    // Small: file path is attacked somewhere, but the block square is safe.
                    path_bonus = 0.025 * adv_scale;
                }

                b += path_bonus;
            }
        }

        // (2) Supported/protected passed pawn (pawn-protected).
        if (is_protected_by_pawn(s))
        {
            b += (0.03 + 0.04 * adv_scale);
        }

        // (3) Rook-behind bonus/penalty (clear file behind the pawn).
        //     Reward our rook behind; penalize enemy rook behind.
        if (has_rook_behind(s, our_rooks))
        {
            b += 0.06 * adv_scale;
        }
        if (has_rook_behind(s, enemy_rooks))
        {
            b -= 0.06 * adv_scale;
        }

        // Step 2.4 (block-square support): reward passers whose block square is defended, and
        // penalize passers whose block square is controlled by the opponent and not defended.
        if (block_sq >= 0 && block_sq < 64)
        {
            const bool us_defends = is_sq_attacked_by(c, block_sq);
            const bool them_controls = is_sq_attacked_by(them, block_sq);

            if (us_defends)
                b += 0.03 * adv_scale;
            if (them_controls && !us_defends)
                b -= 0.03 * adv_scale;
        }

        // Step 2.5 (king proximity refinement): add block-square distance in addition to promotion-square distance.
        if (block_sq >= 0 && block_sq < 64)
        {
            int diff_block = chebyshev_dist(enemy_king_sq, block_sq) - chebyshev_dist(our_king_sq, block_sq);
            diff_block = std::clamp(diff_block, -4, 4);
            b += static_cast<double>(diff_block) * (0.03 * adv_scale);

            // Also consider the second push square (SF12-style refinement), lightly scaled.
            if (adv_clamped <= 5)
            {
                const int block_sq2 = (c == Color::WHITE) ? (block_sq + 8) : (block_sq - 8);
                if (block_sq2 >= 0 && block_sq2 < 64)
                {
                    const int d2 = std::min(chebyshev_dist(our_king_sq, block_sq2), 5);
                    b -= static_cast<double>(d2) * (0.01 * adv_scale);
                }
            }
        }

        // Step 2.6 (enemy rook in front): strong stopper when the first blocker ahead is an enemy rook.
        if (has_enemy_rook_in_front(s, enemy_rooks))
        {
            b -= 0.08 * adv_scale;
        }
        // (4) King proximity to the promotion square (Chebyshev distance).
        //     Only a small adjustment; endgame-weighted outside this function.
        const int promo_sq = (c == Color::WHITE) ? (7 * 8 + f) : (0 * 8 + f);
        int diff = chebyshev_dist(enemy_king_sq, promo_sq) - chebyshev_dist(our_king_sq, promo_sq);
        diff = std::clamp(diff, -4, 4);
        b += static_cast<double>(diff) * (0.02 * adv_scale);

        // Keep stable: passed pawn bonus should not become negative here.
        if (b < 0.0)
            b = 0.0;

        bonus += b;
    }

    return bonus*1.5;
}

double passed_pawn_term_white_minus_black(const Board &board)
{
    const double w = passed_pawn_term_for_color(board, Color::WHITE);
    const double b = passed_pawn_term_for_color(board, Color::BLACK);
    return w - b;
}

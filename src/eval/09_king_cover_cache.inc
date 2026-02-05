// Evaluation module: king cover cache
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

// ---------------------- King cover cache ----------------------
//
// King cover depends only on (own pawns, king square). It is reused heavily.
struct KingCoverHashEntry
{
    std::uint64_t pawns = 0ULL;
    std::uint8_t king_sq = 0;
    std::uint8_t used = 0;
    double penalty = 0.0; // in pawn units
};

struct KingCoverHashBucket
{
    KingCoverHashEntry e[2];
};

class KingCoverHashTable
{
public:
    KingCoverHashTable()
    {
        constexpr std::size_t BUCKETS = 1u << 15; // 32768 buckets, 2-way => 65536 entries
        table_.resize(BUCKETS);
        mask_ = BUCKETS - 1;
    }

    template <typename F>
    double probe(std::uint64_t pawns, std::uint8_t king_sq, F &&compute)
    {
        const std::uint64_t k = splitmix64(pawns ^ (static_cast<std::uint64_t>(king_sq) * 0x9E3779B97F4A7C15ULL));
        KingCoverHashBucket &b = table_[static_cast<std::size_t>(k) & mask_];

        for (auto &ent : b.e)
        {
            if (ent.used && ent.pawns == pawns && ent.king_sq == king_sq)
                return ent.penalty;
        }

        // Miss: compute and store.
        KingCoverHashEntry ne;
        ne.used = 1;
        ne.pawns = pawns;
        ne.king_sq = king_sq;
        ne.penalty = compute();

        // Replacement policy: fill empty slot first, else overwrite slot 0.
        for (auto &ent : b.e)
        {
            if (!ent.used)
            {
                ent = ne;
                return ent.penalty;
            }
        }
        b.e[0] = ne;
        return b.e[0].penalty;
    }

private:
    std::vector<KingCoverHashBucket> table_;
    std::size_t mask_ = 0;
};

static inline KingCoverHashTable &king_cover_hash_table()
{
    static KingCoverHashTable t;
    return t;
}

double king_cover_penalty_for_color_at_ksq(const Board &board, Color us, const Square ksq)
{
    if (!ksq.is_valid())
        return 0.0;

    const Bitboard pawns_bb = board.pieces(PieceType::PAWN, us);
    const std::uint8_t ksqi = static_cast<std::uint8_t>(ksq.index());
    const std::uint64_t pawns = pawns_bb.getBits();

    return king_cover_hash_table().probe(pawns, ksqi, [&]() -> double
                                         { return king_cover_penalty_for_color_slow_at_ksq(board, us, ksq); });
}

// --- Pawn storm (not cached: depends on enemy pawns) ---
//
// Returns a *penalty* in pawn units: 0 is best, larger is worse.
double pawn_storm_penalty_for_color_at_ksq(const Board &board, Color us, const Square ksq)
{
    const Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;

    const Bitboard our_pawns = board.pieces(PieceType::PAWN, us);
    const Bitboard their_pawns = board.pieces(PieceType::PAWN, them);

    const int k_idx = ksq.index();
    const int k_file = k_idx & 7;
    const int k_rank = k_idx >> 3;

    double pen = 0.0;

    for (int df = -1; df <= 1; ++df)
    {
        const int f = k_file + df;
        if (f < 0 || f > 7)
            continue;

        const Bitboard file_mask = chess::attacks::MASK_FILE[f];

        // Our closest cover pawn on this file (same as in king cover).
        int best_cover_r = (us == Color::WHITE) ? 99 : -99;
        {
            Bitboard file_pawns = our_pawns & file_mask;
            Bitboard tmp = file_pawns;
            while (tmp)
            {
                const int s = static_cast<int>(tmp.pop());
                const int r = s >> 3;
                if (us == Color::WHITE)
                {
                    if (r >= k_rank && r < best_cover_r)
                        best_cover_r = r;
                }
                else
                {
                    if (r <= k_rank && r > best_cover_r)
                        best_cover_r = r;
                }
            }
        }

        int cover_idx = 0; // 0 => no cover pawn
        if ((us == Color::WHITE && best_cover_r != 99) || (us == Color::BLACK && best_cover_r != -99))
        {
            const int rr = (us == Color::WHITE) ? best_cover_r : (7 - best_cover_r);
            cover_idx = std::min(6, rr);
        }

        // Their most advanced pawn on this file (toward our king), considering
        // pawns on the king's rank or in front of it from our perspective.
        int best_storm_r = (us == Color::WHITE) ? 99 : -99;
        {
            Bitboard file_pawns = their_pawns & file_mask;
            Bitboard tmp = file_pawns;
            while (tmp)
            {
                const int s = static_cast<int>(tmp.pop());
                const int r = s >> 3;

                if (us == Color::WHITE)
                {
                    if (r >= k_rank && r < best_storm_r)
                        best_storm_r = r; // closer to White home (smaller r) is more dangerous
                }
                else
                {
                    if (r <= k_rank && r > best_storm_r)
                        best_storm_r = r; // closer to Black home (larger r) is more dangerous
                }
            }
        }

        if ((us == Color::WHITE && best_storm_r == 99) || (us == Color::BLACK && best_storm_r == -99))
            continue; // no storm pawn on this file

        const int storm_idx = (us == Color::WHITE) ? best_storm_r : (7 - best_storm_r); // 0..7

        int storm_cp = 0;
        if (cover_idx == 0)
        {
            storm_cp = DONNA_STORM_PENALTY_CP[storm_idx];
        }
        else if (storm_idx == cover_idx + 1)
        {
            storm_cp = DONNA_STORM_BLOCKED_PENALTY_CP[storm_idx];
        }
        else
        {
            storm_cp = DONNA_STORM_UNBLOCKED_PENALTY_CP[storm_idx];
        }

        pen += static_cast<double>(storm_cp) * CP_TO_PAWNS;
    }

    return pen;
}

// Combined Donna-style cover + storm penalty.
double king_cover_storm_penalty_for_color_at_ksq(const Board &board, Color us, const Square ksq)
{
    return king_cover_penalty_for_color_at_ksq(board, us, ksq) +
           pawn_storm_penalty_for_color_at_ksq(board, us, ksq);
}

// If we still have castling rights, evaluate the *best* cover/storm among current and
// potential castling destinations (Donna does this to encourage keeping castling pawns intact).
double king_cover_storm_penalty_for_color(const Board &board, Color us)
{
    const Square ksq = board.kingSq(us);
    if (!ksq.is_valid())
        return 0.0;

    if (!king_is_homeish(us, ksq))
        return 0.0;

    double best = king_cover_storm_penalty_for_color_at_ksq(board, us, ksq);

    const auto cr = board.castlingRights();
    using CastleSide = chess::Board::CastlingRights::Side;

    if (cr.has(us, CastleSide::KING_SIDE))
    {
        const Square king_to = Square::castling_king_square(true, us);
        best = std::min(best, king_cover_storm_penalty_for_color_at_ksq(board, us, king_to));
    }

    if (cr.has(us, CastleSide::QUEEN_SIDE))
    {
        const Square king_to = Square::castling_king_square(false, us);
        best = std::min(best, king_cover_storm_penalty_for_color_at_ksq(board, us, king_to));
    }

    return best;
}

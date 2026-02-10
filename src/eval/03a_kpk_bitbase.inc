// Evaluation module: KPK bitbase (K+P vs K) for exact win/draw
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.
//
// Rationale:
//   - KPK outcomes are cheap to probe, perfectly stable, and remove a major source
//     of static-eval error in pawn endgames.
//   - This is especially search-friendly because your pruning gates rely on static eval.

namespace kpk
{
    // Small WDL bitbase for KPK (pawn side can only win or draw).
    // The table is built once (lazy init) using a monotone fixpoint.
    //
    // Normalization:
    //   - Pawn side is always treated as "White".
    //   - If the pawn is on files e..h, we mirror along the vertical axis (a<->h) to
    //     reduce pawn files to a..d (4 files) while mirroring both kings too.
    //   - Pawn ranks are 2..7 (internal 0-based ranks 1..6), so there are 4*6=24 pawn squares.

    constexpr int PAWN_FILES = 4;  // a..d after mirroring
    constexpr int PAWN_RANKS = 6;  // ranks 2..7 (0-based 1..6)
    constexpr int PAWN_SQ_N = PAWN_FILES * PAWN_RANKS; // 24

    constexpr int WK_N = 64;
    constexpr int BK_N = 64;
    constexpr int STM_N = 2; // 0 = pawn side to move, 1 = defender to move

    constexpr int TABLE_N = STM_N * WK_N * BK_N * PAWN_SQ_N;

    enum : std::uint8_t
    {
        DRAW = 0,
        WIN = 1
    };

    static inline int file_of(int sq) noexcept { return sq & 7; }
    static inline int rank_of(int sq) noexcept { return sq >> 3; }

    static inline int sq_mirror_file(int sq) noexcept
    {
        // Mirror a<->h (file flip), rank unchanged.
        return sq ^ 7;
    }

    static inline int sq_rotate_180(int sq) noexcept
    {
        // 180-degree rotation: (f,r)->(7-f,7-r). For 0=a1 indexing this is 63 - sq.
        return sq ^ 63;
    }

    static inline bool kings_adjacent(int wk, int bk) noexcept
    {
        const int df = std::abs(file_of(wk) - file_of(bk));
        const int dr = std::abs(rank_of(wk) - rank_of(bk));
        return std::max(df, dr) <= 1;
    }

    static inline bool pawn_attacks_square(int pawn_sq, int target_sq) noexcept
    {
        // Pawn is normalized as "white" moving north (+8).
        const int pf = file_of(pawn_sq);
        const int pr = rank_of(pawn_sq);
        const int tf = file_of(target_sq);
        const int tr = rank_of(target_sq);
        return (tr == pr + 1) && (tf == pf - 1 || tf == pf + 1);
    }

    static inline bool attacked_by_pawn_side(int target_sq, int wk, int pawn_sq) noexcept
    {
        // White king attacks adjacent squares; pawn attacks diagonals forward.
        const int df = std::abs(file_of(target_sq) - file_of(wk));
        const int dr = std::abs(rank_of(target_sq) - rank_of(wk));
        if (std::max(df, dr) <= 1)
            return true;
        if (pawn_sq >= 0 && pawn_attacks_square(pawn_sq, target_sq))
            return true;
        return false;
    }

    static inline int pawn_sq_from_index(int pawn24) noexcept
    {
        const int pf = pawn24 & 3;
        const int pr = 1 + (pawn24 >> 2); // 1..6
        return pr * 8 + pf;
    }

    static inline int pawn_index_from_sq(int pawn_sq) noexcept
    {
        const int pf = file_of(pawn_sq);    // 0..3 after mirroring
        const int pr = rank_of(pawn_sq);    // 1..6
        return (pr - 1) * 4 + pf;
    }

    static inline int idx(int stm, int wk, int bk, int pawn24) noexcept
    {
        // Layout: (((stm * 64 + wk) * 64 + bk) * 24 + pawn24)
        return (((stm * WK_N + wk) * BK_N + bk) * PAWN_SQ_N + pawn24);
    }

    struct KpkBitbase
    {
        std::array<std::uint8_t, TABLE_N> wdl{};

        KpkBitbase() { build(); }

        static inline bool invalid_state(int stm, int wk, int bk, int pawn_sq) noexcept
        {
            (void)stm;
            if (wk == bk)
                return true;
            if (wk == pawn_sq || bk == pawn_sq)
                return true;
            if (kings_adjacent(wk, bk))
                return true;

            const int pr = rank_of(pawn_sq);
            // Pawn should never be on rank 1 (0) or rank 8 (7) in the table.
            if (pr <= 0 || pr >= 7)
                return true;

            // A position is legal even if a king is in check; we don't filter for that.
            return false;
        }

        static inline bool is_pawn_promotion_move(int pawn_sq, int wk, int bk) noexcept
        {
            const int pr = rank_of(pawn_sq);
            if (pr != 6)
                return false;
            const int dst = pawn_sq + 8;
            // Promotion square must be empty (only kings can occupy squares).
            if (dst == wk || dst == bk)
                return false;
            return true;
        }

        static inline bool pawn_can_push(int pawn_sq, int wk, int bk) noexcept
        {
            const int pr = rank_of(pawn_sq);
            if (pr >= 6)
                return false;
            const int dst = pawn_sq + 8;
            return dst != wk && dst != bk;
        }

        static inline bool pawn_can_double(int pawn_sq, int wk, int bk) noexcept
        {
            const int pr = rank_of(pawn_sq);
            if (pr != 1)
                return false;
            const int d1 = pawn_sq + 8;
            const int d2 = pawn_sq + 16;
            if (d1 == wk || d1 == bk)
                return false;
            if (d2 == wk || d2 == bk)
                return false;
            return true;
        }

        void build() noexcept
        {
            // Monotone fixpoint: states transition from DRAW -> WIN only.
            bool changed = true;
            while (changed)
            {
                changed = false;

                for (int stm = 0; stm < STM_N; ++stm)
                {
                    for (int wk = 0; wk < 64; ++wk)
                    {
                        for (int bk = 0; bk < 64; ++bk)
                        {
                            for (int p24 = 0; p24 < PAWN_SQ_N; ++p24)
                            {
                                const int sidx = idx(stm, wk, bk, p24);
                                if (wdl[sidx] == WIN)
                                    continue;

                                const int pawn_sq = pawn_sq_from_index(p24);
                                if (invalid_state(stm, wk, bk, pawn_sq))
                                    continue;

                                if (stm == 0)
                                {
                                    // Pawn side to move: WIN if it has any move to a WIN state,
                                    // or an immediate promotion.
                                    bool can_win = false;

                                    // Immediate promotion.
                                    if (is_pawn_promotion_move(pawn_sq, wk, bk))
                                    {
                                        can_win = true;
                                    }

                                    // Pawn pushes (non-promotion).
                                    if (!can_win && pawn_can_push(pawn_sq, wk, bk))
                                    {
                                        const int np = pawn_sq + 8;
                                        const int np24 = pawn_index_from_sq(np);
                                        if (wdl[idx(1, wk, bk, np24)] == WIN)
                                            can_win = true;
                                    }

                                    // Pawn double push.
                                    if (!can_win && pawn_can_double(pawn_sq, wk, bk))
                                    {
                                        const int np = pawn_sq + 16;
                                        const int np24 = pawn_index_from_sq(np);
                                        if (wdl[idx(1, wk, bk, np24)] == WIN)
                                            can_win = true;
                                    }

                                    // King moves.
                                    if (!can_win)
                                    {
                                        const int wf = file_of(wk);
                                        const int wr = rank_of(wk);
                                        for (int df = -1; df <= 1 && !can_win; ++df)
                                        {
                                            for (int dr = -1; dr <= 1 && !can_win; ++dr)
                                            {
                                                if (df == 0 && dr == 0)
                                                    continue;
                                                const int nf = wf + df;
                                                const int nr = wr + dr;
                                                if (nf < 0 || nf > 7 || nr < 0 || nr > 7)
                                                    continue;
                                                const int nwk = nr * 8 + nf;
                                                if (nwk == bk || nwk == pawn_sq)
                                                    continue;
                                                if (kings_adjacent(nwk, bk))
                                                    continue;
                                                // Legal move.
                                                if (wdl[idx(1, nwk, bk, p24)] == WIN)
                                                    can_win = true;
                                            }
                                        }
                                    }

                                    if (can_win)
                                    {
                                        wdl[sidx] = WIN;
                                        changed = true;
                                    }
                                }
                                else
                                {
                                    // Defender to move: WIN only if ALL legal moves lead to WIN.
                                    // If the defender can capture the pawn, it's a DRAW sink.
                                    bool has_legal = false;
                                    bool all_win = true;

                                    const int bf = file_of(bk);
                                    const int br = rank_of(bk);

                                    for (int df = -1; df <= 1; ++df)
                                    {
                                        for (int dr = -1; dr <= 1; ++dr)
                                        {
                                            if (df == 0 && dr == 0)
                                                continue;
                                            const int nf = bf + df;
                                            const int nr = br + dr;
                                            if (nf < 0 || nf > 7 || nr < 0 || nr > 7)
                                                continue;
                                            const int nbk = nr * 8 + nf;
                                            if (nbk == wk)
                                                continue;
                                            if (kings_adjacent(wk, nbk))
                                                continue;
                                            // Can't move into pawn-side attacks (king or pawn).
                                            if (attacked_by_pawn_side(nbk, wk, pawn_sq))
                                                continue;

                                            // Capture pawn -> draw sink.
                                            if (nbk == pawn_sq)
                                            {
                                                has_legal = true;
                                                all_win = false;
                                                df = dr = 2; // break both loops
                                                break;
                                            }

                                            has_legal = true;
                                            if (wdl[idx(0, wk, nbk, p24)] != WIN)
                                                all_win = false;
                                        }
                                    }

                                    if (!has_legal)
                                    {
                                        // No legal moves: checkmate is WIN, stalemate is DRAW.
                                        // In KPK, the only possible check is by the pawn.
                                        const bool in_check = pawn_attacks_square(pawn_sq, bk);
                                        if (in_check)
                                        {
                                            wdl[sidx] = WIN;
                                            changed = true;
                                        }
                                    }
                                    else if (all_win)
                                    {
                                        wdl[sidx] = WIN;
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        inline std::uint8_t probe_normalized(int stm, int wk, int bk, int pawn_sq) const noexcept
        {
            // Precondition: pawn side is White; pawn and kings are already mirrored into files a..d.
            const int p24 = pawn_index_from_sq(pawn_sq);
            if (p24 < 0 || p24 >= PAWN_SQ_N)
                return DRAW;
            const int sidx = idx(stm, wk, bk, p24);
            return wdl[sidx];
        }
    };

    static inline const KpkBitbase &bitbase() noexcept
    {
        static const KpkBitbase bb;
        return bb;
    }

    // Probe KPK result for an actual position.
    // Returns true if the position is a strict KPK (exactly one pawn total, no other pieces).
    // If true, 'pawn_side_wins' is set to whether the pawn side has a theoretical win.
    static inline bool probe_kpk_pawn_side_wins(const Board &board,
                                                Color pawn_color,
                                                bool &pawn_side_wins) noexcept
    {
        using chess::PieceType;

        const int white_p = static_cast<int>(board.pieces(PieceType::PAWN, Color::WHITE).count());
        const int black_p = static_cast<int>(board.pieces(PieceType::PAWN, Color::BLACK).count());
        if (white_p + black_p != 1)
            return false;

        // No pieces besides kings and pawns.
        const int non_pieces =
            static_cast<int>(board.pieces(PieceType::KNIGHT).count() +
                             board.pieces(PieceType::BISHOP).count() +
                             board.pieces(PieceType::ROOK).count() +
                             board.pieces(PieceType::QUEEN).count());
        if (non_pieces != 0)
            return false;

        const Square wk_sq0 = board.kingSq(Color::WHITE);
        const Square bk_sq0 = board.kingSq(Color::BLACK);
        if (!wk_sq0.is_valid() || !bk_sq0.is_valid())
            return false;

        Bitboard pb = board.pieces(PieceType::PAWN, pawn_color);
        if (pb.count() != 1)
            return false;
        const int pawn_sq0 = static_cast<int>(pb.pop());

        // Normalize so pawn side is "White" moving north.
        int n_wk = 0;
        int n_bk = 0;
        int n_pawn = 0;
        int stm = 0;

        if (pawn_color == Color::WHITE)
        {
            n_wk = wk_sq0.index();
            n_bk = bk_sq0.index();
            n_pawn = pawn_sq0;
            stm = (board.sideToMove() == Color::WHITE) ? 0 : 1;
        }
        else
        {
            // Rotate 180 degrees and swap kings so pawn side becomes "White".
            n_wk = sq_rotate_180(bk_sq0.index());
            n_bk = sq_rotate_180(wk_sq0.index());
            n_pawn = sq_rotate_180(pawn_sq0);
            stm = (board.sideToMove() == Color::BLACK) ? 0 : 1;
        }

        // Mirror files to reduce pawn to a..d.
        if (file_of(n_pawn) > 3)
        {
            n_pawn = sq_mirror_file(n_pawn);
            n_wk = sq_mirror_file(n_wk);
            n_bk = sq_mirror_file(n_bk);
        }

        const int pr = rank_of(n_pawn);
        if (pr <= 0 || pr >= 7)
            return false;

        pawn_side_wins = (bitbase().probe_normalized(stm, n_wk, n_bk, n_pawn) == WIN);
        return true;
    }

} // namespace kpk

// Public-facing helper for evaluation.
// If the position is KPK (kings + exactly one pawn), return an exact score.
static inline bool probe_kpk_exact_eval_white_pov(const Board &board, Score &out_white_pov) noexcept
{
    using chess::Color;
    using chess::PieceType;

    const int white_p = static_cast<int>(board.pieces(PieceType::PAWN, Color::WHITE).count());
    const int black_p = static_cast<int>(board.pieces(PieceType::PAWN, Color::BLACK).count());
    if (white_p + black_p != 1)
        return false;

    const Color pawn_color = (white_p == 1) ? Color::WHITE : Color::BLACK;
    bool pawn_side_wins = false;
    if (!kpk::probe_kpk_pawn_side_wins(board, pawn_color, pawn_side_wins))
        return false;

    // Map WDL to a conservative, search-friendly centipawn score.
    // - WIN: treat as equivalent to an eventual queen advantage.
    // - DRAW: 0.
    constexpr Score KPK_WIN_CP = 800; // ~queen-pawn (8 pawns) in centipawns.

    if (!pawn_side_wins)
    {
        out_white_pov = 0;
        return true;
    }

    out_white_pov = (pawn_color == Color::WHITE) ? KPK_WIN_CP : -KPK_WIN_CP;
    return true;
}

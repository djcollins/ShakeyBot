// Evaluation module: endgame scaling + king crowding
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

// ---------------------- Endgame scaling ----------------------
//
// Purpose:
//   Reduce over-optimistic evaluation in drawish endgames by scaling the score
//   toward 0 when endgame characteristics strongly favor the defender.
//
// Semantics:
//   - scale factor in [0.0, 1.0], where 1.0 means "no scaling" and 0.0 means "draw".
//   - Applied progressively with phase: final multiplier = mg_weight + eg_weight * scale.
//
// This is intentionally conservative (high ROI / low tuning risk) and focuses on a
// few classic cases that commonly leak Elo if untreated.

static inline int chebyshev_distance(int a_sq, int b_sq) noexcept
{
    const int af = a_sq & 7, ar = a_sq >> 3;
    const int bf = b_sq & 7, br = b_sq >> 3;
    const int df = std::abs(af - bf);
    const int dr = std::abs(ar - br);
    return std::max(df, dr);
}

static inline bool is_dark_square_idx(int sq) noexcept
{
    // a1 is dark. With 0=a1 indexing: (file+rank) even => dark.
    const int f = sq & 7;
    const int r = sq >> 3;
    return ((f + r) & 1) == 0;
}

static inline bool single_bishop_is_dark(Bitboard bishops) noexcept
{
    // Assumes exactly one bishop.
    const int s = static_cast<int>(bishops.pop());
    return is_dark_square_idx(s);
}

static inline double endgame_scale_factor_for_advantage(const Board &board,
                                                        Color strong,
                                                        Color weak) noexcept
{
    using namespace chess;

    // Default: no scaling.
    double sf = 1.0;

    const Bitboard strong_all = board.us(strong);
    const Bitboard weak_all = board.us(weak);

    const int strong_cnt = static_cast<int>(strong_all.count());
    const int weak_cnt = static_cast<int>(weak_all.count());

    const int strong_pawns = static_cast<int>(board.pieces(PieceType::PAWN, strong).count());
    const int weak_pawns = static_cast<int>(board.pieces(PieceType::PAWN, weak).count());

    const int strong_kn = static_cast<int>(board.pieces(PieceType::KNIGHT, strong).count());
    const int weak_kn = static_cast<int>(board.pieces(PieceType::KNIGHT, weak).count());

    const int strong_bi = static_cast<int>(board.pieces(PieceType::BISHOP, strong).count());
    const int weak_bi = static_cast<int>(board.pieces(PieceType::BISHOP, weak).count());

    const int strong_ro = static_cast<int>(board.pieces(PieceType::ROOK, strong).count());
    const int weak_ro = static_cast<int>(board.pieces(PieceType::ROOK, weak).count());

    const int strong_qu = static_cast<int>(board.pieces(PieceType::QUEEN, strong).count());
    const int weak_qu = static_cast<int>(board.pieces(PieceType::QUEEN, weak).count());

    // --- Case 1: Wrong bishop + rook pawn (K+B+RP vs K) is usually a draw.
    //
    // We only trigger in the clean canonical material case to avoid false positives:
    //   strong: K+B+1 rook pawn, weak: K only.
    if (weak_cnt == 1 && strong_cnt == 2 + strong_pawns &&
        strong_pawns == 1 &&
        strong_bi == 1 &&
        strong_kn == 0 && strong_ro == 0 && strong_qu == 0 &&
        weak_pawns == 0 && weak_bi == 0 && weak_kn == 0 && weak_ro == 0 && weak_qu == 0)
    {
        const Bitboard pbb = board.pieces(PieceType::PAWN, strong);
        const bool pawn_on_a = static_cast<bool>(pbb & attacks::MASK_FILE[0]);
        const bool pawn_on_h = static_cast<bool>(pbb & attacks::MASK_FILE[7]);

        if (pawn_on_a || pawn_on_h)
        {
            // Promotion corner square index.
            const int promo_sq =
                (strong == Color::WHITE)
                    ? (pawn_on_a ? 56 : 63) // a8 or h8
                    : (pawn_on_a ? 0 : 7);  // a1 or h1

            const bool promo_dark = is_dark_square_idx(promo_sq);

            Bitboard bishops = board.pieces(PieceType::BISHOP, strong);
            const bool bishop_dark = single_bishop_is_dark(bishops);

            if (bishop_dark != promo_dark)
            {
                // Defender reaches the corner -> draw. If the defender king is far,
                // keep a small non-zero scale to avoid killing genuine wins where the
                // pawn can promote uncontested.
                const Square ksq = board.kingSq(weak);
                const int k_idx = ksq.is_valid() ? ksq.index() : promo_sq;
                const int d = chebyshev_distance(k_idx, promo_sq);

                if (d <= 1)
                    sf = std::min(sf, 0.0);
                else if (d >= 3)
                    sf = std::min(sf, 0.65);
                else
                    sf = std::min(sf, 0.25);
            }
        }
    }

    // --- Case 2: Opposite-colored bishops with pawns is drawish.
    //
    // Trigger only in simplified endgames (no rooks, no queens, no knights).
    if (strong_ro + weak_ro == 0 &&
        strong_qu + weak_qu == 0 &&
        strong_kn + weak_kn == 0 &&
        strong_bi == 1 && weak_bi == 1 &&
        strong_pawns > 0 && weak_pawns > 0)
    {
        Bitboard sb = board.pieces(PieceType::BISHOP, strong);
        Bitboard wb = board.pieces(PieceType::BISHOP, weak);
        const bool strong_dark = single_bishop_is_dark(sb);
        const bool weak_dark = single_bishop_is_dark(wb);

        if (strong_dark != weak_dark)
        {
            const int pawn_adv = std::max(0, strong_pawns - weak_pawns);

            // Conservative scaling: keep some advantage, but reduce optimism.
            // 0–1 pawn: very drawish; 2 pawns: still tough; 3+ pawns: less drawish.
            const double ob_sf = (pawn_adv <= 1)   ? 0.60
                                 : (pawn_adv == 2) ? 0.72
                                                   : 0.85;
            sf = std::min(sf, ob_sf);
        }
    }

    // Keep within bounds.
    if (sf < 0.0)
        sf = 0.0;
    if (sf > 1.0)
        sf = 1.0;
    return sf;
}

static inline double endgame_scale_factor_white_pov(const Board &board,
                                                    double score_white_pov_pawns) noexcept
{
    // Determine which side is "stronger" based on current evaluation sign.
    if (std::abs(score_white_pov_pawns) < 1e-9)
        return 1.0;

    const Color strong = (score_white_pov_pawns > 0.0) ? Color::WHITE : Color::BLACK;
    const Color weak = (strong == Color::WHITE) ? Color::BLACK : Color::WHITE;

    return endgame_scale_factor_for_advantage(board, strong, weak);
}

// King crowding distance bonuses, in pawns.
// Index: [piece_type_index][distance 0..6]
// piece_type_index: 0=P, 1=N, 2=B, 3=R, 4=Q, 5=K
constexpr int NUM_PT = 6;
constexpr double KING_CROWDING_BONUS[NUM_PT][7] = {
    // PAWN
    {0.0, 0.10, 0.08, 0.06, 0.04, 0.02, 0.01},
    // KNIGHT
    {0.0, 0.25, 0.20, 0.15, 0.10, 0.05, 0.03},
    // BISHOP
    {0.0, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05},
    // ROOK
    {0.0, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05},
    // QUEEN
    {0.0, 0.40, 0.30, 0.20, 0.15, 0.10, 0.05},
    // KING (no crowding contribution)
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
// Per-color king-crowding term, mirroring Python _king_crowding_term.
double king_crowding_term_for_color(const Board &board, Color color)
{
    // Enemy king square (same as Python: enemy_color = not color)
    const Color enemy_color = (color == Color::WHITE ? Color::BLACK : Color::WHITE);
    const Square enemy_king_sq = board.kingSq(enemy_color);
    if (!enemy_king_sq.is_valid())
    {
        return 0.0;
    }

    const int k_idx = enemy_king_sq.index(); // 0..63
    const int k_file = k_idx & 7;            // 0..7
    const int k_rank = k_idx >> 3;           // 0..7

    double total = 0.0;

    // Local helper that exactly mirrors the Python loop over piece types
    auto accumulate_for_pt = [&](PieceType pt, const double *table_row)
    {
        Bitboard bb = board.pieces(pt, color);
        while (bb)
        {
            const std::uint8_t sq_idx = bb.pop(); // 0..63
            const int f = sq_idx & 7;
            const int r = sq_idx >> 3;

            // Chebyshev distance (king distance)
            const int dist = std::max(std::abs(f - k_file), std::abs(r - k_rank));

            // Python: if 1 <= dist <= 5:
            if (dist < 1 || dist > 5)
            {
                continue;
            }

            const double bonus = table_row[dist];
            if (bonus > 0.0)
            {
                total += bonus;
            }
        }
    };

    // KING_CROWDING_BONUS rows:
    // 0 = PAWN, 1 = KNIGHT, 2 = BISHOP, 3 = ROOK, 4 = QUEEN, 5 = KING (all zeros)
    accumulate_for_pt(PieceType::PAWN, KING_CROWDING_BONUS[0]);
    accumulate_for_pt(PieceType::KNIGHT, KING_CROWDING_BONUS[1]);
    accumulate_for_pt(PieceType::BISHOP, KING_CROWDING_BONUS[2]);
    accumulate_for_pt(PieceType::ROOK, KING_CROWDING_BONUS[3]);
    accumulate_for_pt(PieceType::QUEEN, KING_CROWDING_BONUS[4]);
    // No king contribution (Python has no table entry for kings)

    return total;
}

// Symmetric king-crowding term from White's POV,
// matching _king_crowding_term_white_minus_black.
double king_crowding_term_white_minus_black(const Board &board)
{
    const double white_term = king_crowding_term_for_color(board, Color::WHITE);
    const double black_term = king_crowding_term_for_color(board, Color::BLACK);
    return white_term - black_term;
}

// Evaluation module: threats + space
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

// ---------------------- Threats ----------------------
//
// Replaces the older Donna-style "attacked & undefended" heuristic with a
// closer match to SF12's threats model:
//   - strongly protected squares gate (pawn + double-attacks)
//   - weak / defended piece buckets
//   - hanging pieces, weak queen protection, restricted pieces
//   - safe pawn attacks + pawn-push threats
//   - queen-specific threats (knight/slider on queen) with queen-imbalance scaling
//
// All constants below are in centipawns in SF12; we store them in *pawn units*
// (i.e., cp / 100.0) and later taper MG/EG with mg_weight/eg_weight.

struct MgEgPawns
{
    double mg = 0.0;
    double eg = 0.0;
    constexpr MgEgPawns(int mg_cp = 0, int eg_cp = 0) : mg(mg_cp / 100.0), eg(eg_cp / 100.0) {}
};

static inline void add_mgeg(double &mg, double &eg, const MgEgPawns &s, int n = 1)
{
    mg += s.mg * n;
    eg += s.eg * n;
}

constexpr int ALL_IDX = NUM_PT; // extra slot

struct AttackMaps
{
    Bitboard by[2][NUM_PT + 1] = {}; // [color][pieceType or ALL_IDX]
    Bitboard by2[2] = {};            // squares attacked by >= 2 of that side's pieces
};

static inline int cidx(Color c) { return (c == Color::WHITE) ? 0 : 1; }

static inline Bitboard pawn_attacks_bb(Color c, Bitboard pawns)
{
    using namespace chess;
    if (c == Color::WHITE)
        return attacks::shift<Direction::NORTH_WEST>(pawns) | attacks::shift<Direction::NORTH_EAST>(pawns);
    else
        return attacks::shift<Direction::SOUTH_WEST>(pawns) | attacks::shift<Direction::SOUTH_EAST>(pawns);
}

static inline Bitboard pawn_double_attacks_bb(Color c, Bitboard pawns)
{
    using namespace chess;
    if (c == Color::WHITE)
    {
        const Bitboard a = attacks::shift<Direction::NORTH_WEST>(pawns);
        const Bitboard b = attacks::shift<Direction::NORTH_EAST>(pawns);
        return a & b;
    }
    else
    {
        const Bitboard a = attacks::shift<Direction::SOUTH_WEST>(pawns);
        const Bitboard b = attacks::shift<Direction::SOUTH_EAST>(pawns);
        return a & b;
    }
}

static inline Bitboard shift_up(Color c, Bitboard b)
{
    using namespace chess;
    return (c == Color::WHITE) ? attacks::shift<Direction::NORTH>(b)
                               : attacks::shift<Direction::SOUTH>(b);
}

static inline Bitboard mobility_area_for_color(const Board &board, Color us)
{
    using namespace chess;

    const Bitboard occ = board.occ();

    const auto &pe = pawn_hash_probe(board);
    const Bitboard enemy_pawn_attacks = (us == Color::WHITE) ? Bitboard(pe.pawn_attacks_b)
                                                             : Bitboard(pe.pawn_attacks_w);

    // Mobility area semantics (shared by mobility/threats):
    // We count attacks into a filtered area (not legal moves), so do NOT exclude all own pieces.
    // Exclusions: blocked/low pawns, our king/queen, king-blockers (pinned pieces), and enemy pawn attacks.
    const Bitboard our_pawns = board.pieces(PieceType::PAWN, us);
    const Bitboard low_ranks = (us == Color::WHITE)
                                   ? (attacks::MASK_RANK[1] | attacks::MASK_RANK[2])
                                   : (attacks::MASK_RANK[6] | attacks::MASK_RANK[5]);
    const Bitboard blocked_pawns = (us == Color::WHITE)
                                       ? (our_pawns & attacks::shift<Direction::SOUTH>(occ))
                                       : (our_pawns & attacks::shift<Direction::NORTH>(occ));
    const Bitboard pawn_exclusions = our_pawns & (blocked_pawns | low_ranks);

    const Bitboard our_kq = board.pieces(PieceType::KING, us) | board.pieces(PieceType::QUEEN, us);
    const Bitboard king_blockers = blockers_for_king_sf12(board, us);

    return ~(pawn_exclusions | our_kq | king_blockers | enemy_pawn_attacks);
}

static inline void build_attack_maps(const Board &board, AttackMaps &m)
{
    using namespace chess;

    const Bitboard occ = board.occ();
    const Bitboard queens_all = board.pieces(PieceType::QUEEN);
    const Bitboard occ_no_queens = occ ^ queens_all;

    const auto &pe = pawn_hash_probe(board);

    for (Color c : {Color::WHITE, Color::BLACK})
    {
        const int ci = cidx(c);
        // Pawn attacks from the pawn hash (fast, consistent with other eval terms).
        const Bitboard pawns = board.pieces(PieceType::PAWN, c);
        const Bitboard pawn_att = (c == Color::WHITE) ? Bitboard(pe.pawn_attacks_w)
                                                      : Bitboard(pe.pawn_attacks_b);

        m.by[ci][static_cast<int>(PieceType::PAWN)] = pawn_att;
        m.by[ci][ALL_IDX] = pawn_att;
        m.by2[ci] = pawn_double_attacks_bb(c, pawns);

        // King attacks.
        const Bitboard kbb = board.pieces(PieceType::KING, c);
        if (!kbb.empty())
        {
            const Square ksq(static_cast<std::uint8_t>(kbb.lsb()));
            const Bitboard a = attacks::king(ksq);
            m.by2[ci] |= m.by[ci][ALL_IDX] & a;
            m.by[ci][ALL_IDX] |= a;
            m.by[ci][static_cast<int>(PieceType::KING)] = a;
        }

        // Knight attacks.
        {
            Bitboard bb = board.pieces(PieceType::KNIGHT, c);
            while (bb)
            {
                const Square sq(static_cast<std::uint8_t>(bb.pop()));
                const Bitboard a = attacks::knight(sq);
                m.by2[ci] |= m.by[ci][ALL_IDX] & a;
                m.by[ci][ALL_IDX] |= a;
                m.by[ci][static_cast<int>(PieceType::KNIGHT)] |= a;
            }
        }

        // Bishop attacks (x-ray through queens for threats, SF12-style).
        {
            Bitboard bb = board.pieces(PieceType::BISHOP, c);
            while (bb)
            {
                const Square sq(static_cast<std::uint8_t>(bb.pop()));
                const Bitboard a = attacks::bishop(sq, occ_no_queens);
                m.by2[ci] |= m.by[ci][ALL_IDX] & a;
                m.by[ci][ALL_IDX] |= a;
                m.by[ci][static_cast<int>(PieceType::BISHOP)] |= a;
            }
        }

        // Rook attacks (x-ray through queens and our rooks, SF12-style).
        {
            const Bitboard our_rooks = board.pieces(PieceType::ROOK, c);
            const Bitboard occ_xray_rook = occ_no_queens ^ our_rooks;

            Bitboard bb = our_rooks;
            while (bb)
            {
                const Square sq(static_cast<std::uint8_t>(bb.pop()));
                const Bitboard a = attacks::rook(sq, occ_xray_rook);
                m.by2[ci] |= m.by[ci][ALL_IDX] & a;
                m.by[ci][ALL_IDX] |= a;
                m.by[ci][static_cast<int>(PieceType::ROOK)] |= a;
            }
        }

        // Queen attacks.
        {
            Bitboard bb = board.pieces(PieceType::QUEEN, c);
            while (bb)
            {
                const Square sq(static_cast<std::uint8_t>(bb.pop()));
                const Bitboard a = attacks::queen(sq, occ);
                m.by2[ci] |= m.by[ci][ALL_IDX] & a;
                m.by[ci][ALL_IDX] |= a;
                m.by[ci][static_cast<int>(PieceType::QUEEN)] |= a;
            }
        }
    }
}

static inline double threats_term_for_color_sf12_from_maps(const Board &board, Color us, const AttackMaps &m, double mg_weight, double eg_weight)
{
    using namespace chess;

    const Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;
    const int ui = cidx(us);
    const int ti = cidx(them);

    double mg = 0.0;
    double eg = 0.0;

    // SF12 threat constants (cp), stored as pawn-units.
    static constexpr MgEgPawns THREAT_BY_MINOR[NUM_PT] = {
        MgEgPawns(0, 0),    // pawn (unused here)
        MgEgPawns(5, 32),   // knight
        MgEgPawns(55, 41),  // bishop
        MgEgPawns(77, 56),  // rook
        MgEgPawns(89, 119), // queen
        MgEgPawns(79, 162)  // king
    };

    static constexpr MgEgPawns THREAT_BY_ROOK[NUM_PT] = {
        MgEgPawns(0, 0),
        MgEgPawns(3, 44),
        MgEgPawns(37, 68),
        MgEgPawns(42, 60),
        MgEgPawns(0, 39),
        MgEgPawns(58, 43)};

    static constexpr MgEgPawns HANGING_BONUS(69, 36);
    static constexpr MgEgPawns WEAK_Q_PROTECT(14, 0);
    static constexpr MgEgPawns RESTRICTED_PIECE(7, 7);
    static constexpr MgEgPawns THREAT_BY_KING(24, 89);
    static constexpr MgEgPawns THREAT_BY_SAFE_PAWN(173, 94);
    static constexpr MgEgPawns THREAT_BY_PAWN_PUSH(48, 39);
    static constexpr MgEgPawns KNIGHT_ON_QUEEN(16, 11);
    static constexpr MgEgPawns SLIDER_ON_QUEEN(60, 18);

    // stronglyProtected squares: enemy pawn attacks OR (enemy double attacks but not ours)
    const Bitboard strongly_protected = m.by[ti][static_cast<int>(PieceType::PAWN)] | (m.by2[ti] & ~m.by2[ui]);

    // Non-pawn enemy pieces (incl. king) and their attacked squares.
    const Bitboard non_pawn_enemies = board.us(them) & ~board.pieces(PieceType::PAWN, them);

    const Bitboard defended = non_pawn_enemies & strongly_protected;
    const Bitboard weak = board.us(them) & ~strongly_protected & m.by[ui][ALL_IDX];

    if (!(defended | weak).empty())
    {
        // Minor-piece threats.
        Bitboard b = (defended | weak) & (m.by[ui][static_cast<int>(PieceType::KNIGHT)] | m.by[ui][static_cast<int>(PieceType::BISHOP)]);
        while (b)
        {
            const Square s(static_cast<std::uint8_t>(b.pop()));
            const PieceType pt = board.at(s).type();
            add_mgeg(mg, eg, THREAT_BY_MINOR[static_cast<int>(pt.internal())]);
        }

        // Rook threats.
        b = weak & m.by[ui][static_cast<int>(PieceType::ROOK)];
        while (b)
        {
            const Square s(static_cast<std::uint8_t>(b.pop()));
            const PieceType pt = board.at(s).type();
            add_mgeg(mg, eg, THREAT_BY_ROOK[static_cast<int>(pt.internal())]);
        }

        // King threats (only once).
        if (!(weak & m.by[ui][static_cast<int>(PieceType::KING)]).empty())
            add_mgeg(mg, eg, THREAT_BY_KING);

        // Hanging pieces: weak and either undefended OR attacked twice.
        const Bitboard hanging_gate = ~m.by[ti][ALL_IDX] | (non_pawn_enemies & m.by2[ui]);
        add_mgeg(mg, eg, HANGING_BONUS, static_cast<int>((weak & hanging_gate).count()));

        // Weak pieces protected only by the queen.
        add_mgeg(mg, eg, WEAK_Q_PROTECT, static_cast<int>((weak & m.by[ti][static_cast<int>(PieceType::QUEEN)]).count()));
    }

    // "Restricted" pieces: enemy pieces that are attacked by us and are in their own attack set,
    // but are not strongly protected. (Captures the notion of pieces tied down / limited.)
    {
        const Bitboard b = m.by[ti][ALL_IDX] & ~strongly_protected & m.by[ui][ALL_IDX];
        add_mgeg(mg, eg, RESTRICTED_PIECE, static_cast<int>(b.count()));
    }

    // safe squares (SF): not attacked by them OR attacked by us.
    const Bitboard safe = ~m.by[ti][ALL_IDX] | m.by[ui][ALL_IDX];

    // Threatened by a safe pawn attack.
    {
        const Bitboard our_pawns = board.pieces(PieceType::PAWN, us);
        Bitboard b = our_pawns & safe;
        b = pawn_attacks_bb(us, b) & non_pawn_enemies;
        add_mgeg(mg, eg, THREAT_BY_SAFE_PAWN, static_cast<int>(b.count()));
    }

    // Threatened by a pawn push (single or double) that is safe and creates a pawn attack on a non-pawn enemy.
    {
        const Bitboard occ = board.occ();
        const Bitboard our_pawns = board.pieces(PieceType::PAWN, us);

        Bitboard push = shift_up(us, our_pawns) & ~occ;

        // Double push: from rank 2->4 (white) / 7->5 (black). We reuse the SF-style gate:
        // after a single push, the pawn must be on rank 3 (white) / rank 6 (black) to double.
        const Bitboard rank3_gate = (us == Color::WHITE) ? attacks::MASK_RANK[2] : attacks::MASK_RANK[5];
        push |= shift_up(us, push & rank3_gate) & ~occ;

        // Safety gates: don't push into enemy pawn attacks, and ensure the pushed pawn itself is "safe".
        push &= ~m.by[ti][static_cast<int>(PieceType::PAWN)] & safe;

        Bitboard b = pawn_attacks_bb(us, push) & non_pawn_enemies;
        add_mgeg(mg, eg, THREAT_BY_PAWN_PUSH, static_cast<int>(b.count()));
    }

    // Queen-specific threats.
    {
        const Bitboard their_queens = board.pieces(PieceType::QUEEN, them);
        if (their_queens.count() == 1)
        {
            const bool queen_imbalance = (board.pieces(PieceType::QUEEN).count() == 1);
            const Square qsq(static_cast<std::uint8_t>(their_queens.lsb()));

            // Squares that are part of our mobility area, not occupied by our pawns, and not strongly protected.
            const Bitboard mob_area = mobility_area_for_color(board, us);
            const Bitboard safe_sq = mob_area & ~board.pieces(PieceType::PAWN, us) & ~strongly_protected;

            // Knight on queen.
            {
                const Bitboard a = attacks::knight(qsq);
                const int n = static_cast<int>((m.by[ui][static_cast<int>(PieceType::KNIGHT)] & a & safe_sq).count());
                add_mgeg(mg, eg, KNIGHT_ON_QUEEN, n * (1 + (queen_imbalance ? 1 : 0)));
            }

            // Slider on queen, requiring double-attack.
            {
                const Bitboard occ = board.occ();
                const Bitboard b = (m.by[ui][static_cast<int>(PieceType::BISHOP)] & attacks::bishop(qsq, occ)) |
                                   (m.by[ui][static_cast<int>(PieceType::ROOK)] & attacks::rook(qsq, occ));
                const int n = static_cast<int>((b & safe_sq & m.by2[ui]).count());
                add_mgeg(mg, eg, SLIDER_ON_QUEEN, n * (1 + (queen_imbalance ? 1 : 0)));
            }
        }
    }

    // Taper.
    return mg * mg_weight + eg * eg_weight;
}

static inline double threats_term_for_color_sf12(const Board &board, Color us, double mg_weight, double eg_weight)
{
    AttackMaps m;
    build_attack_maps(board, m);
    return threats_term_for_color_sf12_from_maps(board, us, m, mg_weight, eg_weight);
}

static inline double threats_term_white_minus_black_from_maps(const Board &board, const AttackMaps &m, double mg_weight, double eg_weight)
{
    const double w = threats_term_for_color_sf12_from_maps(board, Color::WHITE, m, mg_weight, eg_weight);
    const double b = threats_term_for_color_sf12_from_maps(board, Color::BLACK, m, mg_weight, eg_weight);
    return w - b;
}

double threats_term_white_minus_black(const Board &board, double mg_weight, double eg_weight)
{
    AttackMaps m;
    build_attack_maps(board, m);
    return threats_term_white_minus_black_from_maps(board, m, mg_weight, eg_weight);
}

// --- Space advantage (SF12 evaluate.cpp::space()) ---
//
// Midgame-only space bonus that rewards safe central squares, and (more) safe squares
// behind friendly pawns. The term is intentionally disabled once enough major/minor
// material has been traded (mirrors SF12's SpaceThreshold gate).

static inline double total_non_pawn_material_abs_pawns(const Board &board)
{
    using namespace chess;

    auto count_both = [&](PieceType pt) -> int
    {
        const Bitboard w = board.pieces(pt, Color::WHITE);
        const Bitboard b = board.pieces(pt, Color::BLACK);
        return static_cast<int>(w.count() + b.count());
    };

    double total = 0.0;
    total += static_cast<double>(count_both(PieceType::KNIGHT)) * std::abs(piece_value(PieceType::KNIGHT, Color::WHITE));
    total += static_cast<double>(count_both(PieceType::BISHOP)) * std::abs(piece_value(PieceType::BISHOP, Color::WHITE));
    total += static_cast<double>(count_both(PieceType::ROOK)) * std::abs(piece_value(PieceType::ROOK, Color::WHITE));
    total += static_cast<double>(count_both(PieceType::QUEEN)) * std::abs(piece_value(PieceType::QUEEN, Color::WHITE));
    return total;
}

// Approximation of SF12 pawn hash "blockedCount" (counts blocked pawns for both sides).
static inline int blocked_pawn_count_approx(const Board &board)
{
    using namespace chess;
    const Bitboard pawns_all = board.pieces(PieceType::PAWN);
    const Bitboard pawns_w = board.pieces(PieceType::PAWN, Color::WHITE);
    const Bitboard pawns_b = board.pieces(PieceType::PAWN, Color::BLACK);

    // White pawn is blocked if there is any pawn on its forward square.
    const Bitboard blocked_w = pawns_w & attacks::shift<Direction::SOUTH>(pawns_all);
    // Black pawn is blocked if there is any pawn on its forward square.
    const Bitboard blocked_b = pawns_b & attacks::shift<Direction::NORTH>(pawns_all);

    return static_cast<int>(blocked_w.count() + blocked_b.count());
}

static inline double space_term_for_color_sf12like_from_maps(const Board &board, Color us, const AttackMaps &m, double mg_weight)
{
    using namespace chess;

    // SF12 gate (SpaceThreshold) in your pawn units:
    // Disable space once both queens are traded or when heavy exchanges occur.
    // Start position non-pawn material (both sides) is ~62 pawns.
    constexpr double SPACE_THRESHOLD_TOTAL_NONPAWN_PAWNS = 45.0;
    if (total_non_pawn_material_abs_pawns(board) < SPACE_THRESHOLD_TOTAL_NONPAWN_PAWNS)
        return 0.0;

    const Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;

    // Enemy pawn attacks from pawn hash (cheap and stable).
    const auto &pe = pawn_hash_probe(board);
    const Bitboard enemy_pawn_attacks = (us == Color::WHITE) ? Bitboard(pe.pawn_attacks_b)
                                                             : Bitboard(pe.pawn_attacks_w);
    // Enemy all-attacks for the "behind" double-counting gate.
    const Bitboard enemy_all_attacks = m.by[cidx(them)][ALL_IDX];

    // Central files (C-F) and near-center ranks.
    const Bitboard center_files = attacks::MASK_FILE[2] | attacks::MASK_FILE[3] | attacks::MASK_FILE[4] | attacks::MASK_FILE[5];
    const Bitboard space_ranks = (us == Color::WHITE)
                                     ? (attacks::MASK_RANK[1] | attacks::MASK_RANK[2] | attacks::MASK_RANK[3])
                                     : (attacks::MASK_RANK[4] | attacks::MASK_RANK[5] | attacks::MASK_RANK[6]);
    const Bitboard space_mask = center_files & space_ranks;

    const Bitboard our_pawns = board.pieces(PieceType::PAWN, us);

    // Safe central squares: empty, not occupied by our pawns, and not attacked by enemy pawns.
    Bitboard safe = space_mask & ~our_pawns & ~enemy_pawn_attacks;

    // Squares behind our pawns (up to two squares behind).
    Bitboard behind = our_pawns;
    if (us == Color::WHITE)
    {
        const Bitboard b1 = attacks::shift<Direction::SOUTH>(our_pawns);
        const Bitboard b2 = attacks::shift<Direction::SOUTH>(b1);
        behind |= b1 | b2;
    }
    else
    {
        const Bitboard b1 = attacks::shift<Direction::NORTH>(our_pawns);
        const Bitboard b2 = attacks::shift<Direction::NORTH>(b1);
        behind |= b1 | b2;
    }

    int bonus = static_cast<int>(safe.count());
    bonus += static_cast<int>((behind & safe & ~enemy_all_attacks).count());

    // SF12 weight: (piece count - 3) + min(blockedCount, 9)
    // Your board.us() includes the king; subtract 4 to match SF's opening weight.
    const int piece_cnt_including_king = static_cast<int>(board.us(us).count());
    const int piece_weight = std::max(0, piece_cnt_including_king - 4);
    const int blocked_cnt = std::min(blocked_pawn_count_approx(board), 9);
    const int weight = piece_weight + blocked_cnt;

    const double mg_cp = static_cast<double>(bonus) * static_cast<double>(weight) * static_cast<double>(weight) / 16.0;
    const double mg_pawns = mg_cp / 100.0;
    return mg_weight * mg_pawns;
}

static inline double space_term_for_color_sf12like(const Board &board, Color us, double mg_weight)
{
    AttackMaps m;
    build_attack_maps(board, m);
    return space_term_for_color_sf12like_from_maps(board, us, m, mg_weight);
}

static inline double space_term_white_minus_black_from_maps(const Board &board, const AttackMaps &m, double mg_weight)
{
    const double w = space_term_for_color_sf12like_from_maps(board, Color::WHITE, m, mg_weight);
    const double b = space_term_for_color_sf12like_from_maps(board, Color::BLACK, m, mg_weight);
    return w - b;
}

static inline double space_term_white_minus_black(const Board &board, double mg_weight)
{
    AttackMaps m;
    build_attack_maps(board, m);
    return space_term_white_minus_black_from_maps(board, m, mg_weight);
}

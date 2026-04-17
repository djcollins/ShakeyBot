// Evaluation module: shared evaluation context
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

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

// Mobility-area helper: compute blockers to attacks on our king (i.e. pinned pieces),
// used as part of mobilityArea[Us]. This is intentionally lightweight.
static inline Bitboard blockers_for_king_sf12(const Board &board, Color us)
{
    using namespace chess;

    const Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;

    const Bitboard king_bb = board.pieces(PieceType::KING, us);
    if (king_bb.empty())
        return Bitboard(0ULL);

    const int k_idx = king_bb.lsb();
    const int k_file = k_idx & 7;
    const int k_rank = k_idx >> 3;

    const Bitboard occ_us = board.us(us);
    const Bitboard occ_them = board.us(them);

    const Bitboard enemy_bq = board.pieces(PieceType::BISHOP, PieceType::QUEEN) & occ_them;
    const Bitboard enemy_rq = board.pieces(PieceType::ROOK, PieceType::QUEEN) & occ_them;

    Bitboard pinned(0ULL);

    auto scan_dir = [&](int df, int dr, bool diagonal)
    {
        int f = k_file + df;
        int r = k_rank + dr;

        int pinned_sq = -1;

        while (f >= 0 && f < 8 && r >= 0 && r < 8)
        {
            const int sq = (r << 3) | f;
            const Bitboard sqbb(1ULL << sq);

            if (occ_us & sqbb)
            {
                if (pinned_sq >= 0)
                    return;
                pinned_sq = sq;
            }
            else if (occ_them & sqbb)
            {
                if (pinned_sq >= 0)
                {
                    const bool slider_ok = diagonal ? bool(enemy_bq & sqbb) : bool(enemy_rq & sqbb);
                    if (slider_ok)
                        pinned.set(pinned_sq);
                }
                return;
            }

            f += df;
            r += dr;
        }
    };

    scan_dir(0, 1, false);
    scan_dir(0, -1, false);
    scan_dir(1, 0, false);
    scan_dir(-1, 0, false);

    scan_dir(1, 1, true);
    scan_dir(-1, 1, true);
    scan_dir(1, -1, true);
    scan_dir(-1, -1, true);

    return pinned;
}

struct EvalInfo
{
    const PawnHashEntry *pawn_entry = nullptr;
    Bitboard occ = 0ULL;
    Bitboard queens_all = 0ULL;
    Bitboard occ_no_queens = 0ULL;
    Square king_sq[2] = {};
    std::uint8_t pawn_files[2] = {0, 0};
    AttackMaps attack_maps = {};
    Bitboard mobility_area[2] = {};
    bool attack_maps_ready = false;
    bool mobility_area_ready[2] = {false, false};
    double total_non_pawn_material_abs_pawns = -1.0;
    int blocked_pawn_count = -1;
};

static inline std::uint8_t pawn_file_mask(Bitboard pawns) noexcept
{
    std::uint8_t mask = 0;
    while (pawns)
        mask |= static_cast<std::uint8_t>(1u << (pawns.pop() & 7));
    return mask;
}

static inline EvalInfo make_eval_info(const Board &board)
{
    EvalInfo info;
    info.pawn_entry = &pawn_hash_probe(board);
    info.occ = board.occ();
    info.queens_all = board.pieces(PieceType::QUEEN);
    info.occ_no_queens = info.occ ^ info.queens_all;
    info.king_sq[cidx(Color::WHITE)] = board.kingSq(Color::WHITE);
    info.king_sq[cidx(Color::BLACK)] = board.kingSq(Color::BLACK);
    info.pawn_files[cidx(Color::WHITE)] = pawn_file_mask(board.pieces(PieceType::PAWN, Color::WHITE));
    info.pawn_files[cidx(Color::BLACK)] = pawn_file_mask(board.pieces(PieceType::PAWN, Color::BLACK));
    return info;
}

static inline const Bitboard &mobility_area_for_color(const Board &board, Color us, EvalInfo &info)
{
    const int ui = cidx(us);
    if (info.mobility_area_ready[ui])
        return info.mobility_area[ui];

    using namespace chess;

    const Bitboard enemy_pawn_attacks = (us == Color::WHITE) ? Bitboard(info.pawn_entry->pawn_attacks_b)
                                                             : Bitboard(info.pawn_entry->pawn_attacks_w);

    const Bitboard our_pawns = board.pieces(PieceType::PAWN, us);
    const Bitboard low_ranks = (us == Color::WHITE)
                                   ? (attacks::MASK_RANK[1] | attacks::MASK_RANK[2])
                                   : (attacks::MASK_RANK[6] | attacks::MASK_RANK[5]);
    const Bitboard blocked_pawns = (us == Color::WHITE)
                                       ? (our_pawns & attacks::shift<Direction::SOUTH>(info.occ))
                                       : (our_pawns & attacks::shift<Direction::NORTH>(info.occ));
    const Bitboard pawn_exclusions = our_pawns & (blocked_pawns | low_ranks);
    const Bitboard our_kq = board.pieces(PieceType::KING, us) | board.pieces(PieceType::QUEEN, us);
    const Bitboard king_blockers = blockers_for_king_sf12(board, us);

    info.mobility_area[ui] = ~(pawn_exclusions | our_kq | king_blockers | enemy_pawn_attacks);
    info.mobility_area_ready[ui] = true;
    return info.mobility_area[ui];
}

static inline void build_attack_maps(const Board &board, EvalInfo &info, AttackMaps &m)
{
    using namespace chess;

    const auto &pe = *info.pawn_entry;

    for (Color c : {Color::WHITE, Color::BLACK})
    {
        const int ci = cidx(c);
        const Bitboard pawns = board.pieces(PieceType::PAWN, c);
        const Bitboard pawn_att = (c == Color::WHITE) ? Bitboard(pe.pawn_attacks_w)
                                                      : Bitboard(pe.pawn_attacks_b);

        m.by[ci][static_cast<int>(PieceType::PAWN)] = pawn_att;
        m.by[ci][ALL_IDX] = pawn_att;
        m.by2[ci] = pawn_double_attacks_bb(c, pawns);

        const Bitboard kbb = board.pieces(PieceType::KING, c);
        if (!kbb.empty())
        {
            const Square ksq(static_cast<std::uint8_t>(kbb.lsb()));
            const Bitboard a = attacks::king(ksq);
            m.by2[ci] |= m.by[ci][ALL_IDX] & a;
            m.by[ci][ALL_IDX] |= a;
            m.by[ci][static_cast<int>(PieceType::KING)] = a;
        }

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

        {
            Bitboard bb = board.pieces(PieceType::BISHOP, c);
            while (bb)
            {
                const Square sq(static_cast<std::uint8_t>(bb.pop()));
                const Bitboard a = attacks::bishop(sq, info.occ_no_queens);
                m.by2[ci] |= m.by[ci][ALL_IDX] & a;
                m.by[ci][ALL_IDX] |= a;
                m.by[ci][static_cast<int>(PieceType::BISHOP)] |= a;
            }
        }

        {
            const Bitboard our_rooks = board.pieces(PieceType::ROOK, c);
            const Bitboard occ_xray_rook = info.occ_no_queens ^ our_rooks;
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

        {
            Bitboard bb = board.pieces(PieceType::QUEEN, c);
            while (bb)
            {
                const Square sq(static_cast<std::uint8_t>(bb.pop()));
                const Bitboard a = attacks::queen(sq, info.occ);
                m.by2[ci] |= m.by[ci][ALL_IDX] & a;
                m.by[ci][ALL_IDX] |= a;
                m.by[ci][static_cast<int>(PieceType::QUEEN)] |= a;
            }
        }
    }
}

static inline const AttackMaps &ensure_attack_maps(const Board &board, EvalInfo &info)
{
    if (!info.attack_maps_ready)
    {
        build_attack_maps(board, info, info.attack_maps);
        info.attack_maps_ready = true;
    }
    return info.attack_maps;
}

static inline int file_kind_for_attacker(const EvalInfo &info, Color attacker, int f) noexcept
{
    if (f < 0 || f > 7)
        return 0;
    const std::uint8_t bit = static_cast<std::uint8_t>(1u << f);
    const bool any_pawn = ((info.pawn_files[0] | info.pawn_files[1]) & bit) != 0;
    const bool has_att_pawn = (info.pawn_files[cidx(attacker)] & bit) != 0;
    if (!any_pawn)
        return 2;
    if (!has_att_pawn)
        return 1;
    return 0;
}

static inline double total_non_pawn_material_abs_pawns(const Board &board, EvalInfo &info)
{
    if (info.total_non_pawn_material_abs_pawns >= 0.0)
        return info.total_non_pawn_material_abs_pawns;

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
    info.total_non_pawn_material_abs_pawns = total;
    return total;
}

static inline int blocked_pawn_count_approx(const Board &board, EvalInfo &info)
{
    if (info.blocked_pawn_count >= 0)
        return info.blocked_pawn_count;

    using namespace chess;
    const Bitboard pawns_all = board.pieces(PieceType::PAWN);
    const Bitboard pawns_w = board.pieces(PieceType::PAWN, Color::WHITE);
    const Bitboard pawns_b = board.pieces(PieceType::PAWN, Color::BLACK);
    const Bitboard blocked_w = pawns_w & attacks::shift<Direction::SOUTH>(pawns_all);
    const Bitboard blocked_b = pawns_b & attacks::shift<Direction::NORTH>(pawns_all);

    info.blocked_pawn_count = static_cast<int>(blocked_w.count() + blocked_b.count());
    return info.blocked_pawn_count;
}

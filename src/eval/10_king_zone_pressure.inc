// Evaluation module: king-zone pressure
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

// ---------------------- King-zone pressure (existing) ----------------------
//
// Enemy attacks into a 2-ring king zone. This term is intentionally simple and
// stable; it complements the cover/storm term by capturing piece pressure.

constexpr double KZ_PAWN_W = 0.008;
constexpr double KZ_KNIGHT_W = 0.015;
constexpr double KZ_BISHOP_W = 0.015;
constexpr double KZ_ROOK_W = 0.020;
constexpr double KZ_QUEEN_W = 0.030;

// ---------------------- King danger upgrades: safe checks + open lines ----------------------
//
// Goal: add a cheap, stable signal that correlates with "real" king danger
// (i.e. check threats and clear lines toward the king), while avoiding cliffs.
//
// Units: pawn units (1.00 == 100 centipawns).
constexpr double KSAFECHK_N_W = 0.050; // per safe knight-check move (capped)
constexpr double KSAFECHK_B_W = 0.040; // per safe bishop-check move (capped)
constexpr double KSAFECHK_R_W = 0.050; // per safe rook-check move (capped)
constexpr double KSAFECHK_Q_W = 0.070; // per safe queen-check move (capped)
constexpr int KSAFECHK_CAP_N = 2;
constexpr int KSAFECHK_CAP_B = 2;
constexpr int KSAFECHK_CAP_R = 2;
constexpr int KSAFECHK_CAP_Q = 2;

// Open-line pressure (direct lines to king + semi/open file pressure on king file +/- 1).
constexpr double KOPEN_DIRECT_ROOKQ = 0.080;   // rook/queen has a clear rook-line to king
constexpr double KOPEN_DIRECT_BISHOPQ = 0.060; // bishop/queen has a clear bishop-line to king
constexpr double KOPEN_FILE_OPEN = 0.030;      // rook/queen attacks ring1 on an open file near king
constexpr double KOPEN_FILE_SEMI = 0.020;      // rook/queen attacks ring1 on a semi-open file near king

Bitboard king_zone_2ring(const Square ksq)
{
    Bitboard ring1 = chess::attacks::king(ksq);
    Bitboard ring2;
    Bitboard tmp = ring1;
    while (tmp)
    {
        const std::uint8_t s = tmp.pop();
        ring2 |= chess::attacks::king(Square(s));
    }
    return ring1 | ring2 | Bitboard::fromSquare(ksq.index());
}

static inline Bitboard pawn_attacks_all(const Board &board, Color c)
{
    Bitboard a;
    Bitboard pawns = board.pieces(PieceType::PAWN, c);
    while (pawns)
    {
        const std::uint8_t s = pawns.pop();
        a |= chess::attacks::pawn(c, Square(s));
    }
    return a;
}

static inline double king_safe_check_pressure_against_color(const Board &board, Color defender)
{
    const Color attacker = (defender == Color::WHITE ? Color::BLACK : Color::WHITE);
    const Square ksq = board.kingSq(defender);

    const Bitboard occ = board.occ();
    const Bitboard occ_att = board.us(attacker);

    // "Safe" approximation: ignore checks where the checking destination is trivially
    // recaptured by a pawn or the king.
    const Bitboard def_pawn_att = pawn_attacks_all(board, defender);
    const Bitboard def_king_att = chess::attacks::king(ksq);

    // Squares from which a sliding piece would give check, given the current blockers.
    const Bitboard rook_check_sqs = chess::attacks::rook(ksq, occ);
    const Bitboard bishop_check_sqs = chess::attacks::bishop(ksq, occ);
    const Bitboard knight_check_sqs = chess::attacks::knight(ksq);

    int n_cnt = 0, b_cnt = 0, r_cnt = 0, q_cnt = 0;

    // Knights
    {
        Bitboard bb = board.pieces(PieceType::KNIGHT, attacker);
        while (bb)
        {
            const std::uint8_t s = bb.pop();
            const Square sq(s);
            Bitboard moves = chess::attacks::knight(sq) & ~occ_att;
            Bitboard checks = moves & knight_check_sqs;
            Bitboard safe = checks & ~(def_pawn_att | def_king_att);
            n_cnt += safe.count();
        }
    }

    // Bishops
    {
        Bitboard bb = board.pieces(PieceType::BISHOP, attacker);
        while (bb)
        {
            const std::uint8_t s = bb.pop();
            const Square sq(s);
            Bitboard moves = chess::attacks::bishop(sq, occ) & ~occ_att;
            Bitboard checks = moves & bishop_check_sqs;
            Bitboard safe = checks & ~(def_pawn_att | def_king_att);
            b_cnt += safe.count();
        }
    }

    // Rooks
    {
        Bitboard bb = board.pieces(PieceType::ROOK, attacker);
        while (bb)
        {
            const std::uint8_t s = bb.pop();
            const Square sq(s);
            Bitboard moves = chess::attacks::rook(sq, occ) & ~occ_att;
            Bitboard checks = moves & rook_check_sqs;
            Bitboard safe = checks & ~(def_pawn_att | def_king_att);
            r_cnt += safe.count();
        }
    }

    // Queens
    {
        const Bitboard q_check_sqs = rook_check_sqs | bishop_check_sqs;
        Bitboard bb = board.pieces(PieceType::QUEEN, attacker);
        while (bb)
        {
            const std::uint8_t s = bb.pop();
            const Square sq(s);
            Bitboard moves = chess::attacks::queen(sq, occ) & ~occ_att;
            Bitboard checks = moves & q_check_sqs;
            Bitboard safe = checks & ~(def_pawn_att | def_king_att);
            q_cnt += safe.count();
        }
    }

    // Cap to avoid cliffs and overweighting in crazy tactical positions.
    n_cnt = std::min(n_cnt, KSAFECHK_CAP_N);
    b_cnt = std::min(b_cnt, KSAFECHK_CAP_B);
    r_cnt = std::min(r_cnt, KSAFECHK_CAP_R);
    q_cnt = std::min(q_cnt, KSAFECHK_CAP_Q);

    return (KSAFECHK_N_W * static_cast<double>(n_cnt)) +
           (KSAFECHK_B_W * static_cast<double>(b_cnt)) +
           (KSAFECHK_R_W * static_cast<double>(r_cnt)) +
           (KSAFECHK_Q_W * static_cast<double>(q_cnt));
}

static inline double king_open_lines_pressure_against_color(const Board &board, Color defender)
{
    const Color attacker = (defender == Color::WHITE ? Color::BLACK : Color::WHITE);
    const Square ksq = board.kingSq(defender);

    const Bitboard occ = board.occ();
    const Bitboard ring1 = chess::attacks::king(ksq);
    const Bitboard kbb = Bitboard::fromSquare(ksq.index());

    const Bitboard att_pawns = board.pieces(PieceType::PAWN, attacker);
    const Bitboard def_pawns = board.pieces(PieceType::PAWN, defender);

    const int k_file = ksq.index() & 7;

    auto file_kind_for_attacker = [&](int f) -> int
    {
        if (f < 0 || f > 7)
            return 0;
        const Bitboard m = chess::attacks::MASK_FILE[f];
        const bool any_pawn = static_cast<bool>((att_pawns | def_pawns) & m);
        const bool has_att_pawn = static_cast<bool>(att_pawns & m);
        if (!any_pawn)
            return 2; // open
        if (!has_att_pawn)
            return 1; // semi-open for attacker
        return 0;
    };

    const int fk_m1 = file_kind_for_attacker(k_file - 1);
    const int fk_0 = file_kind_for_attacker(k_file);
    const int fk_p1 = file_kind_for_attacker(k_file + 1);

    auto file_kind = [&](int f) -> int
    {
        if (f == k_file - 1)
            return fk_m1;
        if (f == k_file)
            return fk_0;
        if (f == k_file + 1)
            return fk_p1;
        return 0;
    };

    double pen = 0.0;

    // Rooks (direct rook-line to king + file pressure near king)
    {
        Bitboard bb = board.pieces(PieceType::ROOK, attacker);
        while (bb)
        {
            const std::uint8_t s = bb.pop();
            const Square sq(s);
            const Bitboard a = chess::attacks::rook(sq, occ);

            if (a & kbb)
                pen += KOPEN_DIRECT_ROOKQ;

            const int f = sq.index() & 7;
            const int kind = file_kind(f);
            if (kind && (a & ring1))
                pen += (kind == 2 ? KOPEN_FILE_OPEN : KOPEN_FILE_SEMI);
        }
    }

    // Queens (count rook-line and bishop-line separately, but keep weights conservative)
    {
        Bitboard bb = board.pieces(PieceType::QUEEN, attacker);
        while (bb)
        {
            const std::uint8_t s = bb.pop();
            const Square sq(s);

            const Bitboard aR = chess::attacks::rook(sq, occ);
            const Bitboard aB = chess::attacks::bishop(sq, occ);

            if (aR & kbb)
                pen += KOPEN_DIRECT_ROOKQ;
            if (aB & kbb)
                pen += KOPEN_DIRECT_BISHOPQ;

            const int f = sq.index() & 7;
            const int kind = file_kind(f);
            if (kind && (aR & ring1))
                pen += (kind == 2 ? KOPEN_FILE_OPEN : KOPEN_FILE_SEMI);
        }
    }

    // Bishops (direct diagonal line to king)
    {
        Bitboard bb = board.pieces(PieceType::BISHOP, attacker);
        while (bb)
        {
            const std::uint8_t s = bb.pop();
            const Square sq(s);
            const Bitboard a = chess::attacks::bishop(sq, occ);
            if (a & kbb)
                pen += KOPEN_DIRECT_BISHOPQ;
        }
    }

    return pen;
}

static inline double king_danger_lines_checks_penalty_against_color(const Board &board, Color defender)
{
    // These signals should matter mostly in the middlegame; caller applies mg_weight.
    // Returned value is a *penalty* (larger = worse for defender).
    return king_safe_check_pressure_against_color(board, defender) +
           king_open_lines_pressure_against_color(board, defender);
}

double king_zone_attack_pressure_against_color(const Board &board, Color defender)
{
    using namespace chess;

    const Color attacker = (defender == Color::WHITE ? Color::BLACK : Color::WHITE);
    const Square ksq = board.kingSq(defender);
    if (!ksq.is_valid())
        return 0.0;

    const Bitboard occ = board.occ();
    const Bitboard zone = king_zone_2ring(ksq);

    double pressure = 0.0;
    int attackers_touching = 0;

    auto add_piece_pressure = [&](PieceType pt, double w_per_sq)
    {
        Bitboard bb = board.pieces(pt, attacker);
        while (bb)
        {
            const std::uint8_t s = bb.pop();
            const Square sq(s);
            Bitboard a;
            switch (pt.internal())
            {
            case PieceType::PAWN:
                a = attacks::pawn(attacker, sq);
                break;
            case PieceType::KNIGHT:
                a = attacks::knight(sq);
                break;
            case PieceType::BISHOP:
                a = attacks::bishop(sq, occ);
                break;
            case PieceType::ROOK:
                a = attacks::rook(sq, occ);
                break;
            case PieceType::QUEEN:
                a = attacks::queen(sq, occ);
                break;
            default:
                return;
            }
            const Bitboard hit = a & zone;
            const int c = hit.count();
            if (c > 0)
            {
                attackers_touching += 1;
                pressure += w_per_sq * static_cast<double>(c);
            }
        }
    };

    add_piece_pressure(PieceType::PAWN, KZ_PAWN_W);
    add_piece_pressure(PieceType::KNIGHT, KZ_KNIGHT_W);
    add_piece_pressure(PieceType::BISHOP, KZ_BISHOP_W);
    add_piece_pressure(PieceType::ROOK, KZ_ROOK_W);
    add_piece_pressure(PieceType::QUEEN, KZ_QUEEN_W);

    // Mild attacker-count scaling (common and stable).
    if (attackers_touching >= 4)
        pressure *= 1.45;
    else if (attackers_touching == 3)
        pressure *= 1.30;
    else if (attackers_touching == 2)
        pressure *= 1.15;

    return pressure;
}

// NOTE on weighting:
// This engine's phase_0_256 currently grows as pieces come OFF the board (i.e. it is an
// "endgame-progress" phase). Therefore, the caller must pass a *middlegame-like* weight
// (typically computed as eg_weight = 1 - endgame_progress).
double king_safety_term_for_color(const Board &board, Color color, double mg_weight)
{
    // Penalties are positive; we return a "safety score" (higher is better).
    const double cover_storm_pen = king_cover_storm_penalty_for_color(board, color);
    const double zone_attack_pen = king_zone_attack_pressure_against_color(board, color);

    // New: safe checks + open lines to king (MG-weighted; attacker-material gating is applied
    // in king_safety_term_white_minus_black).
    const double lines_checks_pen = king_danger_lines_checks_penalty_against_color(board, color);

    // Donna cover/storm is MG-weighted; keep king-zone pressure as-is.
    const double total_pen = (cover_storm_pen * mg_weight) + zone_attack_pen + (lines_checks_pen * mg_weight);

    return -total_pen;
}

double king_safety_term_white_minus_black(const Board &board, double mg_weight, const MaterialEntry &mat)
{
    // Donna-style concept (but smoother): king safety should matter in proportion to the
    // attacker's ability to convert "king-zone pressure" into real threats.
    //
    // We replace the previous binary gate with a soft attacker-material weight in [0,1].
    // This avoids evaluation cliffs (especially around Q-trades) that can hurt defense.

    auto attacker_kingsafety_weight = [&](Color attacker) noexcept -> double
    {
        int q = 0, r = 0, minors = 0;
        if (attacker == Color::WHITE)
        {
            q = mat.wQ;
            r = mat.wR;
            minors = mat.w_minors;
        }
        else
        {
            q = mat.bQ;
            r = mat.bR;
            minors = mat.b_minors;
        }

        // Base "activation" by material class (prevents weights being too tiny in common
        // rook+minor attacks while still damping minor-only endgames).
        double base = 0.0;
        if (q > 0)
            base = 0.60; // queen present: attacks are usually real
        else if (r > 0 && minors > 0)
            base = 0.40; // rook+minor: meaningful attacking potential
        else if (r >= 2)
            base = 0.30; // two rooks can create mating nets even without minors
        else if (minors >= 2)
            base = 0.20; // two minors: some king safety relevance
        else
            base = 0.0;

        // Smooth add-on by remaining attacking material.
        const double add = 0.10 * std::min(r, 2) + 0.07 * std::min(minors, 4);

        double w = base + add;
        if (w > 1.0)
            w = 1.0;
        if (w < 0.0)
            w = 0.0;
        return w;
    };

    // Attacker is the opposite color: BLACK attacks White's king; WHITE attacks Black's king.
    const double w_weight = attacker_kingsafety_weight(Color::BLACK);
    const double b_weight = attacker_kingsafety_weight(Color::WHITE);

    const double w = king_safety_term_for_color(board, Color::WHITE, mg_weight) * w_weight;
    const double b = king_safety_term_for_color(board, Color::BLACK, mg_weight) * b_weight;

    return w - b;
}

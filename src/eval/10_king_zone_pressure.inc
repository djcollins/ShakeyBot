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
    const double attack_pen = king_zone_attack_pressure_against_color(board, color);

    // Donna cover/storm is MG-weighted; keep king-zone pressure as-is.
    const double total_pen = (cover_storm_pen * mg_weight) + attack_pen;

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

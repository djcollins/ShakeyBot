inline bool is_quiet_move(const chess::Board &b, const chess::Move &m)
{
    return !b.isCapture(m) && (m.typeOf() != chess::Move::PROMOTION);
}
inline int pt_index(const chess::Piece p) noexcept
{
    if (p == chess::Piece::NONE)
        return -1;
    const int idx = static_cast<int>(p.type());
    return (0 <= idx && idx < NUM_ORDER_PT) ? idx : -1;
}
// Returns victim piece-type index for a capture move (handles en-passant).
inline int victim_pt_index(const chess::Board &board, const chess::Move &move) noexcept
{
    using namespace chess;
    Piece victim = board.at(move.to());
    if (victim == Piece::NONE && move.typeOf() == Move::ENPASSANT)
    {
        const Color stm = board.sideToMove();
        const int to_idx = move.to().index();
        const int victim_idx = (stm == Color::WHITE ? to_idx - 8 : to_idx + 8);
        if (0 <= victim_idx && victim_idx < 64)
            victim = board.at(Square(victim_idx));
    }
    return pt_index(victim);
}
// Heuristic move score (TT move, promotions, MVV-LVA captures, checks).
inline int see_cp(const chess::Board &board, const chess::Move &move);
static int score_move(const chess::Board &board,
                      const chess::Move &move,
                      const chess::Move *tt_move_ptr,
                      int ply,
                      const EngineConfig &config)
{
    using namespace chess;
    int score = 0;
    // Precompute move type flags once.
    const bool is_promo = (move.typeOf() == Move::PROMOTION);
    const bool is_capture = board.isCapture(move) || (move.typeOf() == Move::ENPASSANT);
    const int stm = stm_index(board);
    // 1) TT move first
    if (tt_move_ptr && move == *tt_move_ptr)
        score += 1'000'000;
    // 2) Promotions
    if (is_promo)
    {
        PieceType promo = move.promotionType();
        int idx = static_cast<int>(promo);
        if (0 <= idx && idx < NUM_ORDER_PT)
            score += 800'000 + PIECE_ORDER_VALUE[idx];
        else
            score += 800'000;
    }
    // 3) Captures: MVV-LVA base + capture history
    //    Note: good/bad capture splitting is done in order_moves() via SEE.
    if (is_capture)
    {
        Piece victim = board.at(move.to());
        // En passant: victim not on target square
        if (victim == Piece::NONE && move.typeOf() == Move::ENPASSANT)
        {
            const Color stmC = board.sideToMove();
            const int to_idx = move.to().index();
            const int victim_idx = (stmC == Color::WHITE ? to_idx - 8 : to_idx + 8);
            if (0 <= victim_idx && victim_idx < 64)
                victim = board.at(Square(victim_idx));
        }
        Piece attacker = board.at(move.from());
        if (victim != Piece::NONE && attacker != Piece::NONE)
        {
            const int v_idx = static_cast<int>(victim.type());
            const int a_idx = static_cast<int>(attacker.type());
            if (0 <= v_idx && v_idx < NUM_ORDER_PT &&
                0 <= a_idx && a_idx < NUM_ORDER_PT)
            {
                // Captures: MVV-LVA base. Within captures, MVV-LVA is a stable base.
                score += 200'000 + 200 * PIECE_ORDER_VALUE[v_idx] - PIECE_ORDER_VALUE[a_idx];
                // Capture history is added directly.
                if (config.use_capture_history)
                    score += (config.capture_history_ordering_mult) *
                             (capture_history[stm][a_idx][move.to().index()][v_idx] >> CAPTURE_HISTORY_SCORE_SHIFT_MAIN);
            }
        }
        // SEE weighting and bad-capture penalty (ordering only)
        const int see = see_cp(board, move);
        score += 16 * see;
        const int see_thr = config.good_capture_see_threshold_cp;
        if (config.bad_capture_penalty_cp > 0 && see < see_thr && board.givesCheck(move) == CheckType::NO_CHECK)
            score -= config.bad_capture_penalty_cp * 16;
        return score;
    }
    // 4) Quiet moves (non-promotion, non-capture)
    if (!is_promo)
    {
        // Main (butterfly) history
        if (config.use_history_heuristic)
        {
            const int f = move.from().index();
            const int t = move.to().index();
            score += config.history_ordering_mult * history_heur[stm][f][t];
        }
        // Piece-aware continuation history (Raw sum; dist==5 downweighted; dist>=3 scaled by MULTI_CONT_SCALE_PCT)
        if (config.use_continuation_history && config.continuation_ordering_mult)
        {
            const int cur_pt = pt_index(board.at(move.from()));
            const int cur_to = move.to().index();
            if (cur_pt >= 0 && cur_pt < NUM_ORDER_PT)
            {
                const bool in_check_now = board.inCheck();
                for (int dist = 1; dist <= 6; ++dist)
                {
                    if (in_check_now && dist > 2)
                        break;
                    if (ply < dist)
                        break;
                    const int idx = ply + 1 - dist;
                    const chess::Move &prev = move_stack[idx];
                    if (prev == chess::Move::NO_MOVE)
                        continue;
                    const int prev_pt = moved_pt_stack[idx];
                    if (prev_pt < 0 || prev_pt >= NUM_ORDER_PT)
                        continue;
                    const int prev_to = prev.to().index();
                    int v = cont_history_pc[stm][prev_pt][prev_to][cur_pt][cur_to];
                    if (dist == 5)
                        v /= 3;
                    if (dist >= 3 && MULTI_CONT_SCALE_PCT != 100)
                        v = static_cast<int>((static_cast<std::int64_t>(v) * MULTI_CONT_SCALE_PCT) / 100);
                    score += static_cast<int>((static_cast<std::int64_t>(config.continuation_ordering_mult) * v) / 2);
                }
            }
        }
        // Quiet check bonus
        if (board.givesCheck(move) != CheckType::NO_CHECK)
            score += 16'384;
    }
    return score;
}
// Order a Movelist in-place, best-first, using score_move.
// Order a Movelist in-place using a stage split:
//   1) TT move
//   2) Promotions
//   3) Good captures (SEE >= dynamic threshold OR checking capture)
//   4) Quiets
//   5) Bad captures
//
// This prevents SEE-negative captures from dominating ordering and wasting search effort.
static void order_moves(const chess::Board &board,
                        chess::Movelist &moves,
                        const chess::Move *tt_move_ptr,
                        int ply,
                        const EngineConfig &config)
{
    using namespace chess;
    const int n = static_cast<int>(moves.size());
    if (n <= 1)
        return;

    struct ScoredMove
    {
        Move move;
        int score;
    };

    auto cmp = [](const ScoredMove &a, const ScoredMove &b)
    {
        return a.score > b.score;
    };

    std::array<ScoredMove, constants::MAX_MOVES> tt{};
    std::array<ScoredMove, constants::MAX_MOVES> promos{};
    std::array<ScoredMove, constants::MAX_MOVES> good_caps{};
    std::array<ScoredMove, constants::MAX_MOVES> quiets{};
    std::array<ScoredMove, constants::MAX_MOVES> bad_caps{};
    int tt_n = 0, pr_n = 0, gc_n = 0, q_n = 0, bc_n = 0;

    for (int i = 0; i < n; ++i)
    {
        const Move m = moves[i];

        if (tt_move_ptr && m == *tt_move_ptr)
        {
            tt[tt_n++] = ScoredMove{m, score_move(board, m, tt_move_ptr, ply, config)};
            continue;
        }

        const bool is_promo = (m.typeOf() == Move::PROMOTION);
        const bool is_cap = board.isCapture(m) || (m.typeOf() == Move::ENPASSANT);

        if (is_promo)
        {
            promos[pr_n++] = ScoredMove{m, score_move(board, m, nullptr, ply, config)};
            continue;
        }

        if (is_cap)
        {
            // SEE good/bad split using EngineConfig thresholds. Checking captures are always "good".
            const bool gives_check = (board.givesCheck(m) != CheckType::NO_CHECK);
            bool good = gives_check;
            if (!good)
            {
                const int see = see_cp(board, m);
                good = (see >= config.good_capture_see_threshold_cp);
            }

            if (good)
                good_caps[gc_n++] = ScoredMove{m, score_move(board, m, nullptr, ply, config)};
            else
                bad_caps[bc_n++] = ScoredMove{m, score_move(board, m, nullptr, ply, config)};
            continue;
        }

        quiets[q_n++] = ScoredMove{m, score_move(board, m, nullptr, ply, config)};
    }

    // Sorting policy: full sort at root/small nodes; otherwise only partial sort the top moves per bucket.
    const bool full = (ply == 0 || n <= 12);

    auto sort_bucket = [&](std::array<ScoredMove, constants::MAX_MOVES> &arr, int cnt, int K)
    {
        if (cnt <= 1)
            return;
        if (full || cnt <= K)
            std::sort(arr.begin(), arr.begin() + cnt, cmp);
        else
            std::partial_sort(arr.begin(), arr.begin() + K, arr.begin() + cnt, cmp);
    };

    sort_bucket(promos, pr_n, 8);
    sort_bucket(good_caps, gc_n, 8);
    sort_bucket(quiets, q_n, 12);
    sort_bucket(bad_caps, bc_n, 4);

    int out = 0;
    if (tt_n)
        moves[out++] = tt[0].move;
    for (int i = 0; i < pr_n; ++i)
        moves[out++] = promos[i].move;
    for (int i = 0; i < gc_n; ++i)
        moves[out++] = good_caps[i].move;
    for (int i = 0; i < q_n; ++i)
        moves[out++] = quiets[i].move;
    for (int i = 0; i < bc_n; ++i)
        moves[out++] = bad_caps[i].move;

    // Sanity: preserve total count.
    // If TT move was present, it was also present in input; we replaced it with itself.
    // out should equal n.
}
// -------------------------------------------------------------------------
// Static Exchange Evaluation (SEE)
//
// Returns the material gain (in centipawns) for the side to move, assuming
// optimal recaptures on the target square only. Positive means the initiating
// side wins material in the exchange.
//
// Handles: captures, en-passant, promotions. Filters illegal king captures.
// -------------------------------------------------------------------------
constexpr int SEE_MAX_PLY = 32;
inline int piece_value_cp_idx(const int idx) noexcept
{
    if (idx < 0 || idx >= NUM_ORDER_PT)
        return 0;
    return PIECE_ORDER_VALUE[idx];
}
using PiecesBB = std::array<std::array<chess::Bitboard, NUM_ORDER_PT>, 2>;
inline PiecesBB build_piece_bb(const chess::Board &board)
{
    PiecesBB bb{};
    const chess::Color colors[2] = {chess::Color::WHITE, chess::Color::BLACK};
    for (int c = 0; c < 2; ++c)
    {
        for (int pt = 0; pt < NUM_ORDER_PT; ++pt)
        {
            bb[c][pt] = board.pieces(
                chess::PieceType(static_cast<chess::PieceType::underlying>(pt)),
                colors[c]);
        }
    }
    return bb;
}
inline chess::Bitboard occ_color_bb(const PiecesBB &pieces, int ci) noexcept
{
    chess::Bitboard o{};
    for (int pt = 0; pt < NUM_ORDER_PT; ++pt)
        o |= pieces[ci][pt];
    return o;
}
// A pinned piece may only move along the pin line (king <-> pinner).
inline bool on_pin_line(const chess::Square king_sq,
                        const chess::Square from,
                        const chess::Square to) noexcept
{
    const int kf = king_sq.file();
    const int kr = king_sq.rank();
    const int ff = from.file();
    const int fr = from.rank();
    const int tf = to.file();
    const int tr = to.rank();
    const int df1 = ff - kf;
    const int dr1 = fr - kr;
    const int df2 = tf - kf;
    const int dr2 = tr - kr;
    if (df1 == 0)
        return df2 == 0;
    if (dr1 == 0)
        return dr2 == 0;
    if (std::abs(df1) == std::abs(dr1))
        return std::abs(df2) == std::abs(dr2) && (df1 * dr2 == dr1 * df2);
    return false;
}
// Compute the set of pieces of `side` that are pinned to their king, for the current
// occupancy. This is used by SEE to exclude illegal recaptures.
inline chess::Bitboard pinned_mask_for_side(const chess::Color side,
                                            const chess::Bitboard occ_all,
                                            const PiecesBB &pieces) noexcept
{
    const int si = (side == chess::Color::WHITE) ? 0 : 1;
    const int oi = 1 - si;
    const chess::Square king_sq = pieces[si][5].lsb();
    const chess::Bitboard occ_us = occ_color_bb(pieces, si);
    const chess::Bitboard occ_opp = occ_color_bb(pieces, oi);
    chess::Bitboard pinned{};
    auto add_pins_from = [&](chess::Bitboard pinners)
    {
        while (pinners)
        {
            const int p_idx = pinners.pop();
            const chess::Square p_sq(p_idx);
            // Walk the ray from king to pinner and find the unique blocker (if any).
            const int kf = king_sq.file();
            const int kr = king_sq.rank();
            const int pf = p_sq.file();
            const int pr = p_sq.rank();
            const int df_raw = pf - kf;
            const int dr_raw = pr - kr;
            const int df = (df_raw > 0) - (df_raw < 0);
            const int dr = (dr_raw > 0) - (dr_raw < 0);
            // Should always be aligned for a slider pinner, but be defensive.
            if (!(df == 0 || dr == 0 || std::abs(df_raw) == std::abs(dr_raw)))
                continue;
            int f = kf + df;
            int r = kr + dr;
            int blockers = 0;
            int blocker_sq = -1;
            while (f != pf || r != pr)
            {
                const int sq = r * 8 + f;
                if (occ_all.check(sq))
                {
                    ++blockers;
                    blocker_sq = sq;
                    if (blockers > 1)
                        break;
                }
                f += df;
                r += dr;
            }
            if (blockers == 1 && blocker_sq >= 0 && occ_us.check(blocker_sq))
                pinned.set(blocker_sq);
        }
    };
    // Diagonal pins (bishop/queen)
    {
        const chess::Bitboard sliders = pieces[oi][2] | pieces[oi][4];
        const chess::Bitboard pinners = chess::attacks::bishop(king_sq, occ_opp) & sliders;
        add_pins_from(pinners);
    }
    // Orthogonal pins (rook/queen)
    {
        const chess::Bitboard sliders = pieces[oi][3] | pieces[oi][4];
        const chess::Bitboard pinners = chess::attacks::rook(king_sq, occ_opp) & sliders;
        add_pins_from(pinners);
    }
    return pinned;
}
inline chess::Bitboard attackers_to(const chess::Color c,
                                    const chess::Square to,
                                    const chess::Bitboard occ,
                                    const PiecesBB &pieces)
{
    const int ci = (c == chess::Color::WHITE) ? 0 : 1;
    chess::Bitboard atks{};
    atks |= (pieces[ci][0] & chess::attacks::pawn(~c, to));
    atks |= (pieces[ci][1] & chess::attacks::knight(to));
    atks |= (pieces[ci][2] & chess::attacks::bishop(to, occ));
    atks |= (pieces[ci][3] & chess::attacks::rook(to, occ));
    atks |= (pieces[ci][4] & chess::attacks::queen(to, occ));
    atks |= (pieces[ci][5] & chess::attacks::king(to));
    return atks;
}
struct LvaAttacker
{
    bool valid = false;
    int pt_idx = -1;
    chess::Square sq{};
};
inline LvaAttacker least_valuable_attacker(const chess::Color c,
                                           const chess::Square to,
                                           const chess::Bitboard occ,
                                           const PiecesBB &pieces,
                                           const chess::Bitboard pinned_mask)
{
    const int ci = (c == chess::Color::WHITE) ? 0 : 1;
    const chess::Square king_sq = pieces[ci][5].lsb();
    // Pawns
    {
        chess::Bitboard b = pieces[ci][0] & chess::attacks::pawn(~c, to);
        while (b)
        {
            const int idx = b.pop();
            const chess::Square sq(idx);
            if (pinned_mask.check(idx) && !on_pin_line(king_sq, sq, to))
                continue;
            return {true, 0, sq};
        }
    }
    // Knights
    {
        chess::Bitboard b = pieces[ci][1] & chess::attacks::knight(to);
        while (b)
        {
            const int idx = b.pop();
            const chess::Square sq(idx);
            if (pinned_mask.check(idx) && !on_pin_line(king_sq, sq, to))
                continue;
            return {true, 1, sq};
        }
    }
    // Bishops
    {
        chess::Bitboard b = pieces[ci][2] & chess::attacks::bishop(to, occ);
        while (b)
        {
            const int idx = b.pop();
            const chess::Square sq(idx);
            if (pinned_mask.check(idx) && !on_pin_line(king_sq, sq, to))
                continue;
            return {true, 2, sq};
        }
    }
    // Rooks
    {
        chess::Bitboard b = pieces[ci][3] & chess::attacks::rook(to, occ);
        while (b)
        {
            const int idx = b.pop();
            const chess::Square sq(idx);
            if (pinned_mask.check(idx) && !on_pin_line(king_sq, sq, to))
                continue;
            return {true, 3, sq};
        }
    }
    // Queens
    {
        chess::Bitboard b = pieces[ci][4] & chess::attacks::queen(to, occ);
        while (b)
        {
            const int idx = b.pop();
            const chess::Square sq(idx);
            if (pinned_mask.check(idx) && !on_pin_line(king_sq, sq, to))
                continue;
            return {true, 4, sq};
        }
    }
    // King (last)
    {
        chess::Bitboard b = pieces[ci][5] & chess::attacks::king(to);
        if (b)
            return {true, 5, chess::Square(static_cast<int>(b.pop()))};
    }
    return {};
}
inline int see_cp(const chess::Board &board, const chess::Move &move)
{
    const std::uint16_t mt = move.typeOf();
    const bool is_promo = (mt == chess::Move::PROMOTION);
    const bool is_ep = (mt == chess::Move::ENPASSANT);
    const bool is_cap = board.isCapture(move) || is_ep;
    if (!is_cap && !is_promo)
        return 0;
    const chess::Square from = move.from();
    const chess::Square to = move.to();
    const chess::Color stm = board.sideToMove();
    const chess::Color opp = ~stm;
    const chess::Piece mover = board.at(from);
    if (mover == chess::Piece::NONE)
        return 0;
    const int mover_pt_idx = static_cast<int>(mover.type());
    if (mover_pt_idx < 0 || mover_pt_idx >= NUM_ORDER_PT)
        return 0;
    int victim_cp = 0;
    chess::Square victim_sq = to;
    int victim_pt_idx = -1;
    if (is_cap)
    {
        if (is_ep)
        {
            const int to_idx = to.index();
            const int cap_idx = (stm == chess::Color::WHITE) ? (to_idx - 8) : (to_idx + 8);
            if (cap_idx < 0 || cap_idx >= 64)
                return 0;
            victim_sq = chess::Square(cap_idx);
            victim_pt_idx = 0; // pawn
            victim_cp = piece_value_cp_idx(0);
        }
        else
        {
            const chess::Piece victim = board.at(to);
            if (victim == chess::Piece::NONE)
                return 0;
            victim_pt_idx = static_cast<int>(victim.type());
            victim_cp = piece_value_cp_idx(victim_pt_idx);
        }
    }
    int promo_delta_cp = 0;
    int attacker_pt_idx = mover_pt_idx;
    if (is_promo)
    {
        const chess::PieceType promo = move.promotionType();
        promo_delta_cp = piece_value_cp_idx(static_cast<int>(promo)) - piece_value_cp_idx(0); // minus pawn
        attacker_pt_idx = static_cast<int>(promo);
        if (attacker_pt_idx < 0 || attacker_pt_idx >= NUM_ORDER_PT)
            attacker_pt_idx = mover_pt_idx;
    }
    // Build piece bitboards and mutable occupancy.
    PiecesBB pieces = build_piece_bb(board);
    chess::Bitboard occ = board.occ();
    const chess::Bitboard from_bb = chess::Bitboard::fromSquare(from);
    const chess::Bitboard to_bb = chess::Bitboard::fromSquare(to);
    // Remove mover from "from".
    pieces[(stm == chess::Color::WHITE) ? 0 : 1][mover_pt_idx] &= ~from_bb;
    occ &= ~from_bb;
    // Remove captured victim (if any).
    if (is_cap)
    {
        if (is_ep)
        {
            const chess::Bitboard vbb = chess::Bitboard::fromSquare(victim_sq);
            pieces[(opp == chess::Color::WHITE) ? 0 : 1][0] &= ~vbb;
            occ &= ~vbb;
        }
        else
        {
            if (victim_pt_idx >= 0 && victim_pt_idx < NUM_ORDER_PT)
                pieces[(opp == chess::Color::WHITE) ? 0 : 1][victim_pt_idx] &= ~to_bb;
            occ &= ~to_bb;
        }
    }
    // Place (possibly promoted) attacker on "to".
    pieces[(stm == chess::Color::WHITE) ? 0 : 1][attacker_pt_idx] |= to_bb;
    occ |= to_bb;
    struct Occupant
    {
        chess::Color c;
        int pt_idx;
    };
    Occupant occ_on_to{stm, attacker_pt_idx};
    int gain[SEE_MAX_PLY];
    int depth = 0;
    gain[0] = victim_cp + promo_delta_cp;
    auto apply_capture = [&](PiecesBB &p,
                             chess::Bitboard &o,
                             Occupant &occ_to,
                             const chess::Color side,
                             const LvaAttacker &att)
    {
        const int si = (side == chess::Color::WHITE) ? 0 : 1;
        const int oi = (occ_to.c == chess::Color::WHITE) ? 0 : 1;
        const chess::Bitboard from_att_bb = chess::Bitboard::fromSquare(att.sq);
        // Remove attacker from its origin square.
        p[si][att.pt_idx] &= ~from_att_bb;
        o &= ~from_att_bb;
        // Remove current occupant from "to".
        p[oi][occ_to.pt_idx] &= ~to_bb;
        // Put attacker on "to".
        p[si][att.pt_idx] |= to_bb;
        o |= to_bb;
        occ_to = {side, att.pt_idx};
    };
    chess::Color side = opp;
    while (depth + 1 < SEE_MAX_PLY)
    {
        const chess::Bitboard pinned = pinned_mask_for_side(side, occ, pieces);
        const LvaAttacker att = least_valuable_attacker(side, to, occ, pieces, pinned);
        if (!att.valid)
            break;
        const int captured_val = piece_value_cp_idx(occ_on_to.pt_idx);
        const int next_gain = captured_val - gain[depth];
        // King captures must be legal (king cannot capture into check).
        if (att.pt_idx == 5)
        {
            PiecesBB p2 = pieces;
            chess::Bitboard o2 = occ;
            Occupant occ2 = occ_on_to;
            apply_capture(p2, o2, occ2, side, att);
            if (attackers_to(~side, to, o2, p2))
                break; // illegal king capture
            pieces = p2;
            occ = o2;
            occ_on_to = occ2;
        }
        else
        {
            apply_capture(pieces, occ, occ_on_to, side, att);
        }
        gain[++depth] = next_gain;
        side = ~side;
    }
    // Back-propagate with minimax. After the initial capture, the opponent
    // chooses whether to recapture; then we may choose whether to continue,
    // etc. The formulation below yields the correct sign
    // for losing exchanges (e.g., QxP ... xQ).
    for (int i = depth - 1; i >= 0; --i)
    {
        gain[i] = -std::max(-gain[i], gain[i + 1]);
    }
    return gain[0];
}

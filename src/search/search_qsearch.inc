Score qsearch(chess::Board &board,
              int ply,
              Score alpha,
              Score beta,
              SearchStats &stats,
              const EngineConfig &config,
              SearchControl *control)
{
    // Quiescence keeps search tactical at the frontier:
    // stand-pat baseline, then captures/promotions (or evasions if in check).
    stats.nodes++;
    if (periodic_hard_stop(control, stats))
    {
        return 0;
    }
    // Hard draws
    if (board.isInsufficientMaterial())
    {
        return 0;
    }
    if (board.isHalfMoveDraw() || board.isRepetition(2))
    {
        return draw_score_stm(board, config);
    }
    if (board.isRepetition(1))
    {
        const Score ds = draw_score_stm(board, config);
        // 2-fold repetition is not yet a draw; shape it to discourage the winning side
        // from steering into repetition, but do not weaken the losing side's defensive
        // tendency to repeat when appropriate.
        return (ds < 0) ? mul_div_round(ds, TWO_FOLD_REP_NUM, TWO_FOLD_REP_DEN) : ds;
    }
    // Helper: piece value in pawns (your eval scale is pawns).
    auto value_cp = [](chess::PieceType pt) -> Score
    {
        const int idx = static_cast<int>(pt);
        if (idx < 0 || idx >= NUM_ORDER_PT)
            return 0;
        return static_cast<Score>(PIECE_ORDER_VALUE[idx]);
    };
    // Mate distance pruning: clamp window and early-exit if empty.
    clamp_mate_window(alpha, beta, ply);
    if (alpha >= beta)
        return alpha;
    // Max-ply safety (Stockfish-like): avoid overflowing ply-indexed stacks/tables.
    // At MAX_PLY, we do not expand further. If in check, return mate if no evasion else draw.
    if (ply >= MAX_PLY)
    {
        if (board.inCheck())
        {
            chess::Movelist evasions;
            chess::movegen::legalmoves(evasions, board);
            if (evasions.empty())
                return -MATE_SCORE + ply;
            return draw_score_stm(board, config);
        }
        Score e = eval_stm_no_game_over(board, config);
        e += corr_hist_probe(board, config);
        return e;
    }
    // If side to move is in check, we must search ALL legal evasions.
    if (board.inCheck())
    {
        chess::Movelist evasions;
        chess::movegen::legalmoves(evasions, board);
        if (evasions.empty())
        {
            // Checkmate: losing mate with distance.
            return -MATE_SCORE + ply;
        }
        // Order evasions using the same heuristic pipeline (TT/history).
        order_moves(board, evasions, nullptr, ply, config);
        for (int i = 0; i < evasions.size(); ++i)
        {
            if (hard_stop(control, stats))
                return 0;
            const chess::Move move = evasions[i];
            if (ply + 1 <= MAX_PLY)
            {
                move_stack[ply + 1] = move;
                moved_pt_stack[ply + 1] = pt_index(board.at(move.from()));
                prior_capture_stack[ply + 1] = capture_like_for_stack(board, move);
                parent_move_count_stack[ply + 1] = i + 1;
                tt_hit_stack[ply + 1] = false;
                in_check_stack[ply + 1] = false;
            }
            board.makeMove(move);
            const Score score = -qsearch(board, ply + 1, -beta, -alpha, stats, config, control);
            board.unmakeMove(move);
            if (stats.stopped)
                return 0;
            if (score >= beta)
                return score;
            if (score > alpha)
                alpha = score;
        }
        return alpha;
    }
    // Stand-pat (not in check)
    Score stand_pat = eval_stm_no_game_over(board, config);
    stand_pat += corr_hist_probe(board, config);
    if (stand_pat >= beta)
        return stand_pat;
    if (stand_pat > alpha)
        alpha = stand_pat;
    // Tactical move generation only: captures + quiet promotions.
    // (Non-tactical quiets are intentionally excluded to prevent horizon explosion.)
    chess::Movelist captures;
    chess::movegen::legalmoves<chess::movegen::MoveGenType::CAPTURE>(captures, board);
    // Only generate quiet promotions (non-captures). Capture promotions are already in `captures`.
    // This avoids generating *all* pawn quiet moves just because a single promotion pawn exists.
    const chess::Color stm = board.sideToMove();
    const std::uint64_t pawns_bits = board.pieces(chess::PieceType::PAWN, stm).getBits();
    constexpr std::uint64_t WHITE_PROMO_PAWNS = 0x00FF000000000000ULL; // rank 7 pawns (from White POV)
    constexpr std::uint64_t BLACK_PROMO_PAWNS = 0x000000000000FF00ULL; // rank 2 pawns (from Black POV)
    const std::uint64_t promo_pawns_bits = (stm == chess::Color::WHITE)
                                               ? (pawns_bits & WHITE_PROMO_PAWNS)
                                               : (pawns_bits & BLACK_PROMO_PAWNS);
    struct ScoredMove
    {
        chess::Move move;
        int score;
    };
    std::array<ScoredMove, chess::constants::MAX_MOVES> scored;
    int count = 0;
    auto push_scored = [&](const chess::Move &m)
    {
        // Note: this is a qsearch-specific ordering: promotions > MVV-LVA captures > checks.
        int s = 0;
        const bool is_promo = (m.typeOf() == chess::Move::PROMOTION);
        const bool is_cap = board.isCapture(m) || (m.typeOf() == chess::Move::ENPASSANT);
        if (is_promo)
        {
            const chess::PieceType promo = m.promotionType();
            s += 800'000;
            const int idx = static_cast<int>(promo);
            if (0 <= idx && idx < NUM_ORDER_PT)
                s += PIECE_ORDER_VALUE[idx];
        }
        if (is_cap)
        {
            chess::Piece victim = board.at(m.to());
            // En passant: victim is not on the target square.
            if (victim == chess::Piece::NONE && m.typeOf() == chess::Move::ENPASSANT)
            {
                const chess::Color stm = board.sideToMove();
                const int to_idx = m.to().index();
                const int victim_idx = (stm == chess::Color::WHITE ? to_idx - 8 : to_idx + 8);
                if (0 <= victim_idx && victim_idx < 64)
                    victim = board.at(chess::Square(victim_idx));
            }
            const chess::Piece attacker = board.at(m.from());
            if (victim != chess::Piece::NONE && attacker != chess::Piece::NONE)
            {
                const int v = PIECE_ORDER_VALUE[static_cast<int>(victim.type())];
                const int a = PIECE_ORDER_VALUE[static_cast<int>(attacker.type())];
                // MVV-LVA style.
                s += 100'000 + (50 * v - a);
                // Refine capture ordering with SEE + capture history. This helps reduce q-node blowups
                // in tactically dense positions without changing correctness.
                int see = see_cp(board, m);
                if (see > 2000)
                    see = 2000;
                else if (see < -2000)
                    see = -2000;
                s += see * 12;
                if (config.use_capture_history)
                {
                    const int stm = (board.sideToMove() == chess::Color::WHITE) ? 0 : 1;
                    const int v_idx = static_cast<int>(victim.type());
                    const int a_idx = static_cast<int>(attacker.type());
                    if (0 <= v_idx && v_idx < NUM_ORDER_PT && 0 <= a_idx && a_idx < NUM_ORDER_PT)
                        s += config.capture_history_ordering_mult * (capture_history[stm][a_idx][m.to().index()][v_idx] >> CAPTURE_HISTORY_SCORE_SHIFT_QS);
                }
            }
            else
            {
                s += 100'000;
            }
        }
        if (board.givesCheck(m) != chess::CheckType::NO_CHECK)
            s += 5'000;
        if (count < static_cast<int>(scored.size()))
            scored[count++] = ScoredMove{m, s};
    };
    for (int i = 0; i < captures.size(); ++i)
        push_scored(captures[i]);
    // Add promotion moves (quiet + capture).
    // Capture promotions are normally included by CAPTURE move generation, but we generate them here
    // as well and de-duplicate defensively, to avoid relying on generator details.
    auto in_captures = [&](const chess::Move &m) -> bool
    {
        for (int j = 0; j < captures.size(); ++j)
            if (captures[j] == m)
                return true;
        return false;
    };

    if (promo_pawns_bits != 0ULL)
    {
        const chess::Square king_sq = board.kingSq(stm);
        chess::Bitboard promo_pawns(promo_pawns_bits);

        auto try_promo = [&](const chess::Square from_sq, const int to_idx, const bool is_capture_target)
        {
            if (to_idx < 0 || to_idx >= 64)
                return;

            const chess::Square to_sq(to_idx);

            if (is_capture_target)
            {
                const chess::Piece target = board.at(to_sq);
                if (target == chess::Piece::NONE || target.color() == stm)
                    return;
            }
            else
            {
                // Quiet promotions require an empty target square.
                if (board.at(to_sq) != chess::Piece::NONE)
                    return;
            }

            constexpr chess::PieceType promo_types[4] = {
                chess::PieceType::QUEEN,
                chess::PieceType::ROOK,
                chess::PieceType::BISHOP,
                chess::PieceType::KNIGHT};

            for (const chess::PieceType pt : promo_types)
            {
                const chess::Move m = chess::Move::make<chess::Move::PROMOTION>(from_sq, to_sq, pt);

                // Avoid duplicates if movegen already produced capture promotions.
                if (is_capture_target && in_captures(m))
                    continue;

                // Ensure legality (primarily pins / king exposure). The library's movegen already
                // handles this when generating all pawn moves, but we're generating only promotions
                // here for performance.
                board.makeMove(m);
                const bool illegal = board.isAttacked(king_sq, board.sideToMove());
                board.unmakeMove(m);

                if (!illegal)
                    push_scored(m);
            }
        };

        while (promo_pawns)
        {
            const int from_idx = promo_pawns.pop();
            const int file = from_idx & 7;

            const chess::Square from_sq(from_idx);

            // Forward (quiet) promotion.
            const int fwd = (stm == chess::Color::WHITE) ? (from_idx + 8) : (from_idx - 8);
            try_promo(from_sq, fwd, /*is_capture_target=*/false);

            // Capture promotions.
            if (stm == chess::Color::WHITE)
            {
                if (file > 0)
                    try_promo(from_sq, from_idx + 7, /*is_capture_target=*/true);
                if (file < 7)
                    try_promo(from_sq, from_idx + 9, /*is_capture_target=*/true);
            }
            else
            {
                if (file > 0)
                    try_promo(from_sq, from_idx - 9, /*is_capture_target=*/true);
                if (file < 7)
                    try_promo(from_sq, from_idx - 7, /*is_capture_target=*/true);
            }
        }
    }
    if (count == 0)
    {
        // No tactical moves available -> quiet position.
        return alpha;
    }
    std::sort(scored.begin(), scored.begin() + count,
              [](const ScoredMove &a, const ScoredMove &b)
              {
                  return a.score > b.score;
              });
    constexpr Score DELTA_MARGIN = 50; // centipawns; conservative start
    for (int i = 0; i < count; ++i)
    {
        if (hard_stop(control, stats))
            return 0;
        const chess::Move move = scored[i].move;
        // Delta pruning (skip hopeless tactical moves), but do not delta-prune checking moves.
        {
            Score gain = 0;
            if (board.isCapture(move) || (move.typeOf() == chess::Move::ENPASSANT))
            {
                chess::Piece victim = board.at(move.to());
                if (victim == chess::Piece::NONE && move.typeOf() == chess::Move::ENPASSANT)
                {
                    victim = chess::Piece(chess::PieceType::PAWN, ~board.sideToMove());
                }
                if (victim != chess::Piece::NONE)
                    gain += value_cp(victim.type());
            }
            if (move.typeOf() == chess::Move::PROMOTION)
            {
                const chess::PieceType promo = move.promotionType();
                gain += (value_cp(promo) - value_cp(chess::PieceType::PAWN));
            }
            if (stand_pat + gain + DELTA_MARGIN <= alpha)
            {
                if (board.givesCheck(move) == chess::CheckType::NO_CHECK)
                {
                    continue;
                }
            }
        }
        // SEE pruning complements delta pruning: it filters exchanges that are tactically
        // unsound even if the raw captured material looked tempting.
        // Never apply this filter to checking moves.
        {
            const bool is_cap = board.isCapture(move) || (move.typeOf() == chess::Move::ENPASSANT);
            if (is_cap && board.givesCheck(move) == chess::CheckType::NO_CHECK)
            {
                constexpr int SEE_QS_PRUNE_CP = 100;
                if (see_cp(board, move) < -SEE_QS_PRUNE_CP)
                {
                    continue;
                }
            }
        }
        if (ply + 1 <= MAX_PLY)
        {
            move_stack[ply + 1] = move;
            moved_pt_stack[ply + 1] = pt_index(board.at(move.from()));
            prior_capture_stack[ply + 1] = capture_like_for_stack(board, move);
            parent_move_count_stack[ply + 1] = i + 1;
            tt_hit_stack[ply + 1] = false;
            in_check_stack[ply + 1] = false;
        }
        board.makeMove(move);
        const Score score = -qsearch(board, ply + 1, -beta, -alpha, stats, config, control);
        board.unmakeMove(move);
        if (stats.stopped)
            return 0;
        if (score >= beta)
            return score;
        if (score > alpha)
            alpha = score;
    }
    return alpha;
}

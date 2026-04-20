
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
    // Max-ply safety : avoid overflowing ply-indexed stacks/tables.
    // At MAX_PLY, we do not expand further. If in check, return mate if no evasion else draw.
    if (ply >= MAX_PLY)
    {
        if (board.inCheck())
        {
            chess::Movelist evasions;
            chess::movegen::legalmoves(evasions, board);
            record_legal_movegen(stats, evasions);
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
        record_legal_movegen(stats, evasions);
        if (evasions.empty())
        {
            // Checkmate: losing mate with distance.
            return -MATE_SCORE + ply;
        }
        // Order evasions using the same heuristic pipeline as before.
        order_moves(board, evasions, nullptr, ply, config);
        for (int i = 0; i < evasions.size(); ++i)
        {
            if (hard_stop(control, stats))
                return 0;
            const chess::Move move = evasions[i];
            if (ply + 1 <= MAX_PLY)
                set_search_stack_entry(ply + 1, board, move, i + 1);
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
        int see = 0;
        bool has_see = false;
        bool gives_check = false;
    };
    std::array<ScoredMove, chess::constants::MAX_MOVES> scored;
    int count = 0;
    auto push_scored = [&](const chess::Move &m)
    {
        // Note: this is a qsearch-specific ordering: promotions > MVV-LVA captures > checks.
        int s = 0;
        const bool is_promo = (m.typeOf() == chess::Move::PROMOTION);
        const bool is_cap = board.isCapture(m) || (m.typeOf() == chess::Move::ENPASSANT);
        int move_see = 0;
        bool have_see = false;
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
                move_see = see_cp(board, m);
                have_see = true;
                int ordering_see = move_see;
                if (ordering_see > 2000)
                    ordering_see = 2000;
                else if (ordering_see < -2000)
                    ordering_see = -2000;
                s += ordering_see * 12;
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
                move_see = see_cp(board, m);
                have_see = true;
            }
        }
        const bool gives_check = (board.givesCheck(m) != chess::CheckType::NO_CHECK);
        if (gives_check)
            s += 5'000;
        if (count < static_cast<int>(scored.size()))
            scored[count++] = ScoredMove{m, s, move_see, have_see, gives_check};
    };
    for (int i = 0; i < captures.size(); ++i)
        push_scored(captures[i]);
    if (promo_pawns_bits != 0ULL)
    {
        chess::Bitboard promo_pawns(promo_pawns_bits);
        constexpr chess::PieceType PROMO_TYPES[4] = {
            chess::PieceType::QUEEN,
            chess::PieceType::ROOK,
            chess::PieceType::BISHOP,
            chess::PieceType::KNIGHT,
        };
        const int push = (stm == chess::Color::WHITE) ? 8 : -8;
        while (promo_pawns)
        {
            const chess::Square from = chess::Square(promo_pawns.pop());
            const int to_idx = from.index() + push;
            if (to_idx < 0 || to_idx >= 64)
                continue;
            const chess::Square to = chess::Square(to_idx);
            if (board.at(to) != chess::Piece::NONE)
                continue;
            for (const chess::PieceType promo : PROMO_TYPES)
            {
                const chess::Move move = chess::Move::make<chess::Move::PROMOTION>(from, to, promo);
                board.makeMove(move);
                const bool legal = !board.isAttacked(board.kingSq(~board.sideToMove()), board.sideToMove());
                board.unmakeMove(move);
                if (legal)
                    push_scored(move);
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
    constexpr Score DELTA_MARGIN = 100; // centipawns; slightly less aggressive (safer tactics)
    for (int i = 0; i < count; ++i)
    {
        if (hard_stop(control, stats))
            return 0;
        const chess::Move move = scored[i].move;
        const bool is_cap = board.isCapture(move) || (move.typeOf() == chess::Move::ENPASSANT);
        const bool gives_check = scored[i].gives_check;
        // Delta pruning (skip hopeless tactical moves), but do not delta-prune checking moves.
        {
            Score gain = 0;
            if (is_cap)
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
                if (!gives_check)
                    continue;
            }
        }
        // SEE pruning complements delta pruning: it filters exchanges that are tactically
        // unsound even if the raw captured material looked tempting.
        // Never apply this filter to checking moves.
        {
            if (is_cap)
            {
                constexpr int SEE_QS_PRUNE_CP = 200; // prune only clearly losing captures
                const int move_see = scored[i].has_see ? scored[i].see : see_cp(board, move);
                if (move_see < -SEE_QS_PRUNE_CP && !gives_check)
                {
                    continue;
                }
            }
        }
        if (ply + 1 <= MAX_PLY)
            set_search_stack_entry(ply + 1, board, move, i + 1);
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

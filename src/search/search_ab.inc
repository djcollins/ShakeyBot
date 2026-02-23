
// --- Pawn-push criticality (endgame / passed pawn awareness) ---
// Purpose: protect passer creation/advances from the harshest late pruning and reductions.
static inline int color_index(chess::Color c) noexcept
{
    return static_cast<int>(c.internal());
}

static inline std::uint64_t passed_mask_for_search(chess::Color us, int sq) noexcept
{
    static bool inited = false;
    static std::uint64_t masks[2][64]; // [WHITE/BLACK][square]
    if (!inited)
    {
        for (int s = 0; s < 64; ++s)
        {
            const int f = s & 7;
            const int r = s >> 3;
            std::uint64_t mw = 0, mb = 0;
            for (int ff = std::max(0, f - 1); ff <= std::min(7, f + 1); ++ff)
            {
                for (int rr = r + 1; rr < 8; ++rr)
                    mw |= 1ULL << (rr * 8 + ff);
                for (int rr = r - 1; rr >= 0; --rr)
                    mb |= 1ULL << (rr * 8 + ff);
            }
            masks[0][s] = mw; // WHITE
            masks[1][s] = mb; // BLACK
        }
        inited = true;
    }
    return masks[color_index(us)][sq];
}

static inline int passed_progress_search(chess::Color us, int sq) noexcept
{
    const int r = sq >> 3;
    return (color_index(us) == static_cast<int>(chess::Color::WHITE)) ? r : (7 - r);
}

static inline bool is_critical_endgame_pawn_push_search(const chess::Board &board, const chess::Move &move) noexcept
{
    // Promotions are always critical and should not be subject to the harshest pruning/reductions.
    if (move.typeOf() == chess::Move::PROMOTION)
        return true;

    const chess::Piece p = board.at(move.from());
    if (p.type() != chess::PieceType::PAWN)
        return false;

    // "Endgame-ish" gate: focus this protection on queenless positions.
    if (board.pieces(chess::PieceType::QUEEN))
        return false;

    const chess::Color us = board.sideToMove();
    const chess::Color them = chess::Color(~us.internal());

    const int from = move.from().index();
    const int to = move.to().index();

    const std::uint64_t enemy_pawns = board.pieces(chess::PieceType::PAWN, them).getBits();

    const bool passed_before = ((enemy_pawns & passed_mask_for_search(us, from)) == 0ULL);
    const bool passed_after = ((enemy_pawns & passed_mask_for_search(us, to)) == 0ULL);

    if (!passed_after)
        return false;

    const int prog_to = passed_progress_search(us, to);

    // Strong signals:
    //  - Any passed pawn reaching the 6th+ (from own POV) is usually critical in queenless endgames.
    //  - Creating a new passer on the 5th+ is also typically critical.
    if (prog_to >= 5)
        return true;

    if (!passed_before && prog_to >= 4)
        return true;

    // Advancing an existing passer into the "danger zone" (5th+) is also worth protecting.
    if (passed_before && prog_to >= 4)
        return true;

    return false;
}

Score negamax(chess::Board &board,
              int depth,
              int ply,
              Score alpha,
              Score beta,
              bool pv,
              SearchStats &stats,
              const EngineConfig &config,
              bool use_quiescence,
              bool allow_iid,
              TranspositionTable *tt,
              SearchControl *control)
{
    // Core fail-soft alpha-beta:
    // 1) handle terminal/pruning gates, 2) order and search moves (PVS/LMR),
    // 3) update learning tables, 4) store TT bound/exact result.
    // Track max search depth reached (root-relative depth, not ply)
    // depth_reached is used for reporting only; keep it stable and meaningful.
    stats.depth_reached = std::max(stats.depth_reached, stats.depth_requested - depth);
    const bool leaf_to_qsearch = (use_quiescence && depth <= 0);
    if (!leaf_to_qsearch)
        stats.nodes++;
    if (periodic_hard_stop(control, stats))
    {
        return 0;
    }
    // Max-ply safety: prevent ply-stack overflow in pathological lines.
    if (ply >= MAX_PLY)
    {
        chess::Movelist mvs;
        chess::movegen::legalmoves(mvs, board);
        if (mvs.empty())
            return board.inCheck() ? (-MATE_SCORE + ply) : 0;
        if (board.inCheck())
            return draw_score_stm(board, config);
        Score e = eval_stm_no_game_over(board, config);
        e += corr_hist_probe(board, config);
        return e;
    }
    std::uint64_t key = 0;
    chess::Move tt_best_move{};
    bool have_tt_best = false;
    // Singular verification can request "search this node without one move";
    // when active, bypass TT at this exact ply to avoid self-confirming cutoffs.
    const bool exclude_this_node = (g_excluded_move_ply == ply && g_excluded_move != chess::Move::NO_MOVE);
    TranspositionTable *tt_here = exclude_this_node ? nullptr : tt;
    bool have_tt_entry = false;
    int tt_entry_depth = 0;
    TTFlag tt_entry_flag = TT_UPPERBOUND;
    Score tt_entry_value = 0;
    // --- Draws / repetition ---
    // IMPORTANT: only mark stats.is_draw at ROOT (ply == 0). Deep repetitions should
    // not set the final "is_draw" output.
    // Also IMPORTANT: use isRepetition(1) as early warning to prevent perpetual loops.
    if (board.isInsufficientMaterial())
    {
        if (ply == 0)
            stats.is_draw = true;
        return 0;
    }
    if (board.isHalfMoveDraw() || board.isRepetition(2))
    {
        if (ply == 0)
            stats.is_draw = true;
        return draw_score_stm(board, config);
    }
    // 2-fold repetition warning (NOT a draw yet), but we treat it as drawish so the
    // winning side avoids stepping into repetition cycles.
    if (board.isRepetition(1))
    {
        const Score ds = draw_score_stm(board, config);
        // 2-fold repetition is not yet a draw; shape it to discourage the winning side
        // from steering into repetition, but do not weaken the losing side's defensive
        // tendency to repeat when appropriate.
        return (ds < 0) ? mul_div_round(ds, TWO_FOLD_REP_NUM, TWO_FOLD_REP_DEN) : ds;
    }
    // Mate distance pruning: clamp window and early-exit if empty.
    // Safe because mate scores are represented as +/-MATE_SCORE with ply distance.
    clamp_mate_window(alpha, beta, ply);
    if (alpha >= beta)
        return alpha;
    // TT cutoffs are only safe/reliable in null-window (cut) nodes.
    // Do not attempt to infer PV-ness from window width beyond this gate; PV semantics are handled elsewhere.
    const bool is_null_window_tt = (beta - alpha) <= ONE_CP;
    // Initialize LMR tables and root aspiration delta (used in LMR reduction formula).
    init_lmr_tables_once();
    if (ply == 0)
        lmr_root_delta = std::max(1, static_cast<int>(beta - alpha));
    // TT probe (with mate-distance restore)
    if (tt_here)
    {
        key = board.hash();
        tt_hit_stack[ply] = false;
        if (auto entry_opt = tt_here->probe(key))
        {
            stats.tt_hits++;
            tt_hit_stack[ply] = true;
            const TTEntry &e = *entry_opt;
            const Score tt_value = from_tt_score(e.value, ply);
            have_tt_entry = true;
            tt_entry_depth = e.depth;
            tt_entry_flag = e.flag;
            tt_entry_value = tt_value;
            if (e.hasMove)
            {
                tt_best_move = e.bestMove;
                have_tt_best = true;
            }
            // Early TT cutoff (non-PV / null-window nodes only).
            // Important: do not tighten alpha/beta here; only return if the TT entry already
            // implies the node fails high/low at the current beta.
            //
            // This keeps the searched window stable for downstream pruning gates (razoring, NMP, etc.).
            if (is_null_window_tt && depth > 0)
            {
                const bool ttLower = (e.flag == TT_LOWERBOUND || e.flag == TT_EXACT);
                const bool ttUpper = (e.flag == TT_UPPERBOUND || e.flag == TT_EXACT);
                //  Gate: ttDepth > depth - (ttValue <= beta)
                //  - If tt_value  < beta (fail-low), requires ttDepth >= depth
                //  - If tt_value == beta (borderline fail-high), requires ttDepth >= depth
                //  - If tt_value  > beta (clear fail-high), requires ttDepth >= depth + 1
                const int depthMargin = (tt_value <= beta) ? 1 : 0;
                const bool ttDepthOk = (e.depth > depth - depthMargin);
                if (ttDepthOk)
                {
                    const bool failHigh = (tt_value >= beta);
                    if ((failHigh && ttLower) || (!failHigh && ttUpper))
                    {
                        // TT fail-high learning hook.
                        // If the TT entry causes an immediate fail-high cutoff at this node (non-PV/null-window),
                        // 1) still update quiet histories for the TT move
                        // 2) apply a continuation penalty to early quiet parent moves (when applicable).
                        if (failHigh)
                        {
                            // 1) Quiet TT move bonus (main history + continuation histories)
                            if (config.use_history_heuristic && e.hasMove)
                            {
                                const chess::Move hm = e.bestMove;
                                if (hm != chess::Move::NO_MOVE)
                                {
                                    const bool promoQ = (hm.typeOf() == chess::Move::PROMOTION && hm.promotionType() == chess::PieceType::QUEEN);
                                    const bool capture_stage = board.isCapture(hm) || (hm.typeOf() == chess::Move::ENPASSANT) || promoQ;
                                    if (!capture_stage)
                                    {
                                        const int s = stm_index(board);
                                        const int f = hm.from().index();
                                        const int t = hm.to().index();
                                        const int bonus_tt = std::min(120 * depth - 75, 1241); // TO BE TUNED
                                        update_main_history_entry(history_heur[s][f][t], bonus_tt);
                                        if (config.use_continuation_history && ply > 0)
                                        {
                                            const int cur_pt = pt_index(board.at(hm.from()));
                                            const int cont_tt = (bonus_tt * 1004) / 1024; // TO BE TUNED
                                            static constexpr std::array<int, 7> W = {0, 1102, 658, 325, 535, 125, 475};
                                            const bool in_check_tt = board.inCheck();
                                            for (int i = 1; i <= 6; ++i)
                                            {
                                                // Only update the first 2 continuation contexts if in check.
                                                if (in_check_tt && i > 2)
                                                    break;
                                                if (ply < i)
                                                    break;
                                                const int idx = ply + 1 - i;
                                                const chess::Move &prev = move_stack[idx];
                                                if (prev == chess::Move::NO_MOVE)
                                                    continue;
                                                const int prev_to = prev.to().index();
                                                const int delta = (cont_tt * W[i]) / 1024;
                                                if (delta == 0)
                                                    continue;
                                                update_cont_history_entry(cont_history[s][prev_to][f][t], delta);
                                                const int prev_pt = moved_pt_stack[idx];
                                                if (prev_pt >= 0 && cur_pt >= 0)
                                                    update_cont_history_entry(cont_history_pc[s][prev_pt][prev_to][cur_pt][t], delta);
                                            }
                                        }
                                    }
                                }
                            }
                            // 2) Parent early quiet move continuation penalty
                            // Condition: prevSq != NONE, (ss-1)->moveCount <= 3, and !priorCapture
                            if (config.use_continuation_history && ply > 0 && parent_move_count_stack[ply] <= 3 && !prior_capture_stack[ply])
                            {
                                const chess::Move pm = move_stack[ply];
                                if (pm != chess::Move::NO_MOVE)
                                {
                                    const int s_parent = 1 - stm_index(board);
                                    const int cur_pt = moved_pt_stack[ply];
                                    const int cur_from = pm.from().index();
                                    const int cur_to = pm.to().index();
                                    const int pen_base = -std::min(809 * (depth + 1) - 249, 3052);
                                    const int pen_cont = (pen_base * 1004) / 1024;
                                    static constexpr std::array<int, 7> W = {0, 1103, 659, 323, 533, 121, 474};
                                    for (int i = 1; i <= 6; ++i)
                                    {
                                        const int idx = ply - i;
                                        if (idx < 0)
                                            break;
                                        const chess::Move &prev = move_stack[idx];
                                        if (prev == chess::Move::NO_MOVE)
                                            continue;
                                        const int prev_to = prev.to().index();
                                        const int prev_pt = moved_pt_stack[idx];
                                        const int delta = (pen_cont * W[i]) / 1024;
                                        if (delta == 0)
                                            continue;
                                        update_continuation_no_board(s_parent, prev_pt, prev_to, cur_pt, cur_from, cur_to, delta);
                                    }
                                }
                            }
                        }
                        return tt_value;
                    }
                }
            }
        }
        else
        {
            stats.tt_misses++;
            tt_hit_stack[ply] = false;
        }
    }
    // Keep the original window for TT flagging and learning updates at node exit.
    // (We intentionally do not tighten the window from TT bounds.)
    const Score alpha_window = alpha;
    const Score beta_window = beta;
    const bool is_null_window = (beta_window - alpha_window) <= ONE_CP;
    const bool in_check = board.inCheck();
    const Score node_corr_eval = corrected_eval_stm(board, config);
    if (0 <= ply && ply <= MAX_PLY)
        in_check_stack[ply] = in_check;
    // --- Razoring (very conservative; depth 2-3 only; null-window only) ---
    // Trigger: if static-eval is sufficiently below alpha, do a null-window qsearch probe.
    // If the probe also fails low, we can return early and avoid a full ply expansion.
    if (config.use_razoring &&
        config.use_quiescence &&
        is_null_window &&
        !in_check &&
        ply > 0 &&
        (depth == 2 || depth == 3) &&
        allow_razoring_by_material(board) &&
        !tactically_volatile_for_razoring(board) &&
        alpha < MATE_BOUND &&
        alpha > -MATE_BOUND)
    {
        Score se = node_corr_eval;
        const Score margin = static_cast<Score>(460 + 310 * depth * depth); // TO BE TUNED
        if (se + margin <= alpha)
        {
            ++stats.razor_attempts;
            // Return qsearch at the node window if the static-eval gate already fails low.
            const Score qs = qsearch(board, ply, alpha, beta, stats, config, control);
            if (stats.stopped)
                return 0;
            if (qs <= alpha)
            {
                ++stats.razor_cutoffs;
                return qs;
            }
        }
    }
    // --- Null-move pruning (adaptive reduction + verification) ---
    //   - Upgrade from fixed R=2:
    //   - Adaptive reduction based on depth (and a small "above-beta" signal)
    //   - Verification search at higher depths with null-move disabled (zugzwang safety)
    // Null-move pruning is intended for cut-nodes (null-window).
    // Applying it in PV / wide-window nodes (incl. root under aspiration) can cause tactical
    // oversights and unstable PV selection.
    const bool excluded_move_active = (g_excluded_move_ply == ply && g_excluded_move != chess::Move::NO_MOVE);
    const bool prev_was_null = (ply > 0 && move_stack[ply] == chess::Move::NO_MOVE);
    if (config.use_null_move_pruning && g_allow_null_move &&
        is_null_window &&
        ply > 0 &&
        ply >= g_nmp_min_ply &&
        ply + 1 <= MAX_PLY &&
        !excluded_move_active &&
        !prev_was_null &&
        depth >= 3 &&
        !in_check &&
        beta < MATE_BOUND &&
        alpha > -MATE_BOUND)
    {
        auto game = board.isGameOver();
        if (game.second == chess::GameResult::NONE &&
            board.hasNonPawnMaterial(board.sideToMove()))
        {
            Score static_eval = node_corr_eval;
            // Gate: require eval sufficiently above beta, with a margin that increases with depth
            // (higher depth => higher confidence requirement before we prune).
            const Score nmp_margin = static_cast<Score>(std::max(0, 4 * depth - 8));
            if (static_eval >= beta + nmp_margin)
            {
                const int delta = static_cast<int>(static_eval - beta);
                int R = std::min(delta / 235, 6) + depth / 3 + 2;
                // Material-sensitive adjustment: NMP is less reliable in simplified (heavy-piece-less) positions.
                // Reduce the baseline reduction to improve zugzwang safety in minor-only / low-heavy endgames.
                //
                // However, this extra safety can slightly hurt defensive performance under fixed time when
                // the side to move is already clearly worse (because it reduces pruning aggressiveness).
                // To avoid that, apply the endgame-safety reduction only when the side-to-move is not
                // materially/positionally "clearly worse" in absolute terms.
                constexpr Score NMP_ENDGAME_SAFE_MIN_EVAL = -80; // centipawns (stm POV)
                const bool apply_endgame_safe = (static_eval >= NMP_ENDGAME_SAFE_MIN_EVAL);
                if (apply_endgame_safe)
                {
                    const int heavy_cnt = board.pieces(chess::PieceType::QUEEN).count() +
                                          board.pieces(chess::PieceType::ROOK).count();
                    if (heavy_cnt == 0)
                        R -= 2;
                    else if (heavy_cnt == 1)
                        R -= 1;
                    else if (!board.pieces(chess::PieceType::QUEEN) && heavy_cnt <= 2)
                        R -= 1;
                }
                if (R < 2)
                    R = 2;
                int null_depth = depth - R;
                if (null_depth < 0)
                    null_depth = 0;
                // Null-move is not “a real move”: ensure the history stacks do not
                // accidentally treat a stale move_stack/moved_pt_stack entry as the
                // previous move inside the null-move search.
                const chess::Move saved_null_stack_move = move_stack[ply + 1];
                const int saved_null_stack_pt = moved_pt_stack[ply + 1];
                const bool saved_null_stack_prior_cap = prior_capture_stack[ply + 1];
                const int saved_null_stack_parent_mc = parent_move_count_stack[ply + 1];
                const bool saved_null_stack_tt_hit = tt_hit_stack[ply + 1];
                const bool saved_null_stack_in_check = in_check_stack[ply + 1];
                move_stack[ply + 1] = chess::Move::NO_MOVE;
                moved_pt_stack[ply + 1] = -1;
                prior_capture_stack[ply + 1] = false;
                parent_move_count_stack[ply + 1] = 0;
                tt_hit_stack[ply + 1] = false;
                in_check_stack[ply + 1] = false;
                board.makeNullMove();
                const Score null_score = [&]()
                {
                    ScopedNullMoveFlag disable(false);
                    return -negamax(
                        board,
                        null_depth,
                        ply + 1,
                        -beta,
                        -beta + ONE_CP,
                        /*pv=*/false,
                        stats,
                        config,
                        use_quiescence,
                        allow_iid,
                        tt,
                        control);
                }();
                board.unmakeNullMove();
                move_stack[ply + 1] = saved_null_stack_move;
                moved_pt_stack[ply + 1] = saved_null_stack_pt;
                prior_capture_stack[ply + 1] = saved_null_stack_prior_cap;
                parent_move_count_stack[ply + 1] = saved_null_stack_parent_mc;
                tt_hit_stack[ply + 1] = saved_null_stack_tt_hit;
                in_check_stack[ply + 1] = saved_null_stack_in_check;
                if (stats.stopped)
                    return 0;
                if (null_score >= beta && null_score < MATE_BOUND)
                {
                    // Return fail-soft at shallow depths or when already in verification.
                    if (g_nmp_min_ply != 0 || depth < 8)
                        return null_score;
                    int verify_depth = depth - R;
                    if (verify_depth < 1)
                        verify_depth = 1;
                    // Disable null-move pruning until ply exceeds this threshold (zugzwang safety).
                    const int min_ply = ply + (3 * verify_depth) / 4;
                    const Score verify = [&]()
                    {
                        ScopedNmpMinPly scoped_min_ply(min_ply);
                        return negamax(
                            board,
                            verify_depth,
                            ply,
                            beta - ONE_CP,
                            beta,
                            /*pv=*/false,
                            stats,
                            config,
                            use_quiescence,
                            false, // allow_iid
                            tt,
                            control);
                    }();
                    if (stats.stopped)
                        return 0;
                    if (verify >= beta)
                        return null_score;
                }
            }
        }
    }
    // --- ProbCut (null-window only) ---
    // If a reduced search on a strong capture returns a value well above beta, we can prune.
    const Score probCutBeta = beta + static_cast<Score>(185);
    if (config.use_probcut &&
        depth >= 3 &&
        is_null_window &&
        !in_check &&
        beta < MATE_BOUND && alpha > -MATE_BOUND &&
        !(have_tt_entry && tt_entry_value < probCutBeta))
    {
        const int probCutDepth = std::max(depth - 4, 0);
        // Guide candidate selection by using a SEE threshold
        // derived from (probCutBeta - staticEval).
        Score pc_static = node_corr_eval;
        const int minSeeCp = std::max(0, static_cast<int>(probCutBeta - pc_static));
        constexpr int PROBCUT_MAX_TRIES = 8;
        chess::Movelist caps;
        chess::movegen::legalmoves<chess::movegen::MoveGenType::CAPTURE>(caps, board);
        if (!caps.empty())
        {
            const chess::Move *pc_tt_ptr = have_tt_best ? &tt_best_move : nullptr;
            order_moves(board, caps, pc_tt_ptr, ply, config);
            const int tries = std::min<int>(static_cast<int>(caps.size()), PROBCUT_MAX_TRIES);
            for (int i = 0; i < tries; ++i)
            {
                const chess::Move m = caps[i];
                // Skip losing / insufficient captures.
                if (see_cp(board, m) < minSeeCp)
                    continue;
                if (ply + 1 <= MAX_PLY)
                {
                    move_stack[ply + 1] = m;
                    moved_pt_stack[ply + 1] = pt_index(board.at(m.from()));
                    prior_capture_stack[ply + 1] = capture_like_for_stack(board, m);
                    parent_move_count_stack[ply + 1] = i + 1;
                    tt_hit_stack[ply + 1] = false;
                    in_check_stack[ply + 1] = false;
                }
                board.makeMove(m);
                // Preliminary qsearch verification .
                Score score = -qsearch(board, ply + 1, -probCutBeta, -probCutBeta + ONE_CP, stats, config, control);
                // If the qsearch held, perform the reduced-depth search.
                if (score >= probCutBeta && probCutDepth > 0)
                {
                    score = -negamax(board,
                                     probCutDepth,
                                     ply + 1,
                                     -probCutBeta,
                                     -probCutBeta + ONE_CP,
                                     /*pv=*/false,
                                     stats,
                                     config,
                                     use_quiescence,
                                     allow_iid,
                                     tt,
                                     control);
                }
                board.unmakeMove(m);
                if (stats.stopped)
                    return 0;
                if (score >= probCutBeta)
                {
                    // Store a lower bound at an effective depth  (probCutDepth + 1).
                    if (tt_here)
                    {
                        const int store_depth = std::max(depth - 3, 0);
                        TTEntry e(key, store_depth, TT_LOWERBOUND, to_tt_score(beta, ply), m);
                        tt_here->store(e);
                    }
                    return beta; // fail-hard cutoff
                }
            }
        }
    }
    // Leaf: ensure mates at depth==0 still get distance (via qsearch’s mate check)
    if (depth <= 0)
    {
        Score val;
        if (use_quiescence)
        {
            val = qsearch(board, ply, alpha, beta, stats, config, control);
        }
        else
        {
            // If not using quiescence, still detect terminal here:
            chess::Movelist leaf_moves;
            chess::movegen::legalmoves(leaf_moves, board);
            if (leaf_moves.empty())
            {
                val = board.inCheck() ? (-MATE_SCORE + ply) : 0.0;
            }
            else
            {
                val = eval_stm_no_game_over(board, config);
                val += corr_hist_probe(board, config);
            }
        }
        // TT leaf stores must respect the window used (qsearch is not always exact under alpha/beta).
        if (tt_here && !stats.stopped)
        {
            TTFlag leaf_flag;
            if (val <= alpha_window)
                leaf_flag = TT_UPPERBOUND;
            else if (val >= beta_window)
                leaf_flag = TT_LOWERBOUND;
            else
                leaf_flag = TT_EXACT;
            tt_here->store(TTEntry(key, depth, leaf_flag, to_tt_score(val, ply)));
        }
        return val;
    }
    // --- Generate legal moves ---
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    // No legal moves: checkmate or stalemate
    if (moves.empty())
    {
        if (board.inCheck())
        {
            return -MATE_SCORE + ply; // losing mate, distance-aware
        }
        else
        {
            // Stalemate (draw) – apply contempt and only mark draw at root
            if (ply == 0)
                stats.is_draw = true;
            return draw_score_stm(board, config);
        }
    }
    // --- One-reply extension ---
    // If the side to move has only one legal move, the line is forcing.
    // Extend by +1 ply within a conservative depth/ply band.
    if (ply > 0 &&
        depth >= ONE_REPLY_EXT_MIN_DEPTH && depth <= ONE_REPLY_EXT_MAX_DEPTH &&
        ply <= ONE_REPLY_EXT_MAX_PLY &&
        moves.size() == 1)
    {
        depth += 1;
    }
    // --- Internal Iterative Deepening (IID) ---
    // If we have no TT move at a PV node, run a reduced-depth search to seed a TT best-move
    // for move ordering (helps stabilize PV and reduces late PV changes).
    if (allow_iid && config.use_iid && tt_here && !have_tt_best && depth >= 6 && !is_null_window)
    {
        const int iid_depth = (depth > 2 ? depth - 2 : 1);
        (void)negamax(board, iid_depth, ply, alpha, beta, pv, stats, config, use_quiescence,
                      /*allow_iid=*/false, tt, control);
        if (stats.stopped)
            return 0;
        const auto iid_tte_opt = tt_here->probe(key);
        if (iid_tte_opt)
        {
            const TTEntry &iid_tte = *iid_tte_opt;
            if (iid_tte.hasMove)
            {
                have_tt_best = true;
                tt_best_move = iid_tte.bestMove;
            }
        }
    }
    // --- Move ordering: MovePicker staged ordering ---
    const chess::Move *tt_move_ptr = have_tt_best ? &tt_best_move : nullptr;
    MovePicker picker(board, moves, tt_move_ptr, killer_moves[0][ply], killer_moves[1][ply], ply, depth, config);
    // "Improving" signal for LMR: compare static eval with same-side-to-move eval two plies earlier.
    // We compute this once per node and store it for (ply-2) lookups.
    Score node_static_eval = node_corr_eval;
    if (ply >= 0 && ply <= MAX_PLY)
        static_eval_stack[ply] = node_static_eval;
    const bool improving = (ply >= 2 && ply <= MAX_PLY && node_static_eval >= static_eval_stack[ply - 2]);

    // Child-node futility pruning (reverse-futility) ---
    // If the corrected static eval is already comfortably above beta, we can return early
    // at shallow non-PV nodes. This is a fail-high pruning gate.
    if (config.use_quiescence &&
        is_null_window &&
        !in_check &&
        !excluded_move_active &&
        ply > 0 &&
        depth > 0 && depth < 8 &&
        beta < MATE_BOUND &&
        alpha > -MATE_BOUND &&
        node_static_eval < MATE_BOUND &&
        node_static_eval - child_futility_margin(depth, improving) >= beta)
    {
        return node_static_eval;
    }
    // ss->ttPv is an internal PV hint stored in the per-ply stack and driven by
    // a dedicated TT PV bit (not merely by TT bounds). Our TT implementation does not
    // currently carry an explicit PV flag, so we don't approximate ttPv from
    // TT_EXACT / depth — that tends to over-trigger PV-specific logic and can hurt strength.
    const bool tt_pv_node = pv;
    if (picker.bad_caps.count > 0)
    {
        ++stats.badcap_gen_nodes;
        stats.badcap_generated += static_cast<std::uint64_t>(picker.bad_caps.count);
    }
    Score best_score = -INF;
    chess::Move best_move_local{};
    bool have_best = false;
    bool first_move = true;
    Score static_eval = 0;
    bool have_static_eval = false;
    // Static evaluation for shallow pruning gates (futility / extended futility / MCP).
    // Compute only for null-window nodes and shallow depths where these gates are active.
    if (use_quiescence && is_null_window && depth <= 4 && !in_check)
    {
        static_eval = node_corr_eval;
        have_static_eval = true;
    }
    // Track quiet moves that were actually searched at this node, so on a later cutoff we can
    // apply a small negative update to the other tried quiet moves.
    chess::Move quiet_tried[MAX_QUIET_TRIED];
    int quiet_tried_count = 0;
    constexpr int MAX_CAPTURE_TRIED = 32;
    chess::Move capture_tried[MAX_CAPTURE_TRIED];
    int capture_tried_count = 0;
    // Under staged MovePicker ordering, the overall move index (mi) mixes TT/promos/captures/quiets.
    // For futility/LMP/LMR gates that are intended to apply to *late quiet moves*, use a quiet-only index.
    int quiet_idx = 0;
    int quiet_searched = 0; // quiet moves actually searched (after forward-pruning gates)
    bool badcap_stage_seen = false;
    int movesSearched = 0;
    bool skip_quiet_moves = false; // Once moveCount reaches futility_move_count, skip remaining quiet non-checking moves

    // --- Principal Variation Search loop with LMR ---
    chess::Move move{};
    for (int mi = 0; picker.next(move); ++mi)
    {
        // When running singular-extension verification, exclude the specified move at this node only.
        if (g_excluded_move_ply == ply && g_excluded_move != chess::Move::NO_MOVE && move == g_excluded_move)
            continue;
        const bool is_capture = board.isCapture(move) || (move.typeOf() == chess::Move::ENPASSANT);
        const bool is_promo = (move.typeOf() == chess::Move::PROMOTION);
        const bool is_badcap_stage = (picker.stage == MovePicker::ST_BAD_CAP);
        if (is_badcap_stage)
        {
            ++stats.badcap_picked;
            if (!badcap_stage_seen)
            {
                ++stats.badcap_nodes;
                badcap_stage_seen = true;
            }
        }
        const chess::CheckType check_type = board.givesCheck(move);
        const bool gives_chk = (check_type != chess::CheckType::NO_CHECK);
        // Bad captures are searched last. Do not blanket-skip the entire bad-capture stage in cut nodes,
        // as that can miss tactical resources (especially SEE≈0 lines) and cost strength.
        const bool is_quiet = (!is_capture && !is_promo);
        int quiet_index = -1;
        if (is_quiet)
            quiet_index = quiet_idx++;
        const bool is_tt_move = (have_tt_best && move == tt_best_move);

        const bool critical_pawn_push =
            is_quiet && !gives_chk && is_critical_endgame_pawn_push_search(board, move);
        // --- Selective check extension ---
        // Extend credible checking moves by +1 ply within a conservative depth/ply band.
        // Quiet checks are common and many are not forcing; only extend "credible" quiet checks.
        int move_ext = 0;
        if (gives_chk &&
            depth >= CHECK_EXT_MIN_DEPTH && depth <= CHECK_EXT_MAX_DEPTH &&
            ply <= CHECK_EXT_MAX_PLY)
        {
            if (is_capture)
            {
                // Avoid extending too-losing checking captures (SEE is from the current position).
                if (see_cp(board, move) >= CHECK_EXT_MIN_SEE_CP)
                    move_ext = 1;
            }
            else if (is_promo)
            {
                // Checking promotions are rare and usually forcing.
                move_ext = 1;
            }
            else
            {
                bool credible = false;

                // Discovered checks are typically more forcing than direct checks.
                if (check_type == chess::CheckType::DISCOVERY_CHECK)
                    credible = true;

                // TT move: allow extension (already strongly suggested by the search).
                else if (is_tt_move)
                    credible = true;

                // Early quiet checks (by ordering) are more likely to be relevant.
                else if (quiet_index >= 0 && quiet_index < 2)
                    credible = true;

                // Strong history moves: allow extension.
                else if (config.use_history_heuristic)
                {
                    const int s = stm_index(board);
                    const int f = move.from().index();
                    const int t = move.to().index();
                    if (history_heur[s][f][t] > 900)
                        credible = true;
                }

                if (credible)
                    move_ext = 1;
            }
        }
        // After the futility_move_count threshold is reached, we skip remaining quiet non-checking moves.
        // Important: do NOT skip TT move or checking moves.
        if (skip_quiet_moves && is_quiet && !gives_chk && !is_tt_move && !critical_pawn_push)
            continue;
        // --- Singular extension (conservative) ---
        // If the TT indicates the hash move is clearly best (singular), extend it by +1 ply.
        // Verification: search excluding the TT move at reduced depth with a tight window below the TT score.
        if (move_ext == 0 &&
            is_tt_move &&
            have_tt_entry &&
            !is_null_window &&
            !in_check &&
            ply > 0 &&
            depth >= SINGULAR_EXT_MIN_DEPTH &&
            moves.size() > 1 &&
            tt_entry_flag == TT_EXACT &&
            tt_entry_depth >= depth - 1 &&
            std::abs(tt_entry_value) < MATE_BOUND)
        {
            // If TT score is near alpha, singularity is not informative.
            if (tt_entry_value > alpha + 10)
            {
                constexpr int SINGULAR_VERIFY_REDUCTION = 3; // fixed, conservative
                int vdepth = depth - 1 - SINGULAR_VERIFY_REDUCTION;
                if (vdepth < 1)
                    vdepth = 1;
                // Margin grows mildly with depth; keeps the test robust against TT score noise.
                const Score margin = static_cast<Score>(20 + 5 * depth);
                const Score s_beta = tt_entry_value - margin;
                if (s_beta > alpha)
                {
                    const Score s_alpha = s_beta - ONE_CP;
                    ScopedExcludedMove exclude(tt_best_move, ply);
                    const Score alt = negamax(
                        board,
                        vdepth,
                        ply,
                        s_alpha,
                        s_beta,
                        /*pv=*/false,
                        stats,
                        config,
                        use_quiescence,
                        /*allow_iid=*/false,
                        tt,
                        control);
                    if (stats.stopped)
                        return 0;
                    if (alt < s_beta)
                    {
                        move_ext = 1;
                    }
                }
            }
        }
        // History signal for this quiet move (used by move-count pruning and history-aware LMR).
        // Keep this cheap: only compute for quiet moves.
        int main_hist_score = 0;
        int cont12_score = 0;
        int contx_score = 0; // continuation (ply-3..ply-6), scaled by MULTI_CONT_SCALE_PCT
        // History signal for this quiet move:
        //  - pruning gate uses cont1+cont2 (here: cont12_score)
        //  - LMR adjustment uses cont12 + 68*main/32
        int move_hist_score = 0;
        int lmr_hist_score = 0; // local cont(1..2)+main hist for LMR/pruning
        int main_hist_scaled = 0;
        int cont12_scaled = 0;
        if (is_quiet && (config.use_history_heuristic || config.use_continuation_history))
        {
            const int s = stm_index(board);
            const int f = move.from().index();
            const int t = move.to().index();
            if (config.use_history_heuristic)
                main_hist_score = history_heur[s][f][t];
            if (config.use_continuation_history)
            {
                const int cur_pt = pt_index(board.at(move.from()));
                const int cur_to = t;
                if (cur_pt >= 0 && cur_pt < NUM_ORDER_PT)
                {
                    // In check, Only use the first 2 continuation contexts.
                    // We keep cont12_score as (dist=1..2) for pruning gates; additional contexts (dist=3..6)
                    // feed move ordering / LMR via contx_score, scaled by MULTI_CONT_SCALE_PCT.
                    for (int dist = 1; dist <= 6; ++dist)
                    {
                        if (in_check && dist > 2)
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

                        int v = cont_history_pc[s][prev_pt][prev_to][cur_pt][cur_to];
                        // dist==5 downweight to reduce noise.
                        if (dist == 5)
                            v /= 3;

                        if (dist <= 2)
                        {
                            cont12_score += v;
                        }
                        else if (MULTI_CONT_SCALE_PCT != 0)
                        {
                            // Scale only the additional lookbacks (ply-3..ply-6).
                            v = static_cast<int>((static_cast<std::int64_t>(v) * MULTI_CONT_SCALE_PCT) / 100);
                            contx_score += v;
                        }
                    }
                }
            }
            // Combined history term used for LMR adjustment.
            const int cont_score = cont12_score + contx_score;

            // Apply configurable weighting so continuation doesn't over/under-drive ordering and LMR.
            // Defaults preserve prior behavior:
            //   history_ordering_mult=2 => main part = 2*main_hist_score
            //   continuation_ordering_mult=2 => cont part = 1*cont_score
            main_hist_scaled = config.history_ordering_mult * main_hist_score;
            cont12_scaled = static_cast<int>((static_cast<std::int64_t>(config.continuation_ordering_mult) * cont12_score) / 2);
            const int cont_scaled = static_cast<int>((static_cast<std::int64_t>(config.continuation_ordering_mult) * cont_score) / 2);

            move_hist_score = cont_scaled + main_hist_scaled;
            // LMR/history-adjustment should be driven primarily by the local (ply-1..2) continuation signal.
            // Deeper continuation (ply-3..ply-6) is useful for ordering, but is noisier and can overdrive reductions at shallow depths.
            lmr_hist_score = cont12_scaled + main_hist_scaled;
        }
        // Shallow forward pruning gates
        //  - Move-count / LMP:            futility_move_count(improving, depth)
        //  - Continuation history prune:  cont12 < -6500 * depth
        //  - Extended futility (quiet):   staticEval + (bestMove?64 : 162) + 116*lmrDepth (+ small bonus when bestValue << staticEval)
        //
        // Gates: null-window only, not in check, quiet non-checking moves.
        const int prospectiveMoveCount = movesSearched + 1; // MoveCount, before committing to search

        bool history_prune_bad = false; // soft continuation-history pruning: late quiets with very poor cont12

        if (is_null_window && ply > 0 && !in_check && have_best && board.hasNonPawnMaterial(board.sideToMove()) && is_quiet && !gives_chk)
        {
            // Late-move pruning / move-count pruning threshold
            // Here we skip searching the remaining quiet moves once the threshold is exceeded.
            // Futility_move_count threshold is handled by setting skip_quiet_moves after moveCount is committed
            // Continuation history based pruning (uses cont12 only)
            if (!critical_pawn_push && ENABLE_HISTORY_PRUNING && depth <= 6 && quiet_index >= 5 && cont12_scaled < -6500 * depth)
            {
                // Soft history pruning: instead of forward-pruning the move entirely (which is too binary for a slightly
                // immature eval), apply an extra 1-ply reduction later via LMR.
                history_prune_bad = true;
            }

            // Extended futility pruning for quiet moves at shallow depths (2–4)
            if (!critical_pawn_push && ENABLE_EXT_FUTILITY && have_static_eval && depth >= 2 && depth <= 4 && !first_move)
            {
                // Compute lmrDepth for pruning layer:
                const int new_depth = depth - 1;
                const int delta = static_cast<int>(beta - alpha);
                const int move_count = prospectiveMoveCount;

                const int stat_score = main_hist_scaled + cont12_scaled - 3271;
                const int r_fp = lmr_adjusted_fp(improving, depth, move_count, delta, stat_score, tt_pv_node);

                int lmr_depth = new_depth - (r_fp / 1024);
                {
                    int hist_adj = lmr_hist_score / 3593;
                    // At the practical depths this engine reaches most often (<=8), avoid letting noisy negative history
                    // over-reduce critical quiet defenses.
                    if (depth <= 8)
                        hist_adj = std::max(hist_adj, -1);
                    lmr_depth += hist_adj;
                }
                if (lmr_depth < 0)
                    lmr_depth = 0;

                // base = (bestMove ? 48 : 146)
                const int base = have_best ? 64 : 162; // Kinda tuned
                const int extra = (best_score < static_eval - 128) ? 103 : 0;
                const Score safety = static_cast<Score>((depth <= 3) ? 40 : 0);
                const Score futility_value = static_eval + static_cast<Score>(base + 116 * lmr_depth + extra) + safety;

                // Gate: !inCheck && lmrDepth < 12 && futilityValue <= alpha
                if (lmr_depth < 12 && futility_value <= alpha)
                {
                    // Keep a non-search-derived lower bound.
                    if (best_score < futility_value)
                        best_score = futility_value;
                    continue;
                }
            }
        }
        // Move-count pruning (MCP): prune very late quiet moves in null-window nodes.
        // This is forward pruning; keep it conservative and gated by static eval.
        if (ENABLE_MCP && !ENABLE_LMP &&
            config.use_move_count_pruning &&
            is_null_window &&
            !in_check &&
            depth >= 2 &&
            depth <= MCP_MAX_DEPTH &&
            is_quiet &&
            !gives_chk)
        {
            const bool protect = is_tt_move || (move_hist_score >= MCP_GOOD_HIST);
            if (!protect && quiet_index >= 10)
            {
                // Ensure static eval is available for the gate (compute lazily if not).
                if (!have_static_eval)
                {
                    static_eval = node_corr_eval;
                    have_static_eval = true;
                }
                // Only prune when we are clearly failing low already.
                const Score mcp_margin = 100 + 50 * depth; // centipawns //NEED TO TUNE
                if (static_eval + mcp_margin <= alpha)
                {
                    int limit = MCP_BASE + (MCP_SCALE * depth * depth) / MCP_DIV;
                    // Tighten with ply: prune more deeper in the tree.
                    limit -= std::min(6, ply / 2);
                    if (limit < 6)
                        limit = 6;
                    if (quiet_index > limit)
                        continue;
                }
            }
        }
        // Diagnostics: count quiet moves that survive pruning and are actually searched at high-depth nodes.
        if (depth >= 10 && is_quiet)
            ++stats.quiet_searched_ge10;
        const int moveCount = ++movesSearched;
        if (is_quiet)
            ++quiet_searched;
        // Once moveCount reaches the threshold, skip remaining quiet non-checking moves at this node.
        if (is_null_window && ply > 0 && have_best && board.hasNonPawnMaterial(board.sideToMove()) && !skip_quiet_moves && is_quiet && quiet_searched >= 4 && quiet_searched >= sf_futility_move_count(improving, depth) + 2)
        {
            skip_quiet_moves = true;
        }
        // Track previous move for continuation history (child ply reads move_stack[ply]).
        if (ply + 1 <= MAX_PLY)
        {
            move_stack[ply + 1] = move;
            moved_pt_stack[ply + 1] = pt_index(board.at(move.from()));
            prior_capture_stack[ply + 1] = capture_like_for_stack(board, move);
            parent_move_count_stack[ply + 1] = moveCount;
            tt_hit_stack[ply + 1] = false;
            in_check_stack[ply + 1] = false;
        }
        if (is_badcap_stage)
            ++stats.badcap_searched;
        // --- SEE-based capture pruning (v2, safer): only prune clearly losing captures, and only
        // in the bad-capture stage (after good captures + quiets). This reduces SEE overhead and
        // avoids pruning defensive tactical resources too aggressively (notably for Black).
        if (is_null_window && ply > 0 && !in_check && depth <= 6 &&
            is_capture && !gives_chk && !is_tt_move && is_badcap_stage)
        {
            // Conservative threshold (centipawns): prune only very losing exchanges.
            constexpr int SEE_PRUNE_BASE_CP = 150;
            constexpr int SEE_PRUNE_STEP_CP = 50;
            const int see_thr = -(SEE_PRUNE_BASE_CP + SEE_PRUNE_STEP_CP * depth);
            if (see_cp(board, move) < see_thr)
                continue;
        }
        board.makeMove(move);
        if (ply + 1 <= MAX_PLY)
            in_check_stack[ply + 1] = board.inCheck();
        Score score;
        if (first_move)
        {
            // PVS: first move gets the full window; later moves start with null-window probes.
            score = -negamax(
                board,
                (depth - 1) + move_ext,
                ply + 1,
                -beta,
                -alpha,
                /*pv=*/pv,
                stats,
                config,
                use_quiescence,
                allow_iid,
                tt,
                control);
            first_move = false;
        }
        else
        {
            // --- LMR (fixed-point reduction table) ---
            int reduced_depth = depth - 1;
            bool did_reduce = false;
            // Apply LMR only for sufficiently deep nodes and late moves.
            if (
                depth >= 2 &&
                moveCount > 1 &&
                is_quiet &&
                !gives_chk &&
                !in_check)
            {
                //  Compute:
                //   newDepth = depth - 1
                //   delta    = beta - alpha
                //   r        = reduction(improving, depth, moveCount, delta)
                //   lmrDepth = newDepth - r/1024
                // Then for quiet moves:
                //   history = cont12 + 68*main/32
                //   lmrDepth += history / 3593
                //
                // move_count is the number of moves actually searched at this node.
                const int new_depth = depth - 1;
                const int delta = static_cast<int>(beta - alpha);
                const int move_count = moveCount;
                // r -= ss->statScore * 1582 / 16384;
                // For quiet, non-check moves we approximate statScore as:
                //   2 * mainHistory + cont1 + cont2 - 3271
                // where cont12_score is cont1+cont2 in our implementation (pruning signal); 
                // move_hist_score includes scaled ply-3..ply-6 via MULTI_CONT_SCALE_PCT.
                const int stat_score = main_hist_scaled + cont12_scaled - 3271;
                const int r_fp = lmr_adjusted_fp(improving, depth, move_count, delta, stat_score, tt_pv_node);
                int lmr_depth = new_depth - (r_fp / 1024);
                // Apply history adjustment for quiet moves.
                // move_hist_score is computed as: (cont12 + scaled cont(3..6)) + (68 * main_history) / 32
                {
                    int hist_adj = lmr_hist_score / 3593;
                    // At the practical depths this engine reaches most often (<=8), avoid letting noisy negative history
                    // over-reduce critical quiet defenses.
                    if (depth <= 8)
                        hist_adj = std::max(hist_adj, -1);
                    lmr_depth += hist_adj;
                }
                lmr_depth = std::max(1, lmr_depth);
                if (history_prune_bad)
                    lmr_depth = std::max(1, lmr_depth - 1);
                reduced_depth = std::min(new_depth, lmr_depth);
                did_reduce = (reduced_depth < new_depth);

                if (critical_pawn_push && reduced_depth < new_depth)
                {
                    // Passed-pawn creation/advance in queenless positions is often decisive;
                    // protect these moves from the harshest reductions.
                    reduced_depth = std::min(new_depth, reduced_depth + 1);
                    did_reduce = (reduced_depth < new_depth);
                }
            }
            // 1) Reduced-depth null-window probe (or normal-depth if not reduced)
            score = -negamax(
                board,
                reduced_depth + move_ext,
                ply + 1,
                -alpha - ONE_CP,
                -alpha,
                /*pv=*/false,
                stats,
                config,
                use_quiescence,
                allow_iid,
                tt,
                control);
            // If reduced search looks promising, confirm at full depth (still null-window).
            if (score > alpha && did_reduce)
            {
                if (depth >= 10 && is_quiet)
                    ++stats.quiet_researched_ge10;
                score = -negamax(
                    board,
                    (depth - 1) + move_ext,
                    ply + 1,
                    -alpha - ONE_CP,
                    -alpha,
                    /*pv=*/false,
                    stats,
                    config,
                    use_quiescence,
                    allow_iid,
                    tt,
                    control);
            }
            // If it still beats alpha, run full-window re-search to verify exact value.
            if (score > alpha && score < beta)
            {
                if (depth >= 10 && is_quiet)
                    ++stats.quiet_researched_ge10;
                score = -negamax(
                    board,
                    (depth - 1) + move_ext,
                    ply + 1,
                    -beta,
                    -alpha,
                    /*pv=*/pv,
                    stats,
                    config,
                    use_quiescence,
                    allow_iid,
                    tt,
                    control);
            }
        }
        board.unmakeMove(move);
        if (stats.stopped)
        {
            return 0;
        }
        if (score > best_score)
        {
            best_score = score;
            best_move_local = move;
            have_best = true;
        }
        if (score > alpha)
        {
            alpha = score;
        }
        // Collect searched-but-not-best moves for end-of-node malus updates (max 32 searched moves).
        if (moveCount <= 32 && move != best_move_local)
        {
            const bool capture_stage = capture_stage_for_stats(board, move);
            if (!capture_stage)
            {
                if (quiet_tried_count < MAX_QUIET_TRIED)
                    quiet_tried[quiet_tried_count++] = move;
            }
            else
            {
                if (capture_tried_count < MAX_CAPTURE_TRIED)
                    capture_tried[capture_tried_count++] = move;
            }
        }
        // Fail-hard beta cutoff
        if (alpha >= beta)
        {
            // Killers: store quiet beta-cutoff moves per ply (2 slots), non-root.
            // NOTE: Do NOT exclude quiet checking moves here. In this engine they can be critical defensive resources,
            // especially for the side under pressure, and suppressing their learning tends to hurt Black results.
            if (ply > 0 && is_quiet && !is_tt_move)
            {
                chess::Move &k1 = killer_moves[0][ply];
                chess::Move &k2 = killer_moves[1][ply];
                if (move != k1)
                {
                    k2 = k1;
                    k1 = move;
                }

                // Countermove/refutation: remember the quiet move that refuted the parent move.
                const chess::Move pm = move_stack[ply];
                if (pm != chess::Move::NO_MOVE && !prior_capture_stack[ply])
                {
                    const int stm = stm_index(board);
                    counter_moves[stm][pm.from().index()][pm.to().index()] = move;
                }
            }
            break;
        }
    }
    // --- History updates ---
    // Apply learning once per node, based on the best move and the set of searched-but-not-best moves.
    if (!stats.stopped && have_best && best_score > alpha_window)
    {
        const bool best_is_tt = (have_tt_best && best_move_local == tt_best_move);
        const int bonus = std::min(141 * depth - 89, 1613) + (best_is_tt ? 311 : 0);
        const int malus = std::min(695 * depth - 215, 2808) - 31 * (movesSearched - 1);
        const int main_bonus = (bonus * 1129) / 1024;
        const int main_malus = (malus * 1246) / 1024;
        const int cont_bonus = (main_bonus * 1004) / 1024;
        const int cont_malus = (main_malus * 1004) / 1024;
        const int cap_bonus = (bonus * 1187) / 1024;
        const int cap_malus = (malus * 1377) / 1024;
        static constexpr std::array<int, 7> W = {0, 1103, 659, 323, 533, 121, 474};
        const int s = stm_index(board);
        const chess::Move bm = best_move_local;
        const int bf = bm.from().index();
        const int bt = bm.to().index();
        const bool best_capture_stage = capture_stage_for_stats(board, bm);
        // Quiet-history + continuation-history updates
        if (!best_capture_stage)
        {
            if (config.use_history_heuristic)
                update_main_history_entry(history_heur[s][bf][bt], main_bonus);
            if (config.use_continuation_history)
            {
                const int cur_pt = pt_index(board.at(bm.from()));
                for (int i = 1; i <= 6; ++i)
                {
                    if (in_check && i > 2)
                        break;
                    if (ply < i)
                        break;
                    const int idx2 = ply + 1 - i;
                    const chess::Move &prev = move_stack[idx2];
                    if (prev == chess::Move::NO_MOVE)
                        continue;
                    const int prev_to = prev.to().index();
                    const int prev_pt = moved_pt_stack[idx2];
                    const int d = (cont_bonus * W[i]) / 1024;
                    if (d)
                        update_continuation_no_board(s, prev_pt, prev_to, cur_pt, bf, bt, d);
                }
            }
            // Apply malus to searched-but-not-best quiet moves
            if (config.use_history_heuristic)
            {
                for (int qi = 0; qi < quiet_tried_count; ++qi)
                {
                    const chess::Move &qm = quiet_tried[qi];
                    const int qf = qm.from().index();
                    const int qt = qm.to().index();
                    update_main_history_entry(history_heur[s][qf][qt], -main_malus);
                }
            }
            if (config.use_continuation_history)
            {
                for (int qi = 0; qi < quiet_tried_count; ++qi)
                {
                    const chess::Move &qm = quiet_tried[qi];
                    const int qf = qm.from().index();
                    const int qt = qm.to().index();
                    const int cur_pt = pt_index(board.at(qm.from()));
                    for (int i = 1; i <= 6; ++i)
                    {
                        if (in_check && i > 2)
                            break;
                        if (ply < i)
                            break;
                        const int idx2 = ply + 1 - i;
                        const chess::Move &prev = move_stack[idx2];
                        if (prev == chess::Move::NO_MOVE)
                            continue;
                        const int prev_to = prev.to().index();
                        const int prev_pt = moved_pt_stack[idx2];
                        const int d = -(cont_malus * W[i]) / 1024;
                        if (d)
                            update_continuation_no_board(s, prev_pt, prev_to, cur_pt, qf, qt, d);
                    }
                }
            }
        }
        // Capture-history updates
        if (config.use_capture_history)
        {
            if (best_capture_stage)
            {
                const int ca = pt_index(board.at(bm.from()));
                const int cv = victim_pt_index(board, bm);
                if (ca >= 0 && cv >= 0)
                    update_capture_history_entry(capture_history[s][ca][bt][cv], cap_bonus);
            }
            // Apply malus to searched-but-not-best capture-stage moves
            for (int ci = 0; ci < capture_tried_count; ++ci)
            {
                const chess::Move &cm = capture_tried[ci];
                const int ca = pt_index(board.at(cm.from()));
                const int cv = victim_pt_index(board, cm);
                const int ct = cm.to().index();
                if (ca >= 0 && cv >= 0)
                    update_capture_history_entry(capture_history[s][ca][ct][cv], -cap_malus);
            }
        }
        // Refutation penalty for early quiet parent moves.
        if (config.use_continuation_history && ply > 0 && move_stack[ply] != chess::Move::NO_MOVE && !prior_capture_stack[ply])
        {
            const int parent_mc = parent_move_count_stack[ply];
            const int parent_tth = (tt_hit_stack[ply - 1] ? 1 : 0);
            if (parent_mc == 1 + parent_tth)
            {
                const chess::Move pm = move_stack[ply];
                const int pf = pm.from().index();
                const int pt = pm.to().index();
                const int cur_pt = moved_pt_stack[ply];
                const int s_parent = 1 - s;
                const int base = -(malus * 987) / 1024;
                const bool parent_in_check = in_check_stack[ply - 1];
                for (int i = 1; i <= 6; ++i)
                {
                    if (parent_in_check && i > 2)
                        break;
                    if (ply < i)
                        break;
                    const int idx2 = ply - i;
                    const chess::Move &prev = move_stack[idx2];
                    if (prev == chess::Move::NO_MOVE)
                        continue;
                    const int prev_to = prev.to().index();
                    const int prev_pt = moved_pt_stack[idx2];
                    const int d = (base * W[i]) / 1024;
                    if (d)
                        update_continuation_no_board(s_parent, prev_pt, prev_to, cur_pt, pf, pt, d);
                }
            }
        }
    }
    // --- Correction history update (exact nodes only) ---
    // Update with the residual between searched score and raw static eval (no correction applied),
    // so future pruning gates can use static_eval + correction.
    if (config.use_correction_history && !stats.stopped && !in_check &&
        best_score > alpha_window && best_score < beta_window &&
        best_score < MATE_BOUND && best_score > -MATE_BOUND)
    {
        const Score raw_se = eval_stm_no_game_over(board, config);
        corr_hist_update(board, config, best_score - raw_se);
    }
    // --- Store result in TT (if available) ---
    if (tt_here && !stats.stopped)
    {
        TTFlag flag;
        if (best_score <= alpha_window)
        {
            flag = TT_UPPERBOUND;
        }
        else if (best_score >= beta_window)
        {
            flag = TT_LOWERBOUND;
        }
        else
        {
            flag = TT_EXACT;
        }
        const Score store_score = to_tt_score(best_score, ply);
        if (have_best)
        {
            TTEntry e(key, depth, flag, store_score, best_move_local);
            tt_here->store(e);
        }
        else
        {
            TTEntry e(key, depth, flag, store_score);
            tt_here->store(e);
        }
    }
    return best_score;
}

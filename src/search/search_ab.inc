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
    // Max-ply safety (Stockfish-like): prevent ply-stack overflow in pathological lines.
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
    // Initialize Stockfish-style LMR tables and root aspiration delta (used in LMR reduction formula).
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
            // Stockfish-style early TT cutoff (non-PV / null-window nodes only).
            // Important: do not tighten alpha/beta here; only return if the TT entry already
            // implies the node fails high/low at the current beta.
            //
            // This keeps the searched window stable for downstream pruning gates (razoring, NMP, etc.),
            // matching Stockfish's approach.
            if (!pv && is_null_window_tt && depth > 0)
            {
                const bool ttLower = (e.flag == TT_LOWERBOUND || e.flag == TT_EXACT);
                const bool ttUpper = (e.flag == TT_UPPERBOUND || e.flag == TT_EXACT);
                // Stockfish depth gate: ttDepth > depth - (ttValue <= beta)
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
                        // Step 4 (Stockfish-style): TT fail-high learning hook.
                        // If the TT entry causes an immediate fail-high cutoff at this node (non-PV/null-window),
                        // Stockfish still updates quiet histories for the TT move and applies a continuation
                        // penalty to early quiet parent moves (when applicable).
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
                                        // Stockfish: min(120*depth - 75, 1241)
                                        const int bonus_tt = std::min(120 * depth - 75, 1241);
                                        update_main_history_entry(history_heur[s][f][t], bonus_tt);
                                        // Stockfish: update_continuation_histories(..., bonus * 1004/1024)
                                        if (config.use_continuation_history && ply > 0)
                                        {
                                            const int cur_pt = pt_index(board.at(hm.from()));
                                            const int cont_tt = (bonus_tt * 1004) / 1024;
                                            static constexpr std::array<int, 7> W = {0, 1103, 659, 323, 533, 121, 474};
                                            const bool in_check_tt = board.inCheck();
                                            for (int i = 1; i <= 6; ++i)
                                            {
                                                // Stockfish: Only update the first 2 continuation contexts if in check.
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
                            // Stockfish condition: prevSq != NONE, (ss-1)->moveCount <= 3, and !priorCapture
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
        const Score margin = static_cast<Score>(461 + 315 * depth * depth); // Stockfish 17.1 (cp)
        if (se + margin <= alpha)
        {
            ++stats.razor_attempts;
            // Stockfish shape: return qsearch at the node window if the static-eval gate already fails low.
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
            // Stockfish 17.1 gate: require eval close to beta with depth-scaled margin.
            if (static_eval >= beta &&
                static_eval >= beta - static_cast<Score>(19 * depth) + static_cast<Score>(418))
            {
                const int delta = static_cast<int>(static_eval - beta);
                int R = std::min(delta / 232, 6) + depth / 3 + 5;
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
                    // Stockfish 17.1: return fail-soft at shallow depths or when already in verification.
                    if (g_nmp_min_ply != 0 || depth < 16)
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
    // --- ProbCut (Stockfish 17.1 shape, null-window only) ---
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
        // Guide candidate selection similarly to Stockfish's MovePicker by using a SEE threshold
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
                // Preliminary qsearch verification (Stockfish-style).
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
                    // Store a lower bound at an effective depth similar to Stockfish's (probCutDepth + 1).
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
    if (allow_iid && config.use_iid && tt_here && !have_tt_best && depth >= 6 && pv)
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
    // Stockfish-style "improving" signal for LMR: compare static eval with same-side-to-move eval two plies earlier.
    // We compute this once per node and store it for (ply-2) lookups.
    Score node_static_eval = node_corr_eval;
    if (ply >= 0 && ply <= MAX_PLY)
        static_eval_stack[ply] = node_static_eval;
    const bool improving = (ply >= 2 && ply <= MAX_PLY && node_static_eval >= static_eval_stack[ply - 2]);

    // --- Stockfish 12 Step 8: child-node futility pruning (reverse-futility) ---
    // If the corrected static eval is already comfortably above beta, we can return early
    // at shallow non-PV nodes. This is a fail-high pruning gate (~50 Elo in SF12).
    if (config.use_quiescence &&
        !pv &&
        !in_check &&
        !excluded_move_active &&
        ply > 0 &&
        depth > 0 && depth < 8 &&
        beta < MATE_BOUND &&
        alpha > -MATE_BOUND &&
        node_static_eval < MATE_BOUND &&
        node_static_eval - sf_child_futility_margin(depth, improving) >= beta)
    {
        return node_static_eval;
    }
    // Stockfish's ss->ttPv is an internal PV hint stored in the per-ply stack and driven by
    // a dedicated TT PV bit (not merely by TT bounds). Our TT implementation does not
    // currently carry an explicit PV flag, so do NOT try to approximate ttPv from
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
    // apply a small negative update to the other tried quiet moves (Stockfish-style).
    chess::Move quiet_tried[MAX_QUIET_TRIED];
    int quiet_tried_count = 0;
    constexpr int MAX_CAPTURE_TRIED = 32;
    chess::Move capture_tried[MAX_CAPTURE_TRIED];
    int capture_tried_count = 0;
    // Under staged MovePicker ordering, the overall move index (mi) mixes TT/promos/captures/quiets.
    // For futility/LMP/LMR gates that are intended to apply to *late quiet moves*, use a quiet-only index.
    int quiet_idx = 0;
    bool badcap_stage_seen = false;
    int movesSearched = 0;
    bool skip_quiet_moves = false; // Step 14 (SF): once moveCount reaches futility_move_count, skip remaining quiet non-checking moves

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
        // Step 6D (SF): after the futility_move_count threshold is reached, we skip remaining quiet non-checking moves.
        // Important: do NOT skip TT move or checking moves.
        if (skip_quiet_moves && is_quiet && !gives_chk && !is_tt_move)
            continue;
        // --- Singular extension (conservative) ---
        // If the TT indicates the hash move is clearly best (singular), extend it by +1 ply.
        // Verification: search excluding the TT move at reduced depth with a tight window below the TT score.
        if (move_ext == 0 &&
            is_tt_move &&
            have_tt_entry &&
            pv &&
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
        // Stockfish-style history signal for this quiet move:
        //  - pruning gate uses cont1+cont2 (here: cont12_score)
        //  - LMR adjustment uses cont12 + 68*main/32
        int move_hist_score = 0;
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
                    // In check, Stockfish only uses the first 2 continuation contexts.
                    for (int dist = 1; dist <= 2; ++dist)
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
                        cont12_score += cont_history_pc[s][prev_pt][prev_to][cur_pt][cur_to];
                    }
                }
            }
            // Combined history term used for LMR adjustment (Stockfish: history += 68*main/32).
            move_hist_score = cont12_score + (68 * main_hist_score) / 32;
        }
        // Step 6B/6C/6D: Shallow forward pruning gates (Stockfish 17.1–anchored)
        //  - Move-count / LMP:            futility_move_count(improving, depth)
        //  - Continuation history prune:  cont12 < -4348 * depth
        //  - Extended futility (quiet):   staticEval + (bestMove?48:146) + 116*lmrDepth (+ small bonus when bestValue << staticEval)
        //
        // Gates: null-window only, not in check, quiet non-checking moves.
        const int prospectiveMoveCount = movesSearched + 1; // Stockfish-style moveCount, before committing to search

        if (is_null_window && ply > 0 && !in_check && have_best && board.hasNonPawnMaterial(board.sideToMove()) && is_quiet && !gives_chk)
        {
            // Step 6D: Late-move pruning / move-count pruning threshold (Stockfish futility_move_count)
            // In Stockfish this is implemented via MovePicker::skip_quiet_moves(); here we equivalently
            // skip searching the remaining quiet moves once the threshold is exceeded.
            // Step 6D note: futility_move_count threshold is handled by setting skip_quiet_moves after moveCount is committed

            // Step 6C: Continuation history based pruning (uses cont12 only)
            if (ENABLE_HISTORY_PRUNING && cont12_score < -4348 * depth)
            {
                continue;
            }

            // Step 6B: Extended futility pruning for quiet moves at shallow depths (2–4)
            if (ENABLE_EXT_FUTILITY && have_static_eval && depth >= 2 && depth <= 4 && !first_move)
            {
                // Compute the same lmrDepth proxy used by Stockfish's pruning layer:
                //   lmrDepth = (depth-1) - r/1024 + (cont12 + 68*main/32)/3593
                const int new_depth = depth - 1;
                const int delta = static_cast<int>(beta - alpha);
                const int move_count = prospectiveMoveCount;

                const int stat_score = 2 * main_hist_score + cont12_score - 3271;
                const int r_fp = sf_lmr_adjusted_fp(improving, depth, move_count, delta, stat_score, tt_pv_node);

                int lmr_depth = new_depth - (r_fp / 1024);
                lmr_depth += move_hist_score / 3593;
                if (lmr_depth < 0)
                    lmr_depth = 0;

                // Stockfish: base = (bestMove ? 48 : 146)
                const int base = have_best ? 48 : 146;
                const int extra = (best_score < static_eval - 128) ? 103 : 0;
                const Score futility_value = static_eval + static_cast<Score>(base + 116 * lmr_depth + extra);

                // Stockfish gate: !inCheck && lmrDepth < 12 && futilityValue <= alpha
                if (lmr_depth < 12 && futility_value <= alpha)
                {
                    // Keep a non-search-derived lower bound similar to Stockfish's bestValue bump.
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
        // Step 6D (SF Step 14): once moveCount reaches the threshold, skip remaining quiet non-checking moves at this node.
        if (is_null_window && ply > 0 && have_best && board.hasNonPawnMaterial(board.sideToMove()) && !skip_quiet_moves && moveCount >= sf_futility_move_count(improving, depth))
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
            // --- LMR (Stockfish 17.1 shape; fixed-point reduction table) ---
            int reduced_depth = depth - 1;
            bool did_reduce = false;
            // Step 5.3 (Stockfish-aligned): apply LMR only for sufficiently deep nodes and late moves.
            // Stockfish: if (depth >= 2 && moveCount > 1) { ... }
            if (
                depth >= 2 &&
                moveCount > 1 &&
                is_quiet &&
                !gives_chk &&
                !in_check)
            {
                // Stockfish computes:
                //   newDepth = depth - 1
                //   delta    = beta - alpha
                //   r        = reduction(improving, depth, moveCount, delta)
                //   lmrDepth = newDepth - r/1024
                // Then for quiet moves:
                //   history = cont12 + 68*main/32
                //   lmrDepth += history / 3593
                //
                // move_count is the number of moves actually searched at this node (Stockfish-style moveCount).
                const int new_depth = depth - 1;
                const int delta = static_cast<int>(beta - alpha);
                const int move_count = moveCount;
                // Stockfish: r -= ss->statScore * 1582 / 16384;
                // For quiet, non-check moves we approximate statScore as:
                //   2 * mainHistory + cont1 + cont2 - 3271
                // where cont12_score is cont1+cont2 in our implementation.
                const int stat_score = 2 * main_hist_score + cont12_score - 3271;
                const int r_fp = sf_lmr_adjusted_fp(improving, depth, move_count, delta, stat_score, tt_pv_node);
                int lmr_depth = new_depth - (r_fp / 1024);
                // Apply Stockfish history adjustment for quiet moves.
                // move_hist_score is computed as: cont12_score + (68 * main_hist_score) / 32
                lmr_depth += move_hist_score / 3593;
                lmr_depth = std::max(1, lmr_depth);
                reduced_depth = std::min(new_depth, lmr_depth);
                did_reduce = (reduced_depth < new_depth);
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
        // Stockfish-style: collect searched-but-not-best moves for end-of-node malus updates (max 32 searched moves).
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
    // --- Stockfish-style history updates (Step 4, node-exit) ---
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
        // Quiet-history + continuation-history updates (Stockfish update_quiet_histories analogue)
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
        // Capture-history updates (Stockfish captureHistory analogue)
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
        // Step 4C: Refutation penalty for early quiet parent moves (Stockfish: update_continuation_histories(ss-1, ..., -malus*987/1024)).
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

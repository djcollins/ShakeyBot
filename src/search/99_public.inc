bool find_best_move(chess::Board &board,
                    int depth,
                    const EngineConfig &config,
                    bool use_quiescence,
                    bool allow_iid,
                    TranspositionTable &tt,
                    SearchStats &stats,
                    chess::Move &best_move_out,
                    Score &best_score_out,
                    Score alpha,
                    Score beta,
                    SearchControl *control)
{
    (void)allow_iid;
    stats = SearchStats{};
    stats.depth_requested = depth;
    move_stack[0] = chess::Move(chess::Move::NO_MOVE);
    moved_pt_stack[0] = -1;
    // Root-level draw heuristics (50-move, insufficient material, repetition)
    if (board.isHalfMoveDraw() || board.isInsufficientMaterial() || board.isRepetition(2))
    {
        stats.is_draw = true;
    }
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    stats.root_branching_factor = static_cast<int>(moves.size());
    // Try TT-based ordering at the root as well.
    chess::Move root_tt_move{};
    const chess::Move *root_tt_ptr = nullptr;
    {
        auto root_entry = tt.probe(board.hash());
        if (root_entry && root_entry->hasMove)
        {
            root_tt_move = root_entry->bestMove;
            root_tt_ptr = &root_tt_move;
        }
    }
    if (moves.empty())
    {
        stats.nodes++;
        if (board.inCheck())
        {
            stats.is_mate = true;
            stats.is_draw = false;
            stats.has_best_move = false;
            best_score_out = -1000000.0; // root ply = 0, so -MATE_SCORE + 0
            stats.score = best_score_out;
            return false;
        }
        // stalemate
        stats.is_mate = false;
        stats.is_draw = true;
        stats.has_best_move = false;
        best_score_out = 0.0;
        stats.score = 0;
        return false;
    }
    MovePicker root_picker(board, moves, root_tt_ptr, killer_moves[0][0], killer_moves[1][0], 0, depth, config);
    if (root_picker.bad_caps.count > 0)
    {
        ++stats.badcap_gen_nodes;
        stats.badcap_generated += static_cast<std::uint64_t>(root_picker.bad_caps.count);
    }
    bool badcap_stage_seen = false;
    // Score alpha = -INF; //replaced by windowed search
    // Score beta  =  INF;
    chess::Move best_move_local = moves[0];
    Score best_score_local = -INF;
    const int ply_root = 0;
    bool first = true;
    int rootMovesSearched = 0;
    chess::Move move{};
    for (int mi = 0; root_picker.next(move); ++mi)
    {
        if (hard_stop(control, stats))
        {
            stats.has_best_move = false;
            return false;
        }
        const bool is_capture_root = board.isCapture(move) || (move.typeOf() == chess::Move::ENPASSANT);
        const bool is_promo_root = (move.typeOf() == chess::Move::PROMOTION);
        const bool is_quiet_root = (!is_capture_root && !is_promo_root);
        const bool is_badcap_stage = (root_picker.stage == MovePicker::ST_BAD_CAP);
        if (is_badcap_stage)
        {
            ++stats.badcap_picked;
            if (!badcap_stage_seen)
            {
                ++stats.badcap_nodes;
                badcap_stage_seen = true;
            }
        }
        if (depth >= 10 && is_quiet_root)
            ++stats.quiet_searched_ge10;
        if (ply_root + 1 <= MAX_PLY)
        {
            move_stack[ply_root + 1] = move;
            moved_pt_stack[ply_root + 1] = pt_index(board.at(move.from()));
            prior_capture_stack[ply_root + 1] = capture_like_for_stack(board, move);
            const int moveCount = ++rootMovesSearched;
            parent_move_count_stack[ply_root + 1] = moveCount;
            tt_hit_stack[ply_root + 1] = false;
            in_check_stack[ply_root + 1] = false;
        }
        if (is_badcap_stage)
            ++stats.badcap_searched;
        board.makeMove(move);
        Score score;
        if (first)
        {
            score = -negamax(
                board,
                depth - 1,
                ply_root + 1,
                -beta,
                -alpha,
                stats,
                config,
                use_quiescence,
                /*allow_iid=*/true,
                &tt,
                control);
            first = false;
        }
        else
        {
            // null-window probe first
            score = -negamax(
                board,
                depth - 1,
                ply_root + 1,
                -alpha - ONE_CP,
                -alpha,
                stats,
                config,
                use_quiescence,
                /*allow_iid=*/true,
                &tt,
                control);
            // if it improves alpha, confirm with full window
            if (score > alpha && score < beta)
            {
                if (depth >= 10 && is_quiet_root)
                    ++stats.quiet_researched_ge10;
                score = -negamax(
                    board,
                    depth - 1,
                    ply_root + 1,
                    -beta,
                    -alpha,
                    stats,
                    config,
                    use_quiescence,
                    /*allow_iid=*/true,
                    &tt,
                    control);
            }
        }
        board.unmakeMove(move);
        if (stats.stopped)
        {
            stats.has_best_move = false;
            return false;
        }
        if (score > best_score_local)
        {
            best_score_local = score;
            best_move_local = move;
        }
        if (score > alpha)
        {
            alpha = score;
        }
        // Critical: fail-high cutoff for aspiration efficiency
        if (alpha >= beta)
        {
            break;
        }
    }
    best_move_out = best_move_local;
    best_score_out = best_score_local;
    stats.best_move = best_move_local;
    stats.has_best_move = true;
    stats.score = best_score_local;
    return true;
}
void reset_search_heuristics()
{
    const chess::Move none(chess::Move::NO_MOVE);
    for (int s = 0; s < 2; ++s)
        for (int f = 0; f < 64; ++f)
            for (int t = 0; t < 64; ++t)
                history_heur[s][f][t] = 0;
    for (int s = 0; s < 2; ++s)
        for (int f = 0; f < 64; ++f)
            for (int t = 0; t < 64; ++t)
                counter_moves[s][f][t] = none;
    std::memset(cont_history, 0, sizeof(cont_history));
    std::memset(capture_history, 0, sizeof(capture_history));
    std::memset(cont_history_pc, 0, sizeof(cont_history_pc));
    std::memset(corr_hist, 0, sizeof(corr_hist));
    for (int p = 0; p <= MAX_PLY; ++p)
    {
        move_stack[p] = none;
        moved_pt_stack[p] = -1;
        killer_moves[0][p] = none;
        killer_moves[1][p] = none;
        prior_capture_stack[p] = false;
        parent_move_count_stack[p] = 0;
        tt_hit_stack[p] = false;
        in_check_stack[p] = false;
    }
}

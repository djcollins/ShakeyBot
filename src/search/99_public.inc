
namespace
{
    void initialize_root_moves(chess::Board &board,
                               const EngineConfig &config,
                               TranspositionTable &tt,
                               int depth,
                               std::vector<RootMove> &root_moves)
    {
        root_moves.clear();

        chess::Move root_tt_move{};
        const chess::Move *root_tt_ptr = nullptr;
        if (auto root_entry = tt.probe(board.hash()); root_entry && root_entry->hasMove)
        {
            root_tt_move = root_entry->bestMove;
            root_tt_ptr = &root_tt_move;
        }

        MovePicker root_picker(board, root_tt_ptr, killer_moves[0][0], killer_moves[1][0], 0, depth, config);
        root_moves.reserve(static_cast<std::size_t>(root_picker.total_legal_moves()));

        chess::Move move{};
        while (root_picker.next(move))
        {
            RootMove rm{};
            rm.move = move;
            root_moves.push_back(rm);
        }
    }

    bool is_bad_root_capture(const chess::Board &board,
                             const chess::Move &move,
                             int depth,
                             const EngineConfig &config)
    {
        if (move.typeOf() == chess::Move::PROMOTION)
            return false;

        const bool is_capture = board.isCapture(move) || (move.typeOf() == chess::Move::ENPASSANT);
        if (!is_capture)
            return false;

        if (board.givesCheck(move) != chess::CheckType::NO_CHECK)
            return false;

        int thr = config.good_capture_see_threshold_cp;
        if (thr == 0 && depth <= 2)
            thr = 1;

        return see_cp(board, move) < thr;
    }
} // namespace

bool find_best_move(chess::Board &board,
                    int depth,
                    const EngineConfig &config,
                    bool use_quiescence,
                    bool allow_iid,
                    TranspositionTable &tt,
                    std::vector<RootMove> *root_moves,
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
    clear_search_stack_entry(0);

    std::vector<RootMove> local_root_moves;
    std::vector<RootMove> *root_moves_ptr = root_moves ? root_moves : &local_root_moves;
    if (root_moves_ptr->empty())
        initialize_root_moves(board, config, tt, depth, *root_moves_ptr);

    for (RootMove &rm : *root_moves_ptr)
    {
        rm.was_searched = false;
        rm.subtree_nodes = 0;
    }

    // Root-level draw heuristics (50-move, insufficient material, repetition)
    if (board.isHalfMoveDraw() || board.isInsufficientMaterial() || board.isRepetition(2))
    {
        stats.is_draw = true;
    }
    stats.root_branching_factor = static_cast<int>(root_moves_ptr->size());
    std::uint64_t root_badcap_count = 0;
    for (const RootMove &rm : *root_moves_ptr)
    {
        if (is_bad_root_capture(board, rm.move, depth, config))
            ++root_badcap_count;
    }
    if (root_badcap_count > 0)
    {
        ++stats.badcap_gen_nodes;
        stats.badcap_generated += root_badcap_count;
    }
    if (stats.root_branching_factor == 0)
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
    bool badcap_stage_seen = false;
    // Score alpha = -INF; //replaced by windowed search
    // Score beta  =  INF;
    chess::Move best_move_local = chess::Move::NO_MOVE;
    Score best_score_local = -INF;
    const int ply_root = 0;
    bool first = true;
    int rootMovesSearched = 0;
    for (std::size_t mi = 0; mi < root_moves_ptr->size(); ++mi)
    {
        if (hard_stop(control, stats))
        {
            stats.has_best_move = false;
            return false;
        }
        RootMove &root_move = (*root_moves_ptr)[mi];
        const chess::Move move = root_move.move;
        const bool is_capture_root = board.isCapture(move) || (move.typeOf() == chess::Move::ENPASSANT);
        const bool is_promo_root = (move.typeOf() == chess::Move::PROMOTION);
        const bool is_quiet_root = (!is_capture_root && !is_promo_root);
        const bool is_badcap_stage = is_bad_root_capture(board, move, depth, config);
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
            const int moveCount = ++rootMovesSearched;
            set_search_stack_entry(ply_root + 1, board, move, moveCount);
        }
        if (is_badcap_stage)
            ++stats.badcap_searched;
        const std::uint64_t nodes_before = stats.nodes;
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
                 /*pv=*/true,
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
                 /*pv=*/false,
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
                 /*pv=*/true,
                    stats,
                    config,
                    use_quiescence,
                    /*allow_iid=*/true,
                    &tt,
                control);
            }
        }
        board.unmakeMove(move);
        root_move.was_searched = true;
        root_move.last_score = score;
        root_move.subtree_nodes = stats.nodes - nodes_before;
        if (stats.stopped)
        {
            stats.has_best_move = false;
            return false;
        }
        if (score > best_score_local)
        {
            if (best_move_local != chess::Move::NO_MOVE && move != best_move_local)
                ++stats.best_move_changes;
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
    if (best_move_local == chess::Move::NO_MOVE)
        return false;
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
    std::memset(pawn_history, 0, sizeof(pawn_history));
    std::memset(cont_history_pc, 0, sizeof(cont_history_pc));
    std::memset(corr_hist, 0, sizeof(corr_hist));
    for (int p = 0; p <= MAX_PLY; ++p)
    {
        clear_search_stack_entry(p);
        killer_moves[0][p] = none;
        killer_moves[1][p] = none;
    }
}

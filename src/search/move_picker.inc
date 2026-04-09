// -------------------------------------------------------------------------
// MovePicker-style staged ordering
//
// Stages:
//   1) TT move
//   2) Promotions (including capture-promotions)
//   3) Good captures (SEE >= threshold or any checking capture)
//   4) Killers (quiet)
//   5) Countermove/refutation (quiet)
//   6) Quiets (history + continuation)
//   7) Bad captures
//
// Goal: reduce per-node sorting work and avoid scoring quiet moves unless needed,
// improving nps while keeping ordering quality high.
// -------------------------------------------------------------------------
struct MovePicker
{
    static constexpr int NO_SEE_SCORE = std::numeric_limits<int>::min();

    struct StageList
    {
        std::array<chess::Move, chess::constants::MAX_MOVES> moves{};
        std::array<int, chess::constants::MAX_MOVES> scores{};
        std::array<int, chess::constants::MAX_MOVES> see_scores{};
        int count = 0;
        int idx = 0;
        bool scored = false;
        bool preset = false;
        inline void add(const chess::Move m) noexcept
        {
            if (count < static_cast<int>(moves.size()))
            {
                moves[count++] = m;
                see_scores[count - 1] = NO_SEE_SCORE;
            }
        }
        inline void add_scored(const chess::Move m, const int s) noexcept
        {
            if (count < static_cast<int>(moves.size()))
            {
                moves[count] = m;
                scores[count] = s;
                see_scores[count] = NO_SEE_SCORE;
                ++count;
                preset = true;
            }
        }
        inline void add_scored_with_see(const chess::Move m, const int s, const int see) noexcept
        {
            if (count < static_cast<int>(moves.size()))
            {
                moves[count] = m;
                scores[count] = s;
                see_scores[count] = see;
                ++count;
                preset = true;
            }
        }
        inline bool pick(chess::Move &out,
                         const chess::Board &board,
                         int ply,
                         const EngineConfig &config,
                         int *see_out = nullptr,
                         bool *has_see_out = nullptr) noexcept
        {
            if (idx >= count)
                return false;
            if (!scored)
            {
                if (!preset)
                {
                    for (int i = 0; i < count; ++i)
                    {
                        scores[i] = score_move(board, moves[i], nullptr, ply, config);
                        see_scores[i] = NO_SEE_SCORE;
                    }
                }
                scored = true;
            }
            int best = idx;
            int best_score = scores[idx];
            for (int i = idx + 1; i < count; ++i)
            {
                if (scores[i] > best_score)
                {
                    best = i;
                    best_score = scores[i];
                }
            }
            if (best != idx)
            {
                std::swap(moves[idx], moves[best]);
                std::swap(scores[idx], scores[best]);
                std::swap(see_scores[idx], see_scores[best]);
            }
            if (see_out)
                *see_out = see_scores[idx];
            if (has_see_out)
                *has_see_out = (see_scores[idx] != NO_SEE_SCORE);
            out = moves[idx++];
            return true;
        }
    };
    struct KillerList
    {
        std::array<chess::Move, 2> moves{};
        int count = 0;
        int idx = 0;
        inline void add(const chess::Move m) noexcept
        {
            if (count < static_cast<int>(moves.size()))
                moves[count++] = m;
        }
        inline bool pick(chess::Move &out) noexcept
        {
            if (idx >= count)
                return false;
            out = moves[idx++];
            return true;
        }
    };
    const chess::Board &board;
    const EngineConfig &config;
    int ply;
    int depth_remaining;
    chess::Move tt{};
    chess::Move tt_candidate{};
    bool have_tt_candidate = false;
    bool have_tt = false;
    bool tt_verified = false;
    bool done_tt = false;
    bool capture_stages_built = false;
    bool quiet_promos_built = false;
    bool quiet_stages_built = false;
    int capture_move_count = 0;
    int quiet_move_count = 0;
    StageList promos;
    StageList good_caps;
    KillerList killers;
    chess::Move counter{};
    bool have_counter = false;
    bool done_counter = false;
    StageList bad_caps;
    StageList quiets;
    int current_see = NO_SEE_SCORE;
    bool current_has_see = false;
    enum Stage : int
    {
        ST_TT,
        ST_PROMO,
        ST_GOOD_CAP,
        ST_KILLER,
        ST_COUNTER,
        ST_QUIET,
        ST_BAD_CAP,
        ST_DONE
    };
    Stage stage = ST_TT;
    explicit MovePicker(const chess::Board &b, const chess::Move *tt_move_ptr, chess::Move killer1_, chess::Move killer2_, int ply_, int depth_remaining_, const EngineConfig &cfg)
        : board(b), config(cfg), ply(ply_), depth_remaining(depth_remaining_)
    {
        using namespace chess;
        (void)killer1_;
        (void)killer2_;
        const int stm = stm_index(b);
        if (tt_move_ptr && *tt_move_ptr != Move::NO_MOVE)
        {
            tt_candidate = *tt_move_ptr;
            have_tt_candidate = true;
        }

        // Countermove/refutation is a quiet response keyed by the previous move (from,to) and side-to-move.
        Move cm_candidate = Move::NO_MOVE;
        if (ply > 0)
        {
            const Move pm = move_stack[ply];
            if (pm != Move::NO_MOVE && !prior_capture_stack[ply])
                cm_candidate = counter_moves[stm][pm.from().index()][pm.to().index()];
        }
        bool have_cm = false;
        Move cmm = Move::NO_MOVE;
        counter = cmm;
        have_counter = have_cm;
    }
    inline int piece_mask_for_move(const chess::Move &m) const noexcept
    {
        const chess::Piece p = board.at(m.from());
        if (p == chess::Piece::NONE)
            return 0;
        return 1 << static_cast<int>(p.type());
    }
    inline void verify_tt_move() noexcept
    {
        using namespace chess;
        if (tt_verified)
            return;
        tt_verified = true;
        if (!have_tt_candidate)
            return;

        const int piece_mask = piece_mask_for_move(tt_candidate);
        if (piece_mask == 0)
            return;

        Movelist moves;
        const bool is_cap = board.isCapture(tt_candidate) || (tt_candidate.typeOf() == Move::ENPASSANT);
        if (is_cap)
            movegen::legalmoves<movegen::MoveGenType::CAPTURE>(moves, board, piece_mask);
        else
            movegen::legalmoves<movegen::MoveGenType::QUIET>(moves, board, piece_mask);

        for (int i = 0; i < moves.size(); ++i)
        {
            if (moves[i] == tt_candidate)
            {
                tt = moves[i];
                have_tt = true;
                break;
            }
        }
    }
    inline void build_capture_stages() noexcept
    {
        using namespace chess;
        if (capture_stages_built)
            return;
        capture_stages_built = true;

        Movelist caps;
        movegen::legalmoves<movegen::MoveGenType::CAPTURE>(caps, board);
        capture_move_count = static_cast<int>(caps.size());

        const int stm = stm_index(board);
        for (int i = 0; i < caps.size(); ++i)
        {
            const Move m = caps[i];
            // Before the TT stage is consumed, keep the legal TT move out of later
            // buckets so it can be returned first. After the TT stage has already
            // been tried and no TT move was found, do not suppress the candidate
            // from normal stages or it can be dropped entirely.
            if (have_tt_candidate && m == tt_candidate && (have_tt || !done_tt))
            {
                tt = m;
                have_tt = true;
                continue;
            }
            if (m.typeOf() == Move::PROMOTION)
            {
                promos.add(m);
                continue;
            }

            const bool gives_check = (board.givesCheck(m) != CheckType::NO_CHECK);
            int thr = config.good_capture_see_threshold_cp;
            if (thr == 0 && depth_remaining <= 2)
                thr = 1;

            const int see = see_cp(board, m);
            const bool good = gives_check || (see >= thr);

            int cap_score = 0;
            Piece victim = board.at(m.to());
            if (victim == Piece::NONE && m.typeOf() == Move::ENPASSANT)
            {
                const Color stmC = board.sideToMove();
                const int to_idx = m.to().index();
                const int victim_idx = (stmC == Color::WHITE ? to_idx - 8 : to_idx + 8);
                if (0 <= victim_idx && victim_idx < 64)
                    victim = board.at(Square(victim_idx));
            }
            Piece attacker = board.at(m.from());
            if (victim != Piece::NONE && attacker != Piece::NONE)
            {
                const int v_idx = static_cast<int>(victim.type());
                const int a_idx = static_cast<int>(attacker.type());
                if (0 <= v_idx && v_idx < NUM_ORDER_PT && 0 <= a_idx && a_idx < NUM_ORDER_PT)
                {
                    cap_score += 200'000 + 200 * PIECE_ORDER_VALUE[v_idx] - PIECE_ORDER_VALUE[a_idx];
                    if (config.use_capture_history)
                        cap_score += (config.capture_history_ordering_mult) *
                                     (capture_history[stm][a_idx][m.to().index()][v_idx] >> CAPTURE_HISTORY_SCORE_SHIFT_MAIN);
                }
            }
            cap_score += 16 * see;
            if (!gives_check && config.bad_capture_penalty_cp > 0 && see < thr)
                cap_score -= config.bad_capture_penalty_cp * 16;

            if (good)
                good_caps.add_scored_with_see(m, cap_score, see);
            else
                bad_caps.add_scored_with_see(m, cap_score, see);
        }
    }
    inline void build_quiet_promos() noexcept
    {
        using namespace chess;
        if (quiet_promos_built)
            return;
        quiet_promos_built = true;

        Movelist pawn_quiets;
        movegen::legalmoves<movegen::MoveGenType::QUIET>(pawn_quiets, board, PieceGenType::PAWN);
        for (int i = 0; i < pawn_quiets.size(); ++i)
        {
            const Move m = pawn_quiets[i];
            if (have_tt_candidate && m == tt_candidate && (have_tt || !done_tt))
            {
                tt = m;
                have_tt = true;
                continue;
            }
            if (m.typeOf() == Move::PROMOTION)
                promos.add(m);
        }
    }
    inline void build_quiet_stages() noexcept
    {
        using namespace chess;
        if (quiet_stages_built)
            return;
        quiet_stages_built = true;

        const int stm = stm_index(board);
        const Move killer1 = killer_moves[0][ply];
        const Move killer2 = killer_moves[1][ply];
        bool have_k1 = false;
        bool have_k2 = false;
        Move k1m = Move::NO_MOVE;
        Move k2m = Move::NO_MOVE;

        Move cm_candidate = Move::NO_MOVE;
        if (ply > 0)
        {
            const Move pm = move_stack[ply];
            if (pm != Move::NO_MOVE && !prior_capture_stack[ply])
                cm_candidate = counter_moves[stm][pm.from().index()][pm.to().index()];
        }
        bool have_cm = false;
        Move cmm = Move::NO_MOVE;

        Movelist qs;
        movegen::legalmoves<movegen::MoveGenType::QUIET>(qs, board);
        quiet_move_count = static_cast<int>(qs.size());

        for (int i = 0; i < qs.size(); ++i)
        {
            const Move m = qs[i];
            if (have_tt_candidate && m == tt_candidate && (have_tt || !done_tt))
            {
                tt = m;
                have_tt = true;
                continue;
            }
            if (m.typeOf() == Move::PROMOTION)
                continue;
            if (killer1 != Move::NO_MOVE && m == killer1)
            {
                have_k1 = true;
                k1m = m;
                continue;
            }
            if (killer2 != Move::NO_MOVE && m == killer2)
            {
                have_k2 = true;
                k2m = m;
                continue;
            }
            if (cm_candidate != Move::NO_MOVE && m == cm_candidate)
            {
                have_cm = true;
                cmm = m;
                continue;
            }
            quiets.add(m);
        }

        if (have_k1)
            killers.add(k1m);
        if (have_k2 && (!have_k1 || k2m != k1m))
            killers.add(k2m);
        if (have_cm && cmm != Move::NO_MOVE)
        {
            counter = cmm;
            have_counter = true;
        }
    }
    inline bool capture_stages_ready() const noexcept
    {
        return capture_stages_built;
    }
    inline bool has_any_legal_moves() noexcept
    {
        build_capture_stages();
        if (capture_move_count > 0)
            return true;
        build_quiet_stages();
        return quiet_move_count > 0;
    }
    inline bool has_multiple_legal_moves() noexcept
    {
        build_capture_stages();
        if (capture_move_count > 1)
            return true;
        build_quiet_stages();
        return (capture_move_count + quiet_move_count) > 1;
    }
    inline int total_legal_moves() noexcept
    {
        build_capture_stages();
        build_quiet_stages();
        return capture_move_count + quiet_move_count;
    }
    inline bool next(chess::Move &out) noexcept
    {
        current_see = NO_SEE_SCORE;
        current_has_see = false;
        for (;;)
        {
            switch (stage)
            {
            case ST_TT:
                if (!done_tt)
                {
                    verify_tt_move();
                    done_tt = true;
                    if (have_tt)
                    {
                        out = tt;
                        return true;
                    }
                }
                stage = ST_PROMO;
                break;
            case ST_PROMO:
                build_capture_stages();
                build_quiet_promos();
                if (promos.pick(out, board, ply, config, &current_see, &current_has_see))
                    return true;
                stage = ST_GOOD_CAP;
                break;
            case ST_GOOD_CAP:
                build_capture_stages();
                if (good_caps.pick(out, board, ply, config, &current_see, &current_has_see))
                    return true;
                stage = ST_KILLER;
                break;
            case ST_KILLER:
                build_quiet_stages();
                if (killers.pick(out))
                    return true;
                stage = ST_COUNTER;
                break;
            case ST_COUNTER:
                build_quiet_stages();
                if (have_counter && !done_counter)
                {
                    done_counter = true;
                    out = counter;
                    return true;
                }
                stage = ST_QUIET;
                break;
            case ST_QUIET:
                build_quiet_stages();
                if (quiets.pick(out, board, ply, config, &current_see, &current_has_see))
                    return true;
                stage = ST_BAD_CAP;
                break;
            case ST_BAD_CAP:
                build_capture_stages();
                if (bad_caps.pick(out, board, ply, config, &current_see, &current_has_see))
                    return true;
                stage = ST_DONE;
                break;
            default:
                return false;
            }
        }
    }
};

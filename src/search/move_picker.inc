// -------------------------------------------------------------------------
// MovePicker-style staged ordering (Stockfish-inspired)
//
// Stages:
//   1) TT move
//   2) Promotions (including capture-promotions)
//   3) Good captures (SEE >= 0 or any checking capture)
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
    struct StageList
    {
        std::array<chess::Move, chess::constants::MAX_MOVES> moves{};
        std::array<int, chess::constants::MAX_MOVES> scores{};
        int count = 0;
        int idx = 0;
        bool scored = false;
        inline void add(const chess::Move m) noexcept
        {
            if (count < static_cast<int>(moves.size()))
                moves[count++] = m;
        }
        inline bool pick(chess::Move &out,
                         const chess::Board &board,
                         int ply,
                         const EngineConfig &config) noexcept
        {
            if (idx >= count)
                return false;
            if (!scored)
            {
                for (int i = 0; i < count; ++i)
                    scores[i] = score_move(board, moves[i], nullptr, ply, config);
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
            }
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
    bool have_tt = false;
    bool done_tt = false;
    StageList promos;
    StageList good_caps;
    KillerList killers;
    chess::Move counter{};
    bool have_counter = false;
    bool done_counter = false;
    StageList bad_caps;
    StageList quiets;
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
    explicit MovePicker(const chess::Board &b, const chess::Movelist &ml, const chess::Move *tt_move_ptr, chess::Move killer1_, chess::Move killer2_, int ply_, int depth_remaining_, const EngineConfig &cfg)
        : board(b), config(cfg), ply(ply_), depth_remaining(depth_remaining_)
    {
        using namespace chess;
        const int stm = stm_index(b);
        const Move killer1 = killer1_;
        const Move killer2 = killer2_;
        bool have_k1 = false;
        bool have_k2 = false;
        Move k1m = Move::NO_MOVE;
        Move k2m = Move::NO_MOVE;

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

        for (int i = 0; i < ml.size(); ++i)
        {
            const Move m = ml[i];
            if (tt_move_ptr && m == *tt_move_ptr)
            {
                tt = m;
                have_tt = true;
                continue;
            }
            const bool is_promo = (m.typeOf() == Move::PROMOTION);
            const bool is_cap = b.isCapture(m) || (m.typeOf() == Move::ENPASSANT);
            // Promotions first (including capture promotions)
            if (is_promo)
            {
                promos.add(m);
                continue;
            }
            if (is_cap)
            {
                // Split captures into good/bad using Stockfish-shaped dynamic SEE threshold:
                //   good if SEE >= -value/18, where value = 7*VictimValue + CaptureHistory.
                // This keeps a small class of slightly-negative captures in the "good" bucket.
                const int see = see_cp(b, m);
                Piece victim = b.at(m.to());
                if (victim == Piece::NONE && m.typeOf() == Move::ENPASSANT)
                {
                    const Color stmC = b.sideToMove();
                    const int to_idx = m.to().index();
                    const int victim_idx = (stmC == Color::WHITE ? to_idx - 8 : to_idx + 8);
                    if (0 <= victim_idx && victim_idx < 64)
                        victim = b.at(Square(victim_idx));
                }
                Piece attacker = b.at(m.from());
                int good_thr = 0;
                if (victim != Piece::NONE && attacker != Piece::NONE)
                {
                    const int v_idx = static_cast<int>(victim.type());
                    const int a_idx = static_cast<int>(attacker.type());
                    if (0 <= v_idx && v_idx < NUM_ORDER_PT && 0 <= a_idx && a_idx < NUM_ORDER_PT)
                    {
                        int ch = 0;
                        if (config.use_capture_history)
                            ch = (config.capture_history_ordering_mult) *
                                 (capture_history[stm][a_idx][m.to().index()][v_idx] >> CAPTURE_HISTORY_SCORE_SHIFT_MAIN);
                        const int value = 7 * PIECE_ORDER_VALUE[v_idx] + ch;
                        good_thr = -value / 18;
                    }
                }
                if (see >= good_thr)
                    good_caps.add(m);
                else
                    bad_caps.add(m);
            }
            else
            {
                // Killers are quiet beta-cutoff moves tried before the rest of the quiets.
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
                // Countermove/refutation stage (quiet response to parent move)
                if (cm_candidate != Move::NO_MOVE && m == cm_candidate)
                {
                    have_cm = true;
                    cmm = m;
                    continue;
                }
                quiets.add(m);
            }
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
    inline bool next(chess::Move &out) noexcept
    {
        for (;;)
        {
            switch (stage)
            {
            case ST_TT:
                if (have_tt && !done_tt)
                {
                    done_tt = true;
                    out = tt;
                    return true;
                }
                stage = ST_PROMO;
                break;
            case ST_PROMO:
                if (promos.pick(out, board, ply, config))
                    return true;
                stage = ST_GOOD_CAP;
                break;
            case ST_GOOD_CAP:
                if (good_caps.pick(out, board, ply, config))
                    return true;
                stage = ST_KILLER;
                break;
            case ST_KILLER:
                if (killers.pick(out))
                    return true;
                stage = ST_COUNTER;
                break;
            case ST_COUNTER:
                if (have_counter && !done_counter)
                {
                    done_counter = true;
                    out = counter;
                    return true;
                }
                stage = ST_QUIET;
                break;
            case ST_QUIET:
                if (quiets.pick(out, board, ply, config))
                    return true;
                stage = ST_BAD_CAP;
                break;
            case ST_BAD_CAP:
                if (bad_caps.pick(out, board, ply, config))
                    return true;
                stage = ST_DONE;
                break;
            default:
                return false;
            }
        }
    }
};

#include <chrono>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <thread>
#include <unordered_set>

#include "fast_engine/engine.hpp"

namespace fast_engine
{

    TimeBudget compute_time_budget(const SearchLimits &limits,
                                   chess::Color side_to_move,
                                   const EngineConfig &cfg)
    {
        TimeBudget tb{};
        tb.overhead_ms = std::max(0, cfg.move_overhead_ms);

        // Movetime overrides everything else.
        if (limits.movetime_ms >= 0)
        {
            tb.enabled = true;
            const int available = std::max(0, limits.movetime_ms - tb.overhead_ms);
            tb.hard_ms = available;
            tb.soft_ms = (available * 95) / 100; // 95% soft limit
            return tb;
        }

        const int my_time_raw = (side_to_move == chess::Color::WHITE) ? limits.wtime_ms : limits.btime_ms;
        if (my_time_raw < 0)
            return tb;

        tb.enabled = true;

        const int my_time = std::max(0, my_time_raw);
        const int my_inc_raw = (side_to_move == chess::Color::WHITE) ? limits.winc_ms : limits.binc_ms;
        const int my_inc = std::max(0, my_inc_raw);

        const int available = std::max(0, my_time - tb.overhead_ms);

        // Default moves-to-go if unknown.
        // Bias higher so early game time usage is more conservative.
        const int mtg = (limits.movestogo > 0) ? limits.movestogo : 64;

        // Soft budget: time slice plus a fraction of increment.
        long long soft = 0;
        soft += static_cast<long long>(available) / static_cast<long long>(mtg + 1);
        soft += (static_cast<long long>(my_inc) * 6) / 10; // 0.6 * increment

        // Hard budget: allow extension beyond soft, but cap aggressively.
        long long hard = (soft * 2); // 2.0x

        // Caps to prevent pathological long thinks when a lot of time remains.
        hard = std::min<long long>(hard, available);
        hard = std::min<long long>(hard, my_time / 4); // <= 25% of remaining time
        hard = std::min<long long>(hard, soft * 4);    // <= 4x soft

        if (hard < 0)
            hard = 0;
        if (soft < 0)
            soft = 0;
        if (soft > hard)
            soft = hard;

        tb.soft_ms = static_cast<int>(soft * 1);
        tb.hard_ms = static_cast<int>(hard * 1);
        return tb;
    }

    static bool is_legal_move(const chess::Board &board, chess::Move mv)
    {
        if (mv == chess::Move::NO_MOVE)
            return false;
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, const_cast<chess::Board &>(board));
        for (const auto &m : moves)
        {
            if (m == mv)
                return true;
        }
        return false;
    }

    static std::string build_pv_uci(chess::Board root,
                                    const TranspositionTable &tt,
                                    chess::Move root_best,
                                    int max_len)
    {
        std::ostringstream pv;
        int written = 0;

        if (root_best != chess::Move::NO_MOVE && is_legal_move(root, root_best))
        {
            pv << chess::uci::moveToUci(root_best);
            ++written;
            root.makeMove(root_best);
        }
        else
        {
            return "";
        }

        std::unordered_set<std::uint64_t> seen;
        seen.reserve(static_cast<std::size_t>(max_len + 2));
        seen.insert(root.hash());

        while (written < max_len)
        {
            const auto e = tt.probe(root.hash());
            if (!e.has_value() || !e->hasMove)
                break;
            const chess::Move mv = e->bestMove;
            if (!is_legal_move(root, mv))
                break;

            // repetition guard (cheap): stop if we see the same hash again.
            root.makeMove(mv);
            const std::uint64_t h = root.hash();
            if (!seen.insert(h).second)
                break;

            pv << ' ' << chess::uci::moveToUci(mv);
            ++written;
        }

        return pv.str();
    }

    Engine::Engine()
        : config_(),
          tt_()
    {
        resizeTT_MB(static_cast<std::size_t>(config_.hash_mb));
    }

    Engine::Engine(const EngineConfig &cfg)
        : config_(cfg),
          tt_()
    {
        resizeTT_MB(static_cast<std::size_t>(config_.hash_mb));
    }

    void Engine::setConfig(const EngineConfig &cfg)
    {
        config_ = cfg;
    }

    void Engine::clearTT()
    {
        tt_.clear();
    }

    void Engine::resizeTT(std::size_t maxEntries)
    {
        tt_.resize(maxEntries);
    }
    void Engine::resizeTT_MB(std::size_t mb)
    {
        if (mb < 1)
            mb = 1;
        config_.hash_mb = static_cast<int>(mb);
        tt_.resize(TranspositionTable::entries_for_mb(mb));
    }
    bool Engine::search_position(chess::Board &board,
                                 int depth,
                                 SearchResult &result,
                                 std::atomic<bool> *external_stop,
                                 IterationCallback on_iter)
    {
        const int max_depth = (depth > 0 ? depth : config_.search_depth);

        // Depth-only search, but still allow an external stop request (UCI "stop").
        SearchControl control{};
        control.time_enabled = false;
        // Even in non-timed modes (depth / infinite / ponder), we still want sane
        // per-iteration "info time"/"nps". search_position_impl() measures elapsed
        // time from control.start whenever a control object is provided.
        control.start = std::chrono::steady_clock::now();
        control.external_stop = external_stop;

        return search_position_impl(board,
                                    max_depth,
                                    (external_stop ? &control : nullptr),
                                    result,
                                    on_iter,
                                    /*keep_searching_at_max_depth=*/false);
    }

    bool Engine::search_position_fen(const std::string &fen,
                                     int depth,
                                     SearchResult &result)
    {
        chess::Board b;
        if (!b.setFen(fen))
            return false;
        return search_position(b, depth, result);
    }

    bool Engine::search_position(chess::Board &board,
                                 const SearchLimits &limits,
                                 SearchResult &result,
                                 std::atomic<bool> *external_stop,
                                 IterationCallback on_iter)
    {
        // Depth-limited always wins.
        if (limits.depth > 0)
        {
            return search_position(board, limits.depth, result, external_stop, on_iter);
        }

        // "go infinite" and "go ponder": search until externally stopped.
        // If no external stop is provided, we fall back to a large depth-limited search.
        const bool want_infinite = limits.infinite || limits.ponder;
        if (want_infinite)
        {
            constexpr int ABS_MAX_DEPTH = 128; // matches UCI option max and MAX_PLY guards

            SearchControl control{};
            control.time_enabled = false;
            // Initialize start so that per-iteration info time/nps are relative to
            // the beginning of this ponder/infinite search, not the steady_clock epoch.
            control.start = std::chrono::steady_clock::now();
            control.external_stop = external_stop;

            const bool keep = (external_stop != nullptr);
            return search_position_impl(board,
                                        ABS_MAX_DEPTH,
                                        (external_stop ? &control : nullptr),
                                        result,
                                        on_iter,
                                        /*keep_searching_at_max_depth=*/keep);
        }

        const auto stm = board.sideToMove();
        const TimeBudget tb = compute_time_budget(limits, stm, config_);

        // If no time data, fall back to depth-limited defaults.
        if (!tb.enabled)
        {
            return search_position(board, config_.search_depth, result, external_stop, on_iter);
        }

        SearchControl control{};
        control.time_enabled = true;
        control.start = std::chrono::steady_clock::now();
        control.soft_deadline = control.start + std::chrono::milliseconds(tb.soft_ms);
        control.hard_deadline = control.start + std::chrono::milliseconds(tb.hard_ms);
        control.external_stop = external_stop;

        const int max_depth = (config_.max_depth_timed > 0) ? config_.max_depth_timed : config_.search_depth;
        return search_position_impl(board,
                                    max_depth,
                                    &control,
                                    result,
                                    on_iter,
                                    /*keep_searching_at_max_depth=*/false);
    }

    bool Engine::search_position_impl(chess::Board &board,
                                      int max_depth,
                                      SearchControl *control,
                                      SearchResult &result,
                                      const IterationCallback &on_iter,
                                      bool keep_searching_at_max_depth)
    {
        const bool use_quiescence = config_.use_quiescence;

        // Best info from the *deepest* completed iteration
        chess::Move best_move{};
        Score best_score = 0;
        bool has_best = false;

        // Aggregated stats over all iterations
        SearchStats total_stats{};
        total_stats.root_branching_factor = 0;

        const auto start = (control ? control->start : std::chrono::steady_clock::now());

        auto soft_expired = [&]() -> bool
        {
            if (!control || !control->time_enabled)
                return false;
            return std::chrono::steady_clock::now() >= control->soft_deadline;
        };

        // Stockfish-inspired time-management helpers:
        //  - Track root PV[0] stability across iterations (last_pv0_change_depth)
        //  - Track root best-move wobble within an iteration (stats.best_move_changes)
        const int base_soft_ms = (control && control->time_enabled)
                                     ? int(std::chrono::duration_cast<std::chrono::milliseconds>(control->soft_deadline - start).count())
                                     : 0;
        const int base_hard_ms = (control && control->time_enabled)
                                     ? int(std::chrono::duration_cast<std::chrono::milliseconds>(control->hard_deadline - start).count())
                                     : 0;

        chess::Move prev_best_move_all = chess::Move(chess::Move::NO_MOVE);
        int last_pv0_change_depth = 0;
        double previous_time_reduction = 1.0;

        auto update_soft_deadline = [&](int completed_depth,
                                        const SearchStats &iter_stats,
                                        const chess::Move iter_best_move)
        {
            if (!control || !control->time_enabled)
                return;
            if (base_soft_ms <= 0 || base_hard_ms <= 0)
                return;

            // Track how long the root best move (PV[0]) has stayed the same.
            if (prev_best_move_all != chess::Move::NO_MOVE && iter_best_move != prev_best_move_all)
                last_pv0_change_depth = completed_depth;
            prev_best_move_all = iter_best_move;

            // If the best move is stable for many depths, reduce time; otherwise allow more time.
            const double timeReduction = (last_pv0_change_depth + 4 < completed_depth) ? 1.6857 : 0.9; // changed the +8 <.. to +4 <
            const double reduction = (1.4540 + previous_time_reduction) / (2.1593 * timeReduction);

            // More root best-move flips within this iteration => spend more time.
            double bestMoveInstability = 0.9929 + 1.8519 * (double(iter_stats.best_move_changes)); // 1 thread
            bestMoveInstability = std::clamp(bestMoveInstability, 0.50, 3.00);

            double target_ms = double(base_soft_ms) * reduction * bestMoveInstability;

            // Cap used time in case of a single legal move (viewer experience + avoid wasting time).
            if (iter_stats.root_branching_factor == 1)
                target_ms = std::min(target_ms, 500.0);
            // Keep this as a moderate adjustment around the initial optimum time.
            const double min_ms = std::max(1.0, double(base_soft_ms) * 0.60);
            const double max_ms = std::min(double(base_hard_ms), double(base_soft_ms) * 1.15);
            target_ms = std::clamp(target_ms, min_ms, max_ms);

            control->soft_deadline = start + std::chrono::milliseconds(int(target_ms));
            previous_time_reduction = timeReduction;
        };

        bool have_prev = false;
        Score prev_score = 0;

        // Diagnostics: track how often the root PV first move changes late (depth >= 10).
        chess::Move prev_best_move_ge10 = chess::Move(chess::Move::NO_MOVE);
        int prev_best_depth_ge10 = 0;

        constexpr Score INF = SCORE_INF;
        constexpr Score MATE_BOUND = ::fast_engine::MATE_BOUND;

        int cur_depth = 1;
        for (;;)
        {
            if (!keep_searching_at_max_depth && cur_depth > max_depth)
            {
                break;
            }

            // Respect hard deadline at iteration boundaries.
            if (control && control->time_enabled && std::chrono::steady_clock::now() >= control->hard_deadline)
            {
                break;
            }

            // Respect soft deadline at iteration boundaries (keep last completed depth).
            // Always try to complete a couple of iterations before stopping on soft time.
            if (has_best && soft_expired() && cur_depth > 2)
            {
                break;
            }

            // Respect external stop immediately at iteration boundaries (keep last completed depth).
            if (control && control->external_stop &&
                control->external_stop->load(std::memory_order_relaxed))
            {
                break;
            }

            const int depth_to_search = (cur_depth <= max_depth ? cur_depth : max_depth);

            // Start with full window unless we have a prior score
            Score alpha = -INF;
            Score beta = INF;

            Score window = 50; // centipawns
            if (have_prev && std::abs(prev_score) < MATE_BOUND)
            {
                alpha = prev_score - window;
                beta = prev_score + window;
            }

            chess::Move iter_best_move{};
            Score iter_best_score = 0;

            bool ok = false;
            SearchStats last_stats{};
            bool in_window = false;
            // Widen on fail-low / fail-high
            for (int tries = 0; tries < 5; ++tries)
            {
                SearchStats iter_stats{};
                iter_stats.depth_requested = depth_to_search;

                ok = find_best_move(
                    board,
                    depth_to_search,
                    config_,
                    use_quiescence,
                    true /*Use internal iterative deepening*/,
                    tt_,
                    iter_stats,
                    iter_best_move,
                    iter_best_score,
                    alpha,
                    beta,
                    control);

                if (iter_stats.stopped)
                {
                    ok = false;
                }

                // Aggregate stats for every attempt (including re-searches)
                total_stats.nodes += iter_stats.nodes;
                total_stats.tt_hits += iter_stats.tt_hits;
                total_stats.tt_misses += iter_stats.tt_misses;
                total_stats.quiet_searched_ge10 += iter_stats.quiet_searched_ge10;
                total_stats.quiet_researched_ge10 += iter_stats.quiet_researched_ge10;
                total_stats.badcap_nodes += iter_stats.badcap_nodes;
                total_stats.badcap_picked += iter_stats.badcap_picked;
                total_stats.badcap_searched += iter_stats.badcap_searched;
                total_stats.badcap_gen_nodes += iter_stats.badcap_gen_nodes;
                total_stats.badcap_generated += iter_stats.badcap_generated;

                total_stats.razor_attempts += iter_stats.razor_attempts;
                total_stats.razor_cutoffs += iter_stats.razor_cutoffs;

                total_stats.is_mate = iter_stats.is_mate;
                total_stats.is_draw = iter_stats.is_draw;
                total_stats.depth_reached = std::max(total_stats.depth_reached, iter_stats.depth_reached);

                if (total_stats.root_branching_factor == 0)
                {
                    total_stats.root_branching_factor = iter_stats.root_branching_factor;
                }
                total_stats.depth_requested = depth_to_search;

                last_stats = iter_stats;

                if (!ok)
                    break;

                if (iter_best_score <= alpha)
                {
                    // fail-low: widen downward
                    window *= 2;
                    alpha = (have_prev ? prev_score - window : -INF);
                    beta = (have_prev ? prev_score + window : INF);
                    continue;
                }

                if (iter_best_score >= beta)
                {
                    // fail-high: widen upward
                    window *= 2;
                    alpha = (have_prev ? prev_score - window : -INF);
                    beta = (have_prev ? prev_score + window : INF);
                    continue;
                }

                // inside aspiration window -> accept
                in_window = true;
                break;
            }

            // If we exhausted aspiration retries without landing inside the window,
            // force a full-window re-search to avoid accepting a bound as an exact score.
            if (ok && !in_window)
            {
                SearchStats iter_stats{};
                iter_stats.depth_requested = depth_to_search;

                alpha = -INF;
                beta = INF;

                ok = find_best_move(
                    board,
                    depth_to_search,
                    config_,
                    use_quiescence,
                    true /*Use internal iterative deepening*/,
                    tt_,
                    iter_stats,
                    iter_best_move,
                    iter_best_score,
                    alpha,
                    beta,
                    control);

                if (iter_stats.stopped)
                {
                    ok = false;
                }

                total_stats.nodes += iter_stats.nodes;
                total_stats.tt_hits += iter_stats.tt_hits;
                total_stats.tt_misses += iter_stats.tt_misses;
                total_stats.quiet_searched_ge10 += iter_stats.quiet_searched_ge10;
                total_stats.quiet_researched_ge10 += iter_stats.quiet_researched_ge10;
                total_stats.badcap_nodes += iter_stats.badcap_nodes;
                total_stats.badcap_picked += iter_stats.badcap_picked;
                total_stats.badcap_searched += iter_stats.badcap_searched;
                total_stats.is_mate = iter_stats.is_mate;
                total_stats.is_draw = iter_stats.is_draw;
                total_stats.depth_reached = std::max(total_stats.depth_reached, iter_stats.depth_reached);

                if (total_stats.root_branching_factor == 0)
                {
                    total_stats.root_branching_factor = iter_stats.root_branching_factor;
                }
                total_stats.depth_requested = depth_to_search;

                last_stats = iter_stats;
                in_window = ok;
            }

            if (!ok)
            {
                // Keep best_move/best_score/has_best from the last completed iteration.
                break;
            }

            // Successful search at this depth – remember deepest PV
            // Diagnostics: count late PV (first move) changes between consecutive completed iterations.
            if (depth_to_search >= 10 && prev_best_depth_ge10 >= 10 && last_stats.has_best_move &&
                prev_best_move_ge10 != chess::Move::NO_MOVE && iter_best_move != prev_best_move_ge10)
            {
                ++total_stats.pv_firstmove_changes_ge10;
                total_stats.pv_last_change_depth = depth_to_search;
            }
            if (depth_to_search >= 10 && last_stats.has_best_move)
            {
                prev_best_move_ge10 = iter_best_move;
                prev_best_depth_ge10 = depth_to_search;
            }

            has_best = last_stats.has_best_move;
            best_move = iter_best_move;
            best_score = iter_best_score;

            // Update time-management soft deadline based on PV stability and root best-move wobble.
            if (control && control->time_enabled && last_stats.has_best_move)
                update_soft_deadline(depth_to_search, last_stats, iter_best_move);

            // Stockfish-style per-iteration callback (info lines in the UCI layer).
            if (on_iter)
            {
                const auto now = std::chrono::steady_clock::now();
                const double elapsed = std::chrono::duration<double>(now - start).count();
                const double nps = (elapsed > 0.0) ? (static_cast<double>(total_stats.nodes) / elapsed) : 0.0;

                IterationInfo ii{};
                ii.depth = depth_to_search;
                ii.score = iter_best_score;
                ii.best_move = iter_best_move;
                ii.has_best_move = last_stats.has_best_move;
                ii.nodes = total_stats.nodes;
                ii.time_seconds = elapsed;
                ii.nps = nps;
                ii.tt_hits = total_stats.tt_hits;
                ii.tt_misses = total_stats.tt_misses;
                ii.is_mate = total_stats.is_mate;
                ii.is_draw = total_stats.is_draw;

                // PV extraction is best-effort (TT collisions / illegal moves are filtered).
                ii.pv_uci = build_pv_uci(board, tt_, iter_best_move, /*max_len=*/16);

                on_iter(ii);
            }

            have_prev = true;
            prev_score = iter_best_score;

            // Respect soft deadline after completing this depth.
            if (has_best && soft_expired())
            {
                break;
            }

            // In infinite/ponder modes we keep searching until externally stopped.
            // If we have already reached our maximum supported depth, avoid repeatedly
            // re-searching the exact same depth (which can spam UCI "info" output and
            // create pathological behavior in some GUIs). Instead, idle until a stop
            // is requested.
            if (keep_searching_at_max_depth && cur_depth >= max_depth)
            {
                if (control && control->external_stop)
                {
                    while (!control->external_stop->load(std::memory_order_relaxed))
                    {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                }
                break;
            }
            // Next iteration depth
            if (cur_depth < max_depth)
            {
                ++cur_depth;
            }
            else if (!keep_searching_at_max_depth)
            {
                break;
            }
            // else: stay at max_depth and keep searching until externally stopped.
        }

        const auto end = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(end - start).count();
        const double nps = (elapsed > 0.0 ? total_stats.nodes / elapsed : 0.0);

        // --- Fill SearchResult from aggregated stats + deepest PV ---
        result.score = best_score;
        result.best_move = best_move;
        result.has_best_move = has_best;

        result.nodes = total_stats.nodes;
        result.depth_reached = total_stats.depth_reached;
        result.depth_requested = total_stats.depth_requested;
        result.nps = nps;
        result.time_seconds = elapsed;

        result.tt_hits = total_stats.tt_hits;
        result.tt_misses = total_stats.tt_misses;
        const auto total_tt_q = total_stats.tt_hits + total_stats.tt_misses;
        result.tt_hit_rate = (total_tt_q > 0)
                                 ? (100.0 * static_cast<double>(total_stats.tt_hits) /
                                    static_cast<double>(total_tt_q))
                                 : 0.0;

        result.is_mate = total_stats.is_mate;
        result.is_draw = total_stats.is_draw;

        // Diagnostics
        result.quiet_searched_ge10 = total_stats.quiet_searched_ge10;
        result.quiet_researched_ge10 = total_stats.quiet_researched_ge10;

        result.badcap_nodes = total_stats.badcap_nodes;
        result.badcap_picked = total_stats.badcap_picked;
        result.badcap_searched = total_stats.badcap_searched;

        result.badcap_gen_nodes = total_stats.badcap_gen_nodes;
        result.badcap_generated = total_stats.badcap_generated;
        result.pv_firstmove_changes_ge10 = total_stats.pv_firstmove_changes_ge10;
        result.pv_last_change_depth = total_stats.pv_last_change_depth;
        result.razor_attempts = total_stats.razor_attempts;
        result.razor_cutoffs = total_stats.razor_cutoffs;

        return result.has_best_move;
    }

} // namespace fast_engine

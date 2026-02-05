#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <cctype>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>

#include "chess.hpp"
#include "fast_engine/engine.hpp"
#include "fast_engine/config.hpp"
#include "fast_engine/evaluation.hpp"

using fast_engine::Engine;
using fast_engine::EngineConfig;
using fast_engine::IterationCallback;
using fast_engine::IterationInfo;
using fast_engine::Score;
using fast_engine::SearchResult;

constexpr Score UCI_MATE_SCORE = ::fast_engine::MATE_SCORE;
constexpr Score UCI_MATE_BOUND = ::fast_engine::MATE_BOUND;

static bool parse_bool_option(std::string v)
{
    for (char &c : v)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return v == "1" || v == "true" || v == "yes" || v == "on";
}

// ----------------- Score formatting -----------------

static bool score_to_mate_moves(Score score, int &mate_moves_out)
{
    if (score > UCI_MATE_BOUND)
    {
        const long long dPlies = static_cast<long long>(UCI_MATE_SCORE - score);
        mate_moves_out = static_cast<int>((dPlies + 1) / 2);
        return true;
    }
    if (score < -UCI_MATE_BOUND)
    {
        const long long dPlies = static_cast<long long>(UCI_MATE_SCORE + score);
        mate_moves_out = -static_cast<int>((dPlies + 1) / 2);
        return true;
    }
    return false;
}

static void append_uci_score(std::ostringstream &info, Score score)
{
    int mate_moves = 0;
    if (score_to_mate_moves(score, mate_moves))
    {
        info << " score mate " << mate_moves;
    }
    else
    {
        info << " score cp " << static_cast<int>(score);
    }
}

// ----------------- I/O helpers (thread-safe) -----------------

struct UciIO
{
    std::mutex m;

    void log(const std::string &line)
    {
        std::lock_guard<std::mutex> lk(m);
        std::cerr << line << std::endl;
    }

    void send(const std::string &line)
    {
        std::lock_guard<std::mutex> lk(m);
        std::cout << line << std::endl;
    }
};

// ----------------- Iteration "info" output -----------------

static void print_iteration_info(UciIO &io, const IterationInfo &ii)
{
    std::ostringstream info;
    info << "info depth " << ii.depth;
    append_uci_score(info, ii.score);
    info << " nodes " << ii.nodes;
    info << " time " << static_cast<int>(std::llround(ii.time_seconds * 1000.0));
    info << " nps " << static_cast<long long>(std::llround(ii.nps));
    if (!ii.pv_uci.empty())
        info << " pv " << ii.pv_uci;
    io.send(info.str());
}

// ----------------- Trim helpers -----------------

static std::string ltrim(const std::string &s)
{
    std::size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
        ++i;
    return s.substr(i);
}

static std::string rtrim(const std::string &s)
{
    if (s.empty())
        return s;
    std::size_t i = s.size();
    while (i > 0 && std::isspace(static_cast<unsigned char>(s[i - 1])))
        --i;
    return s.substr(0, i);
}

static std::string trim(const std::string &s)
{
    return rtrim(ltrim(s));
}

// ----------------- UCI option handling -----------------

static void handle_setoption(const std::string &line,
                             EngineConfig &config,
                             std::unique_ptr<Engine> &engine)
{
    std::string rest = line.substr(std::string("setoption").size());
    rest = trim(rest);
    if (rest.rfind("name ", 0) != 0)
        return;

    rest = rest.substr(5); // skip "name "

    std::string name;
    std::string value;

    std::size_t value_pos = rest.find(" value ");
    if (value_pos == std::string::npos)
    {
        name = trim(rest);
        value.clear();
    }
    else
    {
        name = trim(rest.substr(0, value_pos));
        value = trim(rest.substr(value_pos + 7));
    }

    auto to_lower = [](std::string s)
    {
        for (char &c : s)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    };

    if (name == "SearchDepth")
    {
        if (!value.empty())
            config.search_depth = std::stoi(value);
    }
    else if (name == "KingCrowdingScale")
    {
        if (!value.empty())
            config.king_crowding_scale = static_cast<double>(std::stoi(value)) / 100.0;
    }
    else if (name == "MobilityScale")
    {
        if (!value.empty())
            config.mobility_scale = static_cast<double>(std::stoi(value)) / 100.0;
    }
    else if (name == "XRayScale")
    {
        if (!value.empty())
            config.xray_scale = static_cast<double>(std::stoi(value)) / 100.0;
    }
    else if (name == "PSTScale")
    {
        if (!value.empty())
            config.pst_scale = static_cast<double>(std::stoi(value)) / 100.0;
    }
    else if (name == "ThreatTerm")
    {
        if (!value.empty())
            config.threat_term = static_cast<double>(std::stoi(value)) / 100.0;
    }
    else if (name == "UseStockPST")
    {
        if (!value.empty())
        {
            const std::string v = to_lower(value);
            config.use_stock_pst = (v == "true" || v == "1");
        }
    }
    else if (name == "UseQuiescence")
    {
        if (!value.empty())
        {
            const std::string v = to_lower(value);
            config.use_quiescence = (v == "true" || v == "1");
        }
    }

    else if (name == "UseRazoring")
    {
        if (!value.empty())
            config.use_razoring = parse_bool_option(value);
    }
    else if (name == "RazorMarginD2")
    {
        if (!value.empty())
        {
            int v = std::stoi(value);
            v = std::max(0, std::min(1000, v));
            config.razor_margin_d2 = v;
        }
    }
    else if (name == "RazorMarginD3")
    {
        if (!value.empty())
        {
            int v = std::stoi(value);
            v = std::max(0, std::min(2000, v));
            config.razor_margin_d3 = v;
        }
    }

    else if (name == "UseHistoryHeuristic")
    {
        if (!value.empty())
            config.use_history_heuristic = parse_bool_option(value);
    }
    else if (name == "UseCaptureHistory")
    {
        if (!value.empty())
            config.use_capture_history = parse_bool_option(value);
    }
    else if (name == "UseContinuationHistory")
    {
        if (!value.empty())
            config.use_continuation_history = parse_bool_option(value);
    }
    else if (name == "UseProbCut")
    {
        if (!value.empty())
            config.use_probcut = parse_bool_option(value);
    }
    else if (name == "GoodCaptureSEEThreshold")
    {
        if (!value.empty())
        {
            int v = std::stoi(value);
            v = std::max(-500, std::min(500, v));
            config.good_capture_see_threshold_cp = v;
        }
    }
    else if (name == "BadCapturePenalty")
    {
        if (!value.empty())
        {
            int v = std::stoi(value);
            v = std::max(0, std::min(2000, v));
            config.bad_capture_penalty_cp = v;
        }
    }
    else if (name == "KillerBonus1")
    {
        if (!value.empty())
        {
            int v = std::stoi(value);
            v = std::max(0, std::min(200000, v));
            config.killer_bonus_1 = v;
        }
    }
    else if (name == "KillerBonus2")
    {
        if (!value.empty())
        {
            int v = std::stoi(value);
            v = std::max(0, std::min(200000, v));
            config.killer_bonus_2 = v;
        }
    }
    else if (name == "CounterMoveBonus")
    {
        if (!value.empty())
        {
            int v = std::stoi(value);
            v = std::max(0, std::min(100000, v));
            config.counter_move_bonus = v;
        }
    }
    else if (name == "HistoryBonusMult")
    {
        if (!value.empty())
        {
            int v = std::stoi(value);
            v = std::max(0, std::min(8, v));
            config.history_ordering_mult = v;
        }
    }
    else if (name == "ContinuationBonusMult")
    {
        if (!value.empty())
        {
            int v = std::stoi(value);
            v = std::max(0, std::min(8, v));
            config.continuation_ordering_mult = v;
        }
    }
    else if (name == "CaptureHistoryMult")
    {
        if (!value.empty())
        {
            int v = std::stoi(value);
            v = std::max(0, std::min(8, v));
            config.capture_history_ordering_mult = v;
        }
    }
    else if (name == "UseNullMovePruning")
    {
        config.use_null_move_pruning = parse_bool_option(value);
    }
    else if (name == "UseMoveCountPruning")
    {
        config.use_move_count_pruning = parse_bool_option(value);
    }
    else if (name == "UseCorrectionHistory")
    {
        config.use_correction_history = parse_bool_option(value);
    }
    else if (name == "CorrectionHistoryScale")
    {
        if (!value.empty())
        {
            int v = std::stoi(value);
            v = std::max(0, std::min(200, v));
            config.correction_history_scale = static_cast<double>(v) / 100.0;
        }
    }
    else if (name == "Hash")
    {
        if (!value.empty())
        {
            int mb = std::max(1, std::stoi(value));
            config.hash_mb = mb;
            if (engine)
                engine->resizeTT_MB(static_cast<std::size_t>(mb));
        }
    }
    else if (name == "MaxDepthTimed")
    {
        if (!value.empty())
            config.max_depth_timed = std::max(1, std::stoi(value));
    }
    else if (name == "MoveOverhead")
    {
        if (!value.empty())
            config.move_overhead_ms = std::max(0, std::stoi(value));
    }
    else if (name == "Ponder")
    {
        if (!value.empty())
        {
            const std::string v = to_lower(value);
            config.ponder = (v == "true" || v == "1");
        }
    }

    if (engine)
        engine->setConfig(config);
}

// ----------------- Position parsing -----------------

static int handle_position(const std::string &line, chess::Board &board)
{
    std::string rest = line.substr(std::string("position").size());
    rest = trim(rest);

    std::istringstream iss(rest);
    int moves_applied = 0;
    std::string token;
    iss >> token;

    if (token == "startpos")
    {
        board = chess::Board();
    }
    else if (token == "fen")
    {
        std::string f1, f2, f3, f4, f5, f6;
        if (!(iss >> f1 >> f2 >> f3 >> f4 >> f5 >> f6))
            return moves_applied;
        const std::string fen = f1 + " " + f2 + " " + f3 + " " + f4 + " " + f5 + " " + f6;
        board = chess::Board(fen);
    }
    else
    {
        return moves_applied;
    }

    std::string token2;
    if (!(iss >> token2))
        return moves_applied;
    if (token2 != "moves")
        return moves_applied;

    std::string move_str;
    while (iss >> move_str)
    {
        chess::Move m = chess::uci::uciToMove(board, move_str);
        if (m == chess::Move::NO_MOVE)
            break;
        board.makeMove(m);
        ++moves_applied;
    }
    return moves_applied;
}

static chess::Move pick_fallback_legal_move(const chess::Board &board)
{
    chess::Movelist ml;
    chess::movegen::legalmoves(ml, board);
    if (ml.empty())
        return chess::Move::NO_MOVE;
    return ml[0];
}

static chess::Move ensure_legal_or_fallback(const chess::Board &board, chess::Move candidate)
{
    chess::Movelist ml;
    chess::movegen::legalmoves(ml, board);
    if (ml.empty())
        return chess::Move::NO_MOVE;

    for (const auto &m : ml)
        if (m == candidate)
            return candidate;

    return ml[0];
}

// ----------------- Search worker -----------------

enum class StopReason : int
{
    None = 0,
    StopCmd = 1,
    PonderHit = 2,
    Internal = 3
};

struct SearchWorker
{
    std::atomic<bool> stop{false};
    std::atomic<int> stop_reason{static_cast<int>(StopReason::None)};
    std::atomic<bool> suppress_bestmove{false};
    std::thread th;

    // Protected state
    std::mutex state_m;
    bool running = false;
    bool pondering = false;

    // Stored for go ponder / ponderhit flow
    fast_engine::SearchLimits last_go_ponder_limits{};
    bool have_go_ponder_limits = false;

    // Predicted opponent move from last completed normal search (bestmove ponder ...)
    chess::Move last_ponder_move = chess::Move::NO_MOVE;
    bool have_last_ponder_move = false;

    void join_if_running()
    {
        if (th.joinable())
            th.join();
        stop.store(false, std::memory_order_relaxed);
        stop_reason.store(static_cast<int>(StopReason::None), std::memory_order_relaxed);
        suppress_bestmove.store(false, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(state_m);
        running = false;
        pondering = false;
    }

    void request_stop()
    {
        stop.store(true, std::memory_order_relaxed);
    }
};

static chess::Move compute_ponder_move_depth1(Engine &engine, const chess::Board &root, const chess::Move best)
{
    chess::Board tmp = root;
    tmp.makeMove(best);

    SearchResult pr{};
    // Depth 1 is fast and generally enough to produce a plausible reply move.
    engine.search_position(tmp, 1, pr, nullptr);
    if (pr.has_best_move)
        return pr.best_move;
    return chess::Move::NO_MOVE;
}

static fast_engine::SearchLimits parse_go_limits(const std::string &line)
{
    fast_engine::SearchLimits limits{};

    std::string rest = line.substr(std::string("go").size());
    rest = trim(rest);
    std::istringstream iss(rest);
    std::string token;

    while (iss >> token)
    {
        if (token == "depth")
        {
            int d;
            if (iss >> d)
                limits.depth = d;
        }
        else if (token == "movetime")
        {
            int ms;
            if (iss >> ms)
                limits.movetime_ms = ms;
        }
        else if (token == "wtime")
        {
            int ms;
            if (iss >> ms)
                limits.wtime_ms = ms;
        }
        else if (token == "btime")
        {
            int ms;
            if (iss >> ms)
                limits.btime_ms = ms;
        }
        else if (token == "winc")
        {
            int ms;
            if (iss >> ms)
                limits.winc_ms = ms;
        }
        else if (token == "binc")
        {
            int ms;
            if (iss >> ms)
                limits.binc_ms = ms;
        }
        else if (token == "movestogo")
        {
            int mtg;
            if (iss >> mtg)
                limits.movestogo = mtg;
        }
        else if (token == "infinite")
        {
            limits.infinite = true;
        }
        else if (token == "ponder")
        {
            limits.ponder = true;
        }
        // nodes, mate, searchmoves, etc. are ignored for now.
    }

    return limits;
}

static void print_search_output(UciIO &io,
                                const SearchResult &result,
                                const chess::Move best,
                                const chess::Move ponder,
                                const bool include_ponder)
{
    const int depth_reached = result.depth_reached;
    const std::uint64_t nodes = result.nodes;

    const double t_sec = result.time_seconds;
    const std::uint64_t hits = result.tt_hits;
    const std::uint64_t misses = result.tt_misses;
    const std::uint64_t tt_total = hits + misses;

    const std::uint64_t nps = (t_sec > 0.0) ? static_cast<std::uint64_t>(nodes / t_sec) : 0ULL;

    const double tt_hit_rate = (tt_total > 0)
                                   ? (100.0 * static_cast<double>(hits) / static_cast<double>(tt_total))
                                   : 0.0;

    // Detailed stderr log
    {
        std::ostringstream dbg;
        dbg << std::fixed << std::setprecision(2);

        double branch = 0.0;
        if (depth_reached > 0 && nodes > 1)
        {
            branch = std::pow(static_cast<double>(nodes), 1.0 / static_cast<double>(depth_reached));
        }

        dbg << "[GO] depth=" << depth_reached
            << " score=" << result.score
            << " nodes=" << nodes
            << " time=" << t_sec << "s"
            << " nps=" << static_cast<double>(nps)
            << " branch=" << branch
            << " is_mate=" << (result.is_mate ? 1 : 0)
            << " is_draw=" << (result.is_draw ? 1 : 0)
            << " tt_hits=" << hits
            << " tt_misses=" << misses
            << " tt_hit_rate=" << std::setprecision(1) << tt_hit_rate << "%"
            << " q10=" << result.quiet_searched_ge10
            << " q10r=" << result.quiet_researched_ge10
            << " pvchg10=" << result.pv_firstmove_changes_ge10
            << " pvlast=" << result.pv_last_change_depth << "d";

        const double badcap_per_mn = (nodes > 0)
                                         ? (1e6 * static_cast<double>(result.badcap_searched) / static_cast<double>(nodes))
                                         : 0.0;

        const double badcapG_per_mn = (nodes > 0)
                                          ? (1e6 * static_cast<double>(result.badcap_generated) / static_cast<double>(nodes))
                                          : 0.0;

        dbg << " badcapN=" << result.badcap_nodes
            << " badcapP=" << result.badcap_picked
            << " badcapS=" << result.badcap_searched
            << " badcapS_Mn=" << std::setprecision(2) << badcap_per_mn
            << " badcapGN=" << result.badcap_gen_nodes
            << " badcapG=" << result.badcap_generated
            << " badcapG_Mn=" << std::setprecision(2) << badcapG_per_mn
            << " razorAttempts=" << result.razor_attempts
            << " razorCutoffs=" << result.razor_cutoffs;

        io.log(dbg.str());
    }

    std::ostringstream bm;
    bm << "bestmove " << chess::uci::moveToUci(best);
    if (include_ponder && ponder != chess::Move::NO_MOVE)
    {
        bm << " ponder " << chess::uci::moveToUci(ponder);
    }
    io.send(bm.str());
}

static void start_search_async(SearchWorker &w,
                               UciIO &io,
                               Engine &engine,
                               const EngineConfig &config,
                               const chess::Board &board,
                               const fast_engine::SearchLimits &limits,
                               bool pondering_mode,
                               const chess::Move &ponder_move_to_apply,
                               bool apply_ponder_move)
{
    // Stop any current search (should not happen in normal UCI flow, but keep it robust).
    if (w.th.joinable())
    {
        w.stop_reason.store(static_cast<int>(StopReason::Internal), std::memory_order_relaxed);
        w.suppress_bestmove.store(true, std::memory_order_relaxed);
        w.request_stop();
        w.th.join();
    }
    w.stop.store(false, std::memory_order_relaxed);
    w.stop_reason.store(static_cast<int>(StopReason::None), std::memory_order_relaxed);
    w.suppress_bestmove.store(false, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lk(w.state_m);
        w.running = true;
        w.pondering = pondering_mode;
        if (pondering_mode)
        {
            w.last_go_ponder_limits = limits;
            w.have_go_ponder_limits = true;
        }
    }

    chess::Board search_board = board;
    if (apply_ponder_move && ponder_move_to_apply != chess::Move::NO_MOVE)
    {
        // Apply only if legal in this position.
        chess::Move legal = chess::Move::NO_MOVE;
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, search_board);
        for (const auto &m : moves)
        {
            if (m == ponder_move_to_apply)
            {
                legal = m;
                break;
            }
        }
        if (legal != chess::Move::NO_MOVE)
        {
            search_board.makeMove(legal);
        }
    }

    // Log time management summary at go time.
    const auto stm = search_board.sideToMove();
    const fast_engine::TimeBudget tb = fast_engine::compute_time_budget(limits, stm, config);
    {
        std::ostringstream tm;
        tm << "[TM] stm=" << (stm == chess::Color::WHITE ? 'w' : 'b')
           << " depth=" << (limits.depth > 0 ? limits.depth : config.search_depth)
           << " wtime=" << limits.wtime_ms
           << " btime=" << limits.btime_ms
           << " winc=" << limits.winc_ms
           << " binc=" << limits.binc_ms
           << " movestogo=" << limits.movestogo
           << " movetime=" << limits.movetime_ms
           << " soft=" << (tb.enabled ? tb.soft_ms : 0)
           << " hard=" << (tb.enabled ? tb.hard_ms : 0)
           << " overhead=" << tb.overhead_ms
           << " maxDepthTimed=" << config.max_depth_timed
           << " ponderOpt=" << (config.ponder ? 1 : 0)
           << " goPonder=" << (limits.ponder ? 1 : 0)
           << " infinite=" << (limits.infinite ? 1 : 0);
        io.log(tm.str());
    }

    w.th = std::thread([&w, &io, &engine, config, search_board, limits, pondering_mode]() mutable
                       {
        SearchResult result{};

        // Many GUIs do not expect heavy "info" traffic during ponder and can misbehave
        // (or even stop reading stdout) when a game ends on time. Keep ponder output quiet.
        IterationCallback on_iter = [&io](const IterationInfo &ii) {
            print_iteration_info(io, ii);
        };

        const bool ok = engine.search_position(const_cast<chess::Board &>(search_board),
                                               limits,
                                               result,
                                               &w.stop,
                                               on_iter);

        // If we were pondering, we generally do not output bestmove.
        // However, UCI requires a bestmove reply after a "stop" command even in ponder mode.
        if (pondering_mode)
        {
            const int reason = w.stop_reason.load(std::memory_order_relaxed);
            const bool suppress = w.suppress_bestmove.load(std::memory_order_relaxed);
            if (reason == static_cast<int>(StopReason::StopCmd) && !suppress)
            {
                chess::Move best = (ok && result.has_best_move)
                    ? ensure_legal_or_fallback(search_board, result.best_move)
                    : pick_fallback_legal_move(search_board);

                if (best == chess::Move::NO_MOVE)
                {
                    io.send("bestmove 0000");
                    return;
                }

                std::ostringstream bm;
                bm << "bestmove " << chess::uci::moveToUci(best);
                io.send(bm.str());
            }
            return;
        }

        // Internal stops (reconfig / new position) should not emit spurious bestmove.
        if (w.suppress_bestmove.load(std::memory_order_relaxed))
            return;

        chess::Move best = (ok && result.has_best_move)
            ? ensure_legal_or_fallback(search_board, result.best_move)
            : pick_fallback_legal_move(search_board);

        if (best == chess::Move::NO_MOVE)
        {
            io.send("bestmove 0000");
            return;
        }

        chess::Move ponder = chess::Move::NO_MOVE;
        if (config.ponder)
        {
            ponder = compute_ponder_move_depth1(engine, search_board, best);
        }

        // Store ponder move for potential future "go ponder".
        {
            std::lock_guard<std::mutex> lk(w.state_m);
            w.last_ponder_move = ponder;
            w.have_last_ponder_move = (ponder != chess::Move::NO_MOVE);
        }

        print_search_output(io, result, best, ponder, config.ponder); });
}

static void handle_stop(SearchWorker &w, StopReason reason, bool suppress_output)
{
    if (w.th.joinable())
    {
        w.stop_reason.store(static_cast<int>(reason), std::memory_order_relaxed);
        w.suppress_bestmove.store(suppress_output, std::memory_order_relaxed);
        w.request_stop();
        w.th.join();
    }
    w.stop.store(false, std::memory_order_relaxed);
    w.stop_reason.store(static_cast<int>(StopReason::None), std::memory_order_relaxed);
    w.suppress_bestmove.store(false, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(w.state_m);
        w.running = false;
        w.pondering = false;
    }
}

static void handle_ponderhit(SearchWorker &w,
                             UciIO &io,
                             Engine &engine,
                             const EngineConfig &config,
                             chess::Board &board)
{
    // Only meaningful if we are currently pondering.
    bool was_pondering = false;
    fast_engine::SearchLimits saved_limits{};
    chess::Move ponder_move = chess::Move::NO_MOVE;

    {
        std::lock_guard<std::mutex> lk(w.state_m);
        was_pondering = w.pondering && w.th.joinable() && w.have_go_ponder_limits;
        if (was_pondering)
        {
            saved_limits = w.last_go_ponder_limits;
            ponder_move = w.have_last_ponder_move ? w.last_ponder_move : chess::Move::NO_MOVE;
        }
    }

    if (!was_pondering)
        return;
    // Stop the current ponder search (no output).
    handle_stop(w, StopReason::PonderHit, /*suppress_output=*/true);

    // Sync the main board by applying the ponder move if it is legal and not already applied.
    if (ponder_move != chess::Move::NO_MOVE)
    {
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board);
        for (const auto &m : moves)
        {
            if (m == ponder_move)
            {
                board.makeMove(m);
                break;
            }
        }
    }

    // Convert to a normal timed search using the same limits, per your preference.
    saved_limits.ponder = false;
    saved_limits.infinite = false;

    {
        std::lock_guard<std::mutex> lk(w.state_m);
        w.pondering = false;
        w.running = true;
    }

    // Start a normal search that will output bestmove.
    start_search_async(w,
                       io,
                       engine,
                       config,
                       board,
                       saved_limits,
                       /*pondering_mode=*/false,
                       chess::Move::NO_MOVE,
                       /*apply_ponder_move=*/false);
}

int main()
{
    UciIO io;

    EngineConfig config;
    std::unique_ptr<Engine> engine;
    chess::Board board; // startpos
    // Note: ordering heuristics (history/continuation/capture history) should persist within a game.
    // They are reset only on the UCI "ucinewgame" command.

    SearchWorker worker;

    std::string line;
    while (std::getline(std::cin, line))
    {
        line = trim(line);
        if (line.empty())
            continue;

        if (line == "uci")
        {
            auto to_cp = [](double x) -> int
            {
                return static_cast<int>(std::lround(x * 100.0));
            };
            auto as_bool = [](bool b)
            {
                return b ? "true" : "false";
            };

            io.send("id name ShakeyBot 1");
            io.send("id author Daniel Collins");

            io.send("option name SearchDepth type spin default " + std::to_string(config.search_depth) + " min 1 max 20");
            io.send("option name MaxDepthTimed type spin default " + std::to_string(config.max_depth_timed) + " min 1 max 128");
            io.send("option name MoveOverhead type spin default " + std::to_string(config.move_overhead_ms) + " min 0 max 2000");
            io.send("option name Ponder type check default " + std::string(as_bool(config.ponder)));

            io.send("option name KingCrowdingScale type spin default " + std::to_string(to_cp(config.king_crowding_scale)) + " min 0 max 100");
            io.send("option name MobilityScale type spin default " + std::to_string(to_cp(config.mobility_scale)) + " min 0 max 100");
            io.send("option name XRayScale type spin default " + std::to_string(to_cp(config.xray_scale)) + " min 0 max 30");
            io.send("option name PSTScale type spin default " + std::to_string(to_cp(config.pst_scale)) + " min 0 max 150");
            io.send("option name Hash type spin default " + std::to_string(std::lround(config.hash_mb)) + " min 1 max 4096");
            // io.send("option name UseStockPST type check default " + std::string(as_bool(config.use_stock_pst)));
            io.send("option name ThreatTerm type spin default " + std::to_string(to_cp(config.threat_term)) + " min 0 max 300");

            io.send("option name UseQuiescence type check default " + std::string(as_bool(config.use_quiescence)));
            io.send("option name UseRazoring type check default " + std::string(as_bool(config.use_razoring)));
            io.send("option name RazorMarginD2 type spin default " + std::to_string(config.razor_margin_d2) + " min 0 max 1000");
            io.send("option name RazorMarginD3 type spin default " + std::to_string(config.razor_margin_d3) + " min 0 max 2000");

            // Phase-1 toggles (for A/B isolation)
            io.send("option name UseHistoryHeuristic type check default " + std::string(as_bool(config.use_history_heuristic)));
            io.send("option name UseCaptureHistory type check default " + std::string(as_bool(config.use_capture_history)));
            io.send("option name UseContinuationHistory type check default " + std::string(as_bool(config.use_continuation_history)));
            io.send("option name UseProbCut type check default " + std::string(as_bool(config.use_probcut)));
            io.send("option name GoodCaptureSEEThreshold type spin default " + std::to_string(config.good_capture_see_threshold_cp) + " min -100 max 100");
            io.send("option name BadCapturePenalty type spin default " + std::to_string(config.bad_capture_penalty_cp) + " min 0 max 250");
            io.send("option name KillerBonus1 type spin default " + std::to_string(config.killer_bonus_1) + " min 0 max 150000");
            io.send("option name KillerBonus2 type spin default " + std::to_string(config.killer_bonus_2) + " min 0 max 150000");
            io.send("option name CounterMoveBonus type spin default " + std::to_string(config.counter_move_bonus) + " min 0 max 30000");
            io.send("option name HistoryBonusMult type spin default " + std::to_string(config.history_ordering_mult) + " min 0 max 3");
            io.send("option name ContinuationBonusMult type spin default " + std::to_string(config.continuation_ordering_mult) + " min 0 max 3");
            io.send("option name CaptureHistoryMult type spin default " + std::to_string(config.capture_history_ordering_mult) + " min 0 max 3");
            io.send("option name UseNullMovePruning type check default " + std::string(as_bool(config.use_null_move_pruning)));
            io.send("option name UseMoveCountPruning type check default " + std::string(as_bool(config.use_move_count_pruning)));
            io.send("option name UseCorrectionHistory type check default " + std::string(as_bool(config.use_correction_history)));
            io.send("option name CorrectionHistoryScale type spin default " + std::to_string(static_cast<int>(std::lround(config.correction_history_scale * 100.0))) + " min 0 max 200");

            io.send("uciok");
        }
        else if (line == "isready")
        {
            io.send("readyok");
        }
        else if (line.rfind("setoption", 0) == 0)
        {
            // If a search is running, stop it first to avoid reconfiguring mid-search.
            handle_stop(worker, StopReason::Internal, /*suppress_output=*/true);
            handle_setoption(line, config, engine);
        }
        else if (line == "ucinewgame")
        {
            handle_stop(worker, StopReason::Internal, /*suppress_output=*/true);
            board = chess::Board();

            // Per UCI spec, ucinewgame is emitted once for a new game. Reset stateful
            // move-ordering heuristics and clear the TT here (and only here).
            fast_engine::reset_search_heuristics();
            if (engine)
                engine->clearTT();

            // Clear evaluation caches to keep per-game behavior stable.
            fast_engine::clear_eval_cache();
        }
        else if (line.rfind("position", 0) == 0)
        {
            // Position updates typically arrive when not searching; stop if needed.
            handle_stop(worker, StopReason::Internal, /*suppress_output=*/true);
            (void)handle_position(line, board);
        }
        else if (line.rfind("go", 0) == 0)
        {
            if (!engine)
                engine = std::make_unique<Engine>(config);

            fast_engine::SearchLimits limits = parse_go_limits(line);

            // Robustness: some GUIs can still send "go ponder" even when the engine's
            // Ponder option is disabled. Treat that as a normal timed search; otherwise
            // we'd enter an effectively-infinite search and never reply.
            if (limits.ponder && !config.ponder)
            {
                limits.ponder = false;
            }

            // go ponder only if the GUI enabled Ponder and we have something plausible to ponder on.
            if (limits.ponder && config.ponder)
            {
                chess::Move ponder_move = chess::Move::NO_MOVE;
                {
                    std::lock_guard<std::mutex> lk(worker.state_m);
                    if (worker.have_last_ponder_move)
                        ponder_move = worker.last_ponder_move;
                }

                // If we don't have a stored ponder move, we can still attempt a guess:
                // pick the opponent's depth-1 best move and ponder after it.
                if (ponder_move == chess::Move::NO_MOVE)
                {
                    SearchResult pr{};
                    engine->search_position(board, 1, pr, nullptr);
                    if (pr.has_best_move)
                        ponder_move = pr.best_move;
                }

                start_search_async(worker,
                                   io,
                                   *engine,
                                   config,
                                   board,
                                   limits,
                                   /*pondering_mode=*/true,
                                   ponder_move,
                                   /*apply_ponder_move=*/true);
            }
            else
            {
                start_search_async(worker,
                                   io,
                                   *engine,
                                   config,
                                   board,
                                   limits,
                                   /*pondering_mode=*/false,
                                   chess::Move::NO_MOVE,
                                   /*apply_ponder_move=*/false);
            }
        }
        else if (line == "stop")
        {
            // Stop current search; per UCI, a bestmove reply is required even in ponder mode.
            handle_stop(worker, StopReason::StopCmd, /*suppress_output=*/false);
        }
        else if (line == "ponderhit")
        {
            if (engine)
                handle_ponderhit(worker, io, *engine, config, board);
        }
        else if (line == "quit")
        {
            handle_stop(worker, StopReason::Internal, /*suppress_output=*/true);
            break;
        }
    }

    handle_stop(worker, StopReason::Internal, /*suppress_output=*/true);
    return 0;
}

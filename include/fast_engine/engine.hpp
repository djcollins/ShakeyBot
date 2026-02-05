#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

#include "chess.hpp"
#include "fast_engine/config.hpp"
#include "fast_engine/search.hpp"
#include "fast_engine/transposition.hpp"
#include "fast_engine/types.hpp"

namespace fast_engine
{

    // -----------------------------
    // Search output structures
    // -----------------------------

    struct SearchResult
    {
        // Final evaluation (side-to-move POV) and best move
        Score score = 0;
        chess::Move best_move{};
        bool has_best_move = false;

        // Time and speed
        double time_seconds = 0.0;
        double nps = 0.0;

        // Aggregated search stats (over all iterations / re-searches)
        std::uint64_t nodes = 0;
        int depth_requested = 0;
        int depth_reached = 0;

        std::uint64_t tt_hits = 0;
        std::uint64_t tt_misses = 0;
        double tt_hit_rate = 0.0;

        // Diagnostics (optional):
        // Count of quiet moves that were actually searched at nodes with remaining depth >= 10.
        std::uint64_t quiet_searched_ge10 = 0;
        // Count of additional searches (PVS/LMR re-searches) for quiet moves at nodes with remaining depth >= 10.
        std::uint64_t quiet_researched_ge10 = 0;
        // How often the root PV first move changes between consecutive completed iterations at depth >= 10.
        std::uint64_t pv_firstmove_changes_ge10 = 0;
        // Depth of the last such PV change (0 if none).
        int pv_last_change_depth = 0;

        // MovePicker diagnostics
        std::uint64_t badcap_nodes = 0;
        std::uint64_t badcap_picked = 0;
        std::uint64_t badcap_searched = 0;

        std::uint64_t badcap_gen_nodes = 0;
        std::uint64_t badcap_generated = 0;

        std::uint64_t razor_attempts = 0;
        std::uint64_t razor_cutoffs = 0;

        bool is_mate = false;
        bool is_draw = false;
    };

    // Iteration-by-iteration reporting (for UCI "info" output).
    struct IterationInfo
    {
        int depth = 0;

        Score score = 0;
        chess::Move best_move{};
        bool has_best_move = false;

        // Cumulative nodes/time up to this completed iteration.
        std::uint64_t nodes = 0;
        double time_seconds = 0.0;
        double nps = 0.0;

        std::uint64_t tt_hits = 0;
        std::uint64_t tt_misses = 0;

        bool is_mate = false;
        bool is_draw = false;

        // Principal variation in UCI move strings (space-separated). May be empty.
        std::string pv_uci;
    };

    using IterationCallback = std::function<void(const IterationInfo &)>;

    // -----------------------------
    // UCI time control structures
    // -----------------------------

    // UCI "go" limits / constraints. Time values are milliseconds.
    struct SearchLimits
    {
        int depth = 0;        // "go depth N"
        int movetime_ms = -1; // "go movetime N"

        int wtime_ms = -1;  // "go wtime"
        int btime_ms = -1;  // "go btime"
        int winc_ms = 0;    // "go winc"
        int binc_ms = 0;    // "go binc"
        int movestogo = -1; // "go movestogo"

        bool infinite = false; // "go infinite"
        bool ponder = false;   // "go ponder"
    };

    // Computed time budget for the current move.
    struct TimeBudget
    {
        bool enabled = false;
        int soft_ms = 0;     // preferred stop time
        int hard_ms = 0;     // absolute cutoff
        int overhead_ms = 0; // safety margin
    };

    TimeBudget compute_time_budget(const SearchLimits &limits,
                                   chess::Color side_to_move,
                                   const EngineConfig &cfg);

    // -----------------------------
    // Engine
    // -----------------------------

    class Engine
    {
    public:
        Engine();
        explicit Engine(const EngineConfig &cfg);

        void setConfig(const EngineConfig &cfg);
        EngineConfig &config() { return config_; }
        const EngineConfig &config() const { return config_; }

        void clearTT();
        void resizeTT(std::size_t maxEntries);
        void resizeTT_MB(std::size_t mb);

        // Depth-limited search.
        bool search_position(chess::Board &board,
                             int depth,
                             SearchResult &result,
                             std::atomic<bool> *external_stop = nullptr,
                             IterationCallback on_iter = {});

        // Time-managed (UCI limits) search.
        bool search_position(chess::Board &board,
                             const SearchLimits &limits,
                             SearchResult &result,
                             std::atomic<bool> *external_stop = nullptr,
                             IterationCallback on_iter = {});

        bool search_position_fen(const std::string &fen,
                                 int depth,
                                 SearchResult &result);

    private:
        bool search_position_impl(chess::Board &board,
                                  int max_depth,
                                  SearchControl *control,
                                  SearchResult &result,
                                  const IterationCallback &on_iter,
                                  bool keep_searching_at_max_depth);

        EngineConfig config_;
        TranspositionTable tt_;
    };

} // namespace fast_engine

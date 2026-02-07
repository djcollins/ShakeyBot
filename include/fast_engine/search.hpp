#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <vector>
#include "chess.hpp"
#include "fast_engine/config.hpp"
#include "fast_engine/types.hpp"

namespace fast_engine
{

    constexpr Score SEARCH_INF = SCORE_INF;

    class TranspositionTable;

    struct SearchStats
    {
        std::uint64_t nodes = 0;
        int depth_requested = 0;
        int depth_reached = 0;

        std::uint64_t tt_hits = 0;
        std::uint64_t tt_misses = 0;

        int root_branching_factor = 0;

        Score score = 0;
        chess::Move best_move{};
        bool has_best_move = false;

        double time_seconds = 0.0; // search time
        double nps = 0.0;          // nodes per second (nodes/time_seconds:.3f)

        bool is_mate = false; // root position is checkmated
        bool is_draw = false; // root position is a draw (50-move, stalemate, etc.)

        // Set when time or an external stop request interrupted the search.
        // The engine should ignore results from an incomplete iteration.
        bool stopped = false;

        // Diagnostics (MovePicker / ordering)
        std::uint64_t quiet_searched_ge10 = 0;   // quiet moves that actually got searched at nodes depth>=10
        std::uint64_t quiet_researched_ge10 = 0; // additional searches (PVS/LMR re-search) for quiets at nodes depth>=10
        std::uint64_t pv_firstmove_changes_ge10 = 0;
        int pv_last_change_depth = 0;

        // Root best-move wobble counter: number of times the current best move changed
        // while searching root moves in a *single* completed iteration.
        std::uint64_t best_move_changes = 0;

        // MovePicker diagnostics (to correlate slow positions with ordering pathologies)
        std::uint64_t badcap_nodes = 0;    // nodes where MovePicker reached ST_BAD_CAP
        std::uint64_t badcap_picked = 0;   // moves returned from ST_BAD_CAP (may be pruned before search)
        std::uint64_t badcap_searched = 0; // ST_BAD_CAP moves actually searched (after pruning gates)

        std::uint64_t badcap_gen_nodes = 0; // nodes where at least one capture had SEE<0 (bad_caps list non-empty)
        std::uint64_t badcap_generated = 0; // total number of SEE<0 captures generated (sum of bad_caps sizes)

        // Razoring diagnostics
        std::uint64_t razor_attempts = 0; // nodes where razoring was evaluated (preconditions passed)
        std::uint64_t razor_cutoffs = 0;  // nodes where razoring triggered a cutoff
    };

    // Runtime stop / time control shared across the search stack.
    // Phase 2: single-threaded time enforcement. (External stop is added in Phase 4.)
    struct SearchControl
    {
        bool time_enabled = false;
        std::chrono::steady_clock::time_point start{};
        std::chrono::steady_clock::time_point soft_deadline{}; // preferred stop (checked at iteration boundaries)
        std::chrono::steady_clock::time_point hard_deadline{}; // absolute stop (checked inside the tree)

        // Optional external stop flag (e.g., UCI "stop"). nullptr in Phase 2.
        std::atomic<bool> *external_stop = nullptr;
    };

    // Root move persistence (Stockfish/Donna-style): keep root moves across iterations
    // and reorder by the previous iteration's score.
    //
    // If Engine passes a non-null root move list into find_best_move(), the root search
    // will iterate moves in that provided order and will update RootMove::last_score
    // with the score obtained at the searched depth.
    struct RootMove
    {
        chess::Move move{};
        Score last_score = 0;
    };

    /**
     * Quiescence search.
     *
     * If not in check: stand-pat, then tactical moves (captures/promotions).
     * If in check: search legal evasions (no stand-pat).
     */
    Score qsearch(chess::Board &board,
                  int ply,
                  Score alpha,
                  Score beta,
                  SearchStats &stats,
                  const EngineConfig &config,
                  SearchControl *control = nullptr);

    /**
     * Negamax alpha-beta.
     *
     * depth: remaining depth in plies
     * ply:   distance from root (0 at root)
     */
    Score negamax(
        chess::Board &board,
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
        SearchControl *control = nullptr);

    /**
     * Root search: find the best move for the current side to move.
     *
     * Returns false if there are no legal moves (checkmate/stalemate),
     * true otherwise. best_move_out / best_score_out are only valid when
     * the function returns true.
     */
    bool find_best_move(chess::Board &board,
                        int depth,
                        const EngineConfig &config,
                        bool use_quiescence,
                        bool allow_iid,
                        TranspositionTable &tt,
                        SearchStats &stats,
                        chess::Move &best_move_out,
                        Score &best_score_out,
                        Score alpha = -SEARCH_INF,
                        Score beta = SEARCH_INF,
                        SearchControl *control = nullptr);

    void reset_search_heuristics();
} // namespace fast_engine

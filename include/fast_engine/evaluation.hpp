#pragma once

#include "chess.hpp"
#include "fast_engine/config.hpp"
#include "fast_engine/types.hpp"

namespace fast_engine
{

    // Debug helpers: expose individual eval terms from White's POV (in centipawns).
    Score debug_eval_material_white(const chess::Board &board);
    Score debug_eval_castling_white_minus_black(const chess::Board &board,
                                                const EngineConfig &cfg);
    Score debug_eval_king_crowding_white_minus_black(const chess::Board &board);
    Score debug_eval_mobility_white_minus_black(const chess::Board &board);
    Score debug_eval_xray_white_minus_black(const chess::Board &board);
    Score debug_eval_pst_stock_white_minus_black(const chess::Board &board);

    /// Material-only evaluation, from White's point of view (in centipawns).
    /// Positive = White better, negative = Black better (centipawns).
    Score evaluate_material(const chess::Board &board);

    /// Full evaluation from White's point of view, using EngineConfig.
    /// For now this is material plus stubbed feature terms (all 0.0).
    Score evaluate_white_pov_with_config(const chess::Board &board,
                                         const EngineConfig &cfg);

    /// Evaluation from the point of view of the side to move, using EngineConfig.
    Score evaluate_for_side_to_move_with_config(const chess::Board &board,
                                                const EngineConfig &cfg);

    // Clears the full evaluation cache (used to avoid cross-game cache effects).
    // This does not affect the TT or other search heuristics.
    void clear_eval_cache();

} // namespace fast_engine

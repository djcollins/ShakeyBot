#pragma once
#include "chess.hpp"

namespace fast_engine
{

    struct EngineConfig
    {
        int search_depth = 3;
        bool use_quiescence = true;

        // Phase-1 search ordering / pruning toggles (for clean A/B isolation)
        bool use_history_heuristic = true;
        bool use_capture_history = true;
        bool use_continuation_history = true;
        bool use_probcut = true;
        bool use_iid = true;

        // Move ordering tuning knobs (all values are in centipawns for convenience)
        // SEE threshold used to classify captures as "good" (SEE >= threshold). The code adds +1cp at depth_remaining<=2
        // to preserve the prior behavior (threshold 1 at shallow nodes, else 0) when this is set to 0.
        int good_capture_see_threshold_cp = 0; // 0, 10, 20

        // Additional ordering penalty applied when a capture's SEE is below good_capture_see_threshold_cp.
        // This is applied in move-order scoring (not evaluation), scaled internally by 16 to match the SEE weighting.
        int bad_capture_penalty_cp = 0; // 0, 100, 150, 200

        // History/continuation ordering multipliers (dimensionless). UCI options expose these as integers 0..300,
        // interpreted as (value / 100.0). Example: 150 -> 1.5x.
        double history_ordering_mult = 0.96;      // default 2.0x (UCI default 200)
        double continuation_ordering_mult = 1.52; // default 2.0x (UCI default 200)
        int capture_history_ordering_mult = 1; // base multiplier for capture history (score_move uses 2x this and shifts >>1 (net ~1x range)) 1, 2, 3

        // History update scheme tuning knobs (dimensionless). Exposed as UCI ints 0..300, interpreted as (value / 100.0).
        // These affect LEARNING (the deltas written to history tables), not the scoring multipliers above.
        // Applied only on quiet beta cutoffs (fail-high) to avoid destabilizing exact-node learning.
        double history_cutoff_bonus_mult = 2.00; // default 2.0x (UCI default 200)
        double history_neg_update_mult = 1.25;   // default 1.25x (UCI default 125)
        // Continuation-history update scheme tuning knobs (dimensionless). Exposed as UCI ints 0..300, interpreted as (value / 100.0).
        // Defaults are tied to the main-history knobs above, so behavior is unchanged unless tuned.
        double cont_cutoff_bonus_mult = 2.25; // default 2.0x (UCI default 200)
        double cont_neg_update_mult = 0.5;   // default 1.25x (UCI default 125)


        // LMR history tuning knobs (dimensionless). Exposed as UCI ints 0..300, interpreted as (value / 100.0).
        // These affect only the history-based LMR adjustment:
        //   - Relief: positive history (trusted quiets) -> smaller reduction / deeper reduced search
        //   - Penalty: negative history (untrusted quiets) -> larger reduction
        double lmr_history_relief_mult = 2.25;  // default 1.0x (UCI default 100)
        double lmr_history_penalty_mult = 2.25; // default 1.0x (UCI default 100)

        // LMR base/slope tuning knobs (dimensionless). Exposed as UCI ints 0..300, interpreted as (value / 100.0).
        // These affect the *baseline* LMR reduction curve (intercept and move-index slope) for late quiet moves.
        //  - LMRBase scales an additive fixed-point offset to the reduction (higher => more reduction).
        //  - LMRSlope scales the per-move-index reduction term (higher => reductions grow faster for later moves).
        double lmr_base_mult = 2.25;  // default 1.0x (UCI default 100)
        double lmr_slope_mult = 0.5; // default 1.0x (UCI default 100)

        // Move ordering bonuses (internal ordering score units; not centipawns).
        int killer_bonus_1 = 90000;     // bonus for primary killer move (quiet only) 60000, 90000, 120000
        int killer_bonus_2 = 80000;     // bonus for secondary killer move (quiet only) 50000, 80000, 100000
        int counter_move_bonus = 10000; // bonus if move matches stored counter move (0 disables) 0, 10000, 20000

        bool use_null_move_pruning = true;

        // Razoring (conservative): only intended for shallow non-PV nodes (typically depth 2–3)
        // and only when verified by a null-window quiescence search.
        bool use_razoring = true;
        int razor_margin_d2 = 250; // centipawns
        int razor_margin_d3 = 500; // centipawns

        bool use_move_count_pruning = true;
        // Correction history: a learned static-eval correction used by pruning gates (centipawns).
        // This does not change evaluation itself, only the static eval used for forward-pruning decisions.
        bool use_correction_history = true;
        // Scale for applying the correction (1.0 = full, 0 disables).
        double correction_history_scale = 0.5;

        // Time management / UCI
        // Max depth used when searching under time control (separate from SearchDepth).
        int max_depth_timed = 64;
        // Safety margin (ms) subtracted from available time to avoid losing on time.
        int move_overhead_ms = 20;
        // Standard UCI ponder option (used by GUIs like cutechess).
        bool ponder = false;

        // Evaluation feature scales (in pawns), matching your Python defaults.
        double king_crowding_scale = 0.1;
        double mobility_scale = 0.5;
        double xray_scale = 0.0;
        double pst_scale = 0.5;
        bool use_stock_pst = true;
        // Donna-style threats term scale (in pawns). Set to 0 to disable.
        double threat_term = 0.5;
        // Draw avoidance ("contempt") in pawns.
        // Applied only to draw-by-repetition and 50-move draws (not insufficient material).
        // If side-to-move is ahead, draw is slightly negative; if behind, slightly positive.
        double draw_contempt_scale = 0.20;     // pawns of contempt per pawn of advantage (5cp per 1.0 pawn)
        double draw_contempt_max = 0.80;       // cap (50cp)
        double draw_contempt_threshold = 0.30; // only apply if |eval| >= 0.50 pawn (50cp)
        // Optional deterministic draw "noise" (in pawns), keyed from board.hash().
        // Keep at 0 for fully deterministic testing.
        double draw_noise = 0.0; // e.g. 0.02 = 2cp
        double hash_mb = 256;
    };

    // Base piece value in pawns, from your Python config, with sign applied
    // (+ for White, - for Black).
    double piece_value(chess::PieceType pt, chess::Color color);

    // Convenience: same value, but take a chess::Piece directly.
    double piece_value_signed(chess::Piece piece);

    extern const EngineConfig DEFAULT_CONFIG;

} // namespace fast_engine

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
        int capture_history_ordering_mult = 1;    // base multiplier for capture history

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
        // These affect only the history-based LMR adjustment.
        double lmr_history_relief_mult = 2.25;  // default 1.0x (UCI default 100)
        double lmr_history_penalty_mult = 2.25; // default 1.0x (UCI default 100)

        // LMR base/slope tuning knobs (dimensionless). Exposed as UCI ints 0..300, interpreted as (value / 100.0).
        double lmr_base_mult = 2.25;  // default 1.0x (UCI default 100)
        double lmr_slope_mult = 0.5; // default 1.0x (UCI default 100)

        // Move ordering bonuses (internal ordering score units; not centipawns).
        int killer_bonus_1 = 90000;
        int killer_bonus_2 = 80000;
        int counter_move_bonus = 10000;

        bool use_null_move_pruning = true;

        // Razoring (conservative)
        bool use_razoring = true;
        int razor_margin_d2 = 250; // centipawns
        int razor_margin_d3 = 500; // centipawns

        bool use_move_count_pruning = true;

        // Correction history
        bool use_correction_history = true;
        double correction_history_scale = 0.5;

        // Time management / UCI
        int max_depth_timed = 64;
        int move_overhead_ms = 20;
        bool ponder = false;

        // -------------------- Evaluation tuning knobs --------------------
        // All evaluation sub-terms are accumulated internally in *pawn units* (double)
        // and converted to centipawns only once at the public boundary.
        //
        // The fields below are coarse, top-level scalars for major eval components.
        // They are intentionally linear multipliers so that A/B tuning can treat them
        // as independent "knobs" before micro-tuning the internal heuristics.
        struct EvalScales
        {
            // Material and material-imbalance terms (White POV).
            double material = 1.30;   // matches prior: material_white_pov_pawns(board) * 1.3
            double imbalance = 1.00;  // matches prior: + mat.imbalance_pawns_wmb

            // Piece/king activity.
            double king_crowding = 0.10;
            double mobility = 0.50;
            double xray_pins = 0.00; // 0 disables

            // Midgame positional.
            double space = 1.00;
            double outposts = 1.00;

            // Pawn play.
            double pawn_structure = 1.00;
            double passed_pawns = 1.50; // matches prior: passed_pawn_term * 1.5

            // Pieces.
            double rook_activity = 1.00;
            double bishop_pair_bad_bishop = 1.00;

            // Tactical.
            double threats = 0.50;

            // King safety.
            double king_safety = 1.00;

            // PST/PSQT.
            double pst = 0.50;
            bool use_stock_pst = true;
        } eval;

        // Tempo bonus (centipawns) applied in evaluate_for_side_to_move_with_config().
        int tempo_bonus_cp = 6;

        // Endgame scaling toggle (applied when phase_0_256 >= 128).
        bool enable_endgame_scaling = true;

        // Draw avoidance ("contempt") in pawns.
        // Applied only to draw-by-repetition and 50-move draws (not insufficient material).
        double draw_contempt_scale = 0.20;     // pawns of contempt per pawn of advantage (5cp per 1.0 pawn)
        double draw_contempt_max = 0.80;       // cap (80cp)
        double draw_contempt_threshold = 0.30; // only apply if |eval| >= threshold (pawn units)

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

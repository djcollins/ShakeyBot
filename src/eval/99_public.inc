// Evaluation module: public API (evaluate + debug)
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

// ---------------------- Public functions ----------------------

Score evaluate_material(const Board &board)
{
    // Net from White's point of view, including material imbalance.
    const double base = material_white_pov_pawns(board);
    const double imb = material_probe(board).imbalance_pawns_wmb;
    return pawns_to_cp(base + imb);
}

static Score evaluate_hce_white_pov_uncached(const Board &board,
                                             const EngineConfig &cfg)
{
    // Single phase signal reused by tapered terms:
    // opening-heavy at low values, endgame-heavy near 256.
    const int phase_0_256 = compute_phase_0_256(board);
    // phase_0_256 is an endgame-progress metric: 0 = opening, 256 = king-only endgame.
    const double eg_weight = static_cast<double>(phase_0_256) / 256.0;
    const double mg_weight = 1.0 - eg_weight;
    EvalInfo info = make_eval_info(board);

    // IMPORTANT: all *term* functions here are expressed in pawn units.
    // Keep the internal accumulator in pawn units and convert only once at the end.
    double score = 0.0;

    const MaterialEntry &mat = material_probe(board);

    // 1) Material + imbalance.
    score += cfg.eval.material * material_white_pov_pawns(board);
    score += cfg.eval.imbalance * mat.imbalance_pawns_wmb;

    // 2) King crowding.
    if (cfg.eval.king_crowding != 0.0)
    {
        const double kc_term = king_crowding_term_white_minus_black(board);
        score += cfg.eval.king_crowding * kc_term;
    }

    // 3) Mobility.
    if (cfg.eval.mobility != 0.0)
    {
        const double mob_term = mobility_term_white_minus_black(board, mg_weight, eg_weight, info);
        score += cfg.eval.mobility * mob_term;
    }

    // 3c) X-ray pins (cheap tactical heuristic).
    if (cfg.eval.xray_pins != 0.0)
    {
        const double xr = xray_pins_term_white_minus_black(board);
        score += cfg.eval.xray_pins * xr;
    }

    const bool need_attack_maps = (cfg.eval.space != 0.0) ||
                                  (cfg.eval.threats != 0.0) ||
                                  (cfg.eval.queen_vulnerability != 0.0);
    const AttackMaps *attack_maps = nullptr;
    if (need_attack_maps)
        attack_maps = &ensure_attack_maps(board, info);

    // 3a) Space advantage (midgame-only).
    if (cfg.eval.space != 0.0)
        score += cfg.eval.space * space_term_white_minus_black_from_maps(board, *attack_maps, mg_weight, info);

    // 3b) Minor-piece outposts (pawn-supported, not pawn-chaseable).
    score += cfg.eval.outposts * mg_weight * outpost_term_white_minus_black(board);

    // 3d) Global closedness / openness. Bias knights up and rooks down in closed
    // structures, and the reverse in open ones.
    if (cfg.eval.closedness != 0.0)
        score += cfg.eval.closedness * closedness_term_white_minus_black(board, mat, mg_weight, info);

    // 4a) Pawn structure.
    score += cfg.eval.pawn_structure * pawn_structure_term_white_minus_black(board, mg_weight, eg_weight);

    // 4b) Passed pawns (endgame-weighted).
    score += cfg.eval.passed_pawns * eg_weight * passed_pawn_term_white_minus_black(board);

    // 4c) Rook activity (open/semi-open files + rook on 7th).
    if (ENABLE_ROOK_FILE_ACTIVITY && cfg.eval.rook_activity != 0.0)
    {
        score += cfg.eval.rook_activity * rook_activity_term_white_minus_black(board, mg_weight, eg_weight, info);
    }

    // 4d) Donna-style threats (attacked and undefended pieces).
    if (cfg.eval.threats != 0.0)
    {
        const double thr = threats_term_white_minus_black_from_maps(board, *attack_maps, mg_weight, eg_weight, info);
        score += cfg.eval.threats * thr;
    }

    // 4e) Queen vulnerability. Penalize exposed queens with poor safe mobility.
    if (cfg.eval.queen_vulnerability != 0.0)
    {
        const double qv = queen_vulnerability_term_white_minus_black(board, *attack_maps, mg_weight, info);
        score += cfg.eval.queen_vulnerability * qv;
    }

    // 5) King safety.
    if (cfg.eval.king_safety != 0.0)
    {
        // King-zone pressure remains untapered. The cover/storm component should be weighted toward the middlegame.
        score += cfg.eval.king_safety * king_safety_term_white_minus_black(board, mg_weight, mat, info);
    }

    // 6) Bishop pair + bad bishop.
    if (cfg.eval.bishop_pair_bad_bishop != 0.0)
    {
        score += cfg.eval.bishop_pair_bad_bishop * bishop_pair_bad_bishop_term_white_minus_black(board, mg_weight, eg_weight);
    }

    // 7) Piece-square tables.
    if (cfg.eval.pst != 0.0)
    {
        double pst_term = 0.0;
        if (cfg.eval.use_stock_pst)
            pst_term = piece_square_term_white_minus_black_stock(board);
        else
            pst_term = piece_square_term_white_minus_black_custom(board, phase_0_256);

        score += cfg.eval.pst * pst_term;
    }

    // 8) Endgame scaling.
    if (cfg.enable_endgame_scaling && phase_0_256 >= 128)
    {
        // Scale factor is in [0..1]; final multiplier = mg_weight + eg_weight * scale.
        const double sf = endgame_scale_factor_white_pov(board, score);
        const double factor = mg_weight + eg_weight * sf;
        score *= factor;
    }

    // 8b) Generic complexity scaling: damp optimistic evals in low-material, drawish
    // endgames that are hard to convert even when the static edge is real.
    if (cfg.eval.complexity != 0.0 && phase_0_256 >= 96)
    {
        const double base_scale = complexity_scale_factor_white_pov(board, score, mat, info);
        double complexity_scale = 1.0 + cfg.eval.complexity * (base_scale - 1.0);
        if (complexity_scale < 0.0)
            complexity_scale = 0.0;
        if (complexity_scale > 1.0)
            complexity_scale = 1.0;
        score *= complexity_scale;
    }

    return pawns_to_cp(score);
}

static int count_piece_type_both(const Board &board, PieceType type)
{
    return static_cast<int>(board.pieces(type, Color::WHITE).count() +
                            board.pieces(type, Color::BLACK).count());
}

static bool neural_should_use_hce_endgame_fallback(const Board &board,
                                                   const EngineConfig &cfg)
{
    if (!cfg.neural_endgame_fallback)
        return false;

    const int pawns = count_piece_type_both(board, PieceType::PAWN);
    const int knights = count_piece_type_both(board, PieceType::KNIGHT);
    const int bishops = count_piece_type_both(board, PieceType::BISHOP);
    const int rooks = count_piece_type_both(board, PieceType::ROOK);
    const int queens = count_piece_type_both(board, PieceType::QUEEN);

    const int non_pawn_units = 3 * (knights + bishops) + 5 * rooks + 9 * queens;
    const int total_units = pawns + non_pawn_units;

    if (total_units > cfg.neural_endgame_material_limit)
        return false;
    if (non_pawn_units == 0 && !cfg.neural_pawn_only_fallback)
        return false;

    return true;
}

Score evaluate_white_pov_with_config(const Board &board,
                                     const EngineConfig &cfg)
{
    // Full-eval cache: avoids recomputing expensive terms (mobility, king safety,
    // pawn structure) or neural inference for positions reached repeatedly in the tree.
    const std::uint64_t k = eval_cache_key(board, cfg);
    Score cached = 0;
    if (eval_cache_table().probe(k, cached))
        return cached;

    Score result = 0;
    // Exact KPK (K+P vs K) is stable enough to apply before backend dispatch.
    if (probe_kpk_exact_eval_white_pov(board, result))
    {
        eval_cache_table().store(k, result);
        return result;
    }

    if ((cfg.eval_backend == EvalBackend::NeuralSimple ||
         cfg.eval_backend == EvalBackend::NeuralAccum ||
         cfg.eval_backend == EvalBackend::NeuralQuant ||
         cfg.eval_backend == EvalBackend::NeuralQuantAccum ||
         cfg.eval_backend == EvalBackend::NeuralHalfkp ||
         cfg.eval_backend == EvalBackend::NeuralHalfkpQuant ||
         cfg.eval_backend == EvalBackend::NeuralHalfkpQuantAccum) &&
        neural_should_use_hce_endgame_fallback(board, cfg))
    {
        result = evaluate_hce_white_pov_uncached(board, cfg);
    }
    else if (cfg.eval_backend == EvalBackend::NeuralSimple ||
             cfg.eval_backend == EvalBackend::NeuralAccum)
        result = evaluate_neural_simple_white_pov(board);
    else if (cfg.eval_backend == EvalBackend::NeuralQuant ||
             cfg.eval_backend == EvalBackend::NeuralQuantAccum)
        result = evaluate_neural_quant_white_pov(board);
    else if (cfg.eval_backend == EvalBackend::NeuralHalfkp)
        result = evaluate_neural_halfkp_white_pov(board);
    else if (cfg.eval_backend == EvalBackend::NeuralHalfkpQuant ||
             cfg.eval_backend == EvalBackend::NeuralHalfkpQuantAccum)
        result = evaluate_neural_halfkp_quant_white_pov(board);
    else if (cfg.eval_backend == EvalBackend::NeuralDummy)
        result = evaluate_neural_dummy_white_pov(board);
    else
        result = evaluate_hce_white_pov_uncached(board, cfg);

    eval_cache_table().store(k, result);
    return result;
}

Score evaluate_white_pov_with_accumulator(const Board &board,
                                          const EngineConfig &cfg,
                                          NeuralAccumulator *accum,
                                          NeuralAccumulatorStats *stats)
{
    if (cfg.eval_backend != EvalBackend::NeuralAccum &&
        cfg.eval_backend != EvalBackend::NeuralQuantAccum &&
        cfg.eval_backend != EvalBackend::NeuralHalfkpQuantAccum)
        return evaluate_white_pov_with_config(board, cfg);

    const std::uint64_t k = eval_cache_key(board, cfg);
    Score cached = 0;
    if (eval_cache_table().probe(k, cached))
        return cached;

    Score result = 0;
    if (probe_kpk_exact_eval_white_pov(board, result))
    {
        eval_cache_table().store(k, result);
        return result;
    }

    if (neural_should_use_hce_endgame_fallback(board, cfg))
    {
        result = evaluate_hce_white_pov_uncached(board, cfg);
    }
    else if (cfg.eval_backend == EvalBackend::NeuralAccum)
    {
        NeuralAccumulator local{};
        NeuralAccumulator &use_accum = accum ? *accum : local;
        if (!neural_simple_accumulator_matches_impl(board, use_accum))
        {
            if (stats)
                ++stats->invalid_fallbacks;
            refresh_neural_simple_accumulator_impl(board, use_accum, stats);
        }
        result = evaluate_neural_simple_accumulator_white_pov(use_accum);
        if (cfg.neural_accumulator_check)
        {
            const Score stateless = evaluate_neural_simple_white_pov(board);
            if (std::abs(static_cast<int>(result) - static_cast<int>(stateless)) >
                    static_cast<int>(NEURAL_ACCUMULATOR_FLOAT_TOLERANCE_CP) &&
                stats)
                ++stats->check_failures;
        }
    }
    else if (cfg.eval_backend == EvalBackend::NeuralQuantAccum)
    {
        NeuralAccumulator local{};
        NeuralAccumulator &use_accum = accum ? *accum : local;
        if (!neural_quant_accumulator_matches_impl(board, use_accum))
        {
            if (stats)
                ++stats->invalid_fallbacks;
            refresh_neural_quant_accumulator_impl(board, use_accum, stats);
        }
        result = evaluate_neural_quant_accumulator_white_pov(use_accum);
        if (cfg.neural_accumulator_check)
        {
            const Score stateless = evaluate_neural_quant_white_pov(board);
            if (result != stateless && stats)
                ++stats->check_failures;
        }
    }
    else
    {
        NeuralAccumulator local{};
        NeuralAccumulator &use_accum = accum ? *accum : local;
        if (!neural_halfkp_quant_accumulator_matches_impl(board, use_accum))
        {
            if (stats)
                ++stats->invalid_fallbacks;
            refresh_neural_halfkp_quant_accumulator_impl(board, use_accum, stats);
        }
        result = evaluate_neural_halfkp_quant_accumulator_white_pov(board, use_accum);
        if (cfg.neural_accumulator_check)
        {
            const Score stateless = evaluate_neural_halfkp_quant_white_pov(board);
            if (std::abs(static_cast<int>(result) - static_cast<int>(stateless)) > 1 && stats)
                ++stats->check_failures;
        }
    }

    eval_cache_table().store(k, result);
    return result;
}

Score evaluate_for_side_to_move_with_config(const Board &board,
                                            const EngineConfig &cfg)
{
    // Side-to-move perspective: positive is good for the player to move.
    // Add a small tempo bonus in the middlegame only to avoid endgame zugzwang artifacts.
    const int phase_0_256 = compute_phase_0_256(board);


    const Score white_pov = evaluate_white_pov_with_config(board, cfg);
    const Score stm = (board.sideToMove() == Color::WHITE) ? white_pov : -white_pov;

    const Score tempo = static_cast<Score>((static_cast<std::int64_t>(cfg.tempo_bonus_cp) * (256 - phase_0_256) + 128) / 256);

    return stm + tempo;
}

void clear_eval_cache()
{
    eval_cache_table().clear();
}

bool load_neural_simple_model(const std::string &path, std::string &error)
{
    const bool ok = load_neural_simple_model_impl(path, error);
    clear_eval_cache();
    return ok;
}

void unload_neural_simple_model()
{
    unload_neural_simple_model_impl();
    clear_eval_cache();
}

bool neural_simple_model_loaded()
{
    return neural_simple_model_loaded_impl();
}

const std::string &neural_simple_model_path()
{
    return neural_simple_model_path_impl();
}

bool load_neural_quant_model(const std::string &path, std::string &error)
{
    const bool ok = load_neural_quant_model_impl(path, error);
    clear_eval_cache();
    return ok;
}

void unload_neural_quant_model()
{
    unload_neural_quant_model_impl();
    clear_eval_cache();
}

bool neural_quant_model_loaded()
{
    return neural_quant_model_loaded_impl();
}

const std::string &neural_quant_model_path()
{
    return neural_quant_model_path_impl();
}

bool load_neural_halfkp_model(const std::string &path, std::string &error)
{
    const bool ok = load_neural_halfkp_model_impl(path, error);
    clear_eval_cache();
    return ok;
}

void unload_neural_halfkp_model()
{
    unload_neural_halfkp_model_impl();
    clear_eval_cache();
}

bool neural_halfkp_model_loaded()
{
    return neural_halfkp_model_loaded_impl();
}

const std::string &neural_halfkp_model_path()
{
    return neural_halfkp_model_path_impl();
}

bool load_neural_halfkp_quant_model(const std::string &path, std::string &error)
{
    const bool ok = load_neural_halfkp_quant_model_impl(path, error);
    clear_eval_cache();
    return ok;
}

void unload_neural_halfkp_quant_model()
{
    unload_neural_halfkp_quant_model_impl();
    clear_eval_cache();
}

bool neural_halfkp_quant_model_loaded()
{
    return neural_halfkp_quant_model_loaded_impl();
}

const std::string &neural_halfkp_quant_model_path()
{
    return neural_halfkp_quant_model_path_impl();
}

bool neural_accumulator_backend_active(const EngineConfig &cfg) noexcept
{
    return cfg.eval_backend == EvalBackend::NeuralAccum ||
           cfg.eval_backend == EvalBackend::NeuralQuantAccum ||
           cfg.eval_backend == EvalBackend::NeuralHalfkpQuantAccum;
}

bool neural_simple_accumulator_matches(const Board &board,
                                       const NeuralAccumulator &accum) noexcept
{
    return neural_simple_accumulator_matches_impl(board, accum);
}

void refresh_neural_simple_accumulator(const Board &board,
                                       NeuralAccumulator &accum,
                                       NeuralAccumulatorStats *stats)
{
    refresh_neural_simple_accumulator_impl(board, accum, stats);
}

bool update_neural_simple_accumulator_for_move(const NeuralAccumulator &parent,
                                               const Board &board,
                                               const chess::Move &move,
                                               NeuralAccumulator &child,
                                               NeuralAccumulatorStats *stats)
{
    return update_neural_simple_accumulator_for_move_impl(parent, board, move, child, stats);
}

bool neural_accumulator_matches(const Board &board,
                                const EngineConfig &cfg,
                                const NeuralAccumulator &accum) noexcept
{
    if (cfg.eval_backend == EvalBackend::NeuralQuantAccum)
        return neural_quant_accumulator_matches_impl(board, accum);
    if (cfg.eval_backend == EvalBackend::NeuralHalfkpQuantAccum)
        return neural_halfkp_quant_accumulator_matches_impl(board, accum);
    return neural_simple_accumulator_matches_impl(board, accum);
}

void refresh_neural_accumulator_for_config(const Board &board,
                                           const EngineConfig &cfg,
                                           NeuralAccumulator &accum,
                                           NeuralAccumulatorStats *stats)
{
    if (cfg.eval_backend == EvalBackend::NeuralQuantAccum)
        refresh_neural_quant_accumulator_impl(board, accum, stats);
    else if (cfg.eval_backend == EvalBackend::NeuralHalfkpQuantAccum)
        refresh_neural_halfkp_quant_accumulator_impl(board, accum, stats);
    else
        refresh_neural_simple_accumulator_impl(board, accum, stats);
}

void copy_neural_accumulator_for_config(const NeuralAccumulator &parent,
                                        const EngineConfig &cfg,
                                        NeuralAccumulator &child) noexcept
{
    if (!parent.valid)
    {
        child = NeuralAccumulator{};
        return;
    }
    if (cfg.eval_backend == EvalBackend::NeuralQuantAccum)
        copy_neural_quant_accumulator_impl(parent, child);
    else if (cfg.eval_backend == EvalBackend::NeuralHalfkpQuantAccum)
        copy_neural_halfkp_quant_accumulator_impl(parent, child);
    else
        copy_neural_simple_accumulator_impl(parent, child);
}

bool update_neural_accumulator_for_move(const NeuralAccumulator &parent,
                                        const Board &board,
                                        const chess::Move &move,
                                        const EngineConfig &cfg,
                                        NeuralAccumulator &child,
                                        NeuralAccumulatorStats *stats)
{
    if (cfg.eval_backend == EvalBackend::NeuralQuantAccum)
        return update_neural_quant_accumulator_for_move_impl(parent, board, move, child, stats);
    if (cfg.eval_backend == EvalBackend::NeuralHalfkpQuantAccum)
        return update_neural_halfkp_quant_accumulator_for_move_impl(parent, board, move, child, stats);
    return update_neural_simple_accumulator_for_move_impl(parent, board, move, child, stats);
}

// ----- Debug accessors (White POV) -----

Score debug_eval_material_white(const Board &board)
{
    // Same sign convention as Python evaluate_material: White minus Black.
    return evaluate_material(board);
}

Score debug_eval_king_crowding_white_minus_black(const Board &board)
{
    return pawns_to_cp(king_crowding_term_white_minus_black(board));
}

Score debug_eval_mobility_white_minus_black(const Board &board)
{
    return pawns_to_cp(mobility_term_white_minus_black(board));
}

Score debug_eval_space_white_minus_black(const Board &board)
{
    const int phase_0_256 = compute_phase_0_256(board);
    const double mg_weight = static_cast<double>(256 - phase_0_256) / 256.0;
    return pawns_to_cp(space_term_white_minus_black(board, mg_weight));
}

Score debug_eval_xray_white_minus_black(const Board &board)
{
    return pawns_to_cp(xray_pins_term_white_minus_black(board));
}

Score debug_eval_pst_stock_white_minus_black(const Board &board)
{
    return pawns_to_cp(piece_square_term_white_minus_black_stock(board));
}

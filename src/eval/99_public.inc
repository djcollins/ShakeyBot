// Evaluation module: public API (evaluate + debug)
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

// ---------------------- Public functions ----------------------

Score evaluate_material(const Board &board)
{
    // Net from White's point of view, including Stockfish-style material imbalance.
    const double base = material_white_pov_pawns(board);
    const double imb = material_probe(board).imbalance_pawns_wmb;
    return pawns_to_cp(base + imb);
}

Score evaluate_white_pov_with_config(const Board &board,
                                     const EngineConfig &cfg)
{
    // Full-eval cache: avoids recomputing expensive terms (mobility, king safety,
    // pawn structure) for positions reached repeatedly in the tree.
    const std::uint64_t k = eval_cache_key(board, cfg);
    Score cached = 0;
    if (eval_cache_table().probe(k, cached))
        return cached;

    // Exact KPK (K+P vs K) bitbase probe.
    // This is perfectly stable and prevents static-eval based pruning gates from
    // mis-evaluating pawn endgames.
    {
        Score kpk_eval = 0;
        if (probe_kpk_exact_eval_white_pov(board, kpk_eval))
        {
            eval_cache_table().store(k, kpk_eval);
            return kpk_eval;
        }
    }

    // Exact draw recognizer: minor-only endgames where checkmate is impossible
    // for either side (e.g., K+N vs K, K+B vs K, K+NN vs K, K+B vs K+B).
    // This stabilizes static-eval pruning gates in late endgames.
    {
        Score draw_eval = 0;
        if (probe_minor_only_insufficient_draw_white_pov(board, draw_eval))
        {
            eval_cache_table().store(k, draw_eval);
            return draw_eval;
        }
    }

    // Single phase signal reused by tapered terms:
    // opening-heavy at low values, endgame-heavy near 256.
    // This keeps all tapered terms aligned on the same game-phase estimate.
    const int phase_0_256 = compute_phase_0_256(board);
    // phase_0_256 is an endgame-progress metric: 0 = opening, 256 = king-only endgame.
    const double eg_weight = static_cast<double>(phase_0_256) / 256.0;
    const double mg_weight = 1.0 - eg_weight;
    double score = 0.0;

    const MaterialEntry &mat = material_probe(board);

    // 1) Material is always included.
    // IMPORTANT: all *term* functions in this file are expressed in pawn units.
    // Keep the internal accumulator in pawn units and convert only once at the end.
    score += material_white_pov_pawns(board) * 1.3 + mat.imbalance_pawns_wmb;

    // 2) King crowding (simple, no phase weighting).
    if (cfg.king_crowding_scale != 0.0)
    {
        const double kc_term = king_crowding_term_white_minus_black(board);
        score += cfg.king_crowding_scale * kc_term;
    }

    // 3) Mobility.
    if (cfg.mobility_scale != 0.0)
    {
        const double mob_term = mobility_term_white_minus_black(board);
        score += cfg.mobility_scale * mob_term;
    }

    // 3a) Space advantage (Stockfish 12-style, midgame-only).
    score += space_term_white_minus_black(board, mg_weight);

    // 3b) Minor-piece outposts (pawn-supported, not pawn-chaseable).
    score += mg_weight * outpost_term_white_minus_black(board);

    // 4a) Pawn structure (simple).
    const double midgame_weight = mg_weight;
    const double endgame_weight = eg_weight;
    score += pawn_structure_term_white_minus_black(board, midgame_weight, endgame_weight);

    // 4b) Passed pawns (endgame-weighted).
    score += eg_weight * passed_pawn_term_white_minus_black(board) * 1.5;

    // 4c) Rook activity (open/semi-open files + rook on 7th).
    if (ENABLE_ROOK_FILE_ACTIVITY)
    {
        score += rook_activity_term_white_minus_black(board, mg_weight, eg_weight);
    }

    // 4d) Donna-style threats (attacked and undefended pieces).
    if (cfg.threat_term != 0.0)
    {
        const double thr = threats_term_white_minus_black(board, mg_weight, eg_weight);
        score += cfg.threat_term * thr;
    }

    // 5) King safety (always-on).
    //
    // King-zone pressure remains untapered. The cover/storm component should be weighted toward the middlegame.
    score += king_safety_term_white_minus_black(board, mg_weight, mat);

    // 6) Bishop pair + bad bishop (simple).
    score += bishop_pair_bad_bishop_term_white_minus_black(board, mg_weight, eg_weight);

    // 7) Piece-square tables.
    if (cfg.pst_scale != 0.0)
    {
        double pst_term = 0.0;
        if (cfg.use_stock_pst)
        {
            pst_term = piece_square_term_white_minus_black_stock(board);
        }
        else
        {
            pst_term = piece_square_term_white_minus_black_custom(board, phase_0_256);
        }
        score += cfg.pst_scale * pst_term;
    }

    // 8) Endgame scaling.
    //
    // Apply only once we are past the middlegame to avoid disturbing tactical play.
    // Scale factor is in [0..1]; final multiplier = mg_weight + eg_weight * scale.
    if (phase_0_256 >= 128)
    {
        const double sf = endgame_scale_factor_white_pov(board, score);
        const double factor = mg_weight + eg_weight * sf;
        score *= factor;
    }

    // Convert once at the boundary so sub-terms can stay in pawn units.
    const Score result = pawns_to_cp(score);
    eval_cache_table().store(k, result);
    return result;
}

Score evaluate_for_side_to_move_with_config(const Board &board,
                                            const EngineConfig &cfg)
{
    // Side-to-move perspective: positive is good for the player to move.
    // Add a small tempo bonus in the middlegame only to avoid endgame zugzwang artifacts.
    const int phase_0_256 = compute_phase_0_256(board);

    constexpr Score TEMPO_BONUS_CP = 6; // centipawns

    const Score white_pov = evaluate_white_pov_with_config(board, cfg);
    const Score stm = (board.sideToMove() == Color::WHITE) ? white_pov : -white_pov;

    const Score tempo = static_cast<Score>((static_cast<std::int64_t>(TEMPO_BONUS_CP) * (256 - phase_0_256) + 128) / 256);

    return stm + tempo;
}

void clear_eval_cache()
{
    eval_cache_table().clear();
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

Score debug_eval_xray_white_minus_black(const Board & /*board*/)
{
    // Pinned-piece / xray evaluation has been removed for simplicity.
    return 0;
}

Score debug_eval_pst_stock_white_minus_black(const Board &board)
{
    return pawns_to_cp(piece_square_term_white_minus_black_stock(board));
}

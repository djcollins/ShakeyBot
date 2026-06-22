#pragma once

#include "chess.hpp"
#include "fast_engine/config.hpp"
#include "fast_engine/types.hpp"

#include <array>
#include <cstdint>
#include <string>

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

    // Material-only evaluation, from White's point of view.
    // Positive = White better, negative = Black better.
    Score evaluate_material(const chess::Board &board);

    // Full evaluation from White's point of view, using EngineConfig.
    Score evaluate_white_pov_with_config(const chess::Board &board,
                                         const EngineConfig &cfg);

    // Evaluation from the point of view of the side to move.
    Score evaluate_for_side_to_move_with_config(const chess::Board &board,
                                                const EngineConfig &cfg);

    // Clears the full evaluation cache (used to avoid cross-game cache effects).
    // This does not affect the TT or other search heuristics.
    void clear_eval_cache();

    constexpr int NEURAL_ACCUM_MAX_HIDDEN = 512;

    struct NeuralAccumulator
    {
        alignas(64) std::array<float, NEURAL_ACCUM_MAX_HIDDEN> pre_activation{};
        alignas(64) std::array<std::int32_t, NEURAL_ACCUM_MAX_HIDDEN> quant_pre_activation{};
        alignas(64) std::array<std::array<std::int32_t, NEURAL_ACCUM_MAX_HIDDEN>, 2> halfkp_quant_pre_activation{};
        int hidden_size = 0;
        std::uint64_t board_hash = 0ULL;
        bool quantized = false;
        bool halfkp = false;
        bool valid = false;
    };

    struct NeuralAccumulatorStats
    {
        std::uint64_t refreshes = 0;
        std::uint64_t invalid_fallbacks = 0;
        std::uint64_t delta_updates = 0;
        std::uint64_t check_failures = 0;
    };

    // Loads/unloads the stateless Simple768 neural model used by float neural backends.
    bool load_neural_simple_model(const std::string &path, std::string &error);
    void unload_neural_simple_model();
    bool neural_simple_model_loaded();
    const std::string &neural_simple_model_path();

    // Loads/unloads the quantized Simple768 model used by quantized neural backends.
    bool load_neural_quant_model(const std::string &path, std::string &error);
    void unload_neural_quant_model();
    bool neural_quant_model_loaded();
    const std::string &neural_quant_model_path();

    // Loads/unloads the HalfKP models used by the HalfKP neural backends.
    bool load_neural_halfkp_model(const std::string &path, std::string &error);
    void unload_neural_halfkp_model();
    bool neural_halfkp_model_loaded();
    const std::string &neural_halfkp_model_path();
    bool load_neural_halfkp_quant_model(const std::string &path, std::string &error);
    void unload_neural_halfkp_quant_model();
    bool neural_halfkp_quant_model_loaded();
    const std::string &neural_halfkp_quant_model_path();

    bool neural_accumulator_backend_active(const EngineConfig &cfg) noexcept;
    bool neural_simple_accumulator_matches(const chess::Board &board,
                                           const NeuralAccumulator &accum) noexcept;
    void refresh_neural_simple_accumulator(const chess::Board &board,
                                           NeuralAccumulator &accum,
                                           NeuralAccumulatorStats *stats = nullptr);
    bool update_neural_simple_accumulator_for_move(const NeuralAccumulator &parent,
                                                   const chess::Board &board,
                                                   const chess::Move &move,
                                                   NeuralAccumulator &child,
                                                   NeuralAccumulatorStats *stats = nullptr);
    bool neural_accumulator_matches(const chess::Board &board,
                                    const EngineConfig &cfg,
                                    const NeuralAccumulator &accum) noexcept;
    void refresh_neural_accumulator_for_config(const chess::Board &board,
                                               const EngineConfig &cfg,
                                               NeuralAccumulator &accum,
                                               NeuralAccumulatorStats *stats = nullptr);
    void copy_neural_accumulator_for_config(const NeuralAccumulator &parent,
                                            const EngineConfig &cfg,
                                            NeuralAccumulator &child) noexcept;
    bool update_neural_accumulator_for_move(const NeuralAccumulator &parent,
                                            const chess::Board &board,
                                            const chess::Move &move,
                                            const EngineConfig &cfg,
                                            NeuralAccumulator &child,
                                            NeuralAccumulatorStats *stats = nullptr);
    Score evaluate_white_pov_with_accumulator(const chess::Board &board,
                                              const EngineConfig &cfg,
                                              NeuralAccumulator *accum,
                                              NeuralAccumulatorStats *stats = nullptr);

} // namespace fast_engine

// Evaluation module: quantized Simple768 neural inference
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

struct NeuralQuantModel
{
    AlignedVector<std::int16_t> w1;
    AlignedVector<std::int32_t> b1;
    AlignedVector<std::int16_t> w2;
    AlignedVector<std::int64_t> b2_layer;
    AlignedVector<std::int16_t> w3;
    AlignedVector<std::int64_t> b3_layer;
    AlignedVector<std::int16_t> w4;
    std::int64_t b2 = 0;
    std::int64_t b3 = 0;
    std::int64_t b4 = 0;
    double output_scale_cp = NEURAL_SIMPLE_DEFAULT_OUTPUT_SCALE_CP;
    std::int32_t activation_scale = 1;
    std::int32_t layer2_weight_scale = 1;
    std::int32_t layer3_weight_scale = 1;
    std::int32_t output_weight_scale = 1;
    int hidden_size = 0;
    int layer2_size = 0;
    int layer3_size = 0;
    bool loaded = false;
    std::string path;
};

#if defined(__SSE2__)
static inline void neural_quant_add_i16_row_to_i32_sse2(std::int32_t *dst,
                                                        const std::int16_t *row,
                                                        int hidden_size) noexcept
{
    const __m128i zero = _mm_setzero_si128();
    int h = 0;
    for (; h + 8 <= hidden_size; h += 8)
    {
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(row + h));
        const __m128i sign = _mm_cmpgt_epi16(zero, packed);
        const __m128i lo = _mm_unpacklo_epi16(packed, sign);
        const __m128i hi = _mm_unpackhi_epi16(packed, sign);

        __m128i acc_lo = _mm_loadu_si128(reinterpret_cast<const __m128i *>(dst + h));
        __m128i acc_hi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(dst + h + 4));
        acc_lo = _mm_add_epi32(acc_lo, lo);
        acc_hi = _mm_add_epi32(acc_hi, hi);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + h), acc_lo);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + h + 4), acc_hi);
    }
    for (; h < hidden_size; ++h)
        dst[h] += static_cast<std::int32_t>(row[h]);
}

static inline void neural_quant_sub_i16_row_from_i32_sse2(std::int32_t *dst,
                                                          const std::int16_t *row,
                                                          int hidden_size) noexcept
{
    const __m128i zero = _mm_setzero_si128();
    int h = 0;
    for (; h + 8 <= hidden_size; h += 8)
    {
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(row + h));
        const __m128i sign = _mm_cmpgt_epi16(zero, packed);
        const __m128i lo = _mm_unpacklo_epi16(packed, sign);
        const __m128i hi = _mm_unpackhi_epi16(packed, sign);

        __m128i acc_lo = _mm_loadu_si128(reinterpret_cast<const __m128i *>(dst + h));
        __m128i acc_hi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(dst + h + 4));
        acc_lo = _mm_sub_epi32(acc_lo, lo);
        acc_hi = _mm_sub_epi32(acc_hi, hi);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + h), acc_lo);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + h + 4), acc_hi);
    }
    for (; h < hidden_size; ++h)
        dst[h] -= static_cast<std::int32_t>(row[h]);
}
#endif

static inline void neural_quant_add_i16_row_to_i32(std::int32_t *dst,
                                                   const std::int16_t *row,
                                                   int hidden_size) noexcept
{
#if defined(__SSE2__)
    neural_quant_add_i16_row_to_i32_sse2(dst, row, hidden_size);
#else
    for (int h = 0; h < hidden_size; ++h)
        dst[h] += static_cast<std::int32_t>(row[h]);
#endif
}

static inline void neural_quant_sub_i16_row_from_i32(std::int32_t *dst,
                                                     const std::int16_t *row,
                                                     int hidden_size) noexcept
{
#if defined(__SSE2__)
    neural_quant_sub_i16_row_from_i32_sse2(dst, row, hidden_size);
#else
    for (int h = 0; h < hidden_size; ++h)
        dst[h] -= static_cast<std::int32_t>(row[h]);
#endif
}

NeuralQuantModel &neural_quant_model() noexcept
{
    static NeuralQuantModel model;
    return model;
}

bool neural_quant_model_loaded_impl() noexcept
{
    return neural_quant_model().loaded;
}

const std::string &neural_quant_model_path_impl() noexcept
{
    return neural_quant_model().path;
}

void unload_neural_quant_model_impl() noexcept
{
    neural_quant_model() = NeuralQuantModel{};
}

static bool read_neural_i64(std::istream &in, std::int64_t &value, std::string &error)
{
    if (!(in >> value))
    {
        error = "unexpected end of quantized model file";
        return false;
    }
    return true;
}

bool load_neural_quant_model_impl(const std::string &path, std::string &error)
{
    if (path.empty())
    {
        unload_neural_quant_model_impl();
        return true;
    }

    std::ifstream in(path);
    if (!in)
    {
        error = "could not open quantized model file";
        return false;
    }

    std::string token;
    if (!(in >> token) ||
        (token != "SHAKEYBOT_SIMPLE768_QUANT_V1" &&
         token != "SHAKEYBOT_SIMPLE768_2L_QUANT_V1" &&
         token != "SHAKEYBOT_SIMPLE768_3L_QUANT_V1"))
    {
        error = "bad quantized model header";
        return false;
    }
    const bool two_layer = token == "SHAKEYBOT_SIMPLE768_2L_QUANT_V1" || token == "SHAKEYBOT_SIMPLE768_3L_QUANT_V1";
    const bool three_layer = token == "SHAKEYBOT_SIMPLE768_3L_QUANT_V1";

    int feature_count = 0;
    int hidden = 0;
    int layer2 = 0;
    int layer3 = 0;
    if (!(in >> feature_count >> hidden))
    {
        error = "missing quantized model dimensions";
        return false;
    }
    if (two_layer && !(in >> layer2))
    {
        error = "missing 2L quantized model dimensions";
        return false;
    }
    if (three_layer && !(in >> layer3))
    {
        error = "missing 3L quantized model dimensions";
        return false;
    }
    if (feature_count != SIMPLE_768_FEATURE_COUNT ||
        hidden <= 0 ||
        hidden > NEURAL_SIMPLE_MAX_HIDDEN ||
        (two_layer && layer2 <= 0) ||
        layer2 < 0 ||
        layer2 > NEURAL_SIMPLE_MAX_HIDDEN ||
        (three_layer && layer3 <= 0) ||
        layer3 < 0 ||
        layer3 > NEURAL_SIMPLE_MAX_HIDDEN)
    {
        std::ostringstream oss;
        oss << "unsupported quantized model dimensions " << feature_count << " " << hidden << " " << layer2 << " " << layer3;
        error = oss.str();
        return false;
    }

    NeuralQuantModel candidate;
    candidate.hidden_size = hidden;
    candidate.layer2_size = layer2;
    candidate.layer3_size = layer3;
    candidate.w1.resize(static_cast<std::size_t>(SIMPLE_768_FEATURE_COUNT) * hidden);
    candidate.b1.resize(hidden);
    if (three_layer)
    {
        candidate.w2.resize(static_cast<std::size_t>(hidden) * layer2);
        candidate.b2_layer.resize(layer2);
        candidate.w3.resize(static_cast<std::size_t>(layer2) * layer3);
        candidate.b3_layer.resize(layer3);
        candidate.w4.resize(layer3);
    }
    else if (two_layer)
    {
        candidate.w2.resize(static_cast<std::size_t>(hidden) * layer2);
        candidate.b2_layer.resize(layer2);
        candidate.w3.resize(layer2);
    }
    else
    {
        candidate.w2.resize(hidden);
    }

    if (!(in >> token) || token != "output_scale_cp")
    {
        error = "missing output_scale_cp";
        return false;
    }
    if (!(in >> candidate.output_scale_cp) || !std::isfinite(candidate.output_scale_cp) ||
        candidate.output_scale_cp <= 0.0)
    {
        error = "invalid output_scale_cp";
        return false;
    }

    if (!(in >> token) || token != "activation_scale")
    {
        error = "missing activation_scale";
        return false;
    }
    if (!(in >> candidate.activation_scale) || candidate.activation_scale <= 0)
    {
        error = "invalid activation_scale";
        return false;
    }

    if (two_layer)
    {
        if (!(in >> token) || token != "layer2_weight_scale")
        {
            error = "missing layer2_weight_scale";
            return false;
        }
        if (!(in >> candidate.layer2_weight_scale) || candidate.layer2_weight_scale <= 0)
        {
            error = "invalid layer2_weight_scale";
            return false;
        }
    }

    if (three_layer)
    {
        if (!(in >> token) || token != "layer3_weight_scale")
        {
            error = "missing layer3_weight_scale";
            return false;
        }
        if (!(in >> candidate.layer3_weight_scale) || candidate.layer3_weight_scale <= 0)
        {
            error = "invalid layer3_weight_scale";
            return false;
        }
    }

    if (!(in >> token) || token != "output_weight_scale")
    {
        error = "missing output_weight_scale";
        return false;
    }
    if (!(in >> candidate.output_weight_scale) || candidate.output_weight_scale <= 0)
    {
        error = "invalid output_weight_scale";
        return false;
    }

    if (!(in >> token) || token != "w1_i16")
    {
        error = "missing w1_i16";
        return false;
    }
    for (std::int16_t &value : candidate.w1)
    {
        std::int64_t v = 0;
        if (!read_neural_i64(in, v, error))
            return false;
        if (v < -32768 || v > 32767)
        {
            error = "w1_i16 value out of range";
            return false;
        }
        value = static_cast<std::int16_t>(v);
    }

    if (!(in >> token) || token != "b1_i32")
    {
        error = "missing b1_i32";
        return false;
    }
    for (std::int32_t &value : candidate.b1)
    {
        std::int64_t v = 0;
        if (!read_neural_i64(in, v, error))
            return false;
        if (v < static_cast<std::int64_t>(-2147483647) - 1 ||
            v > static_cast<std::int64_t>(2147483647))
        {
            error = "b1_i32 value out of range";
            return false;
        }
        value = static_cast<std::int32_t>(v);
    }

    if (!(in >> token) || token != "w2_i16")
    {
        error = "missing w2_i16";
        return false;
    }
    for (std::int16_t &value : candidate.w2)
    {
        std::int64_t v = 0;
        if (!read_neural_i64(in, v, error))
            return false;
        if (v < -32768 || v > 32767)
        {
            error = "w2_i16 value out of range";
            return false;
        }
        value = static_cast<std::int16_t>(v);
    }

    if (!(in >> token) || token != "b2_i64")
    {
        error = "missing b2_i64";
        return false;
    }
    if (two_layer)
    {
        for (std::int64_t &value : candidate.b2_layer)
        {
            if (!read_neural_i64(in, value, error))
                return false;
        }

        if (!(in >> token) || token != "w3_i16")
        {
            error = "missing w3_i16";
            return false;
        }
        for (std::int16_t &value : candidate.w3)
        {
            std::int64_t v = 0;
            if (!read_neural_i64(in, v, error))
                return false;
            if (v < -32768 || v > 32767)
            {
                error = "w3_i16 value out of range";
                return false;
            }
            value = static_cast<std::int16_t>(v);
        }

        if (!(in >> token) || token != "b3_i64")
        {
            error = "missing b3_i64";
            return false;
        }
        if (three_layer)
        {
            for (std::int64_t &value : candidate.b3_layer)
            {
                if (!read_neural_i64(in, value, error))
                    return false;
            }

            if (!(in >> token) || token != "w4_i16")
            {
                error = "missing w4_i16";
                return false;
            }
            for (std::int16_t &value : candidate.w4)
            {
                std::int64_t v = 0;
                if (!read_neural_i64(in, v, error))
                    return false;
                if (v < -32768 || v > 32767)
                {
                    error = "w4_i16 value out of range";
                    return false;
                }
                value = static_cast<std::int16_t>(v);
            }

            if (!(in >> token) || token != "b4_i64")
            {
                error = "missing b4_i64";
                return false;
            }
            if (!read_neural_i64(in, candidate.b4, error))
                return false;
        }
        else
        {
            if (!read_neural_i64(in, candidate.b3, error))
                return false;
        }
    }
    else
    {
        if (!read_neural_i64(in, candidate.b2, error))
            return false;
    }

    if (in >> token)
    {
        error = "extra tokens after b2_i64";
        return false;
    }

    candidate.loaded = true;
    candidate.path = path;
    neural_quant_model() = std::move(candidate);
    return true;
}

static inline Score neural_quant_output_to_cp(const NeuralQuantModel &model, std::int64_t output_q)
{
    long double denom =
        static_cast<long double>(model.activation_scale) * static_cast<long double>(model.output_weight_scale);
    if (model.layer2_size > 0)
        denom *= static_cast<long double>(model.layer2_weight_scale);
    if (model.layer3_size > 0)
        denom *= static_cast<long double>(model.layer3_weight_scale);
    long double cp = (static_cast<long double>(output_q) * model.output_scale_cp) / denom;
    if (!std::isfinite(static_cast<double>(cp)))
        cp = 0.0L;
    cp = std::max(-static_cast<long double>(NEURAL_SIMPLE_SCORE_CLAMP_CP),
                  std::min(static_cast<long double>(NEURAL_SIMPLE_SCORE_CLAMP_CP), cp));
    return static_cast<Score>(std::llround(cp));
}

static inline std::int64_t neural_quant_output_from_hidden(const NeuralQuantModel &model,
                                                           const std::int32_t *hidden_data)
{
    const int hidden_size = model.hidden_size;
    if (model.layer3_size > 0)
    {
        alignas(NEURAL_CACHELINE_ALIGNMENT) std::array<std::int64_t, NEURAL_SIMPLE_MAX_HIDDEN> hidden2{};
        alignas(NEURAL_CACHELINE_ALIGNMENT) std::array<std::int64_t, NEURAL_SIMPLE_MAX_HIDDEN> hidden3{};
        for (int j = 0; j < model.layer2_size; ++j)
            hidden2[j] = model.b2_layer[j];

        for (int h = 0; h < hidden_size; ++h)
        {
            const std::int32_t acc = hidden_data[h];
            if (acc <= 0)
                continue;
            const std::int16_t *const row = model.w2.data() + static_cast<std::size_t>(h) * model.layer2_size;
            for (int j = 0; j < model.layer2_size; ++j)
                hidden2[j] += static_cast<std::int64_t>(acc) * static_cast<std::int64_t>(row[j]);
        }

        for (int k = 0; k < model.layer3_size; ++k)
            hidden3[k] = model.b3_layer[k];

        for (int j = 0; j < model.layer2_size; ++j)
        {
            const std::int64_t acc = hidden2[j];
            if (acc <= 0)
                continue;
            const std::int16_t *const row = model.w3.data() + static_cast<std::size_t>(j) * model.layer3_size;
            for (int k = 0; k < model.layer3_size; ++k)
                hidden3[k] += acc * static_cast<std::int64_t>(row[k]);
        }

        std::int64_t output_q = model.b4;
        for (int k = 0; k < model.layer3_size; ++k)
        {
            const std::int64_t acc = hidden3[k];
            if (acc > 0)
                output_q += acc * static_cast<std::int64_t>(model.w4[k]);
        }
        return output_q;
    }

    if (model.layer2_size > 0)
    {
        alignas(NEURAL_CACHELINE_ALIGNMENT) std::array<std::int64_t, NEURAL_SIMPLE_MAX_HIDDEN> hidden2{};
        for (int j = 0; j < model.layer2_size; ++j)
            hidden2[j] = model.b2_layer[j];

        for (int h = 0; h < hidden_size; ++h)
        {
            const std::int32_t acc = hidden_data[h];
            if (acc <= 0)
                continue;
            const std::int16_t *const row = model.w2.data() + static_cast<std::size_t>(h) * model.layer2_size;
            for (int j = 0; j < model.layer2_size; ++j)
                hidden2[j] += static_cast<std::int64_t>(acc) * static_cast<std::int64_t>(row[j]);
        }

        std::int64_t output_q = model.b3;
        for (int j = 0; j < model.layer2_size; ++j)
        {
            const std::int64_t acc = hidden2[j];
            if (acc > 0)
                output_q += acc * static_cast<std::int64_t>(model.w3[j]);
        }
        return output_q;
    }

    std::int64_t output_q = model.b2;
    const std::int16_t *const w2_data = model.w2.data();
#if SHAKEYBOT_HAS_AVX512_KERNELS
    if (neural_cpu_supports_avx512())
    {
        output_q += neural_dot_relu_i32_i16_avx512(hidden_data, w2_data, hidden_size);
        return output_q;
    }
#endif
#if SHAKEYBOT_HAS_AVX2_KERNELS
    if (neural_cpu_supports_avx2())
    {
        output_q += neural_dot_relu_i32_i16_avx2(hidden_data, w2_data, hidden_size);
        return output_q;
    }
#endif
    for (int h = 0; h < hidden_size; ++h)
    {
        const std::int32_t acc = hidden_data[h];
        if (acc > 0)
            output_q += static_cast<std::int64_t>(acc) * static_cast<std::int64_t>(w2_data[h]);
    }
    return output_q;
}

static inline Score evaluate_neural_quant_white_pov(const Board &board)
{
    const NeuralQuantModel &model = neural_quant_model();
    if (!model.loaded)
        return 0;

    const Simple768Features features = encode_simple_768_features(board);
    const int hidden_size = model.hidden_size;

    alignas(NEURAL_CACHELINE_ALIGNMENT) std::array<std::int32_t, NEURAL_SIMPLE_MAX_HIDDEN> hidden{};
    std::memcpy(hidden.data(),
                model.b1.data(),
                static_cast<std::size_t>(hidden_size) * sizeof(std::int32_t));
    std::int32_t *const hidden_data = hidden.data();
    const std::int16_t *const w1_data = model.w1.data();
    for (int i = 0; i < features.count; ++i)
    {
        const int feature = features.active[i];
        const std::int16_t *const row = w1_data + static_cast<std::size_t>(feature) * hidden_size;
        neural_quant_add_i16_row_to_i32(hidden_data, row, hidden_size);
    }

    return neural_quant_output_to_cp(model, neural_quant_output_from_hidden(model, hidden_data));
}

static inline bool neural_quant_accumulator_model_matches(const NeuralAccumulator &accum) noexcept
{
    const NeuralQuantModel &model = neural_quant_model();
    return model.loaded &&
           accum.valid &&
           accum.quantized &&
           !accum.halfkp &&
           accum.hidden_size == model.hidden_size &&
           accum.hidden_size > 0 &&
           accum.hidden_size <= NEURAL_SIMPLE_MAX_HIDDEN;
}

static inline bool neural_quant_accumulator_matches_impl(const Board &board,
                                                         const NeuralAccumulator &accum) noexcept
{
    return neural_quant_accumulator_model_matches(accum) &&
           accum.board_hash == board.hash();
}

static inline void neural_quant_accumulator_add_feature(NeuralAccumulator &accum,
                                                        int feature,
                                                        int sign) noexcept
{
    const NeuralQuantModel &model = neural_quant_model();
    const int hidden_size = model.hidden_size;
    const std::int16_t *const row = model.w1.data() + static_cast<std::size_t>(feature) * hidden_size;
    std::int32_t *const dst = accum.quant_pre_activation.data();
    if (sign > 0)
        neural_quant_add_i16_row_to_i32(dst, row, hidden_size);
    else
        neural_quant_sub_i16_row_from_i32(dst, row, hidden_size);
}

static inline void neural_quant_accumulator_add_piece(NeuralAccumulator &accum,
                                                      chess::Piece piece,
                                                      Square square,
                                                      int sign) noexcept
{
    if (piece == chess::Piece::NONE)
        return;
    const int feature = simple_768_feature_index(piece.color(), piece.type(), square);
    neural_quant_accumulator_add_feature(accum, feature, sign);
}

static inline Score evaluate_neural_quant_accumulator_white_pov(const NeuralAccumulator &accum)
{
    const NeuralQuantModel &model = neural_quant_model();
    if (!neural_quant_accumulator_model_matches(accum))
        return 0;

    return neural_quant_output_to_cp(
        model,
        neural_quant_output_from_hidden(model, accum.quant_pre_activation.data()));
}

static inline void refresh_neural_quant_accumulator_impl(const Board &board,
                                                         NeuralAccumulator &accum,
                                                         NeuralAccumulatorStats *stats)
{
    const NeuralQuantModel &model = neural_quant_model();
    if (!model.loaded)
    {
        accum.hidden_size = 0;
        accum.board_hash = 0ULL;
        accum.quantized = false;
        accum.halfkp = false;
        accum.valid = false;
        return;
    }

    accum.hidden_size = model.hidden_size;
    accum.board_hash = 0ULL;
    accum.quantized = true;
    accum.halfkp = false;
    accum.valid = false;
    std::memcpy(accum.quant_pre_activation.data(),
                model.b1.data(),
                static_cast<std::size_t>(model.hidden_size) * sizeof(std::int32_t));

    const Simple768Features features = encode_simple_768_features(board);
    for (int i = 0; i < features.count; ++i)
        neural_quant_accumulator_add_feature(accum, features.active[i], 1);

    accum.board_hash = board.hash();
    accum.valid = true;
    if (stats)
        ++stats->refreshes;
}

static inline void copy_neural_quant_accumulator_impl(const NeuralAccumulator &parent,
                                                      NeuralAccumulator &child) noexcept
{
    const int hidden = parent.hidden_size;
    child.hidden_size = hidden;
    child.board_hash = parent.board_hash;
    child.quantized = true;
    child.halfkp = false;
    child.valid = parent.valid;
    std::memcpy(child.quant_pre_activation.data(),
                parent.quant_pre_activation.data(),
                static_cast<std::size_t>(hidden) * sizeof(std::int32_t));
}

static inline bool update_neural_quant_accumulator_for_move_impl(const NeuralAccumulator &parent,
                                                                 const Board &board,
                                                                 const chess::Move &move,
                                                                 NeuralAccumulator &child,
                                                                 NeuralAccumulatorStats *stats)
{
    const NeuralQuantModel &model = neural_quant_model();
    if (!model.loaded || !neural_quant_accumulator_matches_impl(board, parent))
    {
        child = NeuralAccumulator{};
        return false;
    }

    const chess::Piece moving = board.at(move.from());
    if (moving == chess::Piece::NONE)
    {
        child = NeuralAccumulator{};
        return false;
    }

    copy_neural_quant_accumulator_impl(parent, child);
    child.board_hash = 0ULL;

    const chess::Color us = board.sideToMove();
    switch (move.typeOf())
    {
    case chess::Move::CASTLING:
    {
        const chess::Piece rook = board.at(move.to());
        if (rook == chess::Piece::NONE)
        {
            child = NeuralAccumulator{};
            return false;
        }

        const bool king_side = move.to() > move.from();
        const Square king_to = Square::castling_king_square(king_side, us);
        const Square rook_to = Square::castling_rook_square(king_side, us);

        neural_quant_accumulator_add_piece(child, moving, move.from(), -1);
        neural_quant_accumulator_add_piece(child, rook, move.to(), -1);
        neural_quant_accumulator_add_piece(child, moving, king_to, 1);
        neural_quant_accumulator_add_piece(child, rook, rook_to, 1);
        break;
    }
    case chess::Move::PROMOTION:
    {
        neural_quant_accumulator_add_piece(child, board.at(move.to()), move.to(), -1);
        neural_quant_accumulator_add_piece(child, moving, move.from(), -1);
        neural_quant_accumulator_add_piece(child, chess::Piece(move.promotionType(), us), move.to(), 1);
        break;
    }
    case chess::Move::ENPASSANT:
    {
        neural_quant_accumulator_add_piece(child, moving, move.from(), -1);
        neural_quant_accumulator_add_piece(child, moving, move.to(), 1);
        neural_quant_accumulator_add_piece(child, chess::Piece(chess::PieceType::PAWN, ~us), move.to().ep_square(), -1);
        break;
    }
    default:
    {
        neural_quant_accumulator_add_piece(child, board.at(move.to()), move.to(), -1);
        neural_quant_accumulator_add_piece(child, moving, move.from(), -1);
        neural_quant_accumulator_add_piece(child, moving, move.to(), 1);
        break;
    }
    }

    child.hidden_size = model.hidden_size;
    child.quantized = true;
    child.halfkp = false;
    child.valid = true;
    if (stats)
        ++stats->delta_updates;
    return true;
}

// Evaluation module: stateless HalfKP neural inference
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

constexpr int HALFKP_FEATURE_COUNT = 64 * 64 * 5 * 2;
constexpr int HALFKP_MAX_ACTIVE = 30;
constexpr int HALFKP_PERSPECTIVES = 2;
static_assert(HALFKP_FEATURE_COUNT == 40960);

struct HalfkpFeatures
{
    std::array<std::array<int, HALFKP_MAX_ACTIVE>, HALFKP_PERSPECTIVES> active{};
    std::array<int, HALFKP_PERSPECTIVES> count{};
};

struct HalfkpPerspectiveFeatures
{
    std::array<int, HALFKP_MAX_ACTIVE> active{};
    int count = 0;
};

struct NeuralHalfkpModel
{
    AlignedVector<float> w1;
    AlignedVector<float> b1;
    AlignedVector<float> w2;
    AlignedVector<float> b2_layer;
    AlignedVector<float> w3;
    AlignedVector<float> b3_layer;
    AlignedVector<float> w4;
    float b2 = 0.0f;
    float b3 = 0.0f;
    float b4 = 0.0f;
    float output_scale_cp = NEURAL_SIMPLE_DEFAULT_OUTPUT_SCALE_CP;
    int hidden_size = 0;
    int layer2_size = 0;
    int layer3_size = 0;
    bool loaded = false;
    std::string path;
};

struct NeuralHalfkpQuantModel
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

static inline int halfkp_mirror_square_vertical(int square) noexcept
{
    return square ^ 56;
}

static inline int halfkp_feature_index(int king_square,
                                       int piece_square,
                                       int piece_type,
                                       int relative_color) noexcept
{
    return piece_square + ((piece_type * 2 + relative_color) + king_square * 10) * 64;
}

static inline void halfkp_add_piece_feature(HalfkpFeatures &features,
                                            int perspective,
                                            int king_square,
                                            int piece_square,
                                            int piece_type,
                                            int relative_color) noexcept
{
    int &count = features.count[perspective];
    if (count < HALFKP_MAX_ACTIVE)
        features.active[perspective][count++] =
            halfkp_feature_index(king_square, piece_square, piece_type, relative_color);
}

static inline HalfkpFeatures encode_halfkp_features(const Board &board) noexcept
{
    HalfkpFeatures features{};
    const int white_king = board.kingSq(Color::WHITE).index();
    const int black_king = halfkp_mirror_square_vertical(board.kingSq(Color::BLACK).index());

    for (Color color : {Color::WHITE, Color::BLACK})
    {
        for (int piece_index = 0; piece_index < 5; ++piece_index)
        {
            const PieceType type(static_cast<PieceType::underlying>(piece_index));
            Bitboard pieces = board.pieces(type, color);
            while (pieces)
            {
                const Square square(pieces.pop());
                const int sq = square.index();
                const int relative_to_white = (color == Color::WHITE) ? 0 : 1;
                const int relative_to_black = (color == Color::BLACK) ? 0 : 1;
                halfkp_add_piece_feature(features, 0, white_king, sq, piece_index, relative_to_white);
                halfkp_add_piece_feature(features,
                                         1,
                                         black_king,
                                         halfkp_mirror_square_vertical(sq),
                                         piece_index,
                                         relative_to_black);
            }
        }
    }

    return features;
}

static inline HalfkpPerspectiveFeatures encode_halfkp_features_for_perspective(const Board &board,
                                                                               int perspective) noexcept
{
    HalfkpPerspectiveFeatures features{};
    const int king_square = perspective == 0
                                ? board.kingSq(Color::WHITE).index()
                                : halfkp_mirror_square_vertical(board.kingSq(Color::BLACK).index());

    for (Color color : {Color::WHITE, Color::BLACK})
    {
        for (int piece_index = 0; piece_index < 5; ++piece_index)
        {
            const PieceType type(static_cast<PieceType::underlying>(piece_index));
            Bitboard pieces = board.pieces(type, color);
            while (pieces)
            {
                const Square square(pieces.pop());
                const int sq = perspective == 0 ? square.index() : halfkp_mirror_square_vertical(square.index());
                const int relative_color = perspective == 0
                                               ? ((color == Color::WHITE) ? 0 : 1)
                                               : ((color == Color::BLACK) ? 0 : 1);
                if (features.count < HALFKP_MAX_ACTIVE)
                {
                    features.active[features.count++] =
                        halfkp_feature_index(king_square, sq, piece_index, relative_color);
                }
            }
        }
    }

    return features;
}

NeuralHalfkpModel &neural_halfkp_model() noexcept
{
    static NeuralHalfkpModel model;
    return model;
}

NeuralHalfkpQuantModel &neural_halfkp_quant_model() noexcept
{
    static NeuralHalfkpQuantModel model;
    return model;
}

bool neural_halfkp_model_loaded_impl() noexcept
{
    return neural_halfkp_model().loaded;
}

bool neural_halfkp_quant_model_loaded_impl() noexcept
{
    return neural_halfkp_quant_model().loaded;
}

const std::string &neural_halfkp_model_path_impl() noexcept
{
    return neural_halfkp_model().path;
}

const std::string &neural_halfkp_quant_model_path_impl() noexcept
{
    return neural_halfkp_quant_model().path;
}

void unload_neural_halfkp_model_impl() noexcept
{
    neural_halfkp_model() = NeuralHalfkpModel{};
}

void unload_neural_halfkp_quant_model_impl() noexcept
{
    neural_halfkp_quant_model() = NeuralHalfkpQuantModel{};
}

bool load_neural_halfkp_model_impl(const std::string &path, std::string &error)
{
    if (path.empty())
    {
        unload_neural_halfkp_model_impl();
        return true;
    }

    std::ifstream in(path);
    if (!in)
    {
        error = "could not open HalfKP model file";
        return false;
    }

    std::string token;
    if (!(in >> token) ||
        (token != "SHAKEYBOT_HALFKP_WP_V1" &&
         token != "SHAKEYBOT_HALFKP_WP_2L_V1" &&
         token != "SHAKEYBOT_HALFKP_WP_3L_V1"))
    {
        error = "bad HalfKP model header; expected SHAKEYBOT_HALFKP_WP_V1, SHAKEYBOT_HALFKP_WP_2L_V1, or SHAKEYBOT_HALFKP_WP_3L_V1";
        return false;
    }
    const bool two_layer = token == "SHAKEYBOT_HALFKP_WP_2L_V1" || token == "SHAKEYBOT_HALFKP_WP_3L_V1";
    const bool three_layer = token == "SHAKEYBOT_HALFKP_WP_3L_V1";

    int feature_count = 0;
    int hidden = 0;
    int layer2 = 0;
    int layer3 = 0;
    if (!(in >> feature_count >> hidden))
    {
        error = "missing HalfKP model dimensions";
        return false;
    }
    if (two_layer && !(in >> layer2))
    {
        error = "missing HalfKP 2L model dimensions";
        return false;
    }
    if (three_layer && !(in >> layer3))
    {
        error = "missing HalfKP 3L model dimensions";
        return false;
    }
    if (feature_count != HALFKP_FEATURE_COUNT ||
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
        oss << "unsupported HalfKP model dimensions " << feature_count << " " << hidden << " " << layer2 << " " << layer3;
        error = oss.str();
        return false;
    }

    if (!(in >> token) || token != "output_scale_cp")
    {
        error = "missing output_scale_cp";
        return false;
    }

    NeuralHalfkpModel candidate;
    candidate.hidden_size = hidden;
    candidate.layer2_size = layer2;
    candidate.layer3_size = layer3;
    candidate.w1.resize(static_cast<std::size_t>(HALFKP_FEATURE_COUNT) * hidden);
    candidate.b1.resize(hidden);
    if (three_layer)
    {
        candidate.w2.resize(static_cast<std::size_t>(HALFKP_PERSPECTIVES) * hidden * layer2);
        candidate.b2_layer.resize(layer2);
        candidate.w3.resize(static_cast<std::size_t>(layer2) * layer3);
        candidate.b3_layer.resize(layer3);
        candidate.w4.resize(layer3);
    }
    else if (two_layer)
    {
        candidate.w2.resize(static_cast<std::size_t>(HALFKP_PERSPECTIVES) * hidden * layer2);
        candidate.b2_layer.resize(layer2);
        candidate.w3.resize(layer2);
    }
    else
    {
        candidate.w2.resize(static_cast<std::size_t>(HALFKP_PERSPECTIVES) * hidden);
    }

    if (!read_neural_float(in, candidate.output_scale_cp, error))
        return false;
    if (candidate.output_scale_cp <= 0.0f)
    {
        error = "output_scale_cp must be positive";
        return false;
    }

    if (!(in >> token) || token != "w1")
    {
        error = "missing w1";
        return false;
    }
    for (float &value : candidate.w1)
    {
        if (!read_neural_float(in, value, error))
            return false;
    }

    if (!(in >> token) || token != "b1")
    {
        error = "missing b1";
        return false;
    }
    for (float &value : candidate.b1)
    {
        if (!read_neural_float(in, value, error))
            return false;
    }

    if (!(in >> token) || token != "w2")
    {
        error = "missing w2";
        return false;
    }
    for (float &value : candidate.w2)
    {
        if (!read_neural_float(in, value, error))
            return false;
    }

    if (!(in >> token) || token != "b2")
    {
        error = "missing b2";
        return false;
    }
    if (two_layer)
    {
        for (float &value : candidate.b2_layer)
        {
            if (!read_neural_float(in, value, error))
                return false;
        }

        if (!(in >> token) || token != "w3")
        {
            error = "missing w3";
            return false;
        }
        for (float &value : candidate.w3)
        {
            if (!read_neural_float(in, value, error))
                return false;
        }

        if (!(in >> token) || token != "b3")
        {
            error = "missing b3";
            return false;
        }
        if (three_layer)
        {
            for (float &value : candidate.b3_layer)
            {
                if (!read_neural_float(in, value, error))
                    return false;
            }

            if (!(in >> token) || token != "w4")
            {
                error = "missing w4";
                return false;
            }
            for (float &value : candidate.w4)
            {
                if (!read_neural_float(in, value, error))
                    return false;
            }

            if (!(in >> token) || token != "b4")
            {
                error = "missing b4";
                return false;
            }
            if (!read_neural_float(in, candidate.b4, error))
                return false;
        }
        else
        {
            if (!read_neural_float(in, candidate.b3, error))
                return false;
        }
    }
    else
    {
        if (!read_neural_float(in, candidate.b2, error))
            return false;
    }

    if (in >> token)
    {
        error = "extra tokens after b2";
        return false;
    }

    candidate.loaded = true;
    candidate.path = path;
    neural_halfkp_model() = std::move(candidate);
    return true;
}

bool load_neural_halfkp_quant_model_impl(const std::string &path, std::string &error)
{
    if (path.empty())
    {
        unload_neural_halfkp_quant_model_impl();
        return true;
    }

    std::ifstream in(path);
    if (!in)
    {
        error = "could not open HalfKP quantized model file";
        return false;
    }

    std::string token;
    if (!(in >> token) ||
        (token != "SHAKEYBOT_HALFKP_WP_QUANT_V1" &&
         token != "SHAKEYBOT_HALFKP_WP_2L_QUANT_V1" &&
         token != "SHAKEYBOT_HALFKP_WP_3L_QUANT_V1"))
    {
        error = "bad HalfKP quantized model header; expected SHAKEYBOT_HALFKP_WP_QUANT_V1, SHAKEYBOT_HALFKP_WP_2L_QUANT_V1, or SHAKEYBOT_HALFKP_WP_3L_QUANT_V1";
        return false;
    }
    const bool two_layer = token == "SHAKEYBOT_HALFKP_WP_2L_QUANT_V1" || token == "SHAKEYBOT_HALFKP_WP_3L_QUANT_V1";
    const bool three_layer = token == "SHAKEYBOT_HALFKP_WP_3L_QUANT_V1";

    int feature_count = 0;
    int hidden = 0;
    int layer2 = 0;
    int layer3 = 0;
    if (!(in >> feature_count >> hidden))
    {
        error = "missing HalfKP quantized model dimensions";
        return false;
    }
    if (two_layer && !(in >> layer2))
    {
        error = "missing HalfKP 2L quantized model dimensions";
        return false;
    }
    if (three_layer && !(in >> layer3))
    {
        error = "missing HalfKP 3L quantized model dimensions";
        return false;
    }
    if (feature_count != HALFKP_FEATURE_COUNT ||
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
        oss << "unsupported HalfKP quantized model dimensions " << feature_count << " " << hidden << " " << layer2 << " " << layer3;
        error = oss.str();
        return false;
    }

    NeuralHalfkpQuantModel candidate;
    candidate.hidden_size = hidden;
    candidate.layer2_size = layer2;
    candidate.layer3_size = layer3;
    candidate.w1.resize(static_cast<std::size_t>(HALFKP_FEATURE_COUNT) * hidden);
    candidate.b1.resize(hidden);
    if (three_layer)
    {
        candidate.w2.resize(static_cast<std::size_t>(HALFKP_PERSPECTIVES) * hidden * layer2);
        candidate.b2_layer.resize(layer2);
        candidate.w3.resize(static_cast<std::size_t>(layer2) * layer3);
        candidate.b3_layer.resize(layer3);
        candidate.w4.resize(layer3);
    }
    else if (two_layer)
    {
        candidate.w2.resize(static_cast<std::size_t>(HALFKP_PERSPECTIVES) * hidden * layer2);
        candidate.b2_layer.resize(layer2);
        candidate.w3.resize(layer2);
    }
    else
    {
        candidate.w2.resize(static_cast<std::size_t>(HALFKP_PERSPECTIVES) * hidden);
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
    neural_halfkp_quant_model() = std::move(candidate);
    return true;
}

static inline Score halfkp_float_output_to_cp(float output_norm, float output_scale_cp)
{
    float cp = output_norm * output_scale_cp;
    if (!std::isfinite(cp))
        cp = 0.0f;
    cp = std::max(-static_cast<float>(NEURAL_SIMPLE_SCORE_CLAMP_CP),
                  std::min(static_cast<float>(NEURAL_SIMPLE_SCORE_CLAMP_CP), cp));
    return static_cast<Score>(std::lround(cp));
}

static inline float halfkp_float_output_from_hidden(const NeuralHalfkpModel &model,
                                                    const std::array<std::array<float, NEURAL_SIMPLE_MAX_HIDDEN>, HALFKP_PERSPECTIVES> &hidden)
{
    const int hidden_size = model.hidden_size;
    const int first = 0;
    const int second = 1;
    if (model.layer3_size > 0)
    {
        alignas(NEURAL_CACHELINE_ALIGNMENT) std::array<float, NEURAL_SIMPLE_MAX_HIDDEN> hidden2{};
        alignas(NEURAL_CACHELINE_ALIGNMENT) std::array<float, NEURAL_SIMPLE_MAX_HIDDEN> hidden3{};
        for (int j = 0; j < model.layer2_size; ++j)
            hidden2[j] = model.b2_layer[j];

        for (int h = 0; h < hidden_size; ++h)
        {
            const float a = std::clamp(hidden[first][h], 0.0f, 1.0f);
            const float b = std::clamp(hidden[second][h], 0.0f, 1.0f);
            const float *const row_a = model.w2.data() + static_cast<std::size_t>(h) * model.layer2_size;
            const float *const row_b = model.w2.data() + static_cast<std::size_t>(hidden_size + h) * model.layer2_size;
            for (int j = 0; j < model.layer2_size; ++j)
                hidden2[j] += a * row_a[j] + b * row_b[j];
        }

        for (int k = 0; k < model.layer3_size; ++k)
            hidden3[k] = model.b3_layer[k];

        for (int j = 0; j < model.layer2_size; ++j)
        {
            const float activated = std::clamp(hidden2[j], 0.0f, 1.0f);
            const float *const row = model.w3.data() + static_cast<std::size_t>(j) * model.layer3_size;
            for (int k = 0; k < model.layer3_size; ++k)
                hidden3[k] += activated * row[k];
        }

        float output_norm = model.b4;
        for (int k = 0; k < model.layer3_size; ++k)
            output_norm += std::clamp(hidden3[k], 0.0f, 1.0f) * model.w4[k];
        return output_norm;
    }

    if (model.layer2_size > 0)
    {
        alignas(NEURAL_CACHELINE_ALIGNMENT) std::array<float, NEURAL_SIMPLE_MAX_HIDDEN> hidden2{};
        for (int j = 0; j < model.layer2_size; ++j)
            hidden2[j] = model.b2_layer[j];

        for (int h = 0; h < hidden_size; ++h)
        {
            const float a = std::clamp(hidden[first][h], 0.0f, 1.0f);
            const float b = std::clamp(hidden[second][h], 0.0f, 1.0f);
            const float *const row_a = model.w2.data() + static_cast<std::size_t>(h) * model.layer2_size;
            const float *const row_b = model.w2.data() + static_cast<std::size_t>(hidden_size + h) * model.layer2_size;
            for (int j = 0; j < model.layer2_size; ++j)
                hidden2[j] += a * row_a[j] + b * row_b[j];
        }

        float output_norm = model.b3;
        for (int j = 0; j < model.layer2_size; ++j)
            output_norm += std::clamp(hidden2[j], 0.0f, 1.0f) * model.w3[j];
        return output_norm;
    }

    float output_norm = model.b2;
    for (int h = 0; h < hidden_size; ++h)
    {
        output_norm += std::clamp(hidden[first][h], 0.0f, 1.0f) * model.w2[h];
        output_norm += std::clamp(hidden[second][h], 0.0f, 1.0f) * model.w2[hidden_size + h];
    }
    return output_norm;
}

static inline Score evaluate_neural_halfkp_white_pov(const Board &board)
{
    const NeuralHalfkpModel &model = neural_halfkp_model();
    if (!model.loaded)
        return 0;

    const HalfkpFeatures features = encode_halfkp_features(board);
    const int hidden_size = model.hidden_size;
    alignas(NEURAL_CACHELINE_ALIGNMENT) std::array<std::array<float, NEURAL_SIMPLE_MAX_HIDDEN>, HALFKP_PERSPECTIVES> hidden{};

    for (int p = 0; p < HALFKP_PERSPECTIVES; ++p)
    {
        std::memcpy(hidden[p].data(), model.b1.data(), static_cast<std::size_t>(hidden_size) * sizeof(float));
        for (int i = 0; i < features.count[p]; ++i)
        {
            const int feature = features.active[p][i];
            const float *const row = model.w1.data() + static_cast<std::size_t>(feature) * hidden_size;
            for (int h = 0; h < hidden_size; ++h)
                hidden[p][h] += row[h];
        }
    }

    return halfkp_float_output_to_cp(halfkp_float_output_from_hidden(model, hidden), model.output_scale_cp);
}

static inline Score halfkp_quant_output_to_cp(const NeuralHalfkpQuantModel &model, std::int64_t output_q)
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

static inline void halfkp_quant_accumulate_layer2_from_hidden(
    const NeuralHalfkpQuantModel &model,
    const std::array<std::array<std::int32_t, NEURAL_SIMPLE_MAX_HIDDEN>, HALFKP_PERSPECTIVES> &hidden,
    std::int64_t *hidden2)
{
    const int hidden_size = model.hidden_size;
    const int layer2_size = model.layer2_size;
    const int first = 0;
    const int second = 1;

#if SHAKEYBOT_HAS_AVX2_KERNELS
    if (neural_cpu_supports_avx2() && model.activation_scale <= 32768)
    {
        for (int h = 0; h < hidden_size; ++h)
        {
            const std::int32_t a = std::clamp(hidden[first][h], 0, model.activation_scale);
            const std::int32_t b = std::clamp(hidden[second][h], 0, model.activation_scale);
            if (a == 0 && b == 0)
                continue;
            const std::int16_t *const row_a = model.w2.data() + static_cast<std::size_t>(h) * layer2_size;
            const std::int16_t *const row_b = model.w2.data() + static_cast<std::size_t>(hidden_size + h) * layer2_size;
            neural_accumulate_dual_i32_i16_to_i64_avx2(hidden2, row_a, a, row_b, b, layer2_size);
        }
        return;
    }
#endif

    for (int h = 0; h < hidden_size; ++h)
    {
        const std::int32_t a = std::clamp(hidden[first][h], 0, model.activation_scale);
        const std::int32_t b = std::clamp(hidden[second][h], 0, model.activation_scale);
        const std::int16_t *const row_a = model.w2.data() + static_cast<std::size_t>(h) * layer2_size;
        const std::int16_t *const row_b = model.w2.data() + static_cast<std::size_t>(hidden_size + h) * layer2_size;
        for (int j = 0; j < layer2_size; ++j)
        {
            hidden2[j] += static_cast<std::int64_t>(a) * static_cast<std::int64_t>(row_a[j]);
            hidden2[j] += static_cast<std::int64_t>(b) * static_cast<std::int64_t>(row_b[j]);
        }
    }
}

static inline bool halfkp_quant_layer2_activation_fits_i32(std::int64_t hidden2_scale) noexcept
{
    return hidden2_scale <= static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max());
}

#if SHAKEYBOT_HAS_AVX2_KERNELS
static inline std::int64_t halfkp_quant_layer3_output_from_hidden2_avx2(
    const NeuralHalfkpQuantModel &model,
    const std::int64_t *hidden2,
    std::int64_t hidden2_scale,
    std::int64_t hidden3_scale)
{
    alignas(NEURAL_CACHELINE_ALIGNMENT) std::array<std::int64_t, NEURAL_SIMPLE_MAX_HIDDEN> hidden3_even{};
    alignas(NEURAL_CACHELINE_ALIGNMENT) std::array<std::int64_t, NEURAL_SIMPLE_MAX_HIDDEN> hidden3_odd{};
    for (int k = 0; k < model.layer3_size; ++k)
    {
        if ((k & 1) == 0)
            hidden3_even[k / 2] = model.b3_layer[k];
        else
            hidden3_odd[k / 2] = model.b3_layer[k];
    }

    for (int j = 0; j < model.layer2_size; ++j)
    {
        const std::int64_t acc = std::clamp(hidden2[j], std::int64_t{0}, hidden2_scale);
        if (acc == 0)
            continue;
        const std::int16_t *const row = model.w3.data() + static_cast<std::size_t>(j) * model.layer3_size;
        neural_accumulate_i32_i16_to_i64_evenodd_avx2(
            hidden3_even.data(),
            hidden3_odd.data(),
            row,
            static_cast<std::int32_t>(acc),
            model.layer3_size);
    }

    std::int64_t output_q = model.b4;
    for (int k = 0; k < model.layer3_size; ++k)
    {
        const std::int64_t hidden3_value = ((k & 1) == 0) ? hidden3_even[k / 2] : hidden3_odd[k / 2];
        const std::int64_t acc = std::clamp(hidden3_value, std::int64_t{0}, hidden3_scale);
        output_q += acc * static_cast<std::int64_t>(model.w4[k]);
    }
    return output_q;
}
#endif

static inline std::int64_t halfkp_quant_output_from_hidden(
    const NeuralHalfkpQuantModel &model,
    const std::array<std::array<std::int32_t, NEURAL_SIMPLE_MAX_HIDDEN>, HALFKP_PERSPECTIVES> &hidden)
{
    const int hidden_size = model.hidden_size;
    const int first = 0;
    const int second = 1;
    if (model.layer3_size > 0)
    {
        alignas(NEURAL_CACHELINE_ALIGNMENT) std::array<std::int64_t, NEURAL_SIMPLE_MAX_HIDDEN> hidden2{};
        alignas(NEURAL_CACHELINE_ALIGNMENT) std::array<std::int64_t, NEURAL_SIMPLE_MAX_HIDDEN> hidden3{};
        for (int j = 0; j < model.layer2_size; ++j)
            hidden2[j] = model.b2_layer[j];

        halfkp_quant_accumulate_layer2_from_hidden(model, hidden, hidden2.data());

        const std::int64_t hidden2_scale =
            static_cast<std::int64_t>(model.activation_scale) * static_cast<std::int64_t>(model.layer2_weight_scale);
        const std::int64_t hidden3_scale =
            hidden2_scale * static_cast<std::int64_t>(model.layer3_weight_scale);
#if SHAKEYBOT_HAS_AVX2_KERNELS
        if (neural_cpu_supports_avx2() && halfkp_quant_layer2_activation_fits_i32(hidden2_scale))
            return halfkp_quant_layer3_output_from_hidden2_avx2(model, hidden2.data(), hidden2_scale, hidden3_scale);
#endif

        for (int k = 0; k < model.layer3_size; ++k)
            hidden3[k] = model.b3_layer[k];

        for (int j = 0; j < model.layer2_size; ++j)
        {
            const std::int64_t acc = std::clamp(hidden2[j], std::int64_t{0}, hidden2_scale);
            const std::int16_t *const row = model.w3.data() + static_cast<std::size_t>(j) * model.layer3_size;
            for (int k = 0; k < model.layer3_size; ++k)
                hidden3[k] += acc * static_cast<std::int64_t>(row[k]);
        }

        std::int64_t output_q = model.b4;
        for (int k = 0; k < model.layer3_size; ++k)
        {
            const std::int64_t acc = std::clamp(hidden3[k], std::int64_t{0}, hidden3_scale);
            output_q += acc * static_cast<std::int64_t>(model.w4[k]);
        }
        return output_q;
    }

    if (model.layer2_size > 0)
    {
        alignas(NEURAL_CACHELINE_ALIGNMENT) std::array<std::int64_t, NEURAL_SIMPLE_MAX_HIDDEN> hidden2{};
        for (int j = 0; j < model.layer2_size; ++j)
            hidden2[j] = model.b2_layer[j];

        halfkp_quant_accumulate_layer2_from_hidden(model, hidden, hidden2.data());

        const std::int64_t hidden2_scale =
            static_cast<std::int64_t>(model.activation_scale) * static_cast<std::int64_t>(model.layer2_weight_scale);
        std::int64_t output_q = model.b3;
        for (int j = 0; j < model.layer2_size; ++j)
        {
            const std::int64_t acc = std::clamp(hidden2[j], std::int64_t{0}, hidden2_scale);
            output_q += acc * static_cast<std::int64_t>(model.w3[j]);
        }
        return output_q;
    }

    std::int64_t output_q = model.b2;
#if SHAKEYBOT_HAS_AVX2_KERNELS
    if (neural_cpu_supports_avx2())
    {
        output_q += neural_dot_clamped_i32_i16_avx2(hidden[first].data(),
                                                   model.w2.data(),
                                                   hidden_size,
                                                   model.activation_scale);
        output_q += neural_dot_clamped_i32_i16_avx2(hidden[second].data(),
                                                   model.w2.data() + hidden_size,
                                                   hidden_size,
                                                   model.activation_scale);
        return output_q;
    }
#endif
    for (int h = 0; h < hidden_size; ++h)
    {
        const std::int32_t a = std::clamp(hidden[first][h], 0, model.activation_scale);
        const std::int32_t b = std::clamp(hidden[second][h], 0, model.activation_scale);
        output_q += static_cast<std::int64_t>(a) * static_cast<std::int64_t>(model.w2[h]);
        output_q += static_cast<std::int64_t>(b) * static_cast<std::int64_t>(model.w2[hidden_size + h]);
    }
    return output_q;
}

static inline Score evaluate_neural_halfkp_quant_white_pov(const Board &board)
{
    const NeuralHalfkpQuantModel &model = neural_halfkp_quant_model();
    if (!model.loaded)
        return 0;

    const HalfkpFeatures features = encode_halfkp_features(board);
    const int hidden_size = model.hidden_size;
    alignas(NEURAL_CACHELINE_ALIGNMENT) std::array<std::array<std::int32_t, NEURAL_SIMPLE_MAX_HIDDEN>, HALFKP_PERSPECTIVES> hidden{};

    for (int p = 0; p < HALFKP_PERSPECTIVES; ++p)
    {
        std::memcpy(hidden[p].data(), model.b1.data(), static_cast<std::size_t>(hidden_size) * sizeof(std::int32_t));
        for (int i = 0; i < features.count[p]; ++i)
        {
            const int feature = features.active[p][i];
            const std::int16_t *const row = model.w1.data() + static_cast<std::size_t>(feature) * hidden_size;
            neural_quant_add_i16_row_to_i32(hidden[p].data(), row, hidden_size);
        }
    }

    return halfkp_quant_output_to_cp(model, halfkp_quant_output_from_hidden(model, hidden));
}

static inline bool neural_halfkp_quant_accumulator_model_matches(const NeuralAccumulator &accum) noexcept
{
    const NeuralHalfkpQuantModel &model = neural_halfkp_quant_model();
    return model.loaded &&
           accum.valid &&
           accum.quantized &&
           accum.halfkp &&
           accum.hidden_size == model.hidden_size &&
           accum.hidden_size > 0 &&
           accum.hidden_size <= NEURAL_SIMPLE_MAX_HIDDEN;
}

static inline bool neural_halfkp_quant_accumulator_matches_impl(const Board &board,
                                                                const NeuralAccumulator &accum) noexcept
{
    return neural_halfkp_quant_accumulator_model_matches(accum) &&
           accum.board_hash == board.hash();
}

static inline int halfkp_piece_type_index(chess::Piece piece) noexcept
{
    if (piece == chess::Piece::NONE || piece.type() == chess::PieceType::KING)
        return -1;
    const int piece_type = static_cast<int>(piece.type());
    return piece_type >= 0 && piece_type < 5 ? piece_type : -1;
}

static inline void neural_halfkp_quant_accumulator_add_feature(NeuralAccumulator &accum,
                                                               int perspective,
                                                               int feature,
                                                               int sign) noexcept
{
    const NeuralHalfkpQuantModel &model = neural_halfkp_quant_model();
    const int hidden_size = model.hidden_size;
    const std::int16_t *const row = model.w1.data() + static_cast<std::size_t>(feature) * hidden_size;
    std::int32_t *const dst = accum.halfkp_quant_pre_activation[perspective].data();
    if (sign > 0)
        neural_quant_add_i16_row_to_i32(dst, row, hidden_size);
    else
        neural_quant_sub_i16_row_from_i32(dst, row, hidden_size);
}

static inline int halfkp_perspective_king_square(const Board &board, int perspective) noexcept
{
    if (perspective == 0)
        return board.kingSq(Color::WHITE).index();
    return halfkp_mirror_square_vertical(board.kingSq(Color::BLACK).index());
}

static inline int halfkp_perspective_piece_square(Square square, int perspective) noexcept
{
    const int sq = square.index();
    return perspective == 0 ? sq : halfkp_mirror_square_vertical(sq);
}

static inline int halfkp_perspective_relative_color(chess::Piece piece, int perspective) noexcept
{
    if (perspective == 0)
        return piece.color() == Color::WHITE ? 0 : 1;
    return piece.color() == Color::BLACK ? 0 : 1;
}

static inline void neural_halfkp_quant_accumulator_add_piece_for_perspective(NeuralAccumulator &accum,
                                                                             const Board &board,
                                                                             int perspective,
                                                                             chess::Piece piece,
                                                                             Square square,
                                                                             int sign) noexcept
{
    const int piece_type = halfkp_piece_type_index(piece);
    if (piece_type < 0)
        return;

    const int feature = halfkp_feature_index(halfkp_perspective_king_square(board, perspective),
                                             halfkp_perspective_piece_square(square, perspective),
                                             piece_type,
                                             halfkp_perspective_relative_color(piece, perspective));
    neural_halfkp_quant_accumulator_add_feature(accum, perspective, feature, sign);
}

static inline void neural_halfkp_quant_accumulator_add_piece(NeuralAccumulator &accum,
                                                             const Board &board,
                                                             chess::Piece piece,
                                                             Square square,
                                                             int sign) noexcept
{
    neural_halfkp_quant_accumulator_add_piece_for_perspective(accum, board, 0, piece, square, sign);
    neural_halfkp_quant_accumulator_add_piece_for_perspective(accum, board, 1, piece, square, sign);
}

static inline void refresh_neural_halfkp_quant_accumulator_perspective_impl(const Board &board,
                                                                           NeuralAccumulator &accum,
                                                                           int perspective) noexcept
{
    const NeuralHalfkpQuantModel &model = neural_halfkp_quant_model();
    std::memcpy(accum.halfkp_quant_pre_activation[perspective].data(),
                model.b1.data(),
                static_cast<std::size_t>(model.hidden_size) * sizeof(std::int32_t));

    const HalfkpPerspectiveFeatures features = encode_halfkp_features_for_perspective(board, perspective);
    for (int i = 0; i < features.count; ++i)
        neural_halfkp_quant_accumulator_add_feature(accum, perspective, features.active[i], 1);
}

static inline Score evaluate_neural_halfkp_quant_accumulator_white_pov(const Board &board,
                                                                       const NeuralAccumulator &accum)
{
    (void)board;
    const NeuralHalfkpQuantModel &model = neural_halfkp_quant_model();
    if (!neural_halfkp_quant_accumulator_model_matches(accum))
        return 0;

    return halfkp_quant_output_to_cp(
        model,
        halfkp_quant_output_from_hidden(model, accum.halfkp_quant_pre_activation));
}

static inline void refresh_neural_halfkp_quant_accumulator_impl(const Board &board,
                                                               NeuralAccumulator &accum,
                                                               NeuralAccumulatorStats *stats)
{
    const NeuralHalfkpQuantModel &model = neural_halfkp_quant_model();
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
    accum.halfkp = true;
    accum.valid = false;

    const HalfkpFeatures features = encode_halfkp_features(board);
    for (int perspective = 0; perspective < HALFKP_PERSPECTIVES; ++perspective)
    {
        std::memcpy(accum.halfkp_quant_pre_activation[perspective].data(),
                    model.b1.data(),
                    static_cast<std::size_t>(model.hidden_size) * sizeof(std::int32_t));
        for (int i = 0; i < features.count[perspective]; ++i)
            neural_halfkp_quant_accumulator_add_feature(accum, perspective, features.active[perspective][i], 1);
    }

    accum.board_hash = board.hash();
    accum.valid = true;
    if (stats)
        ++stats->refreshes;
}

static inline void copy_neural_halfkp_quant_accumulator_impl(const NeuralAccumulator &parent,
                                                            NeuralAccumulator &child) noexcept
{
    const int hidden = parent.hidden_size;
    child.hidden_size = hidden;
    child.board_hash = parent.board_hash;
    child.quantized = true;
    child.halfkp = true;
    child.valid = parent.valid;
    for (int p = 0; p < HALFKP_PERSPECTIVES; ++p)
    {
        std::memcpy(child.halfkp_quant_pre_activation[p].data(),
                    parent.halfkp_quant_pre_activation[p].data(),
                    static_cast<std::size_t>(hidden) * sizeof(std::int32_t));
    }
}

static inline bool update_neural_halfkp_quant_accumulator_for_move_impl(const NeuralAccumulator &parent,
                                                                        const Board &board,
                                                                        const chess::Move &move,
                                                                        NeuralAccumulator &child,
                                                                        NeuralAccumulatorStats *stats)
{
    const NeuralHalfkpQuantModel &model = neural_halfkp_quant_model();
    if (!model.loaded || !neural_halfkp_quant_accumulator_matches_impl(board, parent))
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

    copy_neural_halfkp_quant_accumulator_impl(parent, child);
    child.board_hash = 0ULL;

    const chess::Color us = board.sideToMove();
    const int moving_perspective = us == Color::WHITE ? 0 : 1;

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
        const Square rook_to = Square::castling_rook_square(king_side, us);

        Board child_board = board;
        child_board.makeMove(move);
        refresh_neural_halfkp_quant_accumulator_perspective_impl(child_board, child, moving_perspective);

        const int other_perspective = moving_perspective ^ 1;
        neural_halfkp_quant_accumulator_add_piece_for_perspective(child, board, other_perspective, rook, move.to(), -1);
        neural_halfkp_quant_accumulator_add_piece_for_perspective(child, board, other_perspective, rook, rook_to, 1);
        break;
    }
    case chess::Move::PROMOTION:
    {
        neural_halfkp_quant_accumulator_add_piece(child, board, board.at(move.to()), move.to(), -1);
        neural_halfkp_quant_accumulator_add_piece(child, board, moving, move.from(), -1);
        neural_halfkp_quant_accumulator_add_piece(child, board, chess::Piece(move.promotionType(), us), move.to(), 1);
        break;
    }
    case chess::Move::ENPASSANT:
    {
        neural_halfkp_quant_accumulator_add_piece(child, board, moving, move.from(), -1);
        neural_halfkp_quant_accumulator_add_piece(child, board, moving, move.to(), 1);
        neural_halfkp_quant_accumulator_add_piece(child, board, chess::Piece(chess::PieceType::PAWN, ~us), move.to().ep_square(), -1);
        break;
    }
    default:
    {
        if (moving.type() == chess::PieceType::KING)
        {
            Board child_board = board;
            child_board.makeMove(move);
            refresh_neural_halfkp_quant_accumulator_perspective_impl(child_board, child, moving_perspective);

            const int other_perspective = moving_perspective ^ 1;
            neural_halfkp_quant_accumulator_add_piece_for_perspective(
                child, board, other_perspective, board.at(move.to()), move.to(), -1);
        }
        else
        {
            neural_halfkp_quant_accumulator_add_piece(child, board, board.at(move.to()), move.to(), -1);
            neural_halfkp_quant_accumulator_add_piece(child, board, moving, move.from(), -1);
            neural_halfkp_quant_accumulator_add_piece(child, board, moving, move.to(), 1);
        }
        break;
    }
    }

    child.hidden_size = model.hidden_size;
    child.quantized = true;
    child.halfkp = true;
    child.valid = true;
    if (stats)
        ++stats->delta_updates;
    return true;
}

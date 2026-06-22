// Evaluation module: stateless Simple768 neural inference
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

constexpr int NEURAL_SIMPLE_MAX_HIDDEN = 512;
constexpr float NEURAL_SIMPLE_DEFAULT_OUTPUT_SCALE_CP = 1200.0f;
constexpr Score NEURAL_SIMPLE_SCORE_CLAMP_CP = 30000;
constexpr Score NEURAL_ACCUMULATOR_FLOAT_TOLERANCE_CP = 1;
static_assert(NEURAL_SIMPLE_MAX_HIDDEN == NEURAL_ACCUM_MAX_HIDDEN);

struct NeuralSimpleModel
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

NeuralSimpleModel &neural_simple_model() noexcept
{
    static NeuralSimpleModel model;
    return model;
}

bool neural_simple_model_loaded_impl() noexcept
{
    return neural_simple_model().loaded;
}

const std::string &neural_simple_model_path_impl() noexcept
{
    return neural_simple_model().path;
}

void unload_neural_simple_model_impl() noexcept
{
    neural_simple_model() = NeuralSimpleModel{};
}

static bool read_neural_float(std::istream &in, float &value, std::string &error)
{
    if (!(in >> value))
    {
        error = "unexpected end of model file";
        return false;
    }
    if (!std::isfinite(value))
    {
        error = "model contains a non-finite float";
        return false;
    }
    return true;
}

bool load_neural_simple_model_impl(const std::string &path, std::string &error)
{
    if (path.empty())
    {
        unload_neural_simple_model_impl();
        return true;
    }

    std::ifstream in(path);
    if (!in)
    {
        error = "could not open model file";
        return false;
    }

    std::string token;
    if (!(in >> token) ||
        (token != "SHAKEYBOT_SIMPLE768_V1" &&
         token != "SHAKEYBOT_SIMPLE768_2L_V1" &&
         token != "SHAKEYBOT_SIMPLE768_3L_V1"))
    {
        error = "bad model header";
        return false;
    }
    const bool two_layer = token == "SHAKEYBOT_SIMPLE768_2L_V1" || token == "SHAKEYBOT_SIMPLE768_3L_V1";
    const bool three_layer = token == "SHAKEYBOT_SIMPLE768_3L_V1";

    int feature_count = 0;
    int hidden = 0;
    int layer2 = 0;
    int layer3 = 0;
    if (!(in >> feature_count >> hidden))
    {
        error = "missing model dimensions";
        return false;
    }
    if (two_layer && !(in >> layer2))
    {
        error = "missing 2L model dimensions";
        return false;
    }
    if (three_layer && !(in >> layer3))
    {
        error = "missing 3L model dimensions";
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
        oss << "unsupported model dimensions " << feature_count << " " << hidden << " " << layer2 << " " << layer3;
        error = oss.str();
        return false;
    }

    if (!(in >> token) || token != "output_scale_cp")
    {
        error = "missing output_scale_cp";
        return false;
    }

    NeuralSimpleModel candidate;
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
    neural_simple_model() = std::move(candidate);
    return true;
}

static inline Score neural_simple_output_to_cp(float output_norm, float output_scale_cp)
{
    float cp = output_norm * output_scale_cp;
    if (!std::isfinite(cp))
        cp = 0.0f;
    cp = std::max(-static_cast<float>(NEURAL_SIMPLE_SCORE_CLAMP_CP),
                  std::min(static_cast<float>(NEURAL_SIMPLE_SCORE_CLAMP_CP), cp));

    return static_cast<Score>(std::lround(cp));
}

static inline float neural_simple_output_from_hidden(const NeuralSimpleModel &model,
                                                     const float *hidden_data)
{
    const int hidden_size = model.hidden_size;
    if (model.layer3_size > 0)
    {
        alignas(NEURAL_CACHELINE_ALIGNMENT) std::array<float, NEURAL_SIMPLE_MAX_HIDDEN> hidden2{};
        alignas(NEURAL_CACHELINE_ALIGNMENT) std::array<float, NEURAL_SIMPLE_MAX_HIDDEN> hidden3{};
        for (int j = 0; j < model.layer2_size; ++j)
            hidden2[j] = model.b2_layer[j];

        for (int h = 0; h < hidden_size; ++h)
        {
            const float activated = std::max(0.0f, hidden_data[h]);
            if (activated == 0.0f)
                continue;
            const float *const row = model.w2.data() + static_cast<std::size_t>(h) * model.layer2_size;
            for (int j = 0; j < model.layer2_size; ++j)
                hidden2[j] += activated * row[j];
        }

        for (int k = 0; k < model.layer3_size; ++k)
            hidden3[k] = model.b3_layer[k];

        for (int j = 0; j < model.layer2_size; ++j)
        {
            const float activated = std::max(0.0f, hidden2[j]);
            if (activated == 0.0f)
                continue;
            const float *const row = model.w3.data() + static_cast<std::size_t>(j) * model.layer3_size;
            for (int k = 0; k < model.layer3_size; ++k)
                hidden3[k] += activated * row[k];
        }

        float output_norm = model.b4;
        for (int k = 0; k < model.layer3_size; ++k)
            output_norm += std::max(0.0f, hidden3[k]) * model.w4[k];
        return output_norm;
    }

    if (model.layer2_size > 0)
    {
        alignas(NEURAL_CACHELINE_ALIGNMENT) std::array<float, NEURAL_SIMPLE_MAX_HIDDEN> hidden2{};
        for (int j = 0; j < model.layer2_size; ++j)
            hidden2[j] = model.b2_layer[j];

        for (int h = 0; h < hidden_size; ++h)
        {
            const float activated = std::max(0.0f, hidden_data[h]);
            if (activated == 0.0f)
                continue;
            const float *const row = model.w2.data() + static_cast<std::size_t>(h) * model.layer2_size;
            for (int j = 0; j < model.layer2_size; ++j)
                hidden2[j] += activated * row[j];
        }

        float output_norm = model.b3;
        for (int j = 0; j < model.layer2_size; ++j)
            output_norm += std::max(0.0f, hidden2[j]) * model.w3[j];
        return output_norm;
    }

    float output_norm = model.b2;
    const float *const w2_data = model.w2.data();
    for (int h = 0; h < hidden_size; ++h)
        output_norm += std::max(0.0f, hidden_data[h]) * w2_data[h];
    return output_norm;
}

static inline Score evaluate_neural_simple_white_pov(const Board &board)
{
    const NeuralSimpleModel &model = neural_simple_model();
    if (!model.loaded)
        return 0;

    const Simple768Features features = encode_simple_768_features(board);
    const int hidden_size = model.hidden_size;

    alignas(NEURAL_CACHELINE_ALIGNMENT) std::array<float, NEURAL_SIMPLE_MAX_HIDDEN> hidden{};
    std::memcpy(hidden.data(), model.b1.data(), static_cast<std::size_t>(hidden_size) * sizeof(float));
    float *const hidden_data = hidden.data();
    const float *const w1_data = model.w1.data();
    for (int i = 0; i < features.count; ++i)
    {
        const int feature = features.active[i];
        const float *const row = w1_data + static_cast<std::size_t>(feature) * hidden_size;
        for (int h = 0; h < hidden_size; ++h)
            hidden_data[h] += row[h];
    }

    return neural_simple_output_to_cp(neural_simple_output_from_hidden(model, hidden_data), model.output_scale_cp);
}

static inline bool neural_simple_accumulator_model_matches(const NeuralAccumulator &accum) noexcept
{
    const NeuralSimpleModel &model = neural_simple_model();
    return model.loaded &&
           accum.valid &&
           !accum.quantized &&
           !accum.halfkp &&
           accum.hidden_size == model.hidden_size &&
           accum.hidden_size > 0 &&
           accum.hidden_size <= NEURAL_SIMPLE_MAX_HIDDEN;
}

static inline bool neural_simple_accumulator_matches_impl(const Board &board,
                                                          const NeuralAccumulator &accum) noexcept
{
    return neural_simple_accumulator_model_matches(accum) &&
           accum.board_hash == board.hash();
}

static inline void neural_simple_accumulator_add_feature(NeuralAccumulator &accum,
                                                         int feature,
                                                         float sign) noexcept
{
    const NeuralSimpleModel &model = neural_simple_model();
    const int hidden_size = model.hidden_size;
    const float *const row = model.w1.data() + static_cast<std::size_t>(feature) * hidden_size;
    float *const dst = accum.pre_activation.data();
    if (sign > 0.0f)
    {
        for (int h = 0; h < hidden_size; ++h)
            dst[h] += row[h];
    }
    else
    {
        for (int h = 0; h < hidden_size; ++h)
            dst[h] -= row[h];
    }
}

static inline void neural_simple_accumulator_add_piece(NeuralAccumulator &accum,
                                                       chess::Piece piece,
                                                       Square square,
                                                       float sign) noexcept
{
    if (piece == chess::Piece::NONE)
        return;
    const int feature = simple_768_feature_index(piece.color(), piece.type(), square);
    neural_simple_accumulator_add_feature(accum, feature, sign);
}

static inline Score evaluate_neural_simple_accumulator_white_pov(const NeuralAccumulator &accum)
{
    const NeuralSimpleModel &model = neural_simple_model();
    if (!neural_simple_accumulator_model_matches(accum))
        return 0;

    return neural_simple_output_to_cp(
        neural_simple_output_from_hidden(model, accum.pre_activation.data()),
        model.output_scale_cp);
}

static inline void refresh_neural_simple_accumulator_impl(const Board &board,
                                                          NeuralAccumulator &accum,
                                                          NeuralAccumulatorStats *stats)
{
    const NeuralSimpleModel &model = neural_simple_model();
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
    accum.quantized = false;
    accum.halfkp = false;
    accum.valid = false;
    std::memcpy(accum.pre_activation.data(),
                model.b1.data(),
                static_cast<std::size_t>(model.hidden_size) * sizeof(float));

    const Simple768Features features = encode_simple_768_features(board);
    for (int i = 0; i < features.count; ++i)
        neural_simple_accumulator_add_feature(accum, features.active[i], 1.0f);

    accum.board_hash = board.hash();
    accum.valid = true;
    if (stats)
        ++stats->refreshes;
}

static inline void copy_neural_simple_accumulator_impl(const NeuralAccumulator &parent,
                                                       NeuralAccumulator &child) noexcept
{
    const int hidden = parent.hidden_size;
    child.hidden_size = hidden;
    child.board_hash = parent.board_hash;
    child.quantized = false;
    child.halfkp = false;
    child.valid = parent.valid;
    std::memcpy(child.pre_activation.data(),
                parent.pre_activation.data(),
                static_cast<std::size_t>(hidden) * sizeof(float));
}

static inline bool update_neural_simple_accumulator_for_move_impl(const NeuralAccumulator &parent,
                                                                  const Board &board,
                                                                  const chess::Move &move,
                                                                  NeuralAccumulator &child,
                                                                  NeuralAccumulatorStats *stats)
{
    const NeuralSimpleModel &model = neural_simple_model();
    if (!model.loaded || !neural_simple_accumulator_matches_impl(board, parent))
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

    copy_neural_simple_accumulator_impl(parent, child);
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

        neural_simple_accumulator_add_piece(child, moving, move.from(), -1.0f);
        neural_simple_accumulator_add_piece(child, rook, move.to(), -1.0f);
        neural_simple_accumulator_add_piece(child, moving, king_to, 1.0f);
        neural_simple_accumulator_add_piece(child, rook, rook_to, 1.0f);
        break;
    }
    case chess::Move::PROMOTION:
    {
        neural_simple_accumulator_add_piece(child, board.at(move.to()), move.to(), -1.0f);
        neural_simple_accumulator_add_piece(child, moving, move.from(), -1.0f);
        neural_simple_accumulator_add_piece(child, chess::Piece(move.promotionType(), us), move.to(), 1.0f);
        break;
    }
    case chess::Move::ENPASSANT:
    {
        neural_simple_accumulator_add_piece(child, moving, move.from(), -1.0f);
        neural_simple_accumulator_add_piece(child, moving, move.to(), 1.0f);
        neural_simple_accumulator_add_piece(child, chess::Piece(chess::PieceType::PAWN, ~us), move.to().ep_square(), -1.0f);
        break;
    }
    default:
    {
        neural_simple_accumulator_add_piece(child, board.at(move.to()), move.to(), -1.0f);
        neural_simple_accumulator_add_piece(child, moving, move.from(), -1.0f);
        neural_simple_accumulator_add_piece(child, moving, move.to(), 1.0f);
        break;
    }
    }

    child.hidden_size = model.hidden_size;
    child.valid = true;
    if (stats)
        ++stats->delta_updates;
    return true;
}

// Time/stop checks are on the hot path. A slightly larger interval improves nps.
// (Hard deadline is still enforced; this only changes the check cadence.)
constexpr std::uint64_t TIME_CHECK_MASK = 4095ULL; // 4096-nodes interval
inline bool hard_stop(SearchControl *control, SearchStats &stats)
{
    if (!control)
        return false;
    if (control->external_stop &&
        control->external_stop->load(std::memory_order_relaxed))
    {
        stats.stopped = true;
        return true;
    }
    if (control->time_enabled)
    {
        const auto now = std::chrono::steady_clock::now();
        if (now >= control->hard_deadline)
        {
            stats.stopped = true;
            return true;
        }
    }
    return false;
}
inline Score cfg_pawns_to_cp(double pawns)
{
    return static_cast<Score>(std::llround(pawns * 100.0));
}
inline Score mul_div_round(Score s, int num, int den)
{
    const std::int64_t v = static_cast<std::int64_t>(s) * static_cast<std::int64_t>(num);
    if (v >= 0)
        return static_cast<Score>((v + den / 2) / den);
    return static_cast<Score>((v - den / 2) / den);
}
inline bool periodic_hard_stop(SearchControl *control, SearchStats &stats)
{
    if (!control)
        return false;
    if ((stats.nodes & TIME_CHECK_MASK) != 0ULL)
        return false;
    return hard_stop(control, stats);
}
constexpr Score INF = SCORE_INF;    // comfortably larger than any eval in centipawns
constexpr int TWO_FOLD_REP_NUM = 3; // penalize 2-fold repetition (0..1)
constexpr int TWO_FOLD_REP_DEN = 4;
// Shallow forward pruning tuning (pawns, because the engine’s evaluation is pawn-scaled).
// Small increases here can noticeably improve depth-per-second without a large tactical risk,
// as long as they only apply to *late quiet non-checking moves*.
constexpr Score FUTILITY_MARGIN_D1 = 55; // centipawns
// Late move pruning (LMP): keep conservative at depth 2–3 only.
[[maybe_unused]] constexpr int LMP_LIMIT_D2 = 10; // 10 stable baseline
[[maybe_unused]] constexpr int LMP_LIMIT_D3 = 16; // 16 stable baseline
// Move-count pruning (MCP): history-aware LMP for null-window late quiet moves.
// limit = base + scale * depth^2 (depth measured in plies).
constexpr int MCP_MAX_DEPTH = 10;  // 10
constexpr int MCP_BASE = 4;        // 4
constexpr int MCP_SCALE = 1;       // 1
constexpr int MCP_DIV = 2;         // 2
constexpr int MCP_GOOD_HIST = 900; // 900 protect strong history moves
// Multi-ply continuation history: scale the *additional* lookbacks (ply-2..ply-6)
// used in move ordering and history-aware pruning/LMR. 100 = baseline, 0 disables the extra terms.
constexpr int MULTI_CONT_SCALE_PCT = 100; // try 0 (off) .. 200 (aggressive)
// Razoring is controlled via EngineConfig (UCI): config.use_razoring, config.razor_margin_d2/d3.
// Extensions (selective): keep narrow and depth-bounded to avoid node explosions.
//  - Check extension: +1 ply for credible checking moves in the main search.
//  - One-reply extension: +1 ply when the side to move has exactly one legal reply.
constexpr int CHECK_EXT_MIN_DEPTH = 2;
constexpr int CHECK_EXT_MAX_DEPTH = 12;
constexpr int CHECK_EXT_MAX_PLY = 64;
// Extend quiet checks always. For checking captures, extend only if the capture is not too losing.
// (Thresholds around -50 cp tested better than more conservative gates in your recent runs.)
constexpr int CHECK_EXT_MIN_SEE_CP = -50;
constexpr int ONE_REPLY_EXT_MIN_DEPTH = 1;
constexpr int ONE_REPLY_EXT_MAX_DEPTH = 12;
constexpr int ONE_REPLY_EXT_MAX_PLY = 64;
// Singular extensions (hash move): conservative.
// Only consider at relatively deep PV nodes with a confident TT entry.
// Tune knob: lower = more aggressive (more extensions), higher = more conservative.
constexpr int SINGULAR_EXT_MIN_DEPTH = 10; // try 6 (aggressive) .. 10 (very conservative)
// Late move pruning (LMP): forward-prune late quiets in null-window nodes.
constexpr bool ENABLE_LMP = true;
// History pruning: forward-prune late quiets with very poor history/continuation.
constexpr bool ENABLE_HISTORY_PRUNING = true;
// Extended futility pruning for quiets at depths 2-4 (null-window nodes).
constexpr bool ENABLE_EXT_FUTILITY = true;
// Move-count pruning (MCP): master feature toggle for MCP gates in search.
constexpr bool ENABLE_MCP = true;
// Biased draw score from side-to-move POV (contempt).
// Static eval from side-to-move POV, WITHOUT game-over / draw detection.
inline Score eval_stm_no_game_over(const chess::Board &board,
                                   const EngineConfig &config)
{
    const Score white_pov = evaluate_white_pov_with_config(board, config);
    return (board.sideToMove() == chess::Color::WHITE) ? white_pov : -white_pov;
}
inline Score draw_score_stm(const chess::Board &board,
                            const EngineConfig &config)
{
    const Score stm_eval = eval_stm_no_game_over(board, config);
    if (std::abs(stm_eval) < cfg_pawns_to_cp(config.draw_contempt_threshold))
        return 0;
    const Score contempt_cap = cfg_pawns_to_cp(config.draw_contempt_max);
    const Score contempt = std::min<Score>(
        static_cast<Score>(std::llround(static_cast<double>(std::abs(stm_eval)) * config.draw_contempt_scale)),
        contempt_cap);
    Score score = 0;
    if (stm_eval > 0)
        score = -contempt; // winning side dislikes draws
    else if (stm_eval < 0)
        score = +contempt; // losing side likes draws a bit
    if (config.draw_noise > 0.0)
    {
        // Deterministic small noise derived from position hash; helps avoid repetition traps.
        // For fully deterministic testing, keep draw_noise = 0.
        const std::uint64_t x = (board.hash() ^ 0x9E3779B97F4A7C15ULL);
        const int noise = static_cast<int>(x & 0xFFu) - 128; // [-128,127]
        const Score noise_cp = cfg_pawns_to_cp(config.draw_noise);
        score += static_cast<Score>((static_cast<std::int64_t>(noise_cp) * noise) / 128);
    }
    return score;
}
// Convert a search score at a given ply into a TT-storable score.
inline Score to_tt_score(Score score, int ply)
{
    if (score > MATE_BOUND)
        return score + ply; // winning mate
    if (score < -MATE_BOUND)
        return score - ply; // Losing mate: same idea in the other direction.
    return score;
}
// Convert a TT score back into a local score at the current ply.
inline Score from_tt_score(Score score, int ply)
{
    if (score > MATE_BOUND)
        return score - ply;
    if (score < -MATE_BOUND)
        return score + ply;
    return score;
}
inline Score corr_hist_probe(const chess::Board &b, const EngineConfig &cfg) noexcept;
inline bool tt_has_static_eval(const TTEntry &entry) noexcept
{
    return entry.static_eval != TT_NO_STATIC_EVAL;
}
inline Score corrected_static_eval_from_raw(Score raw_eval,
                                            const chess::Board &board,
                                            const EngineConfig &cfg) noexcept
{
    Score e = raw_eval + corr_hist_probe(board, cfg);

    if (e >= MATE_BOUND)
        e = MATE_BOUND - ONE_CP;
    else if (e <= -MATE_BOUND)
        e = -MATE_BOUND + ONE_CP;

    return e;
}
// Clamp alpha/beta to mate bounds at this ply (mate-distance pruning).
// This reduces nodes in forced-mate regions and improves mate conversion/defense.
inline void clamp_mate_window(Score &alpha, Score &beta, int ply)
{
    alpha = std::max(alpha, static_cast<Score>(-MATE_SCORE + ply));
    beta = std::min(beta, static_cast<Score>(MATE_SCORE - ply));
}
constexpr int NUM_ORDER_PT = 6;
constexpr int SEARCH_ALL_IDX = NUM_ORDER_PT;
constexpr int PIECE_ORDER_VALUE[NUM_ORDER_PT] = {
    100,  // PAWN
    320,  // KNIGHT
    330,  // BISHOP
    500,  // ROOK
    900,  // QUEEN
    20000 // KING
};
constexpr int MAX_PLY = 246;
// Singular extension verification needs to search the current node with the TT move excluded.
// We implement this via a thread_local excluded move that is only active at a specific ply
// (so it affects only that node, not its descendants).
thread_local chess::Move g_excluded_move = chess::Move::NO_MOVE;
thread_local int g_excluded_move_ply = -1;
struct ScopedExcludedMove
{
    chess::Move prev_move;
    int prev_ply;
    ScopedExcludedMove(chess::Move m, int ply)
        : prev_move(g_excluded_move), prev_ply(g_excluded_move_ply)
    {
        g_excluded_move = m;
        g_excluded_move_ply = ply;
    }
    ~ScopedExcludedMove()
    {
        g_excluded_move = prev_move;
        g_excluded_move_ply = prev_ply;
    }
};
static int history_heur[2][64][64];
static int cont_history[2][64][64][64];
static int capture_history[2][NUM_ORDER_PT][64][NUM_ORDER_PT];
constexpr int PAWN_HISTORY_SIZE = 8192;
static int pawn_history[2][PAWN_HISTORY_SIZE][NUM_ORDER_PT][64];
struct SearchStackEntry
{
    struct AttackCache
    {
        chess::Bitboard by[2][NUM_ORDER_PT + 1] = {};
        chess::Bitboard by2[2] = {};
    };
    chess::Move move = chess::Move::NO_MOVE;
    int moved_pt = -1;
    int parent_move_count = 0;
    bool prior_capture = false;
    bool tt_hit = false;
    bool in_check = false;
    Score static_eval = 0;
    std::uint64_t attack_cache_key = 0ULL;
    bool attack_cache_ready = false;
    AttackCache attack_cache{};
};
using SearchStack = std::array<SearchStackEntry, MAX_PLY + 1>;
static SearchStack search_stack{};
static chess::Move killer_moves[2][MAX_PLY + 1];
static chess::Move counter_moves[2][64][64]; // [stm][prev_from][prev_to] -> quiet refutation move
                                             // [0]=primary, [1]=secondary (quiet beta-cutoff moves)
// LMR reduction tables (fixed-point scale 1024).
static int lmr_reductions[MAX_PLY + 2];
static std::once_flag lmr_tables_once;
inline void init_lmr_tables_once()
{
    std::call_once(lmr_tables_once, []()
                   {
                lmr_reductions[0] = 0;
                for (int i = 1; i <= MAX_PLY + 1; ++i)
                {
                    //reductions[i] = int(2954 / 128.0 * log(i))
                    lmr_reductions[i] = static_cast<int>((2955.0 / 128.0) * std::log(static_cast<double>(i)));
                } });
}

inline int sf_lmr_reduction_fp(bool improving, int depth, int moveCount, int delta) noexcept
{
    // reductionScale = reductions[depth] * reductions[moveCount]
    const int d = std::clamp(depth, 0, MAX_PLY + 1);
    const int mn = std::clamp(moveCount, 1, MAX_PLY + 1);
    const int reductionScale = lmr_reductions[d] * lmr_reductions[mn];
    // Base reduction grows with depth and move index.
    // Use the *local* node window only: cut nodes can reduce a bit more, while wider
    // PV-ish windows should reduce less. Avoid coupling every node to the root aspiration width.
    int r = reductionScale;
    if (!improving)
        r += (reductionScale * 191) / 512;
    const int local_delta = std::clamp(delta, 1, 64);
    if (local_delta <= ONE_CP)
        r += 192; // null-window / cut nodes: slightly more aggressive
    else
        r -= std::min(384, (local_delta - 1) * 8); // wider windows: back off reductions
    r += 1087;
    return r;
}

inline int lmr_adjusted_fp(bool improving, int depth, int moveCount, int delta, int statScore, bool ttPvNode, double baseMult, double slopeMult)
{
    // Fixed-point LMR:
    //   r = reduction(improving, depth, moveCount, delta)
    //   r -= 32 * moveCount
    //   if (ttPv) reduce less
    //   r -= statScore * coeff(depth) / 16384
    int r = sf_lmr_reduction_fp(improving, depth, moveCount, delta);
    // Tunable move-index slope: scale the per-move reduction term.
    // Base coefficient is 32 (fixed-point), which corresponds to ~0.031 ply per move after /1024.
    const int slope_coeff = std::clamp(static_cast<int>(std::llround(32.0 * slopeMult)), 0, 256);
    r -= slope_coeff * moveCount;

    // Tunable intercept: apply an additive fixed-point offset to the reduction.
    // We scale by 0.5 ply per 1.0x delta so 0..3 maps to [-0.5,+1.0] plies.
    const int base_delta_fp = static_cast<int>(std::llround((baseMult - 1.0) * 512.0));
    r += base_delta_fp;
    if (ttPvNode)
        r -= 512; // PV nodes should reduce less, not more.

    // Depth-dependent symmetric clamp and smooth coefficient:
    //   clamp(statScore) to +/- (256 * depth)
    //   coeff(depth): blends from 1100 at depth 0 to 1582 at depth >= 16
    const int d = std::max(0, depth);
    const int ss = 256 * d;
    const int stat_clamped = std::clamp(statScore, -ss, ss);
    const int d16 = std::clamp(d, 0, 16);
    const int coeff = 1100 + (1582 - 1100) * d16 / 16;

    r -= (stat_clamped * coeff) / 16384;
    if (r < 0)
        r = 0;
    return r;
}


inline int sf_futility_move_count(bool improving, int depth) noexcept
{
    // Movecount = (3 + depth*depth) / (2 - improving)
    const int d = std::max(0, depth);
    const int denom = 2 - (improving ? 1 : 0);
    return (3 + d * d) / denom;
}

inline Score child_futility_margin(int depth, bool improving) noexcept
{
    // 223 * (d - improving)
    // "improving" reduces the margin (more aggressive pruning when the position is improving).
    const int d = std::max(0, depth) - (improving ? 1 : 0);
    return static_cast<Score>(223 * std::max(0, d));
}

static int cont_history_pc[2][NUM_ORDER_PT][64][NUM_ORDER_PT][64]; // [stm][prevPt][prevTo][curPt][curTo]
// Plain continuation table kept on disk for future experiments, but currently unplugged.
constexpr bool ENABLE_PLAIN_CONT_HISTORY = false;
// Correction history (centipawns): learned adjustment for static eval used by pruning gates.
// Indexed by (side-to-move, pawn-key). Reset on ucinewgame for deterministic testing.
constexpr int CORR_HIST_SIZE = 16384; // power of two
static std::int16_t corr_hist[2][CORR_HIST_SIZE];
// History tables are used directly as ordering scores. Keep them bounded to avoid runaway values.
//   - Main (butterfly) history: 7183
//   - Capture history:         10692
//   - Continuation history:    30000
//   - Pawn history:            8192
// This keeps main history from overpowering continuation in quiet move ordering.
constexpr int MAIN_HISTORY_CAP = 7183;
constexpr int MAIN_HISTORY_MIN = -MAIN_HISTORY_CAP;
constexpr int CONT_HISTORY_CAP = 30'000;
constexpr int CONT_HISTORY_MIN = -CONT_HISTORY_CAP;
constexpr int CAPTURE_HISTORY_CAP = 10'692;
constexpr int CAPTURE_HISTORY_MIN = -CAPTURE_HISTORY_CAP;
constexpr int PAWN_HISTORY_CAP = 8192;
constexpr int PAWN_HISTORY_MIN = -PAWN_HISTORY_CAP;
// Max searched quiet moves retained for end-of-node history malus updates.
constexpr int MAX_QUIET_TRIED = 32;
// Capture-history contribution in move ordering.
constexpr int CAPTURE_HISTORY_SCORE_SHIFT_MAIN = 0;
constexpr int CAPTURE_HISTORY_SCORE_SHIFT_QS = 0;
// Razoring is intended as a shallow, middlegame pruning.
// In low-material / endgame phases it often adds nodes without producing cutoffs.
inline int popcount_u64(std::uint64_t x) noexcept
{
    int c = 0;
    while (x)
    {
        x &= (x - 1);
        ++c;
    }
    return c;
}
inline int razoring_phase24(const chess::Board &board) noexcept
{
    auto cnt = [&](chess::PieceType pt, chess::Color c) -> int
    {
        return popcount_u64(board.pieces(pt, c).getBits());
    };
    const int q = cnt(chess::PieceType::QUEEN, chess::Color::WHITE) + cnt(chess::PieceType::QUEEN, chess::Color::BLACK);
    const int r = cnt(chess::PieceType::ROOK, chess::Color::WHITE) + cnt(chess::PieceType::ROOK, chess::Color::BLACK);
    const int b = cnt(chess::PieceType::BISHOP, chess::Color::WHITE) + cnt(chess::PieceType::BISHOP, chess::Color::BLACK);
    const int n = cnt(chess::PieceType::KNIGHT, chess::Color::WHITE) + cnt(chess::PieceType::KNIGHT, chess::Color::BLACK);
    return 4 * q + 2 * r + b + n; // 0..24 (startpos = 24)
}
inline bool allow_razoring_by_material(const chess::Board &board) noexcept
{
    constexpr int RAZOR_MIN_PHASE24 = 9; // disable razoring when phase <= 8 (late/endgame)
    return razoring_phase24(board) >= RAZOR_MIN_PHASE24;
}
// Conservative razoring guardrails
// -------------------------------
// Razoring is intentionally shallow. In tactically volatile positions (exposed kings,
// advanced pawns near promotion, etc.) the razor margin can mis-evaluate and cause
// "random" blunders at fixed time controls.
//
// We follow Donna's principle (avoid sharp positions) using cheap, local features we
// already have available in search: pawn locations and a lightweight king-pawn-shield proxy.
constexpr uint64_t WHITE_ADVANCED_PAWNS_MASK = 0x00FFFF0000000000ULL; // ranks 6-7
constexpr uint64_t BLACK_ADVANCED_PAWNS_MASK = 0x0000000000FFFF00ULL; // ranks 2-3
inline int king_pawn_shield_count(const chess::Board &board, chess::Color c) noexcept
{
    // Count friendly pawns on the 3 files around the king, 1-2 ranks "in front" of it.
    // This is a proxy for king exposure (not a full king-safety evaluation).
    const chess::Square ksq = board.kingSq(c);
    const int kf = static_cast<int>(ksq.file());
    const int kr = static_cast<int>(ksq.rank());
    const int dir = (c == chess::Color::WHITE) ? 1 : -1;
    chess::Bitboard mask{};
    for (int df = -1; df <= 1; ++df)
    {
        const int f = kf + df;
        if (f < 0 || f > 7)
            continue;
        for (int d = 1; d <= 2; ++d)
        {
            const int r = kr + dir * d;
            if (r < 0 || r > 7)
                continue;
            mask |= chess::Bitboard::fromSquare(
                chess::Square(static_cast<chess::File>(f), static_cast<chess::Rank>(r)));
        }
    }
    const chess::Bitboard pawns = board.pieces(chess::PieceType::PAWN, c);
    return (pawns & mask).count();
}
inline bool tactically_volatile_for_razoring(const chess::Board &board) noexcept
{
    // 1) Advanced pawns near promotion tend to create forcing tactics.
    const chess::Bitboard wp = board.pieces(chess::PieceType::PAWN, chess::Color::WHITE);
    const chess::Bitboard bp = board.pieces(chess::PieceType::PAWN, chess::Color::BLACK);
    if ((wp & WHITE_ADVANCED_PAWNS_MASK) || (bp & BLACK_ADVANCED_PAWNS_MASK))
        return true;
    // 2) With queens on board, exposed kings are the primary source of shallow tactical swings.
    const bool queens_present =
        !board.pieces(chess::PieceType::QUEEN, chess::Color::WHITE).empty() ||
        !board.pieces(chess::PieceType::QUEEN, chess::Color::BLACK).empty();
    if (queens_present)
    {
        const int w_shield = king_pawn_shield_count(board, chess::Color::WHITE);
        const int b_shield = king_pawn_shield_count(board, chess::Color::BLACK);
        // Threshold is intentionally conservative: if either king has <= 1 shield pawn,
        // avoid razoring.
        if (w_shield <= 1 || b_shield <= 1)
            return true;
    }
    return false;
}
inline void update_history_entry(int &v, int delta, int cap, int minv) noexcept
{
    // Saturating history update: v <- v + delta - v*|delta|/cap, with clamps.
    // This yields fast learning early, and asymptotically saturates near +/-cap.
    if (delta > cap)
        delta = cap;
    else if (delta < minv)
        delta = minv;
    const int absd = std::abs(delta);
    const std::int64_t vv = static_cast<std::int64_t>(v);
    const std::int64_t adj = (vv * static_cast<std::int64_t>(absd)) / cap;
    std::int64_t nv = vv + static_cast<std::int64_t>(delta) - adj;
    if (nv > cap)
        nv = cap;
    else if (nv < minv)
        nv = minv;
    v = static_cast<int>(nv);
}
inline void update_main_history_entry(int &v, int delta) noexcept
{
    update_history_entry(v, delta, MAIN_HISTORY_CAP, MAIN_HISTORY_MIN);
}
inline void update_cont_history_entry(int &v, int delta) noexcept
{
    update_history_entry(v, delta, CONT_HISTORY_CAP, CONT_HISTORY_MIN);
}
inline void update_capture_history_entry(int &v, int delta) noexcept
{
    update_history_entry(v, delta, CAPTURE_HISTORY_CAP, CAPTURE_HISTORY_MIN);
}
inline void update_pawn_history_entry(int &v, int delta) noexcept
{
    update_history_entry(v, delta, PAWN_HISTORY_CAP, PAWN_HISTORY_MIN);
}
// Helpers:
// These are intentionally board-independent: they operate on precomputed indices,
// so future refutation/TT learning hooks can update continuation history without
// relying on pieces still being present on the board squares.
[[maybe_unused]] inline void update_continuation_no_board(int stm,
                                                          int prev_pt, int prev_to,
                                                          int cur_pt, int cur_from, int cur_to,
                                                          int delta) noexcept
{
    if (stm < 0 || stm > 1)
        return;
    if (prev_to < 0 || prev_to >= 64 || cur_from < 0 || cur_from >= 64 || cur_to < 0 || cur_to >= 64)
        return;
    if constexpr (ENABLE_PLAIN_CONT_HISTORY)
        update_cont_history_entry(cont_history[stm][prev_to][cur_from][cur_to], delta);
    if (0 <= prev_pt && prev_pt < NUM_ORDER_PT && 0 <= cur_pt && cur_pt < NUM_ORDER_PT)
        update_cont_history_entry(cont_history_pc[stm][prev_pt][prev_to][cur_pt][cur_to], delta);
}
[[maybe_unused]] inline bool capture_like_for_stack(const chess::Board &board, const chess::Move &m) noexcept
{
    return board.isCapture(m) || (m.typeOf() == chess::Move::ENPASSANT);
}
[[maybe_unused]] inline int search_color_index(chess::Color c) noexcept
{
    return (c == chess::Color::WHITE) ? 0 : 1;
}
[[maybe_unused]] inline chess::Bitboard search_pawn_attacks_bb(chess::Color c, chess::Bitboard pawns) noexcept
{
    using namespace chess;
    if (c == Color::WHITE)
        return attacks::shift<Direction::NORTH_WEST>(pawns) | attacks::shift<Direction::NORTH_EAST>(pawns);
    return attacks::shift<Direction::SOUTH_WEST>(pawns) | attacks::shift<Direction::SOUTH_EAST>(pawns);
}
[[maybe_unused]] inline chess::Bitboard search_pawn_double_attacks_bb(chess::Color c, chess::Bitboard pawns) noexcept
{
    using namespace chess;
    if (c == Color::WHITE)
    {
        const Bitboard a = attacks::shift<Direction::NORTH_WEST>(pawns);
        const Bitboard b = attacks::shift<Direction::NORTH_EAST>(pawns);
        return a & b;
    }
    const Bitboard a = attacks::shift<Direction::SOUTH_WEST>(pawns);
    const Bitboard b = attacks::shift<Direction::SOUTH_EAST>(pawns);
    return a & b;
}
inline void clear_search_stack_entry(int ply) noexcept
{
    if (ply < 0 || ply > MAX_PLY)
        return;
    search_stack[ply] = SearchStackEntry{};
}
inline void set_search_stack_entry(int ply,
                                   const chess::Board &board,
                                   const chess::Move &move,
                                   int parent_move_count) noexcept
{
    if (ply < 0 || ply > MAX_PLY)
        return;
    SearchStackEntry &ss = search_stack[ply];
    const chess::Piece p = board.at(move.from());
    ss.move = move;
    ss.moved_pt = (p == chess::Piece::NONE) ? -1 : static_cast<int>(p.type());
    ss.parent_move_count = parent_move_count;
    ss.prior_capture = capture_like_for_stack(board, move);
    ss.tt_hit = false;
    ss.in_check = false;
    ss.attack_cache_key = 0ULL;
    ss.attack_cache_ready = false;
}
[[maybe_unused]] inline void build_search_attack_cache(const chess::Board &board,
                                                       SearchStackEntry::AttackCache &cache) noexcept
{
    using namespace chess;

    for (Color c : {Color::WHITE, Color::BLACK})
    {
        const int ci = search_color_index(c);
        const Bitboard pawns = board.pieces(PieceType::PAWN, c);
        const Bitboard pawn_att = search_pawn_attacks_bb(c, pawns);

        cache.by[ci][static_cast<int>(PieceType::PAWN)] = pawn_att;
        cache.by[ci][SEARCH_ALL_IDX] = pawn_att;
        cache.by2[ci] = search_pawn_double_attacks_bb(c, pawns);

        const Bitboard kbb = board.pieces(PieceType::KING, c);
        if (!kbb.empty())
        {
            const Square ksq(static_cast<std::uint8_t>(kbb.lsb()));
            const Bitboard a = attacks::king(ksq);
            cache.by2[ci] |= cache.by[ci][SEARCH_ALL_IDX] & a;
            cache.by[ci][SEARCH_ALL_IDX] |= a;
            cache.by[ci][static_cast<int>(PieceType::KING)] = a;
        }

        {
            Bitboard bb = board.pieces(PieceType::KNIGHT, c);
            while (bb)
            {
                const Square sq(static_cast<std::uint8_t>(bb.pop()));
                const Bitboard a = attacks::knight(sq);
                cache.by2[ci] |= cache.by[ci][SEARCH_ALL_IDX] & a;
                cache.by[ci][SEARCH_ALL_IDX] |= a;
                cache.by[ci][static_cast<int>(PieceType::KNIGHT)] |= a;
            }
        }

        {
            Bitboard bb = board.pieces(PieceType::BISHOP, c);
            while (bb)
            {
                const Square sq(static_cast<std::uint8_t>(bb.pop()));
                const Bitboard a = attacks::bishop(sq, board.occ());
                cache.by2[ci] |= cache.by[ci][SEARCH_ALL_IDX] & a;
                cache.by[ci][SEARCH_ALL_IDX] |= a;
                cache.by[ci][static_cast<int>(PieceType::BISHOP)] |= a;
            }
        }

        {
            Bitboard bb = board.pieces(PieceType::ROOK, c);
            while (bb)
            {
                const Square sq(static_cast<std::uint8_t>(bb.pop()));
                const Bitboard a = attacks::rook(sq, board.occ());
                cache.by2[ci] |= cache.by[ci][SEARCH_ALL_IDX] & a;
                cache.by[ci][SEARCH_ALL_IDX] |= a;
                cache.by[ci][static_cast<int>(PieceType::ROOK)] |= a;
            }
        }

        {
            Bitboard bb = board.pieces(PieceType::QUEEN, c);
            while (bb)
            {
                const Square sq(static_cast<std::uint8_t>(bb.pop()));
                const Bitboard a = attacks::queen(sq, board.occ());
                cache.by2[ci] |= cache.by[ci][SEARCH_ALL_IDX] & a;
                cache.by[ci][SEARCH_ALL_IDX] |= a;
                cache.by[ci][static_cast<int>(PieceType::QUEEN)] |= a;
            }
        }
    }
}
[[maybe_unused]] inline const SearchStackEntry::AttackCache &ensure_search_attack_cache(const chess::Board &board,
                                                                                        int ply) noexcept
{
    if (ply < 0 || ply > MAX_PLY)
        return search_stack[0].attack_cache;
    SearchStackEntry &ss = search_stack[ply];
    const std::uint64_t key = board.hash();
    if (!ss.attack_cache_ready || ss.attack_cache_key != key)
    {
        ss.attack_cache = SearchStackEntry::AttackCache{};
        build_search_attack_cache(board, ss.attack_cache);
        ss.attack_cache_key = key;
        ss.attack_cache_ready = true;
    }
    return ss.attack_cache;
}
[[maybe_unused]] inline bool capture_stage_for_stats(const chess::Board &board, const chess::Move &m) noexcept
{
    return board.isCapture(m) || (m.typeOf() == chess::Move::ENPASSANT) || (m.typeOf() == chess::Move::PROMOTION && m.promotionType() == chess::PieceType::QUEEN);
}
// Null-move pruning is occasionally disabled for verification searches (zugzwang safety)
// and for the null-search itself (avoid consecutive nulls).
static thread_local bool g_allow_null_move = true;
// Verification search disables null-move pruning until a minimum ply.
// This prevents recursive NMP during verification and improves zugzwang safety.
static thread_local int g_nmp_min_ply = 0;
struct ScopedNmpMinPly
{
    int prev;
    explicit ScopedNmpMinPly(int min_ply) noexcept : prev(g_nmp_min_ply)
    {
        g_nmp_min_ply = min_ply;
    }
    ~ScopedNmpMinPly() noexcept
    {
        g_nmp_min_ply = prev;
    }
};
struct ScopedNullMoveFlag
{
    bool prev;
    explicit ScopedNullMoveFlag(bool allow) noexcept : prev(g_allow_null_move)
    {
        g_allow_null_move = allow;
    }
    ~ScopedNullMoveFlag() noexcept
    {
        g_allow_null_move = prev;
    }
};
inline int stm_index(const chess::Board &b)
{
    return (b.sideToMove() == chess::Color::WHITE) ? 0 : 1;
}
// Correction history helpers
inline std::uint64_t splitmix64(std::uint64_t x) noexcept
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
inline std::uint64_t pawn_key(std::uint64_t pw, std::uint64_t pb) noexcept
{
    return splitmix64(pw ^ splitmix64(pb + 0xD1B54A32D192ED03ULL));
}
inline int corr_hist_index(const chess::Board &b) noexcept
{
    const std::uint64_t pw = b.pieces(chess::PieceType::PAWN, chess::Color::WHITE).getBits();
    const std::uint64_t pb = b.pieces(chess::PieceType::PAWN, chess::Color::BLACK).getBits();
    return static_cast<int>(pawn_key(pw, pb) & (CORR_HIST_SIZE - 1));
}
inline int pawn_history_index(const chess::Board &b) noexcept
{
    const std::uint64_t pw = b.pieces(chess::PieceType::PAWN, chess::Color::WHITE).getBits();
    const std::uint64_t pb = b.pieces(chess::PieceType::PAWN, chess::Color::BLACK).getBits();
    return static_cast<int>(pawn_key(pw, pb) & (PAWN_HISTORY_SIZE - 1));
}
inline int probe_pawn_history(const chess::Board &b, int stm, int piece_idx, int to) noexcept
{
    if (stm < 0 || stm > 1 || piece_idx < 0 || piece_idx >= NUM_ORDER_PT || to < 0 || to >= 64)
        return 0;
    return pawn_history[stm][pawn_history_index(b)][piece_idx][to];
}
inline void update_pawn_history_no_board(const chess::Board &b,
                                         int stm,
                                         int piece_idx,
                                         int to,
                                         int delta) noexcept
{
    if (stm < 0 || stm > 1 || piece_idx < 0 || piece_idx >= NUM_ORDER_PT || to < 0 || to >= 64)
        return;
    update_pawn_history_entry(pawn_history[stm][pawn_history_index(b)][piece_idx][to], delta);
}
inline Score corr_hist_probe(const chess::Board &b, const EngineConfig &cfg) noexcept
{
    if (!cfg.use_correction_history || cfg.correction_history_scale <= 0.0)
        return 0;
    const int s = stm_index(b);
    const int idx = corr_hist_index(b);
    const int raw = static_cast<int>(corr_hist[s][idx]); // centipawns
    const int scaled = static_cast<int>(std::lround(static_cast<double>(raw) * cfg.correction_history_scale));
    const int clamped = std::max(-200, std::min(200, scaled));
    return static_cast<Score>(clamped);
}
inline void corr_hist_update(const chess::Board &b, const EngineConfig &cfg, Score residual_cp) noexcept
{
    if (!cfg.use_correction_history)
        return;
    if (residual_cp > 200)
        residual_cp = 200;
    else if (residual_cp < -200)
        residual_cp = -200;
    const int s = stm_index(b);
    const int idx = corr_hist_index(b);
    const int oldv = static_cast<int>(corr_hist[s][idx]);
    const int target = static_cast<int>(residual_cp);
    const int newv = oldv + (target - oldv) / 8; // smooth / low-noise
    const int clamped = std::max(-200, std::min(200, newv));
    corr_hist[s][idx] = static_cast<std::int16_t>(clamped);
}

// Normalized corrected static evaluation (STM POV) used for all eval-based pruning gates.
// This ensures futility/razoring/NMP/ProbCut/etc. all see the same (eval + correction history)
// value and avoids subtle drift from duplicated computations.
inline Score corrected_eval_stm(const chess::Board &b, const EngineConfig &cfg) noexcept
{
    return corrected_static_eval_from_raw(eval_stm_no_game_over(b, cfg), b, cfg);
}

inline void record_legal_movegen(SearchStats &stats, const chess::Movelist &moves) noexcept
{
    ++stats.legal_movegen_calls;
    stats.legal_moves_generated += static_cast<std::uint64_t>(moves.size());
}

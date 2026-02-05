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
// Singular extensions (hash move): conservative Laser/Stockfish-style.
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
// Clamp alpha/beta to mate bounds at this ply (mate-distance pruning).
// This reduces nodes in forced-mate regions and improves mate conversion/defense.
inline void clamp_mate_window(Score &alpha, Score &beta, int ply)
{
    alpha = std::max(alpha, static_cast<Score>(-MATE_SCORE + ply));
    beta = std::min(beta, static_cast<Score>(MATE_SCORE - ply));
}
constexpr int NUM_ORDER_PT = 6;
constexpr int PIECE_ORDER_VALUE[NUM_ORDER_PT] = {
    100,  // PAWN
    320,  // KNIGHT
    330,  // BISHOP
    500,  // ROOK
    900,  // QUEEN
    20000 // KING
};
constexpr int MAX_PLY = 246; // Stockfish 17.1 MAX_PLY
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
static chess::Move move_stack[MAX_PLY + 1];
static int moved_pt_stack[MAX_PLY + 1]; // moved piece-type (index) for move_stack entries
static chess::Move killer_moves[2][MAX_PLY + 1];
static chess::Move counter_moves[2][64][64]; // [stm][prev_from][prev_to] -> quiet refutation move
                                             // [0]=primary, [1]=secondary (quiet beta-cutoff moves)
// Per-ply bookkeeping used for Stockfish-style learning hooks (Step 4+).
// These are written alongside move_stack/moved_pt_stack and will be consumed by
// upcoming Step-4 alignment chunks (TT fail-high learning and refutation penalties).
static int parent_move_count_stack[MAX_PLY + 1]; // parent node moveCount when this ply was entered
static bool prior_capture_stack[MAX_PLY + 1];    // whether the move that entered this ply was a capture
static bool tt_hit_stack[MAX_PLY + 1];           // whether this ply had a TT hit (wired in later steps)
static bool in_check_stack[MAX_PLY + 1];         // whether side-to-move is in check at this ply (wired in later steps)
// Per-ply static evaluation cache (centipawns) used for "improving" signal (Stockfish-style).
// Note: ply parity matters; "improving" compares ss->staticEval with (ss-2)->staticEval (same side to move).
static Score static_eval_stack[MAX_PLY + 1];
// Stockfish-style LMR reduction tables (fixed-point scale 1024).
static int lmr_reductions[MAX_PLY + 2];
static std::once_flag lmr_tables_once;
static thread_local int lmr_root_delta = 1;
inline void init_lmr_tables_once()
{
    std::call_once(lmr_tables_once, []()
                   {
                lmr_reductions[0] = 0;
                for (int i = 1; i <= MAX_PLY + 1; ++i)
                {
                    // Stockfish 17.1: reductions[i] = int(2954 / 128.0 * log(i))
                    lmr_reductions[i] = static_cast<int>((2954.0 / 128.0) * std::log(static_cast<double>(i)));
                } });
}

inline int sf_lmr_reduction_fp(bool improving, int depth, int moveCount, int delta, bool ttPvNode)
{
    (void)ttPvNode;
    // Mirrors Stockfish 17.1 Worker::reduction() (fixed-point 1024), including delta/rootDelta term.
    // reductionScale = reductions[depth] * reductions[moveCount]
    const int d = std::clamp(depth, 0, MAX_PLY + 1);
    const int mn = std::clamp(moveCount, 1, MAX_PLY + 1);
    const int reductionScale = lmr_reductions[d] * lmr_reductions[mn];
    // r = reductionScale - delta*764/rootDelta + (!improving)*reductionScale*191/512 + 1087
    int r = reductionScale;
    r -= (delta * 764) / std::max(1, lmr_root_delta);
    if (!improving)
        r += (reductionScale * 191) / 512;
    r += 1087;
    return r;
}

inline int sf_lmr_adjusted_fp(bool improving, int depth, int moveCount, int delta, int statScore, bool ttPvNode)
{
    // Mirrors Stockfish's fixed-point LMR shape:
    //   r = reduction(improving, depth, moveCount, delta)
    //   r -= 32 * moveCount
    //   if (ttPv) r += 979
    //   r -= statScore * 1582 / 16384
    int r = sf_lmr_reduction_fp(improving, depth, moveCount, delta, ttPvNode);
    r -= 32 * moveCount;
    if (ttPvNode)
        r += 979;
    r -= (statScore * 1582) / 16384;
    return r;
}

inline int sf_futility_move_count(bool improving, int depth) noexcept
{
    // Stockfish 17.1: (3 + depth*depth) / (2 - improving)
    const int d = std::max(0, depth);
    const int denom = 2 - (improving ? 1 : 0);
    return (3 + d * d) / denom;
}

inline Score sf_child_futility_margin(int depth, bool improving) noexcept
{
    // Stockfish 12: 223 * (d - improving)
    // "improving" reduces the margin (more aggressive pruning when the position is improving).
    const int d = std::max(0, depth) - (improving ? 1 : 0);
    return static_cast<Score>(223 * std::max(0, d));
}

static int cont_history_pc[2][NUM_ORDER_PT][64][NUM_ORDER_PT][64]; // [stm][prevPt][prevTo][curPt][curTo]
// Correction history (centipawns): learned adjustment for static eval used by pruning gates.
// Indexed by (side-to-move, pawn-key). Reset on ucinewgame for deterministic testing.
constexpr int CORR_HIST_SIZE = 16384; // power of two
static std::int16_t corr_hist[2][CORR_HIST_SIZE];
// History tables are used directly as ordering scores. Keep them bounded to avoid runaway values.
// Stockfish uses intentionally different ranges for each table:
//   - Main (butterfly) history: 7183
//   - Capture history:         10692
//   - Continuation history:    30000
// This keeps main history from overpowering continuation in quiet move ordering.
constexpr int MAIN_HISTORY_CAP = 7183;
constexpr int MAIN_HISTORY_MIN = -MAIN_HISTORY_CAP;
constexpr int CONT_HISTORY_CAP = 30'000;
constexpr int CONT_HISTORY_MIN = -CONT_HISTORY_CAP;
constexpr int CAPTURE_HISTORY_CAP = 10'692;
constexpr int CAPTURE_HISTORY_MIN = -CAPTURE_HISTORY_CAP;
// Max searched quiet moves retained for end-of-node history malus updates.
constexpr int MAX_QUIET_TRIED = 32;
// Capture-history contribution in move ordering (Stockfish-style: do not downshift).
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
    // Saturating history update (Stockfish-style form): v <- v + delta - v*|delta|/cap, with clamps.
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
// Stockfish-alignment helpers (Step 4+).
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
    update_cont_history_entry(cont_history[stm][prev_to][cur_from][cur_to], delta);
    if (0 <= prev_pt && prev_pt < NUM_ORDER_PT && 0 <= cur_pt && cur_pt < NUM_ORDER_PT)
        update_cont_history_entry(cont_history_pc[stm][prev_pt][prev_to][cur_pt][cur_to], delta);
}
[[maybe_unused]] inline bool capture_like_for_stack(const chess::Board &board, const chess::Move &m) noexcept
{
    return board.isCapture(m) || (m.typeOf() == chess::Move::ENPASSANT);
}
[[maybe_unused]] inline bool capture_stage_for_stats(const chess::Board &board, const chess::Move &m) noexcept
{
    return board.isCapture(m) || (m.typeOf() == chess::Move::ENPASSANT) || (m.typeOf() == chess::Move::PROMOTION && m.promotionType() == chess::PieceType::QUEEN);
}
// Null-move pruning is occasionally disabled for verification searches (zugzwang safety)
// and for the null-search itself (avoid consecutive nulls).
static thread_local bool g_allow_null_move = true;
// Stockfish-style verification search disables null-move pruning until a minimum ply.
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
    Score e = eval_stm_no_game_over(b, cfg);
    e += corr_hist_probe(b, cfg);

    // Pruning gates should never see mate scores; clamp into the non-mate range.
    if (e >= MATE_BOUND)
        e = MATE_BOUND - ONE_CP;
    else if (e <= -MATE_BOUND)
        e = -MATE_BOUND + ONE_CP;

    return e;
}

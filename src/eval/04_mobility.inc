// Evaluation module: Mobility
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

// --- mobility area + piece-type tables ---
//
//  Mobility area excludes squares occupied by own pieces and squares attacked by enemy pawns.
//  Add a small set of exclusions (kept lightweight and compatible with chess.hpp).
//  Use Mobility bonus tables (MG/EG centipawns) tapered by phase.
//
// NOTE: This term is expressed in pawn units (like the other terms in this file).

struct MobScore
{
    int mg_cp;
    int eg_cp;
};

// SF12 MobilityBonus tables (centipawns). We clamp mobility counts to the last entry.
// MOB_PIECETYPE {{early_game_0,late_game_0},{early_game_i,late_game_i},...} represents the penalty or bonus we get
// for a piece having access to 0,1,2...i... unattacked (by pawns) squares in the early or late game
static constexpr MobScore MOB_N[] = {
    {-62, -81}, {-53, -56}, {-12, -31}, {-4, -16}, {3, 5}, {13, 11}, {22, 17}, {28, 20}, {33, 25}};
static constexpr MobScore MOB_B[] = {
    {-48, -59}, {-20, -23}, {16, -3}, {26, 13}, {38, 24}, {51, 42}, {55, 54}, {63, 57}, {63, 65}, {68, 73}, {81, 78}, {81, 86}, {91, 88}, {98, 97}};
static constexpr MobScore MOB_R[] = {
    {-60, -78}, {-20, -17}, {2, 23}, {3, 39}, {3, 70}, {11, 99}, {22, 103}, {31, 121}, {40, 134}, {40, 139}, {41, 158}, {48, 164}, {57, 168}, {57, 169}, {62, 172}};
static constexpr MobScore MOB_Q[] = {
    {-30, -48}, {-12, -30}, {-8, -7}, {-9, 19}, {20, 40}, {23, 55}, {23, 59}, {35, 75}, {38, 78}, {53, 96}, {64, 96}, {65, 100}, {65, 121}, {66, 127}, {67, 131}, {67, 133}, {72, 136}, {72, 141}, {77, 147}, {79, 150}, {93, 151}, {108, 168}, {108, 168}, {108, 171}, {110, 182}, {114, 182}, {114, 192}, {116, 219}};

static inline double mob_bonus_from_table_pawns(const MobScore *t,
                                                int n,
                                                int mob,
                                                double mg_weight,
                                                double eg_weight)
{
    if (mob <= 0)
        mob = 0;
    if (mob >= n)
        mob = n - 1;

    const int mg_cp = t[mob].mg_cp;
    const int eg_cp = t[mob].eg_cp;
    const double blended_cp = mg_weight * static_cast<double>(mg_cp) + eg_weight * static_cast<double>(eg_cp);
    return blended_cp / 100.0; // pawn units
}

static inline double mobility_term_for_color(const Board &board,
                                             Color color,
                                             double mg_weight,
                                             double eg_weight,
                                             EvalInfo &info)
{
    using namespace chess;

    // 4.4 Slider x-ray nuance : for mobility counting only, let bishops see through all queens,
    // and let rooks see through all queens and our own rooks. This values latent line activity without
    // changing legality or move generation elsewhere.
    const Bitboard occ = info.occ;
    const Bitboard occ_no_queens = info.occ_no_queens;
    const Bitboard our_rooks = board.pieces(PieceType::ROOK, color);
    const Bitboard occ_xray_rook = occ_no_queens ^ our_rooks;
    const Bitboard mobility_area = mobility_area_for_color(board, color, info);

    double total = 0.0;

    auto add_pt = [&](PieceType pt, const MobScore *table, int n)
    {
        Bitboard bb = board.pieces(pt, color);
        while (bb)
        {
            const Square sq(static_cast<std::uint8_t>(bb.pop()));

            Bitboard a;
            switch (pt.internal())
            {
            case PieceType::KNIGHT:
                a = attacks::knight(sq);
                break;
            case PieceType::BISHOP:
                a = attacks::bishop(sq, occ_no_queens);
                break;
            case PieceType::ROOK:
                a = attacks::rook(sq, occ_xray_rook);
                break;
            case PieceType::QUEEN:
                a = attacks::queen(sq, occ);
                break;
            default:
                return;
            }

            const int mob = static_cast<int>((a & mobility_area).count());
            total += mob_bonus_from_table_pawns(table, n, mob, mg_weight, eg_weight);
        }
    };

    add_pt(PieceType::KNIGHT, MOB_N, static_cast<int>(sizeof(MOB_N) / sizeof(MOB_N[0])));
    add_pt(PieceType::BISHOP, MOB_B, static_cast<int>(sizeof(MOB_B) / sizeof(MOB_B[0])));
    add_pt(PieceType::ROOK, MOB_R, static_cast<int>(sizeof(MOB_R) / sizeof(MOB_R[0])));
    add_pt(PieceType::QUEEN, MOB_Q, static_cast<int>(sizeof(MOB_Q) / sizeof(MOB_Q[0])));

    return total;
}

static inline double mobility_term_white_minus_black(const Board &board,
                                                     double mg_weight,
                                                     double eg_weight,
                                                     EvalInfo &info)
{
    const double white_mob = mobility_term_for_color(board, Color::WHITE, mg_weight, eg_weight, info);
    const double black_mob = mobility_term_for_color(board, Color::BLACK, mg_weight, eg_weight, info);
    return white_mob - black_mob;
}

double mobility_term_white_minus_black(const Board &board)
{
    const int phase_0_256 = compute_phase_0_256(board);
    const double eg_weight = static_cast<double>(phase_0_256) / 256.0;
    const double mg_weight = 1.0 - eg_weight;
    EvalInfo info = make_eval_info(board);
    return mobility_term_white_minus_black(board, mg_weight, eg_weight, info);
}

static inline int pawn_support_count_for_square(Bitboard our_pawns, Color us, int sq)
{
    // Squares from which OUR pawns attack `sq` are pawn attacks of the opposite color
    // when indexed by the target square (same trick used in Board::isAttacked()).
    const Bitboard attackers = chess::attacks::pawn(~us, Square(static_cast<std::uint8_t>(sq))) & our_pawns;
    return attackers.empty() ? 0 : attackers.count(); // 0..2
}

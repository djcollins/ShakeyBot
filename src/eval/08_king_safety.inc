// Evaluation module: king safety (cover/storm wrapper)
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

// ---------------------- King safety (permanent) ----------------------
//
// Donna-inspired king cover + pawn storm:
//   - King cover: for the king file and adjacent files, find our closest pawn
//     on that file (in front of / on the king's rank). Penalize missing pawns
//     and over-advanced "cover" pawns.
//   - Pawn storm: for the same files, find the opponent pawn that has advanced
//     furthest toward our king. Penalize more if it is unblocked.
//
// We combine this with the existing "king-zone pressure" term (enemy attacks
// into a 2-ring zone around the king).
//
// Practical notes:
// - Cover/storm is primarily a middlegame signal. We taper it by mg_weight (0..1).
// - King-zone pressure remains untapered (your prior experiments found full tapering
//   regressed, especially in defensive scenarios).

// Donna tables (centipawns) ported directly and converted to pawn units.
// Indices are "rank from the defender's point of view", 0-based:
//   - For White: index = rank (A1..H1 => 0, A2..H2 => 1, ..., A8..H8 => 7)
//   - For Black: index = 7 - rank (A8..H8 => 0, ..., A1..H1 => 7)
constexpr int DONNA_COVER_PENALTY_CP[7] = {50, 0, 13, 36, 46, 50, 50};
constexpr int DONNA_STORM_PENALTY_CP[8] = {0, 32, 64, 25, 13, 0, 0, 0};
constexpr int DONNA_STORM_BLOCKED_PENALTY_CP[8] = {0, 0, 32, 12, 6, 0, 0, 0};
constexpr int DONNA_STORM_UNBLOCKED_PENALTY_CP[8] = {13, 16, 48, 19, 10, 0, 0, 0};

constexpr double CP_TO_PAWNS = 0.01;

inline int pov_rank_0_based(Color pov, int sq_idx)
{
    const int r = sq_idx >> 3; // 0..7
    return (pov == Color::WHITE) ? r : (7 - r);
}

inline bool king_is_homeish(Color c, const Square ksq)
{
    const int r = ksq.index() >> 3; // 0..7
    // White king on ranks 1..3, Black king on ranks 8..6.
    return (c == Color::WHITE) ? (r <= 2) : (r >= 5);
}

// --- King cover (cacheable: depends only on own pawns + king square) ---
//
// Returns a *penalty* in pawn units: 0 is best, larger is worse.
double king_cover_penalty_for_color_slow_at_ksq(const Board &board, Color us, const Square ksq)
{
    const Bitboard pawns = board.pieces(PieceType::PAWN, us);

    const int k_idx = ksq.index();
    const int k_file = k_idx & 7;
    const int k_rank = k_idx >> 3;

    double pen = 0.0;

    for (int df = -1; df <= 1; ++df)
    {
        const int f = k_file + df;
        if (f < 0 || f > 7)
            continue;

        Bitboard file_pawns = pawns & chess::attacks::MASK_FILE[f];

        // Find our closest pawn on that file (relative to our side), considering
        // pawns on the king's rank or in front of it.
        int best_r = (us == Color::WHITE) ? 99 : -99;
        if (file_pawns)
        {
            Bitboard tmp = file_pawns;
            while (tmp)
            {
                const int s = static_cast<int>(tmp.pop());
                const int r = s >> 3;

                // "In front of / on" the king's rank from our perspective.
                if (us == Color::WHITE)
                {
                    if (r >= k_rank && r < best_r)
                        best_r = r;
                }
                else
                {
                    if (r <= k_rank && r > best_r)
                        best_r = r;
                }
            }
        }

        int idx = 0; // 0 => missing
        if ((us == Color::WHITE && best_r != 99) || (us == Color::BLACK && best_r != -99))
        {
            const int rr = (us == Color::WHITE) ? best_r : (7 - best_r);
            // Donna cover table has indices 0..6 (ranks beyond 6 are treated as equally bad).
            idx = std::min(6, rr);
        }

        pen += static_cast<double>(DONNA_COVER_PENALTY_CP[idx]) * CP_TO_PAWNS;
    }

    return pen;
}

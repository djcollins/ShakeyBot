// Evaluation module: x-ray/pins helpers
//
// NOTE: This file is included by src/evaluation.cpp (unity build for evaluation).
// Do not compile it as a separate translation unit.

// ---------------------- X-ray / pins (config-scaled) ----------------------
//
// We implement a cheap, proven tactical heuristic: pieces pinned to the king by
// enemy sliders. This is intentionally bitboard-only and avoids move generation.
//
// Note:
// - chess.hpp exposes movegen::pinMask, but it is private. We replicate its
//   logic here using the public attacks::slider and movegen::between helpers.
// - Return value is from White's POV: positive means Black is more pinned.

template <chess::PieceType::underlying PT>
Bitboard pinned_pieces_by_slider_type(const Board &board, Square king_sq, Bitboard occ_opp, Bitboard occ_us)
{
    static_assert(PT == PieceType::BISHOP || PT == PieceType::ROOK, "Only bishop or rook pins supported");

    // Enemy sliders of the given type, plus queens.
    const Bitboard opp_sliders = board.pieces(PieceType(PT), PieceType::QUEEN) & occ_opp;

    // See through our pieces, but stop on enemy pieces (occupied = occ_opp).
    Bitboard pt_attacks = chess::attacks::slider<PT>(king_sq, occ_opp) & opp_sliders;

    const Bitboard occ_all = board.occ();

    auto single_us_blocker_between = [&](int from_idx, int to_idx) -> int
    {
        const int f1 = from_idx & 7;
        const int r1 = from_idx >> 3;
        const int f2 = to_idx & 7;
        const int r2 = to_idx >> 3;

        const int df = f2 - f1;
        const int dr = r2 - r1;

        // Determine step along the line (rook or bishop). If not aligned, return none.
        int step_file = 0;
        int step_rank = 0;
        if (df == 0 && dr != 0)
        {
            step_rank = (dr > 0 ? 1 : -1);
        }
        else if (dr == 0 && df != 0)
        {
            step_file = (df > 0 ? 1 : -1);
        }
        else if (std::abs(df) == std::abs(dr) && df != 0)
        {
            step_file = (df > 0 ? 1 : -1);
            step_rank = (dr > 0 ? 1 : -1);
        }
        else
        {
            return -1;
        }

        const int step = step_file + 8 * step_rank;
        int cur = from_idx + step;

        int pinned_idx = -1;
        while (cur != to_idx)
        {
            if (occ_all.check(cur))
            {
                // If we already saw a blocker, this is not a simple pin.
                if (pinned_idx != -1)
                    return -1;

                if (occ_us.check(cur))
                {
                    pinned_idx = cur;
                }
                else
                {
                    // An enemy piece blocks the line before the attacker.
                    return -1;
                }
            }
            cur += step;
        }

        return pinned_idx;
    };

    Bitboard pinned = 0ull;
    while (pt_attacks)
    {
        const int attacker_idx = static_cast<int>(pt_attacks.pop());
        const int k_idx = king_sq.index();
        const int pinned_idx = single_us_blocker_between(k_idx, attacker_idx);
        if (pinned_idx >= 0)
        {
            pinned.set(pinned_idx);
        }
    }

    return pinned;
}

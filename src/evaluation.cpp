#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <vector>
#include "fast_engine/evaluation.hpp"
#include "fast_engine/config.hpp"

// NOTE:
// This file is the single compilation unit for evaluation.
// The implementation has been split into smaller modules under src/eval/ and
// included here to keep the existing build command working.

namespace fast_engine
{
    static inline Score pawns_to_cp(double pawns)
    {
        return static_cast<Score>(std::llround(pawns * 100.0));
    }

    using chess::Bitboard;
    using chess::Board;
    using chess::Color;
    using chess::PieceType;
    using chess::Square;

    namespace
    {

#include "eval/00_cache.inc"
#include "eval/01_material_phase.inc"
#include "eval/02_material_imbalance_sf12.inc"
#include "eval/03_endgame_scale_king_crowding.inc"
#include "eval/04_mobility.inc"
#include "eval/05_outposts.inc"
#include "eval/06_pawn_structure.inc"
#include "eval/07_pawn_hash_passed.inc"
#include "eval/08_king_safety.inc"
#include "eval/09_king_cover_cache.inc"
#include "eval/10_king_zone_pressure.inc"
#include "eval/11_bishop_pair_bad_bishop.inc"
#include "eval/12_rook_activity.inc"
#include "eval/13_xray_pins.inc"
#include "eval/14_psqt.inc"
#include "eval/15_threats_space.inc"

    } // anonymous namespace

#include "eval/99_public.inc"

} // namespace fast_engine

#pragma once

#include <cstdint>

namespace fast_engine
{

    // All evaluation/search scores are integer centipawns.
    // Positive means "good for side-to-move" once converted to STM perspective.
    using Score = std::int32_t;

    // Large sentinel values used for alpha/beta.
    constexpr Score SCORE_INF = 1'000'000'000;

    // Mate scores are represented as +/-MATE_SCORE, with distance-to-mate encoded
    // as +/- (MATE_SCORE - ply). This keeps mates ordered correctly by distance.
    constexpr Score MATE_SCORE = 1'000'000;
    constexpr Score MATE_BOUND = MATE_SCORE - 1000;

    inline constexpr bool is_mate_score(Score s) noexcept
    {
        return s >= MATE_BOUND || s <= -MATE_BOUND;
    }

} // namespace fast_engine

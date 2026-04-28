#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "chess.hpp"
#include "fast_engine/types.hpp"

namespace fast_engine
{

    enum TTFlag : std::uint8_t
    {
        TT_EXACT = 0,
        TT_LOWERBOUND,
        TT_UPPERBOUND
    };
    constexpr Score TT_NO_STATIC_EVAL = SCORE_INF;

    struct TTEntry
    {
        std::uint64_t key = 0;
        int depth = 0;
        TTFlag flag = TT_EXACT;
        Score value = 0; // centipawns, side-to-move POV
        Score static_eval = TT_NO_STATIC_EVAL; // raw static eval, side-to-move POV
        bool current_generation = false;

        chess::Move bestMove{};
        bool hasMove = false;

        TTEntry() = default;

        TTEntry(std::uint64_t k, int d, TTFlag f, Score v, Score se = TT_NO_STATIC_EVAL)
            : key(k), depth(d), flag(f), value(v), static_eval(se) {}

        TTEntry(std::uint64_t k, int d, TTFlag f, Score v, chess::Move mv, Score se = TT_NO_STATIC_EVAL)
            : key(k), depth(d), flag(f), value(v), static_eval(se), bestMove(mv), hasMove(true) {}
    };

    class TranspositionTable
    {
    public:
        // If you've already done B (associativity), keep your chosen value here.
        // 4-way is a good default.
        static constexpr int CLUSTER_SIZE = 4;

        TranspositionTable() = default;
        explicit TranspositionTable(std::size_t maxEntries) { resize(maxEntries); }

        void resize(std::size_t maxEntries);
        void clear(); // O(1): bump generation
        void new_search(); // bump generation, keep older entries probeable

        std::optional<TTEntry> probe(std::uint64_t key) const;
        void store(const TTEntry &entry);
        static std::size_t entries_for_mb(std::size_t mb);
        static std::size_t mb_for_entries(std::size_t entries);
        std::size_t capacity() const { return capacity_entries_; }

        // D: stored TT bytes per entry (packed), for your MB sizing math.
        static constexpr std::size_t stored_entry_bytes() { return sizeof(PackedEntry); }

    private:
        // D: compact stored entry (this is what lives in the big TT array).
        struct PackedEntry
        {
            std::int32_t value_cp = 0; // centipawns, already mate-adjusted
            std::int32_t static_eval_cp = TT_NO_STATIC_EVAL; // raw static eval or sentinel
            std::uint32_t key32 = 0;   // key signature (top bits)
            std::uint16_t move16 = 0;  // chess::Move raw (0 == NO_MOVE)
            std::uint8_t depth_flag = 0xFF; // low 6 bits depth, high 2 bits TTFlag, 0xFF == empty
            std::uint8_t gen = 0;      // generation tag
        };

        static_assert(sizeof(PackedEntry) == 16, "PackedEntry should be 16 bytes");

        struct Bucket
        {
            std::array<PackedEntry, CLUSTER_SIZE> e{};
        };

        std::vector<Bucket> table_{};
        std::size_t mask_ = 0;             // bucket index mask (power-of-two)
        std::size_t capacity_entries_ = 0; // buckets * CLUSTER_SIZE
        std::uint8_t gen_ = 1;
        std::uint8_t clear_gen_ = 1;

        static std::uint32_t key_signature32(std::uint64_t key)
        {
            return static_cast<std::uint32_t>(key >> 32);
        }

        static bool entry_empty(const PackedEntry &entry)
        {
            return entry.depth_flag == 0xFF;
        }

        static int entry_depth(const PackedEntry &entry)
        {
            return static_cast<int>(entry.depth_flag & 0x3F);
        }

        static TTFlag entry_flag(const PackedEntry &entry)
        {
            return static_cast<TTFlag>(entry.depth_flag >> 6);
        }

        static std::uint8_t pack_depth_flag(int depth, TTFlag flag)
        {
            const int d = std::clamp(depth, 0, 63);
            return static_cast<std::uint8_t>((static_cast<std::uint8_t>(flag) << 6) | static_cast<std::uint8_t>(d));
        }

        bool generation_valid(std::uint8_t gen) const;
        void hard_clear();
        void advance_generation();
    };

} // namespace fast_engine

#pragma once

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

    struct TTEntry
    {
        std::uint64_t key = 0;
        int depth = 0;
        TTFlag flag = TT_EXACT;
        Score value = 0; // centipawns, side-to-move POV

        chess::Move bestMove{};
        bool hasMove = false;

        TTEntry() = default;

        TTEntry(std::uint64_t k, int d, TTFlag f, Score v)
            : key(k), depth(d), flag(f), value(v) {}

        TTEntry(std::uint64_t k, int d, TTFlag f, Score v, chess::Move mv)
            : key(k), depth(d), flag(f), value(v), bestMove(mv), hasMove(true) {}
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
            std::uint16_t key16 = 0;   // key signature (top bits)
            std::uint16_t move16 = 0;  // chess::Move raw (0 == NO_MOVE)
            std::int8_t depth = -1;    // -1 == empty
            std::uint8_t flag = 0;     // TTFlag
            std::uint8_t gen = 0;      // generation tag
            std::uint8_t hasMove = 0;  // 0/1
        };

        static_assert(sizeof(PackedEntry) == 12, "PackedEntry should be 12 bytes");

        struct Bucket
        {
            std::array<PackedEntry, CLUSTER_SIZE> e{};
        };

        std::vector<Bucket> table_{};
        std::size_t mask_ = 0;             // bucket index mask (power-of-two)
        std::size_t capacity_entries_ = 0; // buckets * CLUSTER_SIZE
        std::uint8_t gen_ = 1;

        static std::uint16_t key_signature16(std::uint64_t key)
        {
            return static_cast<std::uint16_t>(key >> 48);
        }
    };

} // namespace fast_engine

#include "fast_engine/transposition.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace fast_engine
{

    static std::size_t next_pow2(std::size_t x)
    {
        if (x <= 1)
            return 1;
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
#if SIZE_MAX > 0xFFFFFFFFu
        x |= x >> 32;
#endif
        return x + 1;
    }

    void TranspositionTable::resize(std::size_t maxEntries)
    {
        if (maxEntries < static_cast<std::size_t>(CLUSTER_SIZE))
            maxEntries = static_cast<std::size_t>(CLUSTER_SIZE);

        std::size_t buckets = maxEntries / static_cast<std::size_t>(CLUSTER_SIZE);
        if (buckets < 1)
            buckets = 1;

        buckets = next_pow2(buckets);

        table_.assign(buckets, Bucket{});
        mask_ = buckets - 1;
        capacity_entries_ = buckets * static_cast<std::size_t>(CLUSTER_SIZE);

        gen_ = 1; // new table: reset generation
        clear_gen_ = 1;
    }

    std::size_t TranspositionTable::entries_for_mb(std::size_t mb)
    {
        if (mb < 1)
            mb = 1;

        const std::size_t bytes = mb * 1024ULL * 1024ULL;

        // Number of buckets we can fit in 'bytes'
        std::size_t buckets = bytes / sizeof(Bucket);
        if (buckets < 1)
            buckets = 1;

        // Keep power-of-two for fast mask indexing
        buckets = next_pow2(buckets);

        return buckets * static_cast<std::size_t>(CLUSTER_SIZE);
    }

    std::size_t TranspositionTable::mb_for_entries(std::size_t entries)
    {
        if (entries < static_cast<std::size_t>(CLUSTER_SIZE))
            entries = static_cast<std::size_t>(CLUSTER_SIZE);

        std::size_t buckets = entries / static_cast<std::size_t>(CLUSTER_SIZE);
        if (buckets < 1)
            buckets = 1;

        buckets = next_pow2(buckets);

        const std::size_t bytes = buckets * sizeof(Bucket);
        return bytes / (1024ULL * 1024ULL);
    }

    bool TranspositionTable::generation_valid(std::uint8_t gen) const
    {
        return gen != 0 && gen >= clear_gen_ && gen <= gen_;
    }

    void TranspositionTable::hard_clear()
    {
        for (auto &b : table_)
        {
            for (auto &e : b.e)
            {
                e = PackedEntry{};
            }
        }
        gen_ = 1;
        clear_gen_ = 1;
    }

    void TranspositionTable::advance_generation()
    {
        ++gen_;
        if (gen_ == 0)
            hard_clear();
    }

    void TranspositionTable::clear()
    {
        // O(1) logical clear: entries from earlier generations are no longer probeable.
        // A hard wipe is still required on 8-bit generation wrap to avoid aliasing.
        advance_generation();
        clear_gen_ = gen_;
    }

    void TranspositionTable::new_search()
    {
        // Keep post-clear older entries probeable between searches, but wipe on wrap so
        // a 255-search-old entry can never become "current" again by generation aliasing.
        advance_generation();
    }

    std::optional<TTEntry> TranspositionTable::probe(std::uint64_t key) const
    {
        if (table_.empty())
            return std::nullopt;

        const std::size_t bucket_index = static_cast<std::size_t>(key) & mask_;
        const std::uint32_t signature = key_signature32(key);

        const Bucket &bucket = table_[bucket_index];
        const PackedEntry *best = nullptr;
        int best_quality = std::numeric_limits<int>::min();

        for (const auto &entry : bucket.e)
        {
            if (entry_empty(entry))
                continue;
            if (!generation_valid(entry.gen))
                continue;
            if (entry.key32 != signature)
                continue;

            int quality = entry_depth(entry) * 8;
            if (entry_flag(entry) == TT_EXACT)
                quality += 4;
            if (entry.move16 != chess::Move::NO_MOVE)
                quality += 1;
            if (entry.gen == gen_)
                quality += 1024;

            if (!best || quality > best_quality)
            {
                best = &entry;
                best_quality = quality;
            }
        }

        if (!best)
            return std::nullopt;

        TTEntry out;
        out.key = key;
        out.depth = entry_depth(*best);
        out.flag = entry_flag(*best);
        out.value = static_cast<Score>(best->value_cp);
        out.static_eval = static_cast<Score>(best->static_eval_cp);
        out.current_generation = (best->gen == gen_);

        out.hasMove = (best->move16 != chess::Move::NO_MOVE);
        if (out.hasMove)
            out.bestMove = chess::Move(best->move16);
        return out;
    }

    void TranspositionTable::store(const TTEntry &entry)
    {
        if (table_.empty())
            return;

        const std::size_t bucket_index = static_cast<std::size_t>(entry.key) & mask_;
        const std::uint32_t signature = key_signature32(entry.key);

        // Store scores as packed centipawns.
        const std::int32_t value_cp = static_cast<std::int32_t>(entry.value);

        Bucket &bucket = table_[bucket_index];

        auto write = [&](PackedEntry &pe)
        {
            pe.gen = gen_;
            pe.key32 = signature;
            pe.depth_flag = pack_depth_flag(entry.depth, entry.flag);
            pe.value_cp = value_cp;
            pe.static_eval_cp = static_cast<std::int32_t>(entry.static_eval);

            pe.move16 = entry.hasMove ? entry.bestMove.move() : chess::Move::NO_MOVE;
        };

        // 1) Same key (signature) in current generation: update smartly.
        for (auto &packed : bucket.e)
        {
            if (packed.gen != gen_)
                continue;
            if (entry_empty(packed))
                continue;
            if (packed.key32 != signature)
                continue;

            const int old_depth = entry_depth(packed);
            const int new_depth = std::clamp(entry.depth, 0, 63);
            const bool old_exact = (entry_flag(packed) == TT_EXACT);
            const bool new_exact = (entry.flag == TT_EXACT);

            const bool replace =
                (new_depth > old_depth) ||
                (new_depth == old_depth && new_exact && !old_exact);

            if (replace)
            {
                write(packed);
            }
            else
            {
                // Keep old eval, but allow best-move fill-in if old had none.
                if (entry.static_eval != TT_NO_STATIC_EVAL && packed.static_eval_cp == TT_NO_STATIC_EVAL)
                    packed.static_eval_cp = static_cast<std::int32_t>(entry.static_eval);
                if (entry.hasMove && packed.move16 == chess::Move::NO_MOVE)
                {
                    packed.move16 = entry.bestMove.move();
                }
            }
            return;
        }

        // 2) Prefer an invalid/old-gen/empty slot first (age/generation).
        for (auto &packed : bucket.e)
        {
            if (packed.gen != gen_ || entry_empty(packed))
            {
                write(packed);
                return;
            }
        }

        // 3) Bucket full: replace the lowest-quality entry.
        auto replacement_quality = [&](const PackedEntry &packed) -> int
        {
            // Lower => more replaceable.
            //
            // Critical: stale generations must be replaced first. Otherwise a cluster can
            // get "poisoned" by high-depth entries from invalid generations, shrinking
            // the effective TT dramatically.
            //
            // Order of preference:
            //   1) empty slots
            //   2) stale-generation entries
            //   3) shallow / non-exact / no-move entries
            if (entry_empty(packed))
                return -1000000; // empty

            int quality = entry_depth(packed) * 4;
            if (entry_flag(packed) == TT_EXACT)
                quality += 2;
            if (packed.move16 != chess::Move::NO_MOVE)
                quality += 1;

            if (packed.gen != gen_)
                quality -= 1000; // stale entries should lose to current-generation entries
            return quality;
        };

        PackedEntry *victim = &bucket.e[0];
        int victim_quality = replacement_quality(bucket.e[0]);

        for (auto &packed : bucket.e)
        {
            const int quality = replacement_quality(packed);
            if (quality < victim_quality)
            {
                victim_quality = quality;
                victim = &packed;
            }
        }

        write(*victim);
    }

} // namespace fast_engine

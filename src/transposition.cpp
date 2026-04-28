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
    void TranspositionTable::clear()
    {
        // O(1) clear: advance generation. Old entries become "invalid".
        ++gen_;

        // If we wrapped, do a real wipe once every 255 clears.
        if (gen_ == 0)
        {
            gen_ = 1;
            for (auto &b : table_)
            {
                for (auto &e : b.e)
                {
                    e = PackedEntry{};
                }
            }
        }
    }

    void TranspositionTable::new_search()
    {
        ++gen_;
        if (gen_ == 0)
            gen_ = 1;
    }

    std::optional<TTEntry> TranspositionTable::probe(std::uint64_t key) const
    {
        if (table_.empty())
            return std::nullopt;

        const std::size_t idx = static_cast<std::size_t>(key) & mask_;
        const std::uint32_t sig = key_signature32(key);

        const Bucket &b = table_[idx];
        const PackedEntry *best = nullptr;
        int best_quality = std::numeric_limits<int>::min();

        for (const auto &pe : b.e)
        {
            if (entry_empty(pe))
                continue;
            if (pe.key32 != sig)
                continue;

            int q = entry_depth(pe) * 8;
            if (entry_flag(pe) == TT_EXACT)
                q += 4;
            if (pe.move16 != chess::Move::NO_MOVE)
                q += 1;
            if (pe.gen == gen_)
                q += 1024;

            if (!best || q > best_quality)
            {
                best = &pe;
                best_quality = q;
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

        const std::size_t idx = static_cast<std::size_t>(entry.key) & mask_;
        const std::uint32_t sig = key_signature32(entry.key);

        // D: store score as centipawns (int), not double.
        const std::int32_t vcp = static_cast<std::int32_t>(entry.value);

        Bucket &b = table_[idx];

        auto write = [&](PackedEntry &pe)
        {
            pe.gen = gen_;
            pe.key32 = sig;
            pe.depth_flag = pack_depth_flag(entry.depth, entry.flag);
            pe.value_cp = vcp;
            pe.static_eval_cp = static_cast<std::int32_t>(entry.static_eval);

            pe.move16 = entry.hasMove ? entry.bestMove.move() : chess::Move::NO_MOVE;
        };

        // 1) Same key (signature) in current generation: update smartly.
        for (auto &pe : b.e)
        {
            if (pe.gen != gen_)
                continue;
            if (entry_empty(pe))
                continue;
            if (pe.key32 != sig)
                continue;

            const int oldDepth = entry_depth(pe);
            const int newDepth = std::clamp(entry.depth, 0, 63);
            const bool oldExact = (entry_flag(pe) == TT_EXACT);
            const bool newExact = (entry.flag == TT_EXACT);

            const bool replace =
                (newDepth > oldDepth) ||
                (newDepth == oldDepth && newExact && !oldExact);

            if (replace)
            {
                write(pe);
            }
            else
            {
                // Keep old eval, but allow best-move fill-in if old had none.
                if (entry.static_eval != TT_NO_STATIC_EVAL && pe.static_eval_cp == TT_NO_STATIC_EVAL)
                    pe.static_eval_cp = static_cast<std::int32_t>(entry.static_eval);
                if (entry.hasMove && pe.move16 == chess::Move::NO_MOVE)
                {
                    pe.move16 = entry.bestMove.move();
                }
            }
            return;
        }

        // 2) Prefer an invalid/old-gen/empty slot first (age/generation).
        for (auto &pe : b.e)
        {
            if (pe.gen != gen_ || entry_empty(pe))
            {
                write(pe);
                return;
            }
        }

        // 3) Bucket full: replace the lowest-quality entry (C).
        auto quality = [&](const PackedEntry &pe) -> int
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
            if (entry_empty(pe))
                return -1000000; // empty

            int q = entry_depth(pe) * 4;
            if (entry_flag(pe) == TT_EXACT)
                q += 2;
            if (pe.move16 != chess::Move::NO_MOVE)
                q += 1;

            if (pe.gen != gen_)
                q -= 1000; // stale => always replaceable
            return q;
        };

        PackedEntry *victim = &b.e[0];
        int victimQ = quality(b.e[0]);

        for (auto &pe : b.e)
        {
            const int q = quality(pe);
            if (q < victimQ)
            {
                victimQ = q;
                victim = &pe;
            }
        }

        write(*victim);
    }

} // namespace fast_engine

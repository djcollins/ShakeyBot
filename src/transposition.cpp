#include "fast_engine/transposition.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

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

    std::optional<TTEntry> TranspositionTable::probe(std::uint64_t key) const
    {
        if (table_.empty())
            return std::nullopt;

        const std::size_t idx = static_cast<std::size_t>(key) & mask_;
        const std::uint16_t sig = key_signature16(key);

        const Bucket &b = table_[idx];

        for (const auto &pe : b.e)
        {
            if (pe.gen != gen_)
                continue;
            if (pe.depth < 0)
                continue;
            if (pe.key16 != sig)
                continue;

            TTEntry out;
            out.key = key;
            out.depth = static_cast<int>(pe.depth);
            out.flag = static_cast<TTFlag>(pe.flag);
            out.value = static_cast<Score>(pe.value_cp);

            out.hasMove = (pe.hasMove != 0);
            if (out.hasMove)
            {
                out.bestMove = chess::Move(pe.move16);
            }
            return out;
        }

        return std::nullopt;
    }

    void TranspositionTable::store(const TTEntry &entry)
    {
        if (table_.empty())
            return;

        const std::size_t idx = static_cast<std::size_t>(entry.key) & mask_;
        const std::uint16_t sig = key_signature16(entry.key);

        // D: store score as centipawns (int), not double.
        const std::int32_t vcp = static_cast<std::int32_t>(entry.value);

        Bucket &b = table_[idx];

        auto write = [&](PackedEntry &pe)
        {
            pe.gen = gen_;
            pe.key16 = sig;
            pe.depth = static_cast<std::int8_t>(std::clamp(entry.depth, 0, 127));
            pe.flag = static_cast<std::uint8_t>(entry.flag);
            pe.value_cp = vcp;

            pe.hasMove = entry.hasMove ? 1 : 0;
            pe.move16 = entry.hasMove ? entry.bestMove.move() : chess::Move::NO_MOVE;
        };

        // 1) Same key (signature) in current generation: update smartly.
        for (auto &pe : b.e)
        {
            if (pe.gen != gen_)
                continue;
            if (pe.depth < 0)
                continue;
            if (pe.key16 != sig)
                continue;

            const int oldDepth = static_cast<int>(pe.depth);
            const bool oldExact = (pe.flag == TT_EXACT);
            const bool newExact = (entry.flag == TT_EXACT);

            const bool replace =
                (entry.depth > oldDepth) ||
                (entry.depth == oldDepth && newExact && !oldExact);

            if (replace)
            {
                write(pe);
            }
            else
            {
                // Keep old eval, but allow best-move fill-in if old had none.
                if (entry.hasMove && pe.hasMove == 0)
                {
                    pe.hasMove = 1;
                    pe.move16 = entry.bestMove.move();
                }
            }
            return;
        }

        // 2) Prefer an invalid/old-gen/empty slot first (age/generation).
        for (auto &pe : b.e)
        {
            if (pe.gen != gen_ || pe.depth < 0)
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
            // get "poisoned" by high-depth entries from old generations that are never
            // probed (probe() filters by gen_), shrinking the effective TT dramatically.
            //
            // Order of preference:
            //   1) empty slots
            //   2) stale-generation entries
            //   3) shallow / non-exact / no-move entries
            if (pe.depth < 0)
                return -1000000; // empty

            int q = static_cast<int>(pe.depth) * 4;
            if (pe.flag == TT_EXACT)
                q += 2;
            if (pe.hasMove)
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

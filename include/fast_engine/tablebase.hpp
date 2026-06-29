#pragma once

#include <vector>

#include "chess.hpp"
#include "fast_engine/config.hpp"

namespace fast_engine
{

    enum class TablebaseWdl : int
    {
        Loss = 0,
        BlessedLoss = 1,
        Draw = 2,
        CursedWin = 3,
        Win = 4
    };

    struct TablebaseRootMove
    {
        chess::Move move{};
        int wdl = -1;
        int dtz = 0;
        int rank = 0;
    };

    struct TablebaseRootResult
    {
        bool available = false;
        int root_wdl = -1;
        std::vector<TablebaseRootMove> moves;
    };

    TablebaseRootResult probe_syzygy_root(const chess::Board &board,
                                          const EngineConfig &config);
    bool initialize_syzygy(const EngineConfig &config);

} // namespace fast_engine

#include "fast_engine/tablebase.hpp"

#include <array>
#include <mutex>
#include <string>

#include "fathom/tbprobe.h"
#include "fast_engine/path_utils.hpp"

namespace fast_engine
{
    namespace
    {
        std::mutex g_tb_mutex;
        std::string g_tb_path;
        bool g_tb_ready = false;

        static int piece_count(const chess::Board &board) noexcept
        {
            return board.occ().count();
        }

        static bool ensure_tb_ready_locked(const std::string &path)
        {
            if (path.empty())
            {
                if (g_tb_ready || !g_tb_path.empty())
                    tb_free();
                g_tb_path.clear();
                g_tb_ready = false;
                return false;
            }

            const std::filesystem::path resolved = resolve_named_directory_upward(path);
            if (resolved.empty())
            {
                if (g_tb_ready || !g_tb_path.empty())
                    tb_free();
                g_tb_path.clear();
                g_tb_ready = false;
                return false;
            }

            const std::string resolved_path = resolved.string();
            if (g_tb_path == resolved_path)
                return g_tb_ready;

            tb_free();
            g_tb_path = resolved_path;
            g_tb_ready = tb_init(resolved_path.c_str());
            return g_tb_ready;
        }

        static unsigned ep_square_for_fathom(const chess::Board &board) noexcept
        {
            const chess::Square ep = board.enpassantSq();
            return ep.is_valid() ? static_cast<unsigned>(ep.index()) : 0u;
        }

        static chess::PieceType promotion_type_from_tb(unsigned promotes) noexcept
        {
            switch (promotes)
            {
            case TB_PROMOTES_QUEEN:
                return chess::PieceType::QUEEN;
            case TB_PROMOTES_ROOK:
                return chess::PieceType::ROOK;
            case TB_PROMOTES_BISHOP:
                return chess::PieceType::BISHOP;
            case TB_PROMOTES_KNIGHT:
                return chess::PieceType::KNIGHT;
            default:
                return chess::PieceType::NONE;
            }
        }

        static bool matches_tb_result(chess::Move move, unsigned result) noexcept
        {
            if (static_cast<unsigned>(move.from().index()) != TB_GET_FROM(result))
                return false;
            if (static_cast<unsigned>(move.to().index()) != TB_GET_TO(result))
                return false;

            const unsigned tb_promotes = TB_GET_PROMOTES(result);
            if (move.typeOf() == chess::Move::PROMOTION)
                return promotion_type_from_tb(tb_promotes) == move.promotionType();
            if (tb_promotes != TB_PROMOTES_NONE)
                return false;

            const bool tb_ep = TB_GET_EP(result) != 0;
            return tb_ep == (move.typeOf() == chess::Move::ENPASSANT);
        }

        static int move_rank_from_tb(int wdl, int dtz) noexcept
        {
            switch (wdl)
            {
            case TB_WIN:
                return 400000 - dtz;
            case TB_CURSED_WIN:
                return 300000 - dtz;
            case TB_DRAW:
                return 200000;
            case TB_BLESSED_LOSS:
                return 100000 + dtz;
            case TB_LOSS:
                return dtz;
            default:
                return -1;
            }
        }
    } // namespace

    TablebaseRootResult probe_syzygy_root(const chess::Board &board,
                                          const EngineConfig &config)
    {
        TablebaseRootResult out{};

        if (!config.syzygy_root_probe || config.syzygy_path.empty())
            return out;

        const int pieces = piece_count(board);
        if (pieces <= 2 || pieces > config.syzygy_probe_limit)
            return out;
        if (!board.castlingRights().isEmpty())
            return out;

        chess::Movelist legal;
        chess::movegen::legalmoves(legal, board);
        if (legal.empty())
            return out;

        std::lock_guard<std::mutex> lock(g_tb_mutex);
        if (!ensure_tb_ready_locked(config.syzygy_path))
            return out;
        if (TB_LARGEST < static_cast<unsigned>(pieces))
            return out;

        std::array<unsigned, TB_MAX_MOVES> results{};
        const std::uint64_t white = board.us(chess::Color::WHITE).getBits();
        const std::uint64_t black = board.us(chess::Color::BLACK).getBits();
        const std::uint64_t kings = board.pieces(chess::PieceType::KING).getBits();
        const std::uint64_t queens = board.pieces(chess::PieceType::QUEEN).getBits();
        const std::uint64_t rooks = board.pieces(chess::PieceType::ROOK).getBits();
        const std::uint64_t bishops = board.pieces(chess::PieceType::BISHOP).getBits();
        const std::uint64_t knights = board.pieces(chess::PieceType::KNIGHT).getBits();
        const std::uint64_t pawns = board.pieces(chess::PieceType::PAWN).getBits();
        const bool white_to_move = board.sideToMove() == chess::Color::WHITE;

        const unsigned root = tb_probe_root(white,
                                            black,
                                            kings,
                                            queens,
                                            rooks,
                                            bishops,
                                            knights,
                                            pawns,
                                            board.halfMoveClock(),
                                            0,
                                            ep_square_for_fathom(board),
                                            white_to_move,
                                            results.data());

        if (root == TB_RESULT_FAILED)
            return out;

        out.available = true;
        out.root_wdl = static_cast<int>(TB_GET_WDL(root));
        out.moves.reserve(legal.size());

        for (const chess::Move &move : legal)
        {
            for (unsigned result : results)
            {
                if (result == TB_RESULT_FAILED)
                    break;
                if (!matches_tb_result(move, result))
                    continue;

                const int wdl = static_cast<int>(TB_GET_WDL(result));
                const int dtz = static_cast<int>(TB_GET_DTZ(result));
                out.moves.push_back(TablebaseRootMove{move, wdl, dtz, move_rank_from_tb(wdl, dtz)});
                break;
            }
        }

        if (out.moves.empty())
            out.available = false;

        return out;
    }

    bool initialize_syzygy(const EngineConfig &config)
    {
        if (!config.syzygy_root_probe || config.syzygy_path.empty())
            return false;

        std::lock_guard<std::mutex> lock(g_tb_mutex);
        return ensure_tb_ready_locked(config.syzygy_path);
    }

} // namespace fast_engine

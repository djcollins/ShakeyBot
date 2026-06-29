#include "fast_engine/path_utils.hpp"

#include <algorithm>
#include <system_error>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fast_engine
{
    namespace
    {
        static bool path_is_directory(const std::filesystem::path &path) noexcept
        {
            std::error_code ec;
            return std::filesystem::is_directory(path, ec);
        }

        static bool path_is_regular_file(const std::filesystem::path &path) noexcept
        {
            std::error_code ec;
            return std::filesystem::is_regular_file(path, ec);
        }

        static std::filesystem::path absolute_path(const std::filesystem::path &path)
        {
            std::error_code ec;
            std::filesystem::path out = std::filesystem::absolute(path, ec);
            return ec ? path : out;
        }

        static std::filesystem::path executable_directory()
        {
#ifdef _WIN32
            std::vector<char> buffer(4096);
            const DWORD len = GetModuleFileNameA(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
            if (len > 0 && len < buffer.size())
                return std::filesystem::path(std::string(buffer.data(), len)).parent_path();
#endif
            return {};
        }

        static std::vector<std::filesystem::path> search_roots()
        {
            std::vector<std::filesystem::path> roots;

            std::error_code ec;
            const std::filesystem::path cwd = std::filesystem::current_path(ec);
            if (!ec && !cwd.empty())
                roots.push_back(cwd);

            const std::filesystem::path exe_dir = executable_directory();
            if (!exe_dir.empty() && std::find(roots.begin(), roots.end(), exe_dir) == roots.end())
                roots.push_back(exe_dir);

            return roots;
        }

        static std::filesystem::path find_child_directory_upward(const std::filesystem::path &start,
                                                                 const std::filesystem::path &dir_name)
        {
            if (start.empty() || dir_name.empty())
                return {};

            std::filesystem::path cur = absolute_path(start);
            for (;;)
            {
                const std::filesystem::path candidate = cur / dir_name;
                if (path_is_directory(candidate))
                    return candidate;

                const std::filesystem::path parent = cur.parent_path();
                if (parent.empty() || parent == cur)
                    break;
                cur = parent;
            }

            return {};
        }
    } // namespace

    std::filesystem::path resolve_named_directory_upward(const std::string &name)
    {
        if (name.empty())
            return {};

        const std::filesystem::path requested(name);
        if (requested.is_absolute() && path_is_directory(requested))
            return requested;
        if (path_is_directory(requested))
            return absolute_path(requested);

        const std::filesystem::path dir_name = requested.filename().empty() ? requested : requested.filename();
        for (const std::filesystem::path &root : search_roots())
        {
            std::filesystem::path found = find_child_directory_upward(root, dir_name);
            if (!found.empty())
                return found;
        }

        return {};
    }

    std::filesystem::path resolve_model_path_upward(const std::string &model_path)
    {
        if (model_path.empty())
            return {};

        const std::filesystem::path requested(model_path);
        if (requested.is_absolute() && path_is_regular_file(requested))
            return requested;
        if (path_is_regular_file(requested))
            return absolute_path(requested);

        const std::filesystem::path file_name = requested.filename();
        if (file_name.empty())
            return {};

        for (const std::filesystem::path &root : search_roots())
        {
            std::filesystem::path models_dir = find_child_directory_upward(root, "models");
            if (models_dir.empty())
                continue;

            const std::filesystem::path candidate = models_dir / file_name;
            if (path_is_regular_file(candidate))
                return candidate;
        }

        return {};
    }

} // namespace fast_engine

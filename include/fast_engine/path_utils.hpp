#pragma once

#include <filesystem>
#include <string>

namespace fast_engine
{

    std::filesystem::path resolve_named_directory_upward(const std::string &name);
    std::filesystem::path resolve_model_path_upward(const std::string &model_path);

} // namespace fast_engine

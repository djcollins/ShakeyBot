#include <algorithm>
#include <array>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include <mutex>
#include "fast_engine/search.hpp"
#include "fast_engine/evaluation.hpp"
#include "fast_engine/transposition.hpp"

namespace fast_engine
{
    constexpr Score ONE_CP = 1;

    namespace
    {

#include "search/search_context.inc"
#include "search/search_ordering_see.inc"

    } // anonymous namespace

#include "search/move_picker.inc"
#include "search/search_qsearch.inc"
#include "search/search_ab.inc"
#include "search/99_public.inc"

} // namespace fast_engine

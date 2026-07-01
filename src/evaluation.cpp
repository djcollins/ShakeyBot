#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <limits>
#include <new>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#if defined(__AVX512F__) || defined(SHAKEYBOT_ENABLE_AVX512_DISPATCH) || \
    defined(__AVX2__) || defined(SHAKEYBOT_ENABLE_AVX2_DISPATCH)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif
#include "fast_engine/evaluation.hpp"
#include "fast_engine/config.hpp"

// Evaluation uses one compilation unit with small internal modules.
// The include order below is part of the eval pipeline.

namespace fast_engine
{
    static inline Score pawns_to_cp(double pawns)
    {
        return static_cast<Score>(std::llround(pawns * 100.0));
    }

    using chess::Bitboard;
    using chess::Board;
    using chess::Color;
    using chess::PieceType;
    using chess::Square;

    namespace
    {
        constexpr std::size_t NEURAL_CACHELINE_ALIGNMENT = 64;

        template <typename T, std::size_t Alignment>
        struct AlignedAllocator
        {
            using value_type = T;
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;

            template <typename U>
            struct rebind
            {
                using other = AlignedAllocator<U, Alignment>;
            };

            AlignedAllocator() noexcept = default;

            template <typename U>
            constexpr AlignedAllocator(const AlignedAllocator<U, Alignment> &) noexcept
            {
            }

            [[nodiscard]] T *allocate(size_type n)
            {
                if (n > std::numeric_limits<size_type>::max() / sizeof(T))
                    throw std::bad_array_new_length();
                return static_cast<T *>(::operator new(n * sizeof(T), std::align_val_t{Alignment}));
            }

            void deallocate(T *p, size_type) noexcept
            {
                ::operator delete(p, std::align_val_t{Alignment});
            }
        };

        template <typename T, typename U, std::size_t Alignment>
        constexpr bool operator==(const AlignedAllocator<T, Alignment> &,
                                  const AlignedAllocator<U, Alignment> &) noexcept
        {
            return true;
        }

        template <typename T, typename U, std::size_t Alignment>
        constexpr bool operator!=(const AlignedAllocator<T, Alignment> &,
                                  const AlignedAllocator<U, Alignment> &) noexcept
        {
            return false;
        }

        template <typename T>
        using AlignedVector = std::vector<T, AlignedAllocator<T, NEURAL_CACHELINE_ALIGNMENT>>;

#if defined(__AVX2__)
#define SHAKEYBOT_HAS_AVX2_KERNELS 1
#define SHAKEYBOT_AVX2_TARGET
#elif defined(SHAKEYBOT_ENABLE_AVX2_DISPATCH) && \
    (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#define SHAKEYBOT_HAS_AVX2_KERNELS 1
#define SHAKEYBOT_AVX2_TARGET __attribute__((target("avx2")))
#else
#define SHAKEYBOT_HAS_AVX2_KERNELS 0
#define SHAKEYBOT_AVX2_TARGET
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
#define SHAKEYBOT_HAS_AVX512_KERNELS 1
#define SHAKEYBOT_AVX512_TARGET
#elif defined(SHAKEYBOT_ENABLE_AVX512_DISPATCH) && \
    (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#define SHAKEYBOT_HAS_AVX512_KERNELS 1
#define SHAKEYBOT_AVX512_TARGET __attribute__((target("avx512f,avx512bw,avx512dq")))
#else
#define SHAKEYBOT_HAS_AVX512_KERNELS 0
#define SHAKEYBOT_AVX512_TARGET
#endif

#if SHAKEYBOT_HAS_AVX2_KERNELS
#if !defined(__AVX2__)
        static const bool NEURAL_CPU_SUPPORTS_AVX2 = []() noexcept {
            __builtin_cpu_init();
            return __builtin_cpu_supports("avx2") != 0;
        }();
#endif

        static inline bool neural_cpu_supports_avx2() noexcept
        {
#if defined(__AVX2__)
            return true;
#else
            return NEURAL_CPU_SUPPORTS_AVX2;
#endif
        }
#else
        static inline bool neural_cpu_supports_avx2() noexcept
        {
            return false;
        }
#endif

#if SHAKEYBOT_HAS_AVX512_KERNELS
#if !(defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__))
        static const bool NEURAL_CPU_SUPPORTS_AVX512 = []() noexcept {
            __builtin_cpu_init();
            return __builtin_cpu_supports("avx512f") != 0 &&
                   __builtin_cpu_supports("avx512bw") != 0 &&
                   __builtin_cpu_supports("avx512dq") != 0;
        }();
#endif

        static inline bool neural_cpu_supports_avx512() noexcept
        {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
            return true;
#else
            return NEURAL_CPU_SUPPORTS_AVX512;
#endif
        }
#else
        static inline bool neural_cpu_supports_avx512() noexcept
        {
            return false;
        }
#endif

#if SHAKEYBOT_HAS_AVX512_KERNELS
        SHAKEYBOT_AVX512_TARGET static inline std::int64_t neural_dot_relu_i32_i16_avx512(
            const std::int32_t *input,
            const std::int16_t *weights,
            int count) noexcept
        {
            const __m512i zero = _mm512_setzero_si512();
            __m512i sum = _mm512_setzero_si512();
            int i = 0;
            for (; i + 16 <= count; i += 16)
            {
                __m512i activations = _mm512_loadu_si512(reinterpret_cast<const void *>(input + i));
                activations = _mm512_max_epi32(activations, zero);
                const __m256i packed_weights =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weights + i));
                const __m512i expanded_weights = _mm512_cvtepi16_epi32(packed_weights);

                sum = _mm512_add_epi64(sum, _mm512_mul_epi32(activations, expanded_weights));
                sum = _mm512_add_epi64(
                    sum,
                    _mm512_mul_epi32(_mm512_srli_epi64(activations, 32),
                                     _mm512_srli_epi64(expanded_weights, 32)));
            }

            alignas(64) std::int64_t lanes[8];
            _mm512_store_si512(reinterpret_cast<void *>(lanes), sum);
            std::int64_t total = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                                 lanes[4] + lanes[5] + lanes[6] + lanes[7];
            for (; i < count; ++i)
            {
                const std::int32_t activation = std::max(input[i], 0);
                total += static_cast<std::int64_t>(activation) * static_cast<std::int64_t>(weights[i]);
            }
            return total;
        }

        SHAKEYBOT_AVX512_TARGET static inline std::int64_t neural_dot_clamped_i32_i16_avx512(
            const std::int32_t *input,
            const std::int16_t *weights,
            int count,
            std::int32_t max_value) noexcept
        {
            const __m512i zero = _mm512_setzero_si512();
            const __m512i max_activation = _mm512_set1_epi32(max_value);
            __m512i sum = _mm512_setzero_si512();
            int i = 0;
            for (; i + 16 <= count; i += 16)
            {
                __m512i activations = _mm512_loadu_si512(reinterpret_cast<const void *>(input + i));
                activations = _mm512_min_epi32(_mm512_max_epi32(activations, zero), max_activation);
                const __m256i packed_weights =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weights + i));
                const __m512i expanded_weights = _mm512_cvtepi16_epi32(packed_weights);

                sum = _mm512_add_epi64(sum, _mm512_mul_epi32(activations, expanded_weights));
                sum = _mm512_add_epi64(
                    sum,
                    _mm512_mul_epi32(_mm512_srli_epi64(activations, 32),
                                     _mm512_srli_epi64(expanded_weights, 32)));
            }

            alignas(64) std::int64_t lanes[8];
            _mm512_store_si512(reinterpret_cast<void *>(lanes), sum);
            std::int64_t total = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                                 lanes[4] + lanes[5] + lanes[6] + lanes[7];
            for (; i < count; ++i)
            {
                const std::int32_t activation = std::clamp(input[i], 0, max_value);
                total += static_cast<std::int64_t>(activation) * static_cast<std::int64_t>(weights[i]);
            }
            return total;
        }

        SHAKEYBOT_AVX512_TARGET static inline void neural_accumulate_dual_i32_i16_to_i64_avx512(
            std::int64_t *dst,
            const std::int16_t *row_a,
            std::int32_t activation_a,
            const std::int16_t *row_b,
            std::int32_t activation_b,
            int count) noexcept
        {
            const __m512i a = _mm512_set1_epi32(activation_a);
            const __m512i b = _mm512_set1_epi32(activation_b);
            int i = 0;
            for (; i + 16 <= count; i += 16)
            {
                const __m256i packed_a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row_a + i));
                const __m256i packed_b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row_b + i));
                const __m512i weights_a = _mm512_cvtepi16_epi32(packed_a);
                const __m512i weights_b = _mm512_cvtepi16_epi32(packed_b);
                const __m512i product32 = _mm512_add_epi32(
                    _mm512_mullo_epi32(weights_a, a),
                    _mm512_mullo_epi32(weights_b, b));

                const __m256i product_lo = _mm512_castsi512_si256(product32);
                const __m256i product_hi = _mm512_extracti64x4_epi64(product32, 1);
                const __m512i product64_lo = _mm512_cvtepi32_epi64(product_lo);
                const __m512i product64_hi = _mm512_cvtepi32_epi64(product_hi);

                __m512i dst_lo = _mm512_loadu_si512(reinterpret_cast<const void *>(dst + i));
                __m512i dst_hi = _mm512_loadu_si512(reinterpret_cast<const void *>(dst + i + 8));
                dst_lo = _mm512_add_epi64(dst_lo, product64_lo);
                dst_hi = _mm512_add_epi64(dst_hi, product64_hi);
                _mm512_storeu_si512(reinterpret_cast<void *>(dst + i), dst_lo);
                _mm512_storeu_si512(reinterpret_cast<void *>(dst + i + 8), dst_hi);
            }

            for (; i < count; ++i)
            {
                dst[i] += static_cast<std::int64_t>(activation_a) * static_cast<std::int64_t>(row_a[i]);
                dst[i] += static_cast<std::int64_t>(activation_b) * static_cast<std::int64_t>(row_b[i]);
            }
        }

        SHAKEYBOT_AVX512_TARGET static inline void neural_accumulate_i32_i16_to_i64_evenodd_avx512(
            std::int64_t *dst_even,
            std::int64_t *dst_odd,
            const std::int16_t *row,
            std::int32_t activation,
            int count) noexcept
        {
            const __m512i a = _mm512_set1_epi32(activation);
            int i = 0;
            for (; i + 16 <= count; i += 16)
            {
                const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row + i));
                const __m512i weights = _mm512_cvtepi16_epi32(packed);
                const __m512i product_even = _mm512_mul_epi32(a, weights);
                const __m512i product_odd = _mm512_mul_epi32(a, _mm512_srli_epi64(weights, 32));

                __m512i even = _mm512_loadu_si512(reinterpret_cast<const void *>(dst_even + i / 2));
                __m512i odd = _mm512_loadu_si512(reinterpret_cast<const void *>(dst_odd + i / 2));
                even = _mm512_add_epi64(even, product_even);
                odd = _mm512_add_epi64(odd, product_odd);
                _mm512_storeu_si512(reinterpret_cast<void *>(dst_even + i / 2), even);
                _mm512_storeu_si512(reinterpret_cast<void *>(dst_odd + i / 2), odd);
            }

            for (; i < count; ++i)
            {
                const std::int64_t product =
                    static_cast<std::int64_t>(activation) * static_cast<std::int64_t>(row[i]);
                if ((i & 1) == 0)
                    dst_even[i / 2] += product;
                else
                    dst_odd[i / 2] += product;
            }
        }
#endif

#if SHAKEYBOT_HAS_AVX2_KERNELS
        SHAKEYBOT_AVX2_TARGET static inline std::int64_t neural_dot_relu_i32_i16_avx2(const std::int32_t *input,
                                                                                      const std::int16_t *weights,
                                                                                      int count) noexcept
        {
            const __m256i zero = _mm256_setzero_si256();
            __m256i sum = _mm256_setzero_si256();
            int i = 0;
            for (; i + 8 <= count; i += 8)
            {
                __m256i activations = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
                activations = _mm256_max_epi32(activations, zero);
                const __m128i packed_weights =
                    _mm_loadu_si128(reinterpret_cast<const __m128i *>(weights + i));
                const __m256i expanded_weights = _mm256_cvtepi16_epi32(packed_weights);

                sum = _mm256_add_epi64(sum, _mm256_mul_epi32(activations, expanded_weights));
                sum = _mm256_add_epi64(
                    sum,
                    _mm256_mul_epi32(_mm256_srli_si256(activations, 4),
                                     _mm256_srli_si256(expanded_weights, 4)));
            }

            alignas(32) std::int64_t lanes[4];
            _mm256_store_si256(reinterpret_cast<__m256i *>(lanes), sum);
            std::int64_t total = lanes[0] + lanes[1] + lanes[2] + lanes[3];
            for (; i < count; ++i)
            {
                const std::int32_t activation = std::max(input[i], 0);
                total += static_cast<std::int64_t>(activation) * static_cast<std::int64_t>(weights[i]);
            }
            return total;
        }

        SHAKEYBOT_AVX2_TARGET static inline std::int64_t neural_dot_clamped_i32_i16_avx2(
            const std::int32_t *input,
            const std::int16_t *weights,
            int count,
            std::int32_t max_value) noexcept
        {
            const __m256i zero = _mm256_setzero_si256();
            const __m256i max_activation = _mm256_set1_epi32(max_value);
            __m256i sum = _mm256_setzero_si256();
            int i = 0;
            for (; i + 8 <= count; i += 8)
            {
                __m256i activations = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
                activations = _mm256_min_epi32(_mm256_max_epi32(activations, zero), max_activation);
                const __m128i packed_weights =
                    _mm_loadu_si128(reinterpret_cast<const __m128i *>(weights + i));
                const __m256i expanded_weights = _mm256_cvtepi16_epi32(packed_weights);

                sum = _mm256_add_epi64(sum, _mm256_mul_epi32(activations, expanded_weights));
                sum = _mm256_add_epi64(
                    sum,
                    _mm256_mul_epi32(_mm256_srli_si256(activations, 4),
                                     _mm256_srli_si256(expanded_weights, 4)));
            }

            alignas(32) std::int64_t lanes[4];
            _mm256_store_si256(reinterpret_cast<__m256i *>(lanes), sum);
            std::int64_t total = lanes[0] + lanes[1] + lanes[2] + lanes[3];
            for (; i < count; ++i)
            {
                const std::int32_t activation = std::clamp(input[i], 0, max_value);
                total += static_cast<std::int64_t>(activation) * static_cast<std::int64_t>(weights[i]);
            }
            return total;
        }

        SHAKEYBOT_AVX2_TARGET static inline void neural_accumulate_dual_i32_i16_to_i64_avx2(
            std::int64_t *dst,
            const std::int16_t *row_a,
            std::int32_t activation_a,
            const std::int16_t *row_b,
            std::int32_t activation_b,
            int count) noexcept
        {
            const __m256i a = _mm256_set1_epi32(activation_a);
            const __m256i b = _mm256_set1_epi32(activation_b);
            int i = 0;
            for (; i + 8 <= count; i += 8)
            {
                const __m128i packed_a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(row_a + i));
                const __m128i packed_b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(row_b + i));
                const __m256i weights_a = _mm256_cvtepi16_epi32(packed_a);
                const __m256i weights_b = _mm256_cvtepi16_epi32(packed_b);
                const __m256i product32 = _mm256_add_epi32(
                    _mm256_mullo_epi32(weights_a, a),
                    _mm256_mullo_epi32(weights_b, b));

                const __m128i product_lo = _mm256_castsi256_si128(product32);
                const __m128i product_hi = _mm256_extracti128_si256(product32, 1);
                const __m256i product64_lo = _mm256_cvtepi32_epi64(product_lo);
                const __m256i product64_hi = _mm256_cvtepi32_epi64(product_hi);

                __m256i dst_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(dst + i));
                __m256i dst_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(dst + i + 4));
                dst_lo = _mm256_add_epi64(dst_lo, product64_lo);
                dst_hi = _mm256_add_epi64(dst_hi, product64_hi);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i), dst_lo);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i + 4), dst_hi);
            }

            for (; i < count; ++i)
            {
                dst[i] += static_cast<std::int64_t>(activation_a) * static_cast<std::int64_t>(row_a[i]);
                dst[i] += static_cast<std::int64_t>(activation_b) * static_cast<std::int64_t>(row_b[i]);
            }
        }

        SHAKEYBOT_AVX2_TARGET static inline void neural_accumulate_i32_i16_to_i64_evenodd_avx2(
            std::int64_t *dst_even,
            std::int64_t *dst_odd,
            const std::int16_t *row,
            std::int32_t activation,
            int count) noexcept
        {
            const __m256i a = _mm256_set1_epi32(activation);
            int i = 0;
            for (; i + 8 <= count; i += 8)
            {
                const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(row + i));
                const __m256i weights = _mm256_cvtepi16_epi32(packed);
                const __m256i product_even = _mm256_mul_epi32(a, weights);
                const __m256i product_odd = _mm256_mul_epi32(_mm256_srli_si256(a, 4),
                                                             _mm256_srli_si256(weights, 4));

                __m256i even = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(dst_even + i / 2));
                __m256i odd = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(dst_odd + i / 2));
                even = _mm256_add_epi64(even, product_even);
                odd = _mm256_add_epi64(odd, product_odd);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_even + i / 2), even);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_odd + i / 2), odd);
            }

            for (; i < count; ++i)
            {
                const std::int64_t product =
                    static_cast<std::int64_t>(activation) * static_cast<std::int64_t>(row[i]);
                if ((i & 1) == 0)
                    dst_even[i / 2] += product;
                else
                    dst_odd[i / 2] += product;
            }
        }
#endif

#include "eval/00_cache.inc"
#include "eval/01_material_phase.inc"
#include "eval/02_material_imbalance_sf12.inc"
#include "eval/03_endgame_scale_king_crowding.inc"
#include "eval/03a_kpk_bitbase.inc"
#include "eval/03b_eval_shared.inc"
#include "eval/04_mobility.inc"
#include "eval/05_outposts.inc"
#include "eval/06_pawn_structure.inc"
#include "eval/07_pawn_hash_passed.inc"
#include "eval/08_king_safety.inc"
#include "eval/09_king_cover_cache.inc"
#include "eval/10_king_zone_pressure.inc"
#include "eval/11_bishop_pair_bad_bishop.inc"
#include "eval/12_rook_activity.inc"
#include "eval/13_xray_pins.inc"
#include "eval/14_psqt.inc"
#include "eval/15_threats_space.inc"
#include "eval/16_closedness.inc"
#include "eval/17_complexity.inc"
#include "eval/18_queen_vulnerability.inc"
#include "eval/19_neural_dummy.inc"
#include "eval/20_neural_simple.inc"
#include "eval/21_neural_quant.inc"
#include "eval/22_neural_halfkp.inc"

    } // anonymous namespace

#include "eval/99_public.inc"

} // namespace fast_engine

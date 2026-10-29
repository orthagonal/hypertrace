#pragma once
#include <cstdint>
#include <cstddef>

// On-disk layout, little-endian. Kept intentionally small and stable for v0.
static constexpr unsigned HDC_DIM_BITS = 1024;
static constexpr unsigned HDC_WORDS = 32;
static constexpr unsigned RECENT_ACTIONS = 8;
static constexpr unsigned FACT_BITS = 12;
static constexpr char DATA_MAGIC[8] = {'H','D','B','S','A','F','E','1'};
struct Record {
    uint32_t split, label, flags, prefix_len;
    uint32_t recent[RECENT_ACTIONS];
};
static_assert(sizeof(Record) == 48, "binary dataset layout differs from Rust");

// Same operation in CUDA and CPU reference; requires no CUDA headers.
#if defined(__CUDACC__)
#define HD __host__ __device__
#else
#define HD
#endif
HD inline uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
HD inline uint32_t hyperword(uint32_t token, uint32_t word) {
    return mix32(token * 0x9e3779b9u + word * 0x85ebca6bu + 0xa5f152bdu);
}
// 12 binary facts (including zero states), plus 8 position-tagged actions.
HD inline uint32_t feature_token(const Record& r, unsigned index) {
    if (index < FACT_BITS) {
        return 1000u + 2u * index + ((r.flags >> index) & 1u);
    }
    unsigned pos = index - FACT_BITS;
    return 2000u + 32u * pos + r.recent[pos];
}
HD inline uint32_t encode_word(const Record& r, unsigned word) {
    // Bitwise majority of 20 item vectors. Tie uses 1 deterministically.
    uint32_t out = 0;
    for (unsigned bit=0; bit<32; bit++) {
        unsigned votes = 0;
        for (unsigned token=0; token<FACT_BITS+RECENT_ACTIONS; token++) {
            votes += (hyperword(feature_token(r, token), word) >> bit) & 1u;
        }
        if (votes * 2 >= FACT_BITS+RECENT_ACTIONS) out |= (1u << bit);
    }
    return out;
}
#undef HD

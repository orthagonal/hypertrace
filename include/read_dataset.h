#pragma once
#include "safety_data.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

inline uint32_t get_u32(std::istream& in) {
    unsigned char buf[4];
    if (!in.read(reinterpret_cast<char*>(buf), 4)) throw std::runtime_error("short binary dataset");
    return (uint32_t)buf[0] | ((uint32_t)buf[1] << 8) |
           ((uint32_t)buf[2] << 16) | ((uint32_t)buf[3] << 24);
}
inline std::vector<Record> load_dataset(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open dataset: " + filename);
    char magic[8];
    if (!in.read(magic, 8) || std::memcmp(magic, DATA_MAGIC, 8))
        throw std::runtime_error("bad dataset magic");
    uint32_t ver = get_u32(in), n = get_u32(in), bits = get_u32(in), words = get_u32(in);
    if (ver != 1 || bits != HDC_DIM_BITS || words != sizeof(Record)/4 || n > 2000000)
        throw std::runtime_error("unsupported version, dimensions, record size or count");
    std::vector<Record> records(n);
    for (Record& r : records) {
        r.split = get_u32(in);
        r.label = get_u32(in);
        r.flags = get_u32(in);
        r.prefix_len = get_u32(in);
        for (auto& action : r.recent) action = get_u32(in);
        if (r.split > 3 || r.label > 1) throw std::runtime_error("invalid split/label");
    }
    if (in.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error("trailing bytes in binary dataset");
    return records;
}

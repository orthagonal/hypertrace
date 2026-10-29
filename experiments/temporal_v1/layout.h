#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace temporal {
constexpr uint32_t WORDS=32, BITS=1024, FANOUT=32, PROPERTIES=8;
enum Property : uint32_t { PrivateRead=1, External=2, Approval=4, Derived=8,
    Sensitive=16, Local=32, Untrusted=64, Filler=128 };
inline const char* property_name(unsigned p) {
    static const char* names[]={"private_read","external","approval","derived",
        "sensitive","local","untrusted","filler"};
    return names[p];
}
struct Event {
    uint32_t action, object, source, flags, step, dependency_begin, dependency_count, reserved;
};
struct Edge { uint32_t consumer_step, producer_step, object; };
struct Leaf { Event event; uint32_t hv[WORDS]; };
struct Node {
    uint32_t hv[WORDS];
    uint32_t start_step, child_index, child_count, properties;
    uint32_t masks[PROPERTIES];
};
static_assert(sizeof(Event)==32 && sizeof(Edge)==12 && sizeof(Leaf)==160 && sizeof(Node)==176,
              "temporal binary layout mismatch");
using Vector=std::array<uint32_t,WORDS>;
struct History {
    uint32_t id=0, pair_id=0, cutoff=0;
    std::vector<Event> events;
    std::vector<Edge> edges;
};
// Never part of History or any CUDA input buffer.
struct Outcome { uint32_t history_id, label, scenario, category; };
struct Dataset { std::vector<History> histories; std::vector<Outcome> outcomes; };
struct Tree {
    std::vector<Leaf> leaves;
    std::vector<std::vector<Node>> levels; // levels[0] is level 1.
};
struct Search { std::vector<uint32_t> events; std::vector<std::array<uint32_t,2>> regions; };
} // namespace temporal

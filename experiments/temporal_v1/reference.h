#pragma once
#include "io.h"
#include <functional>

namespace temporal {
// Independent scalar CPU implementation: no CUDA helpers or production encoder.
inline uint32_t reference_hash(uint32_t v) {
    v=(v^(v>>16))*0x7feb352du;
    v=(v^(v>>15))*0x846ca68bu;
    return v^(v>>16);
}
inline Vector reference_event(const Event& e) {
    Vector result{};
    for (unsigned w=0;w<WORDS;++w) {
        uint32_t tokens[11];
        for (unsigned tag=1;tag<=11;++tag) {
            const uint32_t value=tag==1 ? e.action : tag==2 ? e.object : tag==3 ? e.source : ((e.flags>>(tag-4))&1u);
            tokens[tag-1]=reference_hash(tag*0x9e3779b9u ^ value ^ w*0x85ebca6bu ^ 0xa5f152bdu);
        }
        for (unsigned bit=0;bit<32;++bit) {
            unsigned count=0; for (auto word:tokens) count+=(word>>bit)&1u;
            if (count>=6) result[w]|=1u<<bit;
        }
    }
    return result;
}
inline Tree reference_build(const std::vector<Event>& events) {
    Tree tree;
    for (const auto& event:events) {
        Leaf leaf{}; leaf.event=event; const auto v=reference_event(event);
        std::copy(v.begin(),v.end(),leaf.hv); tree.leaves.push_back(leaf);
    }
    size_t count=events.size(); unsigned level=1;
    while (count>1) {
        std::vector<Node> parents((count+31)/32);
        for (size_t p=0;p<parents.size();++p) {
            auto& node=parents[p]; node.child_index=uint32_t(p*32);
            node.child_count=uint32_t(std::min<size_t>(32,count-p*32));
            node.start_step=level==1 ? tree.leaves[p*32].event.step : tree.levels.back()[p*32].start_step;
            for (unsigned c=0;c<node.child_count;++c) {
                const uint32_t props=level==1 ? tree.leaves[p*32+c].event.flags : tree.levels.back()[p*32+c].properties;
                node.properties|=props;
                for (unsigned property=0;property<PROPERTIES;++property)
                    if (props&(1u<<property)) node.masks[property]|=1u<<c;
            }
            // Scalar bit-address implementation of the position transform.
            for (unsigned w=0;w<WORDS;++w) {
                const uint32_t tie=reference_hash(0xd1b54a35u ^ level*0x94d049bbu ^ w*0x369dea0fu);
                uint32_t bound[32];
                for (unsigned c=0;c<node.child_count;++c) {
                    const auto* hv=level==1 ? tree.leaves[p*32+c].hv : tree.levels.back()[p*32+c].hv;
                    const uint32_t content=hv[(w+(c*7+level)%32)%32];
                    const uint32_t position=reference_hash(0x51ed270bu ^ level*0x9e3779b9u ^ c*0x85ebca6bu ^ w*0xc2b2ae35u);
                    // Explicit bit copying rather than a shared rotation primitive.
                    bound[c]=position;
                    for (unsigned bit=0;bit<32;++bit)
                        bound[c]^=((content>>((bit+32-c)%32))&1u)<<bit;
                }
                for (unsigned bit=0;bit<32;++bit) {
                    unsigned votes=0;
                    for (unsigned c=0;c<node.child_count;++c) votes+=(bound[c]>>bit)&1u;
                    if (2*votes>node.child_count || (2*votes==node.child_count && ((tie>>bit)&1u))) node.hv[w]|=1u<<bit;
                }
            }
        }
        count=parents.size(); tree.levels.push_back(std::move(parents)); ++level;
    }
    return tree;
}
inline unsigned reference_distance(const uint32_t* a,const uint32_t* b) {
    unsigned distance=0;
    for (unsigned w=0;w<WORDS;++w) for (unsigned bit=0;bit<32;++bit)
        distance+=((a[w]>>bit)&1u)!=((b[w]>>bit)&1u);
    return distance;
}
inline Vector root(const Tree& tree) {
    Vector out{};
    if (!tree.levels.empty()) std::copy_n(tree.levels.back()[0].hv,WORDS,out.begin());
    else if (!tree.leaves.empty()) std::copy_n(tree.leaves[0].hv,WORDS,out.begin());
    return out; // Empty root is a documented sentinel, never searched by similarity.
}
inline Search exact_search(const Tree& tree,unsigned property) {
    require(property<PROPERTIES,"invalid query property");
    Search result;
    if (tree.leaves.empty()) return result;
    std::function<void(unsigned,uint32_t)> visit=[&](unsigned level,uint32_t index) {
        if (!level) {
            if (tree.leaves[index].event.flags&(1u<<property)) result.events.push_back(index);
            return;
        }
        const auto& node=tree.levels[level-1][index];
        if (!node.masks[property]) return;
        result.regions.push_back({level,index});
        for (unsigned c=0;c<node.child_count;++c)
            if (node.masks[property]&(1u<<c)) visit(level-1,node.child_index+c);
    };
    visit(unsigned(tree.levels.size()),0);
    return result;
}
inline void verify_masks(const Tree& tree) {
    for (unsigned level=1;level<=tree.levels.size();++level) {
        const auto& nodes=tree.levels[level-1];
        const size_t count=level==1 ? tree.leaves.size() : tree.levels[level-2].size();
        for (size_t i=0;i<nodes.size();++i) {
            const auto& n=nodes[i];
            require(n.child_index==i*32 && n.child_count==std::min<size_t>(32,count-i*32),"invalid exact child range");
            uint32_t combined=0;
            for (unsigned prop=0;prop<PROPERTIES;++prop) {
                uint32_t expected=0;
                for (unsigned c=0;c<n.child_count;++c) {
                    const auto flags=level==1 ? tree.leaves[n.child_index+c].event.flags : tree.levels[level-2][n.child_index+c].properties;
                    if (flags&(1u<<prop)) expected|=1u<<c;
                }
                require(n.masks[prop]==expected,"exact child mask mismatch");
                if (expected) combined|=1u<<prop;
            }
            require(n.properties==combined,"property union mismatch");
        }
    }
    for (unsigned prop=0;prop<PROPERTIES;++prop) {
        const auto found=exact_search(tree,prop).events;
        std::vector<uint32_t> scan;
        for (uint32_t i=0;i<tree.leaves.size();++i) if (tree.leaves[i].event.flags&(1u<<prop)) scan.push_back(i);
        require(found==scan,"mask traversal missed or invented events");
    }
}
} // namespace temporal

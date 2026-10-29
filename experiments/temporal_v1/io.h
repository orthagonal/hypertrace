#pragma once
#include "layout.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace temporal {
inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
inline uint32_t read32(std::istream& in) {
    unsigned char b[4];
    if (!in.read(reinterpret_cast<char*>(b),4)) throw std::runtime_error("short temporal file");
    return uint32_t(b[0]) | (uint32_t(b[1])<<8) | (uint32_t(b[2])<<16) | (uint32_t(b[3])<<24);
}
inline void write32(std::ostream& out, uint32_t n) {
    const unsigned char b[]={static_cast<unsigned char>(n),static_cast<unsigned char>(n>>8),
        static_cast<unsigned char>(n>>16),static_cast<unsigned char>(n>>24)};
    out.write(reinterpret_cast<const char*>(b),4);
}
inline Dataset load(const std::filesystem::path& path) {
    std::ifstream in(path,std::ios::binary);
    require(bool(in),"cannot open temporal dataset");
    char magic[8]; require(bool(in.read(magic,8)) && std::memcmp(magic,"HDBTMP01",8)==0,"bad temporal magic");
    uint32_t h[14]; for (auto& v:h) v=read32(in);
    require(h[0]==1 && h[1]==64 && h[2]<=10000 && h[3]<=2000000 && h[4]<=4000000 &&
        h[5]==32 && h[6]==12 && h[7]==32 && h[8]==64 && h[12]==16 && h[13]==0,"unsupported temporal header");
    const uint64_t event_offset=64ull+32ull*h[2], edge_offset=event_offset+32ull*h[3],
        label_offset=edge_offset+12ull*h[4], file_bytes=label_offset+16ull*h[2];
    require(h[9]==event_offset && h[10]==edge_offset && h[11]==label_offset &&
        std::filesystem::file_size(path)==file_bytes,"invalid temporal offsets or file length");
    Dataset data;
    struct Descriptor { uint32_t words[8]; };
    std::vector<Descriptor> desc(h[2]);
    uint32_t ne=0, nd=0;
    for (uint32_t id=0;id<h[2];++id) {
        auto& d=desc[id].words; for (auto& v:d) v=read32(in);
        require(d[0]==id && d[2]==ne && d[4]==nd && d[3]<=h[3]-ne && d[5]<=h[4]-nd && d[7]==0,
                "invalid history descriptor");
        ne+=d[3]; nd+=d[5];
        History history; history.id=id; history.pair_id=d[1]; history.cutoff=d[6];
        history.events.resize(d[3]); history.edges.resize(d[5]); data.histories.push_back(std::move(history));
    }
    require(ne==h[3] && nd==h[4],"unowned temporal records");
    for (auto& history:data.histories) for (auto& e:history.events) {
        e.action=read32(in); e.object=read32(in); e.source=read32(in); e.flags=read32(in);
        e.step=read32(in); e.dependency_begin=read32(in); e.dependency_count=read32(in); e.reserved=read32(in);
    }
    for (auto& history:data.histories) for (auto& e:history.edges) {
        e.consumer_step=read32(in); e.producer_step=read32(in); e.object=read32(in);
    }
    for (uint32_t id=0;id<h[2];++id) {
        Outcome o{read32(in),read32(in),read32(in),read32(in)};
        require(o.history_id==id && o.label<=1,"invalid outcome metadata"); data.outcomes.push_back(o);
    }
    for (const auto& history:data.histories) {
        std::map<uint32_t,const Event*> observed;
        uint32_t owned_edges=0, previous=0;
        for (size_t i=0;i<history.events.size();++i) {
            const auto& e=history.events[i];
            require(e.reserved==0 && e.action>=1 && e.action<=17 && !(e.flags&~255u) &&
                e.step<history.cutoff && (!i || e.step>previous),"invalid or future observed event");
            require(e.dependency_begin==owned_edges && e.dependency_count<=history.edges.size()-owned_edges,
                "invalid dependency range");
            for (uint32_t j=0;j<e.dependency_count;++j) {
                const auto& edge=history.edges[owned_edges+j];
                const auto source=observed.find(edge.producer_step);
                require(edge.consumer_step==e.step && edge.producer_step<e.step && source!=observed.end() &&
                    source->second->object==edge.object,"future or mismatched dependency");
            }
            owned_edges+=e.dependency_count; previous=e.step; observed[e.step]=&e;
        }
        require(owned_edges==history.edges.size(),"unowned exact edges");
        require(!history.events.empty() || history.cutoff==0,"empty history has nonzero cutoff");
    }
    return data;
}
inline void write_event(std::ostream& out,const Event& e) {
    for (auto n:{e.action,e.object,e.source,e.flags,e.step,e.dependency_begin,e.dependency_count,e.reserved}) write32(out,n);
}
inline void write_tree(const std::filesystem::path& path, const History& history,const Tree& tree) {
    const uint32_t levels=uint32_t(tree.levels.size()+1), table=64,
        leaves=table+levels*24, edges=leaves+uint32_t(tree.leaves.size())*160,
        parents=edges+uint32_t(history.edges.size())*12;
    std::ofstream out(path,std::ios::binary); require(bool(out),"cannot write temporal memory");
    out.exceptions(std::ios::failbit|std::ios::badbit);
    out.write("HDBMEM01",8);
    for (auto n:{1u,64u,history.id,uint32_t(tree.leaves.size()),uint32_t(history.edges.size()),levels,
            160u,176u,12u,24u,table,leaves,edges,parents}) write32(out,n);
    uint32_t offset=parents;
    for (uint32_t level=0;level<levels;++level) {
        const uint32_t count=level ? uint32_t(tree.levels[level-1].size()) : uint32_t(tree.leaves.size());
        const uint32_t size=level ? 176 : 160;
        for (auto n:{level,count,size,level ? offset : leaves,count*size,0u}) write32(out,n);
        if (level) offset+=count*size;
    }
    for (const auto& leaf:tree.leaves) {
        write_event(out,leaf.event); for (auto word:leaf.hv) write32(out,word);
    }
    for (const auto& e:history.edges) for (auto n:{e.consumer_step,e.producer_step,e.object}) write32(out,n);
    for (const auto& level:tree.levels) for (const auto& node:level) {
        for (auto word:node.hv) write32(out,word);
        for (auto n:{node.start_step,node.child_index,node.child_count,node.properties}) write32(out,n);
        for (auto mask:node.masks) write32(out,mask);
    }
    out.close();
}
inline std::string history_file(uint32_t id) {
    return "history_"+std::to_string(id)+".mem";
}
} // namespace temporal

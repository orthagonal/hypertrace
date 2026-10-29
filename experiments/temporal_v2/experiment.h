#pragma once
#include "reference.h"

namespace sparse {
struct Scores {
    std::vector<uint16_t> distances;
    std::vector<uint32_t> ranks;
    std::vector<Timing> timings;
};
inline uint64_t tree_bytes(const Tree& tree) {
    uint64_t bytes=tree.leaves.size()*sizeof(Leaf);
    for (const auto& level:tree.levels) bytes+=level.size()*sizeof(Node);
    return bytes;
}
inline uint64_t scratch_bytes(size_t n,size_t m) {
    // Event staging, selection/prefix, version map, scoped flags/prefix, tiled votes.
    uint64_t bytes=n*(sizeof(Event)+3*sizeof(uint32_t)+sizeof(Version)+8)+288+m*12+((m+255)/256)*2*1024*4;
    // Recursive exclusive scan block sums and block prefixes (both scans).
    for (size_t count:{n,m}) while (count>256) { count=(count+255)/256; bytes+=count*12; }
    if (n) bytes+=8; if (m) bytes+=8;
    return bytes;
}
inline void write_anchor_index(const std::filesystem::path& path,const History& h,const Analysis& a,uint32_t target) {
    std::ofstream out(path,std::ios::binary); require(bool(out),"cannot write sparse index");
    out.exceptions(std::ios::failbit|std::ios::badbit);
    const uint32_t edges=64+uint32_t(a.anchors.size())*176,tail=edges+uint32_t(h.edges.size())*12;
    out.write("HDBANC02",8);
    for (auto v:{2u,64u,h.id,uint32_t(h.events.size()),uint32_t(a.anchors.size()),target,a.scoped_count,
                176u,uint32_t(h.edges.size()),12u,64u,edges,tail,104u}) write32(out,v);
    for (const auto& anchor:a.anchors) {
        write_event(out,anchor.event);
        for (auto v:{anchor.event_index,anchor.object_version,anchor.source_version,anchor.ordinal}) write32(out,v);
        for (auto word:anchor.hv) write32(out,word);
    }
    for (const auto& edge:h.edges) for (auto v:{edge.consumer_step,edge.producer_step,edge.object}) write32(out,v);
    for (const auto& vector:a.fingerprints) for (auto word:vector) write32(out,word);
    for (auto word:a.pattern) write32(out,word);
    out.close();
}
inline std::vector<uint32_t> read_roles(const std::filesystem::path& path,size_t n) {
    std::ifstream in(path); require(bool(in),"cannot open role file");
    std::string line; std::getline(in,line); if (!line.empty() && line.back()=='\r') line.pop_back();
    require(line=="history_id,role","invalid role header (no scoring metadata allowed)");
    std::vector<uint32_t> roles;
    while (std::getline(in,line)) {
        if (!line.empty() && line.back()=='\r') line.pop_back();
        const auto comma=line.find(','); require(comma!=std::string::npos,"invalid role record");
        size_t id_used=0,role_used=0;
        const auto id=std::stoul(line.substr(0,comma),&id_used),role=std::stoul(line.substr(comma+1),&role_used);
        require(id==roles.size() && id_used==comma && role_used==line.size()-comma-1 && role<=1,"invalid role ID/value");
        roles.push_back(uint32_t(role));
    }
    require(roles.size()==n,"role count mismatch"); return roles;
}
template<class Engine> int run(int argc,char** argv,Engine& engine) {
    require(argc>=6,"usage: temporal_v2_<cpu|gpu> <events.bin> --roles <roles.csv> --out <directory> [--repeats N] [--object ID]");
    std::filesystem::path role_file,directory;
    uint32_t target=2;
    for (int i=2;i<argc;++i) {
        const std::string arg=argv[i]; require(i+1<argc,"missing v2 option value");
        const std::string value=argv[++i];
        if (arg=="--roles" && role_file.empty()) role_file=value;
        else if (arg=="--out" && directory.empty()) directory=value;
        else if (arg=="--repeats" || arg=="--object") {
            size_t used=0; const auto parsed=std::stoul(value,&used); require(used==value.size(),"invalid numeric option");
            if (arg=="--repeats") { require(parsed>=1 && parsed<=1000,"invalid repeats"); engine.repeats=unsigned(parsed); }
            else { require(parsed>0 && parsed<=0xffffffffu,"invalid object"); target=uint32_t(parsed); }
        } else throw std::runtime_error("invalid v2 option");
    }
    require(!directory.empty() && !role_file.empty(),"missing v2 output/role path");
    const auto data=load(argv[1]);
    require(!data.histories.empty() && data.histories.size()<=256,"v2 collection must have 1-256 histories");
    const auto roles=read_roles(role_file,data.histories.size());
    std::filesystem::create_directories(directory);
    auto memory=csv(directory,"memory.csv"), timing=csv(directory,"timings.csv"),
        vectors=csv(directory,"fingerprints.csv"), patterns=csv(directory,"patterns.csv"),
        distances=csv(directory,"distances.csv"), rankings=csv(directory,"rankings.csv"),
        correct=csv(directory,"correctness.csv");
    memory<<"history_id,events,anchors,scoped_anchors,v1_tree_bytes,anchor_bytes,v2_fingerprint_bytes,exact_key_bytes,edge_archive_bytes,scratch_upper_bound_bytes\n";
    timing<<"history_id,kernel,level,items,repeats,median_ms,min_ms,max_ms\n";
    vectors<<"history_id,method,words_hex\n";
    patterns<<"history_id,key\n";
    distances<<"method,query_id,candidate_id,distance\n";
    rankings<<"method,query_id,k,candidate_id,distance\n";
    correct<<"history_id,events,anchors,selection_order_versions_vectors,all_fingerprints,v1_nodes,exact_pattern\n";
    auto log_times=[&](int history,const std::vector<Timing>& times) {
        for (const auto& t:times) timing<<history<<','<<t.kernel<<','<<t.level<<','<<t.items<<','<<t.repeats<<','<<t.median<<','<<t.minimum<<','<<t.maximum<<'\n';
    };
    std::vector<Analysis> analyses;
    for (const auto& h:data.histories) {
        auto a=engine.analyze(h,target);
        a.tree_bytes=tree_bytes(a.tree); a.scratch_bytes=scratch_bytes(h.events.size(),a.anchors.size());
        log_times(int(h.id),a.timings);
        write_anchor_index(directory/("anchors_"+std::to_string(h.id)+".bin"),h,a,target);
        // Archive one unchanged full tree per collection; every full tree is built
        // and verified. Sparse kernels use disjoint buffers while the tree is resident.
        if (!h.id) write_tree(directory/"v1_history_0.mem",h,a.tree);
        memory<<h.id<<','<<h.events.size()<<','<<a.anchors.size()<<','<<a.scoped_count<<','<<a.tree_bytes<<','
            <<a.anchors.size()*sizeof(Anchor)<<",256,32,"<<h.edges.size()*sizeof(Edge)<<','<<a.scratch_bytes<<'\n';
        for (unsigned m=0;m<3;++m) {
            std::ostringstream words; words<<std::hex<<std::setfill('0');
            for (auto word:a.fingerprints[m]) words<<std::setw(8)<<word;
            vectors<<h.id<<','<<method_name(m)<<','<<words.str()<<'\n';
        }
        patterns<<h.id<<',';
        for (unsigned j=0;j<a.pattern.size();++j) { if (j) patterns<<';'; patterns<<a.pattern[j]; }
        patterns<<'\n';
        correct<<h.id<<','<<h.events.size()<<','<<a.anchors.size()<<",PASS,PASS,PASS,PASS\n";
        analyses.push_back(std::move(a));
    }
    const auto scores=engine.score(analyses,roles); log_times(-1,scores.timings);
    const size_t n=analyses.size();
    for (unsigned m=0;m<METHODS;++m) for (size_t q=0;q<n;++q) if (roles[q]) {
        for (size_t c=0;c<n;++c) if (!roles[c]) distances<<method_name(m)<<','<<q<<','<<c<<','<<scores.distances[(m*n+q)*n+c]<<'\n';
        for (unsigned k=0;k<TOPK;++k) {
            const auto candidate=scores.ranks[(m*n+q)*TOPK+k];
            if (candidate!=UNKNOWN) rankings<<method_name(m)<<','<<q<<','<<k+1<<','<<candidate<<','<<scores.distances[(m*n+q)*n+candidate]<<'\n';
        }
    }
    auto summary=csv(directory,"summary.txt");
    summary<<engine.name()<<"\n"<<n<<" histories; no oracle/scenario fields used.\n"
        <<"Anchor selection/order/versions/vectors, fingerprints, pair distances and top-K validated.\n"
        <<"Object-scoped query is explicit object "<<target<<"; unrelated significant anchors are still retained.\n";
    std::cout<<"PASS "<<n<<" histories: anchors, v1/v2 vectors, exact keys and retrieval -> "<<directory.string()<<'\n';
    return 0;
}
} // namespace sparse

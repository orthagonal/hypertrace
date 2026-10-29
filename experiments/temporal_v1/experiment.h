#pragma once
#include "reference.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

namespace temporal {
struct Timing {
    std::string kernel;
    unsigned level=0, items=0, repeats=0;
    double median=0, minimum=0, maximum=0;
};
struct Evaluation {
    Tree tree;
    Vector query{};
    std::vector<std::vector<uint16_t>> distances;
    std::array<Search,PROPERTIES> searches;
    std::vector<Timing> timings;
};
inline Vector query_vector(const std::vector<Event>& events) {
    for (const auto& event:events) if (event.flags&PrivateRead) return reference_event(event);
    return reference_event(Event{1,1,0,PrivateRead|Sensitive,0,0,0,0});
}
inline void check_equal(const Tree& gpu,const Tree& cpu) {
    require(gpu.leaves.size()==cpu.leaves.size() && gpu.levels.size()==cpu.levels.size(),"hierarchy size mismatch");
    if (!cpu.leaves.empty()) require(std::memcmp(gpu.leaves.data(),cpu.leaves.data(),cpu.leaves.size()*sizeof(Leaf))==0,
        "GPU leaf fields or vector differ from independent CPU reference");
    for (size_t l=0;l<cpu.levels.size();++l) {
        require(gpu.levels[l].size()==cpu.levels[l].size(),"hierarchy level count mismatch");
        require(std::memcmp(gpu.levels[l].data(),cpu.levels[l].data(),cpu.levels[l].size()*sizeof(Node))==0,
            "GPU summary, range or exact mask differs from independent CPU reference");
    }
    verify_masks(gpu);
}
inline std::ofstream csv(const std::filesystem::path& dir,const char* name) {
    std::ofstream out(dir/name); require(bool(out),"cannot open temporal report");
    out.exceptions(std::ios::failbit|std::ios::badbit);
    out<<std::fixed<<std::setprecision(6); return out;
}
inline const uint32_t* level_vector(const Tree& tree,unsigned level,size_t i) {
    return level ? tree.levels[level-1][i].hv : tree.leaves[i].hv;
}
inline size_t level_count(const Tree& tree,unsigned level) {
    return level ? tree.levels[level-1].size() : tree.leaves.size();
}
inline void sensitivity(std::ostream& out,const History& history,const Tree& a,const Tree& b,const char* variant) {
    const auto ar=root(a), br=root(b);
    const int root_distance=a.leaves.empty() || b.leaves.empty() ? -1 : int(reference_distance(ar.data(),br.data()));
    const unsigned levels=unsigned(std::min(a.levels.size(),b.levels.size())+1);
    for (unsigned level=0;level<levels;++level) {
        const size_t n=std::min(level_count(a,level),level_count(b,level));
        uint64_t sum=0; unsigned max=0;
        for (size_t i=0;i<n;++i) {
            const auto d=reference_distance(level_vector(a,level,i),level_vector(b,level,i));
            sum+=d; max=std::max(max,d);
        }
        out<<history.id<<','<<history.pair_id<<','<<a.leaves.size()<<','<<variant<<','<<b.leaves.size()<<',';
        if (root_distance<0) out<<"NA"; else out<<root_distance;
        out<<','<<level<<','<<n<<','; if (n) out<<double(sum)/n; else out<<"NA";
        out<<','<<max<<'\n';
    }
}
template<class Engine> int run_experiment(int argc,char** argv,Engine& engine) {
    require(argc>=4,"usage: temporal_v1_<cpu|gpu> <dataset> --out <directory> [--repeats N]");
    std::filesystem::path directory;
    for (int i=2;i<argc;++i) {
        std::string arg=argv[i];
        if (arg=="--out" && directory.empty() && i+1<argc) directory=argv[++i];
        else if (arg=="--repeats" && i+1<argc) {
            const std::string value=argv[++i]; size_t used=0;
            const auto repeats=std::stoul(value,&used);
            require(used==value.size() && repeats>=1 && repeats<=1000,"invalid repetition count");
            engine.repeats=unsigned(repeats);
        } else throw std::runtime_error("invalid temporal experiment arguments");
    }
    require(!directory.empty(),"missing --out directory");
    const auto data=load(argv[1]);
    std::filesystem::create_directories(directory);
    auto memory=csv(directory,"memory.csv"), timing=csv(directory,"timings.csv"),
        retrieval=csv(directory,"retrieval.csv"), locations=csv(directory,"retrieved_events.csv"),
        regions=csv(directory,"regions.csv"), queries=csv(directory,"query_distances.csv"),
        changes=csv(directory,"sensitivity.csv"), correctness=csv(directory,"correctness.csv");
    memory<<"history_id,events,level,records,record_bytes,total_bytes\n";
    timing<<"history_id,variant,kernel,level,items,repeats,median_ms,min_ms,max_ms\n";
    retrieval<<"history_id,events,property,exact_matches,regions,scan_matches,verified\n";
    locations<<"history_id,property,event_index,step,action,object,source,flags\n";
    regions<<"history_id,property,level,node_index,start_step,child_index,child_count,child_mask\n";
    queries<<"history_id,level,node_index,hamming_distance\n";
    changes<<"history_id,pair_id,events,variant,other_events,root_hamming,level,aligned_nodes,mean_hamming,max_hamming\n";
    correctness<<"history_id,events,internal_nodes,levels,leaf_fields_and_vectors,all_internal_fields,all_masks,queries,exact_search\n";
    auto log_timing=[&](uint32_t id,const char* variant,const Evaluation& evaluated) {
        for (const auto& t:evaluated.timings) timing<<id<<','<<variant<<','<<t.kernel<<','<<t.level<<','<<t.items
            <<','<<t.repeats<<','<<t.median<<','<<t.minimum<<','<<t.maximum<<'\n';
    };
    uint64_t total_events=0,total_nodes=0,total_hits=0;
    for (size_t index=0;index<data.histories.size();++index) {
        const auto& h=data.histories[index];
        auto evaluated=engine.evaluate(h.events,true);
        const auto& tree=evaluated.tree;
        log_timing(h.id,"original",evaluated);
        write_tree(directory/history_file(h.id),h,tree);
        uint64_t nodes=0;
        for (unsigned l=0;l<=tree.levels.size();++l) {
            const auto n=level_count(tree,l); const unsigned bytes=l ? sizeof(Node) : sizeof(Leaf);
            memory<<h.id<<','<<h.events.size()<<','<<l<<','<<n<<','<<bytes<<','<<n*bytes<<'\n';
            if (l) nodes+=n;
            for (size_t i=0;i<evaluated.distances[l].size();++i)
                queries<<h.id<<','<<l<<','<<i<<','<<evaluated.distances[l][i]<<'\n';
        }
        for (unsigned p=0;p<PROPERTIES;++p) {
            const auto& found=evaluated.searches[p];
            unsigned scan=0; for (const auto& e:h.events) scan+=bool(e.flags&(1u<<p));
            require(found.events.size()==scan,"exact retrieval count mismatch");
            retrieval<<h.id<<','<<h.events.size()<<','<<property_name(p)<<','<<found.events.size()<<','
                <<found.regions.size()<<','<<scan<<",PASS\n";
            for (auto i:found.events) {
                const auto& e=h.events[i];
                locations<<h.id<<','<<property_name(p)<<','<<i<<','<<e.step<<','<<e.action<<','<<e.object<<','<<e.source<<','<<e.flags<<'\n';
            }
            for (auto region:found.regions) {
                const auto& n=tree.levels[region[0]-1][region[1]];
                regions<<h.id<<','<<property_name(p)<<','<<region[0]<<','<<region[1]<<','<<n.start_step<<','
                    <<n.child_index<<','<<n.child_count<<','<<n.masks[p]<<'\n';
            }
            total_hits+=found.events.size();
        }
        correctness<<h.id<<','<<h.events.size()<<','<<nodes<<','<<tree.levels.size()+1<<",PASS,PASS,PASS,PASS,PASS\n";
        // Paired original histories differ only in approval/rewrite order and its exact dependencies.
        if (index+1<data.histories.size() && data.histories[index+1].pair_id==h.pair_id) {
            const auto other=engine.evaluate(data.histories[index+1].events,false);
            sensitivity(changes,h,tree,other.tree,"paired_action_order");
            log_timing(h.id,"paired_action_order",other);
        }
        // Controlled perturbations have no oracle labels or dependency edges. They
        // isolate token order/removal/gaps from the generator's policy scenarios.
        if (!(h.id%2) && h.events.size()>=2) {
            std::vector<Event> control(h.events.size());
            for (size_t i=0;i<control.size();++i) control[i]=Event{17,1000,0,Filler,uint32_t(i),0,0,0};
            control[0]=Event{1,1,0,PrivateRead|Sensitive,0,0,0,0};
            control[1]=Event{11,2,0,Approval,1,0,0,0};
            const auto base=engine.evaluate(control,false);
            log_timing(h.id,"controlled_base",base);
            const char* names[]={"controlled_reorder", "controlled_remove_approval", "controlled_gap"};
            for (unsigned mode=0;mode<3;++mode) {
                auto altered=control;
                if (mode==0) {
                    std::swap(altered[0],altered[1]); altered[0].step=0; altered[1].step=1;
                } else if (mode==1) altered.erase(altered.begin()+1);
                else {
                    std::swap(altered[1],altered.back()); altered[1].step=1;
                    altered.back().step=uint32_t(altered.size()-1);
                }
                const auto changed=engine.evaluate(altered,false);
                sensitivity(changes,h,base.tree,changed.tree,names[mode]);
                log_timing(h.id,names[mode],changed);
            }
        }
        total_events+=h.events.size(); total_nodes+=nodes;
        std::cout<<"PASS history "<<h.id<<": "<<h.events.size()<<" events, "<<nodes<<" parents, "
            <<tree.levels.size()+1<<" levels; all fields/vectors/masks/query distances/retrieval verified\n";
    }
    auto summary=csv(directory,"summary.txt");
    summary<<engine.name()<<"\nObserved events: "<<total_events<<"\nInternal nodes: "<<total_nodes
        <<"\nExact property hits (properties may overlap): "<<total_hits
        <<"\nAll hierarchy nodes and exact searches verified. Empty roots have no similarity distance.\n"
        <<"Outcomes are read only by the file parser, never encoded or sent to CUDA.\n"
        <<"No classifier or trained prototype is used. Timings exclude allocation/transfers/CPU verification.\n";
    std::cout<<"Artifacts: "<<directory.string()<<"\n";
    return 0;
}
} // namespace temporal

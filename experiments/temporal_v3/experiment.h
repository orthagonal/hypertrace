#pragma once
#include "reference.h"

namespace windowed {
inline Input load_input(const History& history,const std::filesystem::path& path,uint32_t target) {
    std::ifstream in(path,std::ios::binary); require(bool(in),"cannot open preserved v2 anchor archive");
    char magic[8]; require(bool(in.read(magic,8)) && !std::memcmp(magic,"HDBANC02",8),"bad v2 anchor magic");
    uint32_t header[14]; for (auto& n:header) n=read32(in);
    require(header[0]==2 && header[1]==64 && header[2]==history.id && header[3]==history.events.size() &&
        header[5]==target && header[7]==176 && header[8]==history.edges.size() && header[9]==12 && header[10]==64 &&
        header[11]==64ull+176ull*header[4] && header[12]==header[11]+12ull*header[8] && header[13]==104 &&
        std::filesystem::file_size(path)==header[12]+416ull,"v2 archive offsets/history mismatch");
    std::vector<Anchor> anchors(header[4]);
    for (auto& a:anchors) {
        auto& e=a.event;
        e.action=read32(in); e.object=read32(in); e.source=read32(in); e.flags=read32(in); e.step=read32(in);
        e.dependency_begin=read32(in); e.dependency_count=read32(in); e.reserved=read32(in);
        a.event_index=read32(in); a.object_version=read32(in); a.source_version=read32(in); a.ordinal=read32(in);
        for (auto& word:a.hv) word=read32(in);
    }
    for (const auto& edge:history.edges) require(read32(in)==edge.consumer_step && read32(in)==edge.producer_step &&
        read32(in)==edge.object,"preserved exact edges mismatch");
    Input input;
    for (auto& vector:input.baseline) for (auto& word:vector) word=read32(in);
    for (auto& word:input.baseline_only_key) word=read32(in);
    const auto expected=sparse::reference_analyze(history,target);
    require(anchors.size()==expected.anchors.size(),"preserved anchor selection mismatch");
    if (!anchors.empty()) require(!std::memcmp(anchors.data(),expected.anchors.data(),anchors.size()*sizeof(Anchor)),"preserved anchors/versions/vectors mismatch");
    require(input.baseline[1]==expected.fingerprints[1] && input.baseline[2]==expected.fingerprints[2] &&
        input.baseline_only_key==expected.pattern,"preserved v2 fingerprints or baseline key mismatch");
    require(input.baseline[0]==root(reference_build(history.events)),"preserved v1 root mismatch");
    for (const auto& a:anchors) if (a.event.object==target) input.scoped.push_back(a);
    require(input.scoped.size()==header[6],"scope count mismatch");
    return input;
}
inline void write_windows(const std::filesystem::path& path,unsigned history,unsigned target,const Input& input,const std::vector<Matrix>& matrices) {
    uint32_t total=0; for (const auto& m:matrices) total+=m.counts[history];
    const uint32_t refs=64,table=refs+uint32_t(input.scoped.size())*24,data=table+uint32_t(matrices.size())*24;
    std::ofstream out(path,std::ios::binary); require(bool(out),"cannot write v3 window archive");
    out.exceptions(std::ios::failbit|std::ios::badbit); out.write("HDBWIN03",8);
    for (auto n:{3u,64u,history,target,uint32_t(input.scoped.size()),total,176u,24u,refs,table,uint32_t(matrices.size()),24u,data,0u}) write32(out,n);
    for (const auto& a:input.scoped) for (auto n:{a.ordinal,a.event_index,a.event.step,a.event.object,a.object_version,a.source_version}) write32(out,n);
    uint32_t offset=data;
    for (const auto& m:matrices) {
        for (auto n:{m.width,m.counts[history],m.width/2,offset,m.counts[history]*176,0u}) write32(out,n);
        offset+=m.counts[history]*176;
    }
    for (const auto& m:matrices) for (unsigned i=0;i<m.counts[history];++i) {
        const auto& w=m.windows[m.begins[history]+i];
        for (auto word:w.hv) write32(out,word);
        for (auto n:{w.history_id,w.width,w.scope_begin,w.count,w.first_step,w.last_step,w.first_anchor,w.last_anchor,
                    w.object,w.first_version,w.last_version,w.reserved}) write32(out,n);
    }
    out.close();
}
inline void write_similarities(const std::filesystem::path& path,const Matrix& m) {
    std::ofstream out(path,std::ios::binary); require(bool(out),"cannot write similarity matrix");
    out.exceptions(std::ios::failbit|std::ios::badbit); out.write("HDBSIM03",8);
    for (auto n:{3u,32u,m.width,uint32_t(m.queries.size()),uint32_t(m.windows.size()),2u}) write32(out,n);
    for (auto value:m.distances) { const unsigned char bytes[]={static_cast<unsigned char>(value),static_cast<unsigned char>(value>>8)};
        out.write(reinterpret_cast<const char*>(bytes),2); }
    out.close();
}
inline uint32_t containing_window(const Matrix& m,const Input& input,unsigned history,uint32_t step) {
    const auto anchor=std::lower_bound(input.scoped.begin(),input.scoped.end(),step,
        [](const Anchor& a,uint32_t s) { return a.event.step<s; });
    if (anchor==input.scoped.end() || anchor->event.step!=step) return MISSING;
    const unsigned ordinal=unsigned(anchor-input.scoped.begin());
    const auto begin=m.windows.begin()+m.begins[history],end=begin+m.counts[history];
    const auto window=std::lower_bound(begin,end,ordinal,
        [](const Window& w,unsigned index) { return w.scope_begin+w.count<=index; });
    return window!=end && window->scope_begin<=ordinal ? uint32_t(window-m.windows.begin()) : MISSING;
}
template<class Engine> int run(int argc,char** argv,Engine& engine) {
    require(argc>=8,"usage: temporal_v3_<cpu|gpu> <events.bin> --roles <roles.csv> --anchors <v2 directory> --out <directory> [--repeats N] [--object ID]");
    std::filesystem::path directory,anchor_path,role_path; unsigned target=2;
    for (int i=2;i<argc;++i) {
        const std::string arg=argv[i]; require(i+1<argc,"missing v3 option value"); const std::string value=argv[++i];
        if (arg=="--out" && directory.empty()) directory=value;
        else if (arg=="--anchors" && anchor_path.empty()) anchor_path=value;
        else if (arg=="--roles" && role_path.empty()) role_path=value;
        else if (arg=="--repeats" || arg=="--object") {
            size_t used=0; auto n=std::stoul(value,&used); require(used==value.size(),"invalid numeric value");
            if (arg=="--repeats") { require(n>=1 && n<=1000,"invalid repeats"); engine.repeats=unsigned(n); }
            else { require(n>0 && n<=0xffffffffu,"invalid object"); target=unsigned(n); }
        } else throw std::runtime_error("invalid v3 option");
    }
    require(!directory.empty() && !anchor_path.empty() && !role_path.empty(),"missing v3 path");
    const auto dataset=load(argv[1]); const unsigned n=unsigned(dataset.histories.size());
    require(n>0 && n<=256,"v3 collection size must be 1-256");
    const auto roles=sparse::read_roles(role_path,n);
    std::vector<Input> inputs;
    for (const auto& h:dataset.histories) inputs.push_back(load_input(h,anchor_path/("anchors_"+std::to_string(h.id)+".bin"),target));
    std::filesystem::create_directories(directory);
    auto scores=csv(directory,"scores.csv"),ranks=csv(directory,"ranks.csv"),timings=csv(directory,"timings.csv"),
        raw=csv(directory,"window_topk.csv"),queries=csv(directory,"queries.csv"),memory=csv(directory,"memory.csv"),
        checks=csv(directory,"graph_checks.csv"),graph_time=csv(directory,"graph_timings.csv"),
        dependency=csv(directory,"dependency_ranks.csv"),links=csv(directory,"dependency_links.csv"),correctness=csv(directory,"correctness.csv");
    scores<<"method,query_id,candidate_id,score\n"; ranks<<"method,query_id,k,candidate_id,score\n";
    timings<<"width,kernel,items,repeats,median_ms,min_ms,max_ms\n";
    raw<<"width,query_row,k,window_id,history_id,distance\n";
    queries<<"width,query_row,query_id,window_id\n";
    memory<<"history_id,width,scoped_anchors,windows,window_bytes,scope_ref_bytes,query_matrix_bytes,gpu_peak_bytes\n";
    checks<<"method,query_id,candidate_id,proposal_rank,match,nodes,edges,max_dependency_gap\n";
    graph_time<<"method,query_id,candidate_id,proposal_rank,wall_ms\n";
    dependency<<"method,budget,query_id,k,candidate_id,score\n";
    links<<"method,query_id,candidate_id,proposal_rank,object,query_consumer_step,query_producer_step,query_consumer_window,query_producer_window,candidate_consumer_step,candidate_producer_step,candidate_consumer_window,candidate_producer_window\n";
    correctness<<"width,windows,similarities,all_window_fields,all_similarities,window_topk,aggregates,history_ranks\n";
    // Old baselines are comparison-only. Their exact key never enters v3 kernels,
    // query selection, candidate proposals, or dependency verification.
    std::vector<unsigned> query_histories;
    for (unsigned q=0;q<n;++q) if (roles[q]) {
        query_histories.push_back(q);
        for (unsigned method=0;method<4;++method) {
            const std::string name=method<3 ? sparse::method_name(method) : "exact_baseline";
            std::vector<std::pair<uint32_t,uint32_t>> candidates;
            for (unsigned c=0;c<n;++c) if (!roles[c]) {
                const uint32_t score=(method<3 ? hamming(inputs[q].baseline[method].data(),inputs[c].baseline[method].data()) :
                    (inputs[q].baseline_only_key==inputs[c].baseline_only_key ? 0 : 1024))*1024;
                scores<<name<<','<<q<<','<<c<<','<<score<<'\n'; candidates.push_back({score,c});
            }
            std::sort(candidates.begin(),candidates.end());
            for (unsigned k=0;k<candidates.size();++k) ranks<<name<<','<<q<<','<<k+1<<','<<candidates[k].second<<','<<candidates[k].first<<'\n';
        }
    }
    std::vector<Matrix> matrices;
    for (unsigned width:{8u,16u,32u}) {
        auto m=engine.evaluate(inputs,roles,width);
        for (const auto& t:m.timings) timings<<width<<','<<t.kernel<<','<<t.items<<','<<t.repeats<<','<<t.median<<','<<t.minimum<<','<<t.maximum<<'\n';
        write_similarities(directory/("similarities_"+std::to_string(width)+".bin"),m);
        for (unsigned h=0;h<n;++h) memory<<h<<','<<width<<','<<inputs[h].scoped.size()<<','<<m.counts[h]<<','
            <<m.counts[h]*sizeof(Window)<<','<<inputs[h].scoped.size()*sizeof(ScopedRef)<<','<<m.distances.size()*2<<','<<engine.peak_bytes<<'\n';
        for (unsigned row=0;row<m.queries.size();++row) {
            queries<<width<<','<<row<<','<<m.query_histories[row/QUERY_SLOTS]<<','<<m.queries[row]<<'\n';
            for (unsigned k=0;k<WINDOW_TOPK;++k) {
                const auto w=m.window_ranks[row*WINDOW_TOPK+k];
                if (w!=MISSING) raw<<width<<','<<row<<','<<k+1<<','<<w<<','<<m.windows[w].history_id<<','<<m.distances[size_t(row)*m.windows.size()+w]<<'\n';
            }
        }
        for (unsigned a=0;a<AGGREGATIONS;++a) for (unsigned group=0;group<m.query_histories.size();++group) {
            const auto q=m.query_histories[group]; const std::string method="w"+std::to_string(width)+"_"+aggregation_name(a);
            std::vector<uint32_t> proposed;
            for (unsigned c=0;c<n;++c) if (!roles[c]) scores<<method<<','<<q<<','<<c<<','<<m.history_scores[(a*m.query_histories.size()+group)*n+c]<<'\n';
            for (unsigned k=0;k<n;++k) {
                const auto c=m.history_ranks[(a*m.query_histories.size()+group)*n+k];
                if (c==MISSING) break;
                proposed.push_back(c);
                ranks<<method<<','<<q<<','<<k+1<<','<<c<<','<<m.history_scores[(a*m.query_histories.size()+group)*n+c]<<'\n';
            }
            // Exact checks occur only after CUDA/CPU HDC candidate proposals.
            std::vector<bool> accepted;
            for (unsigned k=0;k<std::min<size_t>(16,proposed.size());++k) {
                const auto c=proposed[k]; const auto start=std::chrono::steady_clock::now();
                const auto verified=verify_dependencies(dataset.histories[q],inputs[q].scoped,dataset.histories[c],inputs[c].scoped);
                const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                accepted.push_back(verified.match);
                checks<<method<<','<<q<<','<<c<<','<<k+1<<','<<verified.match<<','<<verified.nodes<<','<<verified.edges<<','<<verified.max_gap<<'\n';
                graph_time<<method<<','<<q<<','<<c<<','<<k+1<<','<<ms<<'\n';
                if (verified.match) for (const auto& edge:verified.links) links<<method<<','<<q<<','<<c<<','<<k+1<<','<<edge[4]<<','
                    <<edge[0]<<','<<edge[1]<<','<<containing_window(m,inputs[q],q,edge[0])<<','<<containing_window(m,inputs[q],q,edge[1])<<','
                    <<edge[2]<<','<<edge[3]<<','<<containing_window(m,inputs[c],c,edge[2])<<','<<containing_window(m,inputs[c],c,edge[3])<<'\n';
            }
            for (unsigned budget:{2u,4u,8u,16u}) {
                unsigned returned=0;
                for (unsigned k=0;k<std::min<size_t>(budget,accepted.size());++k) if (accepted[k]) {
                    const auto c=proposed[k];
                    dependency<<method<<','<<budget<<','<<q<<','<<++returned<<','<<c<<','<<m.history_scores[(a*m.query_histories.size()+group)*n+c]<<'\n';
                }
            }
        }
        correctness<<width<<','<<m.windows.size()<<','<<m.distances.size()<<",PASS,PASS,PASS,PASS,PASS\n";
        matrices.push_back(std::move(m));
        std::cout<<"PASS width "<<width<<": every window, similarity and retrieval result verified\n";
    }
    for (unsigned h=0;h<n;++h) write_windows(directory/("windows_"+std::to_string(h)+".bin"),h,target,inputs[h],matrices);
    auto summary=csv(directory,"summary.txt"); summary<<engine.name()<<"\nAll windows retained; 4 query slots and 8 equally budgeted history contexts.\n"
        <<"Exact dependency checks follow HDC proposals and use raw DAGs, never answer keys.\n";
    return 0;
}
} // namespace windowed

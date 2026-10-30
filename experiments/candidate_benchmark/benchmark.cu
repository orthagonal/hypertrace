#include "gpu.cuh"
#include <iomanip>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif

namespace candidate {
inline uint64_t resident_bytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX info{};
    if(K32GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&info),sizeof(info))) return info.WorkingSetSize;
#endif
    return 0;
}
struct ScalarEncoder {
    std::unordered_map<Type,Vector,TypeHash> cache;
    Vector encode(const Context& c) {
        std::vector<sparse::Anchor> a;
        for(unsigned i=0;i<c.count;++i) {
            auto key=type(c.events[i]); key[4]=0; auto found=cache.find(key);
            if(found==cache.end()) found=cache.emplace(key,reference_event(c.events[i])).first;
            sparse::Anchor x{}; x.event=c.events[i]; std::copy(found->second.begin(),found->second.end(),x.hv); a.push_back(x);
        } return sparse::reference_fingerprint(a,0,false);
    }
};
struct OutcomeCounts {
    unsigned candidate_hits=0,final_hits=0,false_matches=0,returned=0,verified=0,rejected=0;
    uint64_t nodes=0,edges=0; double verification_ms=0;
};
inline OutcomeCounts evaluate(const Observed& o,const std::vector<Truth>& truth,unsigned qi,
        const std::vector<unsigned>& rank,unsigned k,bool verify,const ExactIndex& index,
        const std::vector<sparse::Anchor>& qr) {
    OutcomeCounts r; const auto qh=o.queries[qi];
    for(unsigned j=0;j<std::min<unsigned>(k,unsigned(rank.size()));++j) {
        const unsigned g=rank[j],ch=o.gallery[g]; bool accept=true;
        if(relevant(truth[qh],truth[ch])) ++r.candidate_hits;
        if(verify) {
            auto begin=Clock::now(); auto checked=windowed::verify_dependencies(o.histories[qh],qr,o.histories[ch],index.roots[g]);
            r.verification_ms+=ms(begin); ++r.verified; r.nodes+=checked.nodes; r.edges+=checked.edges;
            accept=checked.match; if(!accept) ++r.rejected;
        }
        if(accept && r.returned<2) { ++r.returned;
            if(relevant(truth[qh],truth[ch])) ++r.final_hits; else ++r.false_matches; }
    } return r;
}
inline void benchmark(Corpus& corpus,const std::filesystem::path& out,unsigned repeats) {
    const auto& o=corpus.observed; const auto& truth=corpus.truth; const auto ng=unsigned(o.gallery.size()),nq=unsigned(o.queries.size());
    require(ng>=16 && nq>0,"invalid gallery/query cardinality");
    for(auto q:o.queries) { unsigned n=0; for(auto g:o.gallery) if(relevant(truth[q],truth[g])) ++n; require(n==2,"each query must have exactly two relevant histories"); }
    std::filesystem::create_directories(out);
    std::ofstream manifest(out/"histories.csv"); manifest<<"history_id,role,family,order,replica,event_count,edge_count,observed_fnv64\n";
    uint64_t raw_bytes=0;
    for(unsigned h=0;h<o.histories.size();++h) {
        const auto& v=o.histories[h]; const auto& t=truth[h]; raw_bytes+=v.events.capacity()*sizeof(Event)+v.edges.capacity()*sizeof(Edge);
        manifest<<v.id<<','<<(std::find(o.queries.begin(),o.queries.end(),h)!=o.queries.end())<<','<<t.family<<','<<t.order<<','<<t.replica<<','
                <<v.events.size()<<','<<v.edges.size()<<','<<observed_hash(v)<<'\n';
    }
    auto begin=Clock::now(); ExactIndex index(o); double index_ms=ms(begin);
    const auto index_rss=resident_bytes();
    begin=Clock::now(); std::vector<Context> gc,qc; std::vector<std::vector<sparse::Anchor>> qr;
    for(auto h:o.gallery) { auto a=contexts(o.histories[h],GCONTEXTS); gc.insert(gc.end(),a.begin(),a.end()); }
    const double gallery_scope_ms=ms(begin);
    begin=Clock::now(); for(auto h:o.queries) { auto a=contexts(o.histories[h],QCONTEXTS); qc.insert(qc.end(),a.begin(),a.end()); qr.push_back(endpoints(o.histories[h])); }
    const double query_scope_ms=ms(begin);
    check(cudaFree(nullptr)); // Exclude one-time driver/context startup, explicitly reported by runner.
    begin=Clock::now(); Gpu gpu(gc,nq); const double gpu_index_ms=ms(begin)+gallery_scope_ms;
    gpu.snapshot(); // Reference-only download, outside construction timing.
    begin=Clock::now(); auto qvectors=encode_queries(qc); const double query_encode_ms=ms(begin);
    ScalarEncoder scalar;
    for(unsigned i=0;i<gc.size();++i) require(scalar.encode(gc[i])==gpu.gallery_host[i],"GPU gallery fingerprint differs from independent CPU");
    for(unsigned i=0;i<qc.size();++i) require(scalar.encode(qc[i])==qvectors[i],"GPU query fingerprint differs from independent CPU");
    std::vector<uint8_t> validq(nq); for(unsigned q=0;q<nq;++q) validq[q]=qc[q*QCONTEXTS].count>0;
    const auto batch=gpu.search(qvectors,validq);
    std::vector<std::vector<uint32_t>> cpu_scores(nq);
    std::ofstream scorefile(out/"scores.csv"); scorefile<<"query_id,candidate_id,score\n";
    for(unsigned q=0;q<nq;++q) {
        std::vector<Vector> qv(qvectors.begin()+q*QCONTEXTS,qvectors.begin()+(q+1)*QCONTEXTS);
        cpu_scores[q]=scalar_scores(qv,gpu.gallery_host);
        if(!validq[q]) std::fill(cpu_scores[q].begin(),cpu_scores[q].end(),1024*1024);
        for(unsigned g=0;g<ng;++g) {
            if(!gc[g*GCONTEXTS].count) cpu_scores[q][g]=1024*1024;
            require(cpu_scores[q][g]==batch.scores[q*ng+g],"GPU score differs from independent scalar CPU");
            scorefile<<o.histories[o.queries[q]].id<<','<<o.histories[o.gallery[g]].id<<','<<cpu_scores[q][g]<<'\n';
        }
        const auto rank=score_rank(cpu_scores[q]);
        for(unsigned j=0;j<MAXK;++j) require(rank[j]==batch.ranks[q*MAXK+j],"GPU top K differs from independent CPU");
    }
    // Untimed warmups before single-query and batched measurements.
    gpu.search(qvectors,validq);
    double batch_wall=0,batch_similarity=0,batch_topk=0;
    for(unsigned r=0;r<repeats;++r) { auto b=gpu.search(qvectors,validq); batch_wall+=b.wall_ms; batch_similarity+=b.similarity_ms; batch_topk+=b.topk_ms; }
    std::ofstream construction(out/"construction.csv");
    construction<<"gallery,queries,events,edges,raw_cpu_bytes,exact_index_estimated_bytes,cpu_working_set_bytes,gpu_resident_bytes,gpu_build_peak_bytes,exact_index_ms,hdc_index_wall_ms,hdc_scope_ms,hdc_gpu_build_ms,hdc_gpu_kernel_ms,query_scope_ms,query_encode_ms,batch_wall_ms,batch_similarity_ms,batch_topk_ms,batch_queries_per_second,cpu_gpu_fingerprints_checked,cpu_gpu_scores_checked,cpu_gpu_ranks_checked\n";
    construction<<std::setprecision(12)<<ng<<','<<nq<<','<<index.event_count<<','<<index.edge_count<<','<<raw_bytes<<','<<index.bytes()<<','<<index_rss<<','<<gpu.bytes()<<','
        <<gpu.bytes()+gc.size()*sizeof(Context)<<','<<index_ms<<','<<gpu_index_ms<<','<<gallery_scope_ms<<','<<gpu.build_wall_ms<<','<<gpu.build_kernel_ms<<','
        <<query_scope_ms<<','<<query_encode_ms<<','<<batch_wall/repeats<<','<<batch_similarity/repeats<<','<<batch_topk/repeats<<','
        <<1000.0*nq*repeats/batch_wall<<','<<gc.size()+qc.size()<<','<<ng*nq<<','<<MAXK*nq<<'\n';
    std::ofstream rows(out/"queries.csv"),details(out/"candidates.csv");
    rows<<"strategy,query_id,budget,relevant_total,candidate_hits,candidate_recall,final_hits,final_recall2,false_matches,returned,abstention,partial,verifications,rejected,verifier_nodes,verifier_edges,shortlist_nodes,shortlist_edges,postings_examined,shortlist_pool,ranked_candidates,query_prepare_ms,ranking_ms,shortlist_ms,gpu_similarity_ms,gpu_topk_ms,verification_ms,total_query_ms,score_ties_at_cutoff\n";
    details<<"strategy,query_id,budget,rank,candidate_id,relevant,accepted,returned\n";
    const std::vector<std::string> strategies={"random_only","hdc_only","inverted_only","dependency_only","random_verified","hdc_verified","inverted_verified","dependency_verified","hybrid_verified","hybrid_inverted_verified"};
    for(unsigned q=0;q<nq;++q) {
        std::vector<Vector> qv(qvectors.begin()+q*QCONTEXTS,qvectors.begin()+(q+1)*QCONTEXTS);
        std::vector<uint8_t> valid={validq[q]};
        for(const auto& strategy:strategies) for(unsigned k:{1u,2u,4u,8u,16u}) {
            const bool verify=strategy.find("verified")!=std::string::npos;
            double wall=0,prepare_ms=0,ranking_ms=0,shortlist_ms=0,similarity=0,topk=0,verify_ms=0; OutcomeCounts counts; Work work;
            std::vector<unsigned> rank; unsigned ties=0;
            for(unsigned r=0;r<repeats;++r) {
                begin=Clock::now(); Shortlist selected; SearchResult search;
                const bool hybrid=strategy.find("hybrid")==0;
                const bool hdc=hybrid || strategy.find("hdc")!=std::string::npos;
                const auto& query_history=o.histories[o.queries[q]];
                std::vector<sparse::Anchor> live_roots; std::vector<Context> live_contexts;
                if(verify || strategy.find("random")==std::string::npos) {
                    const auto s=scope(query_history); live_roots=endpoints_from_scope(query_history,s);
                    if(hdc) live_contexts=contexts_from_scope(query_history,s,QCONTEXTS);
                }
                prepare_ms+=ms(begin); const auto rank_begin=Clock::now();
                if(strategy.find("inverted")!=std::string::npos) selected=index.inverted(query_history,live_roots);
                else if(strategy.find("dependency")!=std::string::npos || hybrid) selected=index.dependency(query_history,live_roots);
                if(strategy=="hybrid_inverted_verified" && selected.rank.size()>64) selected.rank.resize(64);
                if(hybrid) selected.work.pool=unsigned(selected.rank.size());
                if(hybrid) {
                    shortlist_ms+=ms(rank_begin); std::vector<uint8_t> mask(ng); for(auto g:selected.rank) mask[g]=1;
                    search=gpu.search(qv,valid,mask,&live_contexts); rank=search.ranks; rank.erase(std::remove(rank.begin(),rank.end(),NONE),rank.end());
                } else if(strategy.find("hdc")!=std::string::npos) {
                    search=gpu.search(qv,valid,{},&live_contexts); rank=search.ranks;
                } else if(strategy.find("random")!=std::string::npos) {
                    rank.resize(ng); std::iota(rank.begin(),rank.end(),0); std::mt19937 random(173u+o.histories[o.queries[q]].id);
                    for(unsigned j=ng;j>1;--j) std::swap(rank[j-1],rank[random()%j]);
                } else rank=selected.rank;
                ranking_ms+=ms(rank_begin); similarity+=search.similarity_ms; topk+=search.topk_ms; work=selected.work;
                if(!hybrid && (strategy.find("inverted")!=std::string::npos || strategy.find("dependency")!=std::string::npos)) shortlist_ms+=ms(rank_begin);
                // Truth is consulted only here, after the shortlist/ranking is final.
                counts=evaluate(o,truth,q,rank,k,verify,index,live_roots); verify_ms+=counts.verification_ms; wall+=ms(begin);
                // Reference work is strictly outside every latency interval.
                if(hybrid || strategy.find("hdc")!=std::string::npos) {
                    std::vector<uint8_t> eligible;
                    if(hybrid) { eligible.resize(ng); for(auto g:selected.rank) eligible[g]=1; }
                    const auto expected=score_rank(cpu_scores[q],eligible);
                    require(rank.size()==std::min<size_t>(MAXK,expected.size()),"GPU rank count mismatch");
                    for(unsigned j=0;j<rank.size();++j) require(rank[j]==expected[j],"GPU rank mismatch");
                    for(unsigned g=0;g<ng;++g) require(search.scores[g]==(!hybrid || eligible[g] ? cpu_scores[q][g]:NONE),"GPU score mismatch");
                }
            }
            if(strategy.find("hdc")!=std::string::npos || strategy.find("hybrid")==0) {
                if(rank.size()>=k) { auto cutoff=cpu_scores[q][rank[k-1]]; for(unsigned g=0;g<ng;++g) if(cpu_scores[q][g]==cutoff) ++ties; }
            }
            const auto qid=o.histories[o.queries[q]].id;
            rows<<std::setprecision(12)<<strategy<<','<<qid<<','<<k<<",2,"<<counts.candidate_hits<<','<<counts.candidate_hits/2.0<<','<<counts.final_hits<<','
                <<counts.final_hits/2.0<<','<<counts.false_matches<<','<<counts.returned<<','<<(counts.returned==0)<<','<<(counts.returned<2)<<','
                <<counts.verified<<','<<counts.rejected<<','<<counts.nodes<<','<<counts.edges<<','<<work.nodes<<','<<work.edges<<','<<work.postings<<','
                <<work.pool<<','<<(strategy.find("hdc")!=std::string::npos ? ng:strategy.find("hybrid")==0 ? work.pool:unsigned(rank.size()))<<','
                <<prepare_ms/repeats<<','<<ranking_ms/repeats<<','<<shortlist_ms/repeats<<','<<similarity/repeats<<','<<topk/repeats<<','<<verify_ms/repeats<<','<<wall/repeats<<','<<ties<<'\n';
            unsigned returned=0;
            for(unsigned j=0;j<std::min<unsigned>(k,unsigned(rank.size()));++j) {
                auto g=rank[j]; bool accepted=!verify || windowed::verify_dependencies(o.histories[o.queries[q]],qr[q],o.histories[o.gallery[g]],index.roots[g]).match;
                bool ret=accepted && returned<2; if(ret) ++returned;
                details<<strategy<<','<<qid<<','<<k<<','<<j+1<<','<<o.histories[o.gallery[g]].id<<','<<relevant(truth[o.queries[q]],truth[o.gallery[g]])<<','<<accepted<<','<<ret<<'\n';
            }
        }
    }
    std::cout<<"PASS: "<<ng<<" candidates, "<<nq<<" queries; all GPU contexts/scores/ranks match scalar CPU; "<<out.string()<<'\n';
}
} // namespace candidate

int main(int argc,char** argv) {
    try {
        using namespace candidate; std::map<std::string,std::string> args;
        for(int i=1;i<argc;i+=2) { require(i+1<argc,"arguments require values"); args[argv[i]]=argv[i+1]; }
        require(args.count("--out"),"--out required"); unsigned repeats=args.count("--repeats") ? unsigned(std::stoul(args.at("--repeats"))):5;
        require(repeats>0,"repeats must be positive"); Corpus corpus;
        auto start=Clock::now();
        if(args.count("--original")) corpus=load_original(args.at("--original"));
        else corpus=scaled(unsigned(std::stoul(args.at("--gallery"))),unsigned(std::stoul(args.at("--length"))),args.at("--profile"),unsigned(std::stoul(args.at("--seed"))));
        std::cout<<"Corpus ready in "<<ms(start)<<" ms\n";
        benchmark(corpus,args.at("--out"),repeats); return 0;
    } catch(const std::exception& e) { std::cerr<<"Candidate benchmark failure: "<<e.what()<<'\n'; return 1; }
}

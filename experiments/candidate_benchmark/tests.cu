#include "gpu.cuh"
using namespace candidate;

void test_index() {
    for(const std::string profile:{"sparse","dense_unrelated","dense_target","dense_target_far"}) {
        auto corpus=scaled(128,1024,profile,2027); const auto& o=corpus.observed; ExactIndex index(o);
        for(auto q:o.queries) {
            auto qr=endpoints(o.histories[q]); auto shortlist=index.dependency(o.histories[q],qr);
            unsigned relevant_count=0,accepted=0;
            for(unsigned g=0;g<o.gallery.size();++g) {
                bool match=windowed::verify_dependencies(o.histories[q],qr,o.histories[o.gallery[g]],index.roots[g]).match;
                bool rel=relevant(corpus.truth[q],corpus.truth[o.gallery[g]]);
                require(match==rel,"scaled same-family/order ground truth disagrees with unchanged verifier");
                if(rel) ++relevant_count;
                if(match) { ++accepted; require(std::find(shortlist.rank.begin(),shortlist.rank.end(),g)!=shortlist.rank.end(),"semijoin lost an exact match"); }
            }
            require(relevant_count==2 && accepted==2,"incorrect relevance denominator");
            require(index.inverted(o.histories[q],qr).rank.size()==128,"shared primitive identifiers unexpectedly unique");
        }
        // Matched order pairs have identical raw action multisets, not answer tags.
        std::map<std::pair<unsigned,unsigned>,std::array<std::vector<std::array<unsigned,4>>,2>> pairs;
        for(unsigned h=0;h<o.histories.size();++h) {
            const auto& t=corpus.truth[h]; auto& bag=pairs[{t.family,t.replica}][t.order];
            for(const auto& e:o.histories[h].events) bag.push_back({e.action,e.object,e.source,e.flags});
            std::sort(bag.begin(),bag.end());
        }
        for(const auto& pair:pairs) require(pair.second[0]==pair.second[1],"order pair changed action multiset");
    }
    auto a=scaled(16,32768,"dense_target_far",42),b=scaled(128,32768,"dense_target_far",42);
    std::map<unsigned,uint64_t> hashes; for(const auto& h:b.observed.histories) hashes[h.id]=observed_hash(h);
    for(const auto& h:a.observed.histories) require(hashes.at(h.id)==observed_hash(h),"gallery growth changes existing histories");
    ExactIndex index(a.observed);
    for(auto q:a.observed.queries) {
        const auto qr=endpoints(a.observed.histories[q]); auto selected=index.dependency(a.observed.histories[q],qr);
        require(selected.rank.size()==2,"far writer chain not recovered");
    }
    // A semijoin can match parents independently while missing their shared
    // identity across roots. The full, unchanged verifier must reject this.
    const std::vector<std::array<unsigned,4>> actions={{1,1,0,PrivateRead|Sensitive},{4,2,1,Derived|Sensitive},
        {11,2,0,Approval},{16,2,1,Derived|Sensitive},{13,2,0,External|Sensitive}};
    auto changed=actions; changed.insert(changed.begin()+3,{1,1,0,PrivateRead|Sensitive});
    Observed aliases; aliases.histories={wire(actions,0),wire(changed,1),wire(actions,2)}; aliases.gallery={0,1}; aliases.queries={2};
    ExactIndex ai(aliases); auto ar=endpoints(aliases.histories[2]); auto shortlist=ai.dependency(aliases.histories[2],ar);
    require(shortlist.rank.size()==2,"alias control must exercise semijoin false positive");
    require(!windowed::verify_dependencies(aliases.histories[2],ar,aliases.histories[1],ai.roots[1]).match,"full verifier accepted source-version alias mismatch");
    History future; future.cutoff=1; future.events.push_back({4,2,1,Derived,1,0,0,0}); bool rejected=false;
    try { scope(future); } catch(const std::exception&) { rejected=true; } require(rejected,"future event admitted");
    future.cutoff=2; future.events[0].step=0; future.events[0].dependency_count=1; future.edges.push_back({0,1,2}); rejected=false;
    try { parents(future,0); } catch(const std::exception&) { rejected=true; } require(rejected,"future dependency admitted");
}

void test_gpu() {
    std::vector<History> histories;
    for(unsigned n:{0u,1u,7u,8u,9u,11u,31u,32u,1024u,32768u}) {
        std::vector<std::array<unsigned,4>> a(n);
        for(unsigned i=0;i<n;++i) a[i]={i%2 ? 3u:4u,2,0,Derived|Sensitive};
        histories.push_back(wire(a,unsigned(histories.size())));
    }
    std::vector<Context> gc,qc;
    // 1024 candidates exercises the former 256-slot shared-memory limitation.
    for(unsigned i=0;i<1024;++i) { auto a=contexts(histories[i%histories.size()],8); gc.insert(gc.end(),a.begin(),a.end()); }
    for(const auto& h:histories) { auto a=contexts(h,4); qc.insert(qc.end(),a.begin(),a.end()); }
    Gpu gpu(gc,unsigned(histories.size())); gpu.snapshot(); auto qv=encode_queries(qc);
    for(unsigned i=0;i<gc.size();++i) require(scalar_window(gc[i])==gpu.gallery_host[i],"gallery vector mismatch");
    for(unsigned i=0;i<qc.size();++i) require(scalar_window(qc[i])==qv[i],"query vector mismatch");
    // Compare selected contexts with full, preserved v3 planning, including tails.
    for(unsigned i=0;i<histories.size();++i) {
        windowed::Input input;
        for(auto j:scope(histories[i])) { sparse::Anchor a{}; a.event=histories[i].events[j]; a.event_index=j;
            auto v=reference_event(a.event); std::copy(v.begin(),v.end(),a.hv); input.scoped.push_back(a); }
        const auto plan=windowed::reference_plan({input},{1},8);
        for(unsigned j=0;j<4;++j) if(plan.queries[j]!=NONE) {
            const auto* expected=plan.windows[plan.queries[j]].hv;
            require(std::equal(qv[i*4+j].begin(),qv[i*4+j].end(),expected),"selected context differs from full v3 plan");
        }
    }
    std::vector<uint8_t> valid(histories.size()); for(unsigned q=0;q<valid.size();++q) valid[q]=qc[q*4].count>0;
    for(bool masked:{false,true}) {
        std::vector<uint8_t> mask;
        if(masked) { mask.resize(histories.size()*1024); for(unsigned q=0;q<histories.size();++q) for(unsigned g=0;g<1024;++g) mask[q*1024+g]=(q!=0 && g%17==0); }
        auto results=gpu.search(qv,valid,mask);
        for(unsigned q=0;q<histories.size();++q) {
            std::vector<Vector> query(qv.begin()+q*4,qv.begin()+(q+1)*4); auto scores=scalar_scores(query,gpu.gallery_host);
            std::vector<uint8_t> eligible(1024,1);
            for(unsigned g=0;g<1024;++g) {
                if(!valid[q] || !gc[g*8].count) scores[g]=1024*1024;
                if(masked) { eligible[g]=mask[q*1024+g]; if(!eligible[g]) scores[g]=NONE; }
                require(scores[g]==results.scores[q*1024+g],"score mismatch");
            }
            auto rank=score_rank(scores,eligible);
            for(unsigned j=0;j<16;++j) require(results.ranks[q*16+j]==(j<rank.size() ? rank[j]:NONE),"rank/tie/empty-mask mismatch");
        }
    }
    // Raw writer-step/dependency metadata cannot affect HDC vectors.
    auto changed=qc; for(auto& c:changed) for(auto& e:c.events) { e.step+=100000; e.dependency_begin^=42; e.dependency_count^=7; e.reserved=99; }
    require(encode_queries(changed)==qv,"non-feature writer metadata leaked into HDC");
}
int main() {
    try { test_index(); test_gpu(); std::cout<<"PASS: shared identifiers, hard negatives, nested galleries, exact-version joins, 32768 events, future rejection, full v3 parity, partial/empty contexts, 1024-way ranking, masked/empty shortlist, stable ties, non-feature metadata isolation\n"; return 0; }
    catch(const std::exception& e) { std::cerr<<"Candidate test failure: "<<e.what()<<'\n'; return 1; }
}

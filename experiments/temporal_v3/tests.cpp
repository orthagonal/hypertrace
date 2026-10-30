#include "experiment.h"
#include <iostream>

using namespace windowed;
static History trace(const std::vector<std::array<uint32_t,4>>& actions,const std::vector<uint32_t>& steps={}) {
    History h; std::map<unsigned,unsigned> writer,approval;
    for (unsigned i=0;i<actions.size();++i) {
        const auto& a=actions[i]; const unsigned step=steps.empty() ? i : steps[i];
        Event e{a[0],a[1],a[2],a[3],step,uint32_t(h.edges.size()),0,0};
        if (e.source) h.edges.push_back({step,writer.at(e.source),e.source});
        else if (!sparse::reference_write(e.action) && writer.count(e.object)) h.edges.push_back({step,writer[e.object],e.object});
        if (sparse::reference_write(e.action)) { writer[e.object]=step; approval.erase(e.object); }
        else if (e.action==11) approval[e.object]=step;
        else if ((e.action==13 || e.action==5) && approval.count(e.object)) h.edges.push_back({step,approval[e.object],e.object});
        e.dependency_count=uint32_t(h.edges.size())-e.dependency_begin; h.events.push_back(e);
    }
    h.cutoff=h.events.empty() ? 0 : h.events.back().step+1; return h;
}
static std::vector<Anchor> scope(const History& h) {
    auto a=sparse::reference_analyze(h,2); std::vector<Anchor> result;
    for (const auto& anchor:a.anchors) if (anchor.event.object==2) result.push_back(anchor); return result;
}
int main() {
    try {
        for (unsigned n:{0u,1u,7u,8u,9u,15u,16u,17u,31u,32u,33u,1025u,32768u}) for (unsigned width:{8u,16u,32u}) {
            const auto starts=reference_starts(n,width); std::vector<bool> covered(n,false);
            if (!n) require(starts.empty(),"empty scope made a window");
            for (auto begin:starts) {
                require(begin<n,"window starts outside complete scope");
                for (unsigned i=begin;i<std::min(n,begin+width);++i) covered[i]=true;
            }
            require(std::all_of(covered.begin(),covered.end(),[](bool x){return x;}),"long history was truncated");
            if (n>=width) require(std::find(starts.begin(),starts.end(),n-width)!=starts.end(),"complete tail context missing");
        }
        const std::vector<std::array<uint32_t,4>> unsafe={{1,1,0,PrivateRead|Sensitive},{4,2,1,Derived|Sensitive},
            {11,2,0,Approval},{16,2,1,Derived|Sensitive},{13,2,0,External|Untrusted|Sensitive}};
        const auto q=trace(unsafe),gapped=trace(unsafe,{0,1,1001,4096,32767});
        const auto qa=scope(q),ga=scope(gapped);
        const auto w=reference_window(qa,0,8,0),gapw=reference_window(ga,0,8,0);
        require(!std::memcmp(w.hv,gapw.hv,128) && gapw.last_step==32767,"original steps lost or gaps leaked into window HDC");
        require(verify_dependencies(q,qa,gapped,ga).match,"distant exact versions failed");
        auto reordered=unsafe; std::swap(reordered[2],reordered[3]);
        const auto safe=trace(reordered); const auto sa=scope(safe);
        const auto sw=reference_window(sa,0,8,0);
        require(hamming(w.hv,sw.hv)>0,"identical multisets in different orders encoded identically");
        require(!verify_dependencies(q,qa,safe,sa).match,"approval of a different writer version was accepted");
        // The old eight-word baseline key cannot see this source-version alias
        // change. Raw dependency verification must reject it anyway.
        auto changed=unsafe; changed.insert(changed.begin()+3,{1,1,0,PrivateRead|Sensitive});
        const auto alias=trace(changed); const auto aa=scope(alias);
        require(sparse::reference_analyze(q,2).pattern==sparse::reference_analyze(alias,2).pattern,"alias control does not isolate baseline limitation");
        require(!verify_dependencies(q,qa,alias,aa).match,"shared source version alias was not enforced");
        auto edge_order=gapped;
        const auto safegap=trace(reordered,{0,1,1001,4096,32767}); edge_order=safegap;
        std::swap(edge_order.edges[edge_order.events.back().dependency_begin],edge_order.edges.back());
        require(verify_dependencies(safe,sa,edge_order,scope(edge_order)).match,"edge serialization order affected exact graph match");
        Matrix m; m.windows.resize(8); m.queries={0,1,2,3}; m.support={0,1,2,3,4,5,6,7};
        std::vector<uint16_t> costs(4*8,1024);
        costs[0]=0; costs[8+2]=0; costs[16+5]=0; costs[24+7]=0;
        require(reference_aggregate(m,costs,0,0,2)==0,"ordered context match failed");
        for (unsigned s=0;s<8;++s) std::swap(costs[8+s],costs[16+s]);
        require(reference_aggregate(m,costs,0,0,0)==0 && reference_aggregate(m,costs,0,0,2)>0,"ordered aggregation ignored chronology");
        std::fill(costs.begin(),costs.end(),uint16_t(193));
        for (unsigned a=0;a<3;++a) require(reference_aggregate(m,costs,0,0,a)==193*1024,"aggregation favored counts instead of normalized contexts");
        // No recursive depth/hop cap in exact dependency traversal.
        History chain; const unsigned n=32768;
        for (unsigned i=0;i<n;++i) {
            Event e{i==0 ? 1u : i==n-1 ? 13u : 4u,2,i>0 && i<n-1 ? 2u : 0u,
                i==0 ? PrivateRead|Sensitive : i==n-1 ? External|Sensitive : Derived|Sensitive,i,uint32_t(chain.edges.size()),i ? 1u : 0u,0};
            if (i) chain.edges.push_back({i,i-1,2}); chain.events.push_back(e);
        }
        Anchor first{},last{}; first.event=chain.events.front(); first.event_index=0;
        last.event=chain.events.back(); last.event_index=n-1;
        const auto deep=verify_dependencies(chain,{first,last},chain,{first,last});
        require(deep.match && deep.nodes==n && deep.edges==n-1,"long exact chain was truncated");
        std::cout<<"PASS: overlapping/partial/full-tail windows; no truncation; order-aware normalization; distant versions; alias rejection; 32768-node DAG\n";
        return 0;
    } catch (const std::exception& e) { std::cerr<<"V3 unit failure: "<<e.what()<<'\n'; return 1; }
}

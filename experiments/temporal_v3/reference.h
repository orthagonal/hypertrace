#pragma once
#include "../temporal_v2/experiment.h"
#include <set>

namespace windowed {
using namespace temporal;
using sparse::Anchor;
constexpr unsigned QUERY_SLOTS=4, SUPPORT_SLOTS=8, WINDOW_TOPK=5, AGGREGATIONS=3;
constexpr uint32_t MISSING=0xffffffffu, INF=0x3fffffffu;
inline const char* aggregation_name(unsigned a) {
    static const char* names[]={"best","normalized_top2","ordered"}; return names[a];
}
struct Window {
    uint32_t hv[32];
    uint32_t history_id,width,scope_begin,count,first_step,last_step,first_anchor,last_anchor;
    uint32_t object,first_version,last_version,reserved;
};
static_assert(sizeof(Window)==176,"window layout mismatch");
struct ScopedRef { uint32_t anchor_index,event_index,step,object,object_version,source_version; };
static_assert(sizeof(ScopedRef)==24,"scope reference layout mismatch");
struct Input {
    std::vector<Anchor> scoped;
    std::array<Vector,3> baseline;
    sparse::Pattern baseline_only_key{}; // Never supplied to v3 scoring/verification.
};
struct Matrix {
    unsigned width=0;
    std::vector<Window> windows;
    std::vector<uint32_t> begins,counts,queries,support,query_histories;
    std::vector<uint16_t> distances;
    std::vector<uint32_t> window_ranks,history_scores,history_ranks;
    std::vector<Timing> timings;
};
inline uint64_t allocation_bytes(const std::vector<Input>& inputs,const Matrix& m) {
    uint64_t anchors=0; for (const auto& input:inputs) anchors+=input.scoped.size();
    const uint64_t n=inputs.size(),q=m.queries.size(),g=m.query_histories.size(),w=m.windows.size();
    return anchors*sizeof(Anchor)+n*5*4+w*sizeof(Window)+m.support.size()*4+q*4+q*w*2+
        q*WINDOW_TOPK*4+AGGREGATIONS*g*n*8;
}
inline unsigned pop(uint32_t x) {
    x=x-((x>>1)&0x55555555u);
    x=(x&0x33333333u)+((x>>2)&0x33333333u);
    return (((x+(x>>4))&0x0f0f0f0fu)*0x01010101u)>>24;
}
inline unsigned hamming(const uint32_t* a,const uint32_t* b) {
    unsigned n=0; for (unsigned w=0;w<32;++w) n+=pop(a[w]^b[w]); return n;
}
inline std::vector<unsigned> reference_starts(unsigned n,unsigned width) {
    if (!n) return {};
    if (n<width) return {0};
    std::set<unsigned> starts;
    for (unsigned begin=0;begin<n;begin+=width/2) starts.insert(begin);
    starts.insert(n-width); // Full tail context, even when not on the stride grid.
    return {starts.begin(),starts.end()};
}
inline Window reference_window(const std::vector<Anchor>& anchors,unsigned begin,unsigned width,unsigned history) {
    Window out{};
    const unsigned count=std::min<unsigned>(width,unsigned(anchors.size())-begin);
    std::vector<Anchor> local(anchors.begin()+begin,anchors.begin()+begin+count);
    const auto vector=sparse::reference_fingerprint(local,0,false);
    std::copy(vector.begin(),vector.end(),out.hv);
    const auto& first=anchors[begin]; const auto& last=anchors[begin+count-1];
    out.history_id=history; out.width=width; out.scope_begin=begin; out.count=count;
    out.first_step=first.event.step; out.last_step=last.event.step;
    out.first_anchor=first.ordinal; out.last_anchor=last.ordinal; out.object=first.event.object;
    out.first_version=first.object_version; out.last_version=last.object_version;
    return out;
}
inline std::vector<uint32_t> context_slots(const std::vector<Window>& windows,unsigned begin,unsigned count,unsigned slots) {
    if (!count) return std::vector<uint32_t>(slots,MISSING);
    std::vector<uint32_t> eligible;
    for (unsigned i=0;i<count;++i) if (windows[begin+i].count==windows[begin+i].width) eligible.push_back(begin+i);
    if (eligible.empty()) for (unsigned i=0;i<count;++i) eligible.push_back(begin+i);
    std::vector<uint32_t> output;
    for (unsigned s=0;s<slots;++s) output.push_back(eligible[size_t(s)*(eligible.size()-1)/(slots-1)]);
    return output;
}
inline Matrix reference_plan(const std::vector<Input>& inputs,const std::vector<uint32_t>& roles,unsigned width) {
    Matrix m; m.width=width;
    for (unsigned h=0;h<inputs.size();++h) {
        m.begins.push_back(uint32_t(m.windows.size()));
        for (auto begin:reference_starts(unsigned(inputs[h].scoped.size()),width))
            m.windows.push_back(reference_window(inputs[h].scoped,begin,width,h));
        const unsigned count=unsigned(m.windows.size())-m.begins.back(); m.counts.push_back(count);
        const auto support=context_slots(m.windows,m.begins.back(),count,SUPPORT_SLOTS);
        m.support.insert(m.support.end(),support.begin(),support.end());
        if (roles[h]) {
            m.query_histories.push_back(h);
            const auto queries=context_slots(m.windows,m.begins.back(),count,QUERY_SLOTS);
            m.queries.insert(m.queries.end(),queries.begin(),queries.end());
        }
    }
    return m;
}
inline std::vector<uint16_t> reference_similarities(const Matrix& m) {
    std::vector<uint16_t> distances(m.queries.size()*m.windows.size());
    for (size_t q=0;q<m.queries.size();++q) for (size_t c=0;c<m.windows.size();++c)
        distances[q*m.windows.size()+c]=uint16_t(m.queries[q]==MISSING ? 1024 : hamming(m.windows[m.queries[q]].hv,m.windows[c].hv));
    return distances;
}
inline uint32_t reference_aggregate(const Matrix& m,const std::vector<uint16_t>& scores,unsigned group,unsigned candidate,unsigned strategy) {
    unsigned d[QUERY_SLOTS][SUPPORT_SLOTS];
    for (unsigned q=0;q<QUERY_SLOTS;++q) for (unsigned s=0;s<SUPPORT_SLOTS;++s) {
        const auto c=m.support[candidate*SUPPORT_SLOTS+s];
        d[q][s]=m.queries[group*QUERY_SLOTS+q]==MISSING || c==MISSING ? 1024 : scores[(group*QUERY_SLOTS+q)*m.windows.size()+c];
    }
    if (!strategy) {
        unsigned minimum=1024;
        for (auto& row:d) for (auto v:row) minimum=std::min(minimum,v);
        return minimum*1024;
    }
    if (strategy==1) {
        unsigned sum=0;
        for (auto& row:d) {
            std::array<unsigned,SUPPORT_SLOTS> sorted;
            std::copy_n(row,SUPPORT_SLOTS,sorted.begin()); std::sort(sorted.begin(),sorted.end());
            sum+=sorted[0]+sorted[1];
        }
        return sum*128; // Mean over 4 query slots x 2 candidate contexts, scaled 1024.
    }
    // Ordered, one-to-one slot alignment, with query extent mapped to candidate
    // extent. Exact position/versions are not consulted by approximate matching.
    unsigned previous[SUPPORT_SLOTS],current[SUPPORT_SLOTS];
    std::fill_n(previous,SUPPORT_SLOTS,INF); previous[0]=d[0][0];
    for (unsigned q=1;q<QUERY_SLOTS;++q) {
        std::fill_n(current,SUPPORT_SLOTS,INF);
        for (unsigned s=q;s<SUPPORT_SLOTS;++s) {
            if (q==QUERY_SLOTS-1 && s!=SUPPORT_SLOTS-1) continue;
            for (unsigned before=0;before<s;++before)
                current[s]=std::min(current[s],previous[before]+d[q][s]);
        }
        std::copy_n(current,SUPPORT_SLOTS,previous);
    }
    return previous[SUPPORT_SLOTS-1]*256;
}
inline void reference_retrieve(Matrix& m,const std::vector<uint32_t>& roles) {
    const unsigned n=unsigned(roles.size()),groups=unsigned(m.query_histories.size());
    m.distances=reference_similarities(m);
    m.window_ranks.assign(m.queries.size()*WINDOW_TOPK,MISSING);
    for (unsigned q=0;q<m.queries.size();++q) {
        std::vector<uint32_t> candidates;
        for (unsigned i=0;i<m.windows.size();++i) if (!roles[m.windows[i].history_id]) candidates.push_back(i);
        auto compare=[&](uint32_t a,uint32_t b) {
            const auto da=m.distances[size_t(q)*m.windows.size()+a],db=m.distances[size_t(q)*m.windows.size()+b];
            return da!=db ? da<db : a<b;
        };
        const unsigned k=std::min<unsigned>(WINDOW_TOPK,unsigned(candidates.size()));
        std::partial_sort(candidates.begin(),candidates.begin()+k,candidates.end(),compare);
        for (unsigned i=0;i<k;++i) m.window_ranks[q*WINDOW_TOPK+i]=candidates[i];
    }
    m.history_scores.resize(AGGREGATIONS*groups*n); m.history_ranks.assign(AGGREGATIONS*groups*n,MISSING);
    for (unsigned a=0;a<AGGREGATIONS;++a) for (unsigned q=0;q<groups;++q) {
        std::vector<uint32_t> candidates;
        for (unsigned h=0;h<n;++h) {
            m.history_scores[(a*groups+q)*n+h]=reference_aggregate(m,m.distances,q,h,a);
            if (!roles[h]) candidates.push_back(h);
        }
        std::sort(candidates.begin(),candidates.end(),[&](uint32_t x,uint32_t y) {
            const auto dx=m.history_scores[(a*groups+q)*n+x],dy=m.history_scores[(a*groups+q)*n+y];
            return dx!=dy ? dx<dy : x<y;
        });
        for (unsigned i=0;i<candidates.size();++i) m.history_ranks[(a*groups+q)*n+i]=candidates[i];
    }
}

struct GraphCheck { bool match=false; uint32_t nodes=0,edges=0,max_gap=0; std::vector<std::array<uint32_t,5>> links; };
// No sparse::Pattern or answer signatures enter this verifier. It compares raw
// endpoint-rooted dependency DAGs only AFTER a candidate was proposed by HDC.
inline GraphCheck verify_dependencies(const History& query,const std::vector<Anchor>& qa,
                                       const History& candidate,const std::vector<Anchor>& ca) {
    GraphCheck result;
    if (qa.empty() || ca.empty()) { result.match=qa.empty() && ca.empty(); return result; }
    std::vector<std::pair<uint32_t,uint32_t>> work={{qa.front().event_index,ca.front().event_index},
        {qa.back().event_index,ca.back().event_index}};
    std::map<uint32_t,uint32_t> forward,reverse;
    auto edge_nodes=[](const History& h,const Event& e) {
        std::vector<uint32_t> nodes;
        for (unsigned i=0;i<e.dependency_count;++i) {
            const auto& edge=h.edges[e.dependency_begin+i];
            require(edge.consumer_step==e.step && edge.producer_step<e.step,"invalid exact dependency");
            const auto found=std::lower_bound(h.events.begin(),h.events.end(),edge.producer_step,
                [](const Event& event,uint32_t step) { return event.step<step; });
            require(found!=h.events.end() && found->step==edge.producer_step,"missing exact producer");
            require(found->object==edge.object,"dependency object mismatch");
            nodes.push_back(uint32_t(found-h.events.begin()));
        }
        std::sort(nodes.begin(),nodes.end(),[&](uint32_t a,uint32_t b) {
            const auto& x=h.events[a]; const auto& y=h.events[b];
            return std::tie(x.object,x.action,x.source,x.flags,x.dependency_count,x.step)<
                   std::tie(y.object,y.action,y.source,y.flags,y.dependency_count,y.step);
        });
        return nodes;
    };
    while (!work.empty()) {
        const auto pair=work.back(); work.pop_back();
        const auto seen=forward.find(pair.first);
        if (seen!=forward.end()) { if (seen->second!=pair.second) return result; continue; }
        if (reverse.count(pair.second)) return result;
        const auto& q=query.events[pair.first]; const auto& c=candidate.events[pair.second];
        if (std::tie(q.action,q.object,q.source,q.flags,q.dependency_count)!=
            std::tie(c.action,c.object,c.source,c.flags,c.dependency_count)) return result;
        forward[pair.first]=pair.second; reverse[pair.second]=pair.first; ++result.nodes;
        const auto qe=edge_nodes(query,q),ce=edge_nodes(candidate,c);
        for (unsigned i=0;i<qe.size();++i) {
            work.push_back({qe[i],ce[i]}); ++result.edges;
            result.max_gap=std::max(result.max_gap,q.step-query.events[qe[i]].step);
            result.links.push_back({q.step,query.events[qe[i]].step,c.step,candidate.events[ce[i]].step,query.events[qe[i]].object});
        }
    }
    // Step IDs need not be equal, but the exact mapped chronology and version
    // aliasing must agree, including producers shared by the two distant roots.
    uint32_t previous=0; bool first=true;
    for (const auto& p:forward) {
        const unsigned step=candidate.events[p.second].step;
        if (!first && step<=previous) return result;
        previous=step; first=false;
    }
    result.match=true; return result;
}
} // namespace windowed

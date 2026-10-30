#pragma once
#include "../temporal_v3/reference.h"
#include <chrono>
#include <functional>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <iostream>

namespace candidate {
using namespace temporal;
using Clock=std::chrono::steady_clock;
inline double ms(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
constexpr unsigned WIDTH=8, QCONTEXTS=4, GCONTEXTS=8, MAXK=16;
constexpr uint32_t NONE=0xffffffffu;
struct Observed { std::vector<History> histories; std::vector<unsigned> gallery,queries; };
// This type is confined to construction/evaluation. No ranking API accepts it.
struct Truth { unsigned family,order,replica; };
struct Corpus { Observed observed; std::vector<Truth> truth; };
inline bool relevant(const Truth& a,const Truth& b) { return a.family==b.family && a.order==b.order; }
inline unsigned producer(const History& h,unsigned step) {
    if (step<h.events.size() && h.events[step].step==step) return step;
    auto it=std::lower_bound(h.events.begin(),h.events.end(),step,[](const Event& e,unsigned s){ return e.step<s; });
    require(it!=h.events.end() && it->step==step,"missing observed writer"); return unsigned(it-h.events.begin());
}
using Type=std::array<uint32_t,5>;
inline Type type(const Event& e) { return {e.action,e.object,e.source,e.flags,e.dependency_count}; }
inline std::vector<unsigned> parents(const History& h,unsigned i) {
    const auto& e=h.events[i]; std::vector<unsigned> out;
    for (unsigned j=0;j<e.dependency_count;++j) {
        const auto& d=h.edges[e.dependency_begin+j];
        require(d.consumer_step==e.step && d.producer_step<e.step,"future or invalid dependency");
        auto p=producer(h,d.producer_step); require(h.events[p].object==d.object,"writer object mismatch"); out.push_back(p);
    }
    std::sort(out.begin(),out.end(),[&](unsigned a,unsigned b) {
        const auto& x=h.events[a]; const auto& y=h.events[b];
        return std::tie(x.object,x.action,x.source,x.flags,x.dependency_count,x.step)<
               std::tie(y.object,y.action,y.source,y.flags,y.dependency_count,y.step);
    }); return out;
}
inline std::vector<unsigned> scope(const History& h) {
    std::vector<unsigned> out;
    for (unsigned i=0;i<h.events.size();++i) {
        const auto& e=h.events[i]; require(e.step<h.cutoff,"unobserved event");
        if (e.object==2 && sparse::reference_significant(e)) out.push_back(i);
    } return out;
}
inline std::vector<sparse::Anchor> endpoints_from_scope(const History& h,const std::vector<unsigned>& s) {
    if(s.empty()) return {};
    std::vector<sparse::Anchor> out(2); out[0].event_index=s.front(); out[1].event_index=s.back();
    out[0].event=h.events[s.front()]; out[1].event=h.events[s.back()]; return out;
}
inline std::vector<sparse::Anchor> endpoints(const History& h) { return endpoints_from_scope(h,scope(h)); }
inline std::vector<unsigned> query_nodes(const History& h,const std::vector<sparse::Anchor>& roots) {
    std::set<unsigned> seen; std::vector<unsigned> work;
    if(!roots.empty()) work={roots.front().event_index,roots.back().event_index};
    while(!work.empty()) { const auto i=work.back(); work.pop_back(); if(!seen.insert(i).second) continue;
        for(auto p:parents(h,i)) work.push_back(p); }
    return {seen.begin(),seen.end()};
}
struct Context { Event events[WIDTH]{}; uint32_t count=0; };
inline std::vector<Context> contexts_from_scope(const History& h,const std::vector<unsigned>& s,unsigned slots) {
    std::vector<Context> out(slots);
    if(s.empty()) return out;
    // Same sorted eligible windows as v3, without allocating every window start.
    const unsigned tail=s.size()>=WIDTH ? unsigned(s.size())-WIDTH:0;
    const unsigned eligible=tail/(WIDTH/2)+1+(tail%(WIDTH/2)!=0);
    for(unsigned t=0;t<slots;++t) {
        auto ordinal=unsigned(size_t(t)*(eligible-1)/(slots-1));
        auto b=ordinal==eligible-1 ? tail:ordinal*(WIDTH/2);
        out[t].count=std::min(WIDTH,unsigned(s.size())-b);
        for(unsigned j=0;j<out[t].count;++j) out[t].events[j]=h.events[s[b+j]];
    } return out;
}
inline std::vector<Context> contexts(const History& h,unsigned slots) { return contexts_from_scope(h,scope(h),slots); }
inline Vector scalar_window(const Context& c) {
    // Independent scalar reference uses the preserved v1 and v2 CPU functions.
    std::vector<sparse::Anchor> a;
    for(unsigned i=0;i<c.count;++i) { sparse::Anchor x{}; x.event=c.events[i]; const auto v=reference_event(x.event);
        std::copy(v.begin(),v.end(),x.hv); a.push_back(x); }
    return sparse::reference_fingerprint(a,0,false);
}
inline std::vector<unsigned> score_rank(const std::vector<uint32_t>& scores,const std::vector<uint8_t>& eligible={}) {
    std::vector<unsigned> rank;
    for(unsigned i=0;i<scores.size();++i) if(eligible.empty() || eligible[i]) rank.push_back(i);
    std::sort(rank.begin(),rank.end(),[&](unsigned a,unsigned b){ return scores[a]!=scores[b] ? scores[a]<scores[b] : a<b; }); return rank;
}
inline std::vector<uint32_t> scalar_scores(const std::vector<Vector>& q,const std::vector<Vector>& g) {
    std::vector<uint32_t> scores(g.size()/GCONTEXTS,1024*1024);
    for(unsigned h=0;h<scores.size();++h) for(const auto& a:q) for(unsigned s=0;s<GCONTEXTS;++s)
        scores[h]=std::min(scores[h],windowed::hamming(a.data(),g[h*GCONTEXTS+s].data())*1024);
    return scores;
}
struct TypeHash { size_t operator()(const Type& a) const { size_t x=0; for(auto v:a) x=(x^v)*1099511628211ull; return x; } };
struct Work { uint64_t nodes=0,edges=0,postings=0; unsigned pool=0; };
struct Shortlist { std::vector<unsigned> rank; Work work; };
struct ExactIndex {
    const Observed& o;
    std::unordered_map<Type,unsigned,TypeHash> types;
    std::vector<std::vector<uint64_t>> nodes;
    // Primitive event and one-edge features only, never whole-history signatures.
    std::unordered_map<uint64_t,unsigned> features;
    std::vector<std::vector<unsigned>> postings;
    std::vector<unsigned> seen;
    std::vector<std::vector<sparse::Anchor>> roots;
    uint64_t event_count=0,edge_count=0;
    unsigned intern(const Type& t) {
        auto found=types.find(t); if(found!=types.end()) return found->second;
        const unsigned n=unsigned(nodes.size()); types[t]=n; nodes.emplace_back(); return n;
    }
    unsigned feature(uint64_t key) {
        auto it=features.find(key); if(it!=features.end()) return it->second;
        unsigned n=unsigned(postings.size()); features[key]=n; postings.emplace_back(); seen.push_back(NONE); return n;
    }
    void add(uint64_t key,unsigned g) { unsigned f=feature(key); if(seen[f]!=g) { postings[f].push_back(g); seen[f]=g; } }
    static uint64_t node_key(unsigned t) { return (1ull<<62)|t; }
    static uint64_t edge_key(unsigned child,unsigned parent) { return (2ull<<62)|(uint64_t(child)<<31)|parent; }
    static uint64_t object_key(unsigned object) { return (3ull<<62)|object; }
    explicit ExactIndex(const Observed& input):o(input) {
        for(unsigned g=0;g<o.gallery.size();++g) {
            const auto& h=o.histories[o.gallery[g]]; roots.push_back(endpoints(h));
            for(unsigned i=0;i<h.events.size();++i) {
                const auto& e=h.events[i]; auto t=intern(type(e)); nodes[t].push_back((uint64_t(g)<<32)|i);
                add(node_key(t),g); add(object_key(e.object),g); if(e.source) add(object_key(e.source),g);
                ++event_count;
                for(auto p:parents(h,i)) { auto pt=intern(type(h.events[p])); add(edge_key(t,pt),g); ++edge_count; }
            }
        }
    }
    uint64_t bytes() const {
        uint64_t n=types.size()*(sizeof(Type)+sizeof(unsigned)+32)+features.size()*48;
        for(const auto& v:nodes) n+=v.capacity()*8+sizeof(v);
        for(const auto& v:postings) n+=v.capacity()*4+sizeof(v);
        n+=seen.capacity()*4; for(const auto& r:roots) n+=r.capacity()*sizeof(sparse::Anchor); return n;
    }
    std::vector<unsigned> query_features(const History& q,const std::vector<sparse::Anchor>& qr,Work& w) const {
        std::set<unsigned> out;
        auto insert=[&](uint64_t key){ auto it=features.find(key); if(it!=features.end()) out.insert(it->second); };
        for(auto i:query_nodes(q,qr)) {
            ++w.nodes; const auto& e=q.events[i]; auto ti=types.find(type(e));
            insert(object_key(e.object)); if(e.source) insert(object_key(e.source));
            if(ti==types.end()) continue;
            insert(node_key(ti->second));
            for(auto p:parents(q,i)) { ++w.edges; auto pi=types.find(type(q.events[p]));
                if(pi!=types.end()) insert(edge_key(ti->second,pi->second)); }
        } return {out.begin(),out.end()};
    }
    std::vector<uint32_t> inverted_scores(const History& q,const std::vector<sparse::Anchor>& qr,Work& w) const {
        std::vector<uint32_t> scores(o.gallery.size());
        for(auto f:query_features(q,qr,w)) {
            const auto& post=postings[f]; unsigned weight=1, ratio=unsigned(o.gallery.size()+1)/unsigned(post.size()+1);
            while(ratio>1) { ++weight; ratio>>=1; }
            for(auto g:post) { scores[g]+=weight; ++w.postings; }
        } return scores;
    }
    Shortlist inverted(const History& q,const std::vector<sparse::Anchor>& qr) const {
        Shortlist out; auto scores=inverted_scores(q,qr,out.work);
        for(unsigned g=0;g<scores.size();++g) if(scores[g]) out.rank.push_back(g);
        std::sort(out.rank.begin(),out.rank.end(),[&](unsigned a,unsigned b){ return scores[a]!=scores[b] ? scores[a]>scores[b] : a<b; });
        out.work.pool=unsigned(out.rank.size()); return out;
    }
    Shortlist dependency(const History& q,const std::vector<sparse::Anchor>& qr) const {
        Shortlist out; if(qr.empty()) return out;
        // Relational semijoin: typed node postings joined on actual writer edges.
        // It deliberately does not enforce global injectivity/aliasing/chronology:
        // these remain the unchanged full verifier's job.
        std::map<unsigned,std::vector<uint64_t>> matches;
        const auto qnodes=query_nodes(q,qr);
        for(auto i:qnodes) {
            auto ti=types.find(type(q.events[i])); auto& dest=matches[i];
            if(ti==types.end()) continue;
            const auto qp=parents(q,i);
            for(auto ref:nodes[ti->second]) {
                ++out.work.nodes; const auto g=unsigned(ref>>32),c=unsigned(ref);
                const auto& roots_g=roots[g]; if(roots_g.empty()) continue;
                if(i==qr.front().event_index && c!=roots_g.front().event_index) continue;
                if(i==qr.back().event_index && c!=roots_g.back().event_index) continue;
                const auto& h=o.histories[o.gallery[g]]; auto cp=parents(h,c); bool ok=cp.size()==qp.size();
                for(unsigned j=0;ok && j<qp.size();++j) {
                    ++out.work.edges; const auto& pm=matches.at(qp[j]);
                    ok=std::binary_search(pm.begin(),pm.end(),(uint64_t(g)<<32)|cp[j]);
                }
                if(ok) dest.push_back(ref);
            }
        }
        std::vector<uint8_t> a(o.gallery.size()),b(a.size());
        for(auto r:matches[qr.front().event_index]) a[unsigned(r>>32)]=1;
        for(auto r:matches[qr.back().event_index]) b[unsigned(r>>32)]=1;
        Work iw; auto scores=inverted_scores(q,qr,iw); out.work.postings=iw.postings;
        for(unsigned g=0;g<a.size();++g) if(a[g] && b[g]) out.rank.push_back(g);
        std::sort(out.rank.begin(),out.rank.end(),[&](unsigned x,unsigned y){ return scores[x]!=scores[y] ? scores[x]>scores[y] : x<y; });
        out.work.pool=unsigned(out.rank.size()); return out;
    }
};

inline Corpus load_original(const std::filesystem::path& folder) {
    Corpus c; auto data=temporal::load(folder/"events.bin"); c.observed.histories=std::move(data.histories);
    auto role=sparse::read_roles(folder/"roles.csv",unsigned(c.observed.histories.size()));
    for(unsigned i=0;i<role.size();++i) (role[i] ? c.observed.queries : c.observed.gallery).push_back(i);
    std::ifstream in(folder/"catalog.csv"); require(bool(in),"missing evaluation catalog"); std::string line; std::getline(in,line);
    c.truth.resize(role.size());
    while(std::getline(in,line)) { std::replace(line.begin(),line.end(),',',' '); std::istringstream s(line); unsigned id,f,o,r,unused;
        require(bool(s>>id>>f>>o>>r>>unused) && id<c.truth.size(),"bad catalog"); c.truth[id]={f,o,r}; }
    return c;
}

inline History wire(const std::vector<std::array<unsigned,4>>& actions,unsigned id) {
    History h; h.id=id; h.cutoff=unsigned(actions.size()); h.events.reserve(actions.size()); h.edges.reserve(actions.size()*2);
    std::unordered_map<unsigned,unsigned> writers,approvals;
    for(unsigned i=0;i<actions.size();++i) {
        auto a=actions[i]; Event e{a[0],a[1],a[2],a[3],i,unsigned(h.edges.size()),0,0};
        if(e.source) { require(writers.count(e.source)>0,"missing source"); h.edges.push_back({i,writers.at(e.source),e.source}); }
        else if(!sparse::reference_write(e.action) && writers.count(e.object)) h.edges.push_back({i,writers.at(e.object),e.object});
        if(sparse::reference_write(e.action)) { writers[e.object]=i; approvals.erase(e.object); }
        else if(e.action==11) approvals[e.object]=i;
        else if((e.action==5 || e.action==13 || e.action==14) && approvals.count(e.object)) h.edges.push_back({i,approvals.at(e.object),e.object});
        e.dependency_count=unsigned(h.edges.size())-e.dependency_begin; h.events.push_back(e);
    } return h;
}
inline Corpus scaled(unsigned count,unsigned n,const std::string& profile,unsigned seed) {
    require((count==16 || count==128 || count==1024) && n>=32,"unsupported scaled configuration");
    std::vector<std::array<unsigned,8>> patterns; std::array<unsigned,8> p={3,3,3,3,4,4,16,16};
    do { patterns.push_back(p); } while(std::next_permutation(p.begin(),p.end()));
    std::mt19937 dictionary(seed^0xa47e1u); // Explicit Fisher-Yates, reproducible across STL versions.
    for(unsigned i=unsigned(patterns.size());i>1;--i) std::swap(patterns[i-1],patterns[dictionary()%i]);
    std::vector<std::array<unsigned,3>> entries;
    for(unsigned f=0;f<256;++f) for(unsigned order=0;order<2;++order) for(unsigned r=0;r<2;++r) entries.push_back({f,order,r});
    std::mt19937 ids(seed^0x1732u); for(unsigned i=unsigned(entries.size());i>1;--i) std::swap(entries[i-1],entries[ids()%i]);
    Corpus c;
    auto build=[&](unsigned f,unsigned order,unsigned r,unsigned id) {
        std::mt19937 noise(seed+f*100003+r*1009+n*13); std::vector<std::array<unsigned,4>> a(n);
        for(unsigned i=0;i<n;++i) {
            unsigned x=noise(),obj=1000+(noise()%32),action=std::array<unsigned,3>{6,15,17}[x%3];
            a[i]={action,5000+obj-1000,0,Filler|(action==15 ? Local:0u)};
            if(i>=2 && i<n-4 && profile!="sparse" && (i%2==0 || profile=="dense_target_far")) {
                if(profile=="dense_target_far") { action=std::array<unsigned,4>{5,12,13,14}[x%4];
                    a[i]={action,2,0,Filler|Sensitive|(action==13 ? External:Local)}; }
                else if(profile=="dense_target") { action=std::array<unsigned,3>{4,4,16}[x%3]; a[i]={action,2,1,Filler|Derived|Sensitive}; }
                else { a[i]={4,obj,1,Filler|Derived|Sensitive}; }
            }
        }
        a[0]={1,1,0,PrivateRead|Sensitive}; a[1]={4,2,1,Derived|Sensitive};
        // Every family has exactly the same core event multiset, objects and
        // endpoint types. Only this real, writer-linked action order differs.
        std::vector<unsigned> steps;
        if(profile=="dense_target") for(unsigned i=n-15;i<n-4;++i) steps.push_back(i);
        else for(unsigned j=0;j<11;++j) steps.push_back(2+(profile=="dense_target_far" ? n/3-2:n-18)*j/10);
        a[steps[0]]={4,2,1,Derived|Sensitive};
        for(unsigned j=0;j<10;++j) a[steps[j+1]]={j==0 || j==9 ? 4u:patterns[f][j-1],2,2,Derived|Sensitive};
        unsigned early=n-4,late=n-3;
        if(profile=="dense_target_far") { early=steps.back()+1; late=early+(n>=32768 ? 2048:n>=4096 ? 256:std::max(1u,n/16))+1; }
        const std::array<unsigned,4> rewrite={4,2,2,Derived|Sensitive},grant={11,2,0,Approval};
        a[early]=order ? grant:rewrite; a[late]=order ? rewrite:grant;
        a[n-2]={5,2,0,Local|Sensitive}; a[n-1]={13,2,0,External|Untrusted|Sensitive};
        c.observed.histories.push_back(wire(a,id)); c.truth.push_back({f,order,r});
    };
    for(unsigned id=0;id<entries.size();++id) if(entries[id][0]<count/4) {
        const auto x=entries[id]; c.observed.gallery.push_back(unsigned(c.observed.histories.size())); build(x[0],x[1],x[2],id);
    }
    std::array<unsigned,8> query_ids={0,1,2,3,4,5,6,7};
    for(unsigned i=8;i>1;--i) std::swap(query_ids[i-1],query_ids[ids()%i]);
    for(unsigned f=0;f<4;++f) for(unsigned o=0;o<2;++o) {
        unsigned id=1024+query_ids[f*2+o]; c.observed.queries.push_back(unsigned(c.observed.histories.size())); build(f,o,2,id);
    } return c;
}
inline uint64_t observed_hash(const History& h) {
    uint64_t hash=1469598103934665603ull;
    auto bytes=[&](const void* data,size_t n) { auto p=static_cast<const unsigned char*>(data); for(size_t i=0;i<n;++i) { hash^=p[i]; hash*=1099511628211ull; } };
    bytes(h.events.data(),h.events.size()*sizeof(Event)); bytes(h.edges.data(),h.edges.size()*sizeof(Edge)); return hash;
}
} // namespace candidate

#pragma once
#include "../temporal_v1/experiment.h"

namespace sparse {
using namespace temporal;
constexpr uint32_t UNKNOWN=0xffffffffu;
constexpr unsigned METHODS=4, TOPK=5;
inline const char* method_name(unsigned method) {
    static const char* names[]={"v1_root","v2_global","v2_object_scoped","exact_pattern"};
    return names[method];
}
struct Version { uint32_t object=UNKNOWN, source=UNKNOWN; };
struct Anchor {
    Event event;
    uint32_t event_index, object_version, source_version, ordinal;
    uint32_t hv[32];
};
static_assert(sizeof(Anchor)==176 && sizeof(Version)==8,"sparse layout mismatch");
using Pattern=std::array<uint32_t,8>;
struct Analysis {
    Tree tree;
    std::vector<Anchor> anchors;
    std::vector<Version> versions;
    std::array<Vector,3> fingerprints{};
    Pattern pattern{};
    uint32_t scoped_count=0;
    uint64_t tree_bytes=0, scratch_bytes=0;
    std::vector<Timing> timings;
};
inline bool reference_write(unsigned action) {
    return action==1 || action==2 || action==3 || action==4 || action==8 || action==15 || action==16;
}
inline bool reference_significant(const Event& e) {
    const bool read=(e.action==1 || e.action==2 || e.action==8 || (e.flags&PrivateRead)) && (e.flags&Sensitive);
    const bool transform=e.action==3 || e.action==4 || e.action==16 || (e.flags&Derived);
    const bool save=e.action==5 || e.action==13 || e.action==14;
    const bool permission=e.action==11 || e.action==12 || (e.flags&Approval);
    return read || transform || save || permission || (e.flags&External);
}
inline std::vector<Version> reference_versions(const std::vector<Event>& events) {
    std::map<uint32_t,uint32_t> writers;
    std::vector<Version> versions;
    for (const auto& e:events) {
        Version v;
        const auto source=writers.find(e.source);
        if (e.source && source!=writers.end()) v.source=source->second;
        if (reference_write(e.action) && e.object) writers[e.object]=e.step;
        const auto object=writers.find(e.object);
        if (e.object && object!=writers.end()) v.object=object->second;
        versions.push_back(v);
    }
    return versions;
}
inline Vector reference_fingerprint(const std::vector<Anchor>& anchors,uint32_t target,bool scoped) {
    Vector result{};
    std::vector<const Anchor*> selected;
    for (const auto& a:anchors) if (!scoped || a.event.object==target) selected.push_back(&a);
    if (selected.empty()) return result;
    for (unsigned w=0;w<32;++w) {
        std::array<uint32_t,32> votes{};
        for (unsigned ordinal=0;ordinal<selected.size();++ordinal) {
            const uint32_t content=selected[ordinal]->hv[(w+(ordinal*7u)%32)%32];
            const uint32_t position=reference_hash(0x71a5b39du ^ ordinal*0x9e3779b9u ^ w*0x85ebca6bu);
            for (unsigned bit=0;bit<32;++bit)
                votes[bit]+=((content>>((bit+32-ordinal%32)%32)) ^ (position>>bit))&1u;
        }
        const uint32_t tie=reference_hash(0xe17a1465u ^ w*0xc2b2ae35u);
        for (unsigned bit=0;bit<32;++bit)
            if (2*votes[bit]>selected.size() || (2*votes[bit]==selected.size() && ((tie>>bit)&1u))) result[w]|=1u<<bit;
    }
    return result;
}
inline Pattern reference_pattern(const History& h,const std::vector<Version>& versions,uint32_t target) {
    Pattern key{0,0,2,0,target,0,0,0};
    uint32_t writer=UNKNOWN, approval_step=UNKNOWN, approval_version=UNKNOWN;
    for (unsigned i=0;i<h.events.size();++i) {
        const auto& e=h.events[i];
        if (e.object!=target) continue;
        if (reference_write(e.action)) {
            if (!key[0]) key[0]=e.action;
            key[1]=e.action; key[5]=e.source; writer=e.step;
        }
        if (e.action==11) { approval_step=e.step; approval_version=versions[i].object; }
        if (e.action==13 || e.action==14) { key[3]=e.action; key[6]=e.flags&(Sensitive|Untrusted|External); }
    }
    if (approval_step!=UNKNOWN && writer!=UNKNOWN) {
        key[2]=uint32_t(approval_step>writer);
        key[7]=uint32_t(approval_version==writer);
    }
    return key;
}
inline Analysis reference_analyze(const History& history,uint32_t target) {
    Analysis result;
    result.versions=reference_versions(history.events);
    for (unsigned i=0;i<history.events.size();++i) if (reference_significant(history.events[i])) {
        Anchor a{}; a.event=history.events[i]; a.event_index=i;
        a.object_version=result.versions[i].object; a.source_version=result.versions[i].source;
        a.ordinal=uint32_t(result.anchors.size());
        const auto hv=reference_event(a.event); std::copy(hv.begin(),hv.end(),a.hv);
        result.anchors.push_back(a); result.scoped_count+=uint32_t(a.event.object==target);
    }
    result.fingerprints[1]=reference_fingerprint(result.anchors,target,false);
    result.fingerprints[2]=reference_fingerprint(result.anchors,target,true);
    result.pattern=reference_pattern(history,result.versions,target);
    return result;
}
inline std::vector<uint16_t> reference_distances(const std::vector<Analysis>& analyses) {
    const size_t n=analyses.size();
    std::vector<uint16_t> output(METHODS*n*n);
    for (unsigned m=0;m<METHODS;++m) for (size_t q=0;q<n;++q) for (size_t c=0;c<n;++c) {
        output[(m*n+q)*n+c]=uint16_t(m==3 ? (analyses[q].pattern==analyses[c].pattern ? 0 : 1024) :
            reference_distance(analyses[q].fingerprints[m].data(),analyses[c].fingerprints[m].data()));
    }
    return output;
}
inline std::vector<uint32_t> reference_ranks(const std::vector<uint16_t>& distances,const std::vector<uint32_t>& roles) {
    const size_t n=roles.size();
    std::vector<uint32_t> ranks(METHODS*n*TOPK,UNKNOWN);
    for (unsigned m=0;m<METHODS;++m) for (size_t q=0;q<n;++q) if (roles[q]==1) {
        std::vector<uint32_t> candidates;
        for (uint32_t c=0;c<n;++c) if (!roles[c]) candidates.push_back(c);
        std::sort(candidates.begin(),candidates.end(),[&](uint32_t a,uint32_t b) {
            const auto da=distances[(m*n+q)*n+a],db=distances[(m*n+q)*n+b];
            return da!=db ? da<db : a<b;
        });
        for (unsigned k=0;k<std::min<size_t>(TOPK,candidates.size());++k) ranks[(m*n+q)*TOPK+k]=candidates[k];
    }
    return ranks;
}
} // namespace sparse

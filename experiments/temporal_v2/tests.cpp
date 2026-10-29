#include "reference.h"
#include <iostream>

using namespace sparse;
int main() {
    try {
        Event e{8,1001,0,PrivateRead|Sensitive|Filler,0,0,0,0};
        require(reference_significant(e),"significant filler read was incorrectly dropped");
        e.flags=PrivateRead; require(!reference_significant(e),"non-sensitive read selected without another reason");
        for (unsigned action:{3u,4u,5u,11u,12u,13u,14u,16u}) {
            e.action=action; e.flags=0; require(reference_significant(e),"required action type missing");
        }
        for (auto flag:{External,Approval,Derived}) {
            e.action=17; e.flags=flag; require(reference_significant(e),"significant property ignored");
        }
        History h;
        h.events={{1,1,0,PrivateRead|Sensitive,0,0,0,0},
                  {4,2,1,Derived|Sensitive,1,0,1,0},
                  {11,2,0,Approval,2,1,1,0},
                  {16,2,1,Derived|Sensitive,3,2,1,0},
                  {13,2,0,External|Sensitive|Untrusted,4,3,1,0}};
        h.edges={{1,0,1},{2,1,2},{3,0,1},{4,3,2}}; h.cutoff=5;
        const auto a=reference_analyze(h,2);
        require(a.anchors.size()==5 && a.scoped_count==4,"anchor counts wrong");
        require(a.anchors[1].source_version==0 && a.anchors[1].object_version==1 &&
            a.anchors[2].object_version==1 && a.anchors[3].object_version==3 && a.anchors[4].object_version==3,
            "object/source version reconstruction failed");
        require(a.pattern[2]==0 && a.pattern[7]==0,"stale approval treated as current version");
        for (unsigned i=0;i<a.anchors.size();++i) {
            require(a.anchors[i].event_index==i && a.anchors[i].ordinal==i,"anchor ordering changed");
            require(a.anchors[i].event.dependency_begin==h.events[i].dependency_begin &&
                a.anchors[i].event.dependency_count==h.events[i].dependency_count,"exact edge references lost");
        }
        auto changed=h;
        std::swap(changed.events[2],changed.events[3]);
        changed.events[2].step=2; changed.events[3].step=3;
        changed.edges={{1,0,1},{2,0,1},{3,2,2},{4,2,2},{4,3,2}};
        changed.events[2].dependency_begin=1; changed.events[3].dependency_begin=2;
        changed.events[4].dependency_begin=3; changed.events[4].dependency_count=2;
        const auto b=reference_analyze(changed,2);
        require(a.fingerprints[1]!=b.fingerprints[1] && a.fingerprints[2]!=b.fingerprints[2],"controlled order not distinguishable");
        require(b.pattern[2]==1 && b.pattern[7]==1,"current approval relation missing");
        for (unsigned n:{0u,1u,31u,32u,33u,1024u,4096u,32768u}) {
            History long_h;
            for (unsigned i=0;i<n;++i) long_h.events.push_back({17,5000,0,Filler,i,0,0,0});
            if (n>=2) {
                long_h.events[0]={1,1,0,PrivateRead|Sensitive,0,0,0,0};
                long_h.events[n-1]={11,1,0,Approval,n-1,0,0,0};
            }
            const auto analyzed=reference_analyze(long_h,1);
            require(analyzed.anchors.size()==(n>=2 ? 2u : 0u),"irrelevant gap changed selection");
            if (n>=2) {
                History short_h; short_h.events={long_h.events[0],long_h.events[n-1]};
                short_h.events[1].step=1;
                const auto short_a=reference_analyze(short_h,1);
                require(analyzed.fingerprints[1]==short_a.fingerprints[1] && analyzed.fingerprints[2]==short_a.fingerprints[2],
                    "absolute step/gap leaked into sparse positional encoding");
            } else require(analyzed.fingerprints[1]==Vector{} && analyzed.fingerprints[2]==Vector{},"empty anchor sentinel wrong");
        }
        History unknown; unknown.events={{11,99,0,Approval,55,0,0,0}};
        require(reference_analyze(unknown,99).anchors[0].object_version==UNKNOWN,"unknown writer was invented");
        History non_anchor;
        non_anchor.events={{15,9,0,Local,0,0,0,0},{5,9,0,Local,1,0,1,0},{16,2,9,Derived,2,1,1,0}};
        non_anchor.edges={{1,0,9},{2,0,9}};
        const auto retained=reference_analyze(non_anchor,2);
        require(retained.anchors.size()==2 && retained.anchors[0].event_index==1 && retained.anchors[0].object_version==0 &&
            retained.anchors[1].source_version==0 && retained.anchors[1].event.dependency_begin==1,
            "non-anchor writer/provenance reference was lost");
        std::vector<Analysis> corpus{a,b,a,b};
        const auto distances=reference_distances(corpus);
        const auto ranks=reference_ranks(distances,{0,0,1,1});
        require(ranks[(1*4+2)*TOPK]==0 && ranks[(1*4+3)*TOPK]==1,"retrieval/tie ordering failed");
        std::cout<<"PASS: significance, exact versions/edges, ordering, gap invariance, empty/dense inputs and retrieval\n";
        return 0;
    } catch (const std::exception& e) { std::cerr<<"Sparse test error: "<<e.what()<<'\n'; return 1; }
}

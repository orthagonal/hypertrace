#include "reference.h"
#include <iostream>

using namespace temporal;
int main() {
    try {
        for (unsigned length:{0u,1u,2u,3u,31u,32u,33u,1024u,1025u,4096u,32768u}) {
            std::vector<Event> events(length);
            for (unsigned i=0;i<length;++i) events[i]=Event{17,1000,0,Filler,i*3,0,0,0};
            for (unsigned i:{0u,31u,32u,1023u,1024u,4095u,32767u}) if (i<length)
                events[i]=Event{1,1,0,PrivateRead|Sensitive,i*3,0,0,0};
            const auto tree=reference_build(events);
            verify_masks(tree);
            auto corrupt_summaries=tree;
            for (auto& leaf:corrupt_summaries.leaves) std::fill_n(leaf.hv,WORDS,0xffffffffu);
            for (auto& level:corrupt_summaries.levels) for (auto& node:level) std::fill_n(node.hv,WORDS,0);
            for (unsigned p=0;p<PROPERTIES;++p)
                require(exact_search(tree,p).events==exact_search(corrupt_summaries,p).events,
                    "exact search incorrectly uses lossy vectors");
            if (!tree.levels.empty() && tree.levels[0].back().child_count<32) {
                auto bad=tree;
                bad.levels[0].back().masks[0]|=1u<<31;
                bool rejected=false;
                try { verify_masks(bad); } catch (const std::exception&) { rejected=true; }
                require(rejected,"mask validation missed invalid partial-group bit");
            }
            if (!length) require(tree.levels.empty() && exact_search(tree,0).events.empty(),"empty hierarchy failed");
        }
        Event a{1,1,0,PrivateRead|Sensitive,0,0,0,0},b{11,2,0,Approval,1,0,0,0};
        auto original=reference_build({a,b});
        std::swap(a.step,b.step);
        auto reordered=reference_build({b,a});
        const auto r1=root(original),r2=root(reordered);
        require(reference_distance(r1.data(),r2.data())>0,"controlled two-event order is not distinguishable");
        auto metadata=a;
        metadata.step=999999; metadata.dependency_begin=123; metadata.dependency_count=12;
        require(reference_event(a)==reference_event(metadata),"routing metadata leaks into event encoding");
        auto object_changed=a; object_changed.object=987;
        require(reference_event(a)!=reference_event(object_changed),"object ID is missing from encoding");
        require(reference_distance(r1.data(),r1.data())==0,"identity Hamming distance failed");
        bool invalid=false;
        try { exact_search(original,8); } catch (const std::exception&) { invalid=true; }
        require(invalid,"invalid property was accepted");
        std::cout<<"PASS: empty/singleton/partial/long hierarchies; exact masks; order sensitivity; object IDs; routing isolation\n";
        return 0;
    } catch (const std::exception& e) { std::cerr<<"Temporal test failure: "<<e.what()<<'\n'; return 1; }
}

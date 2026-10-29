#include "experiment.h"

namespace temporal {
struct CpuEngine {
    unsigned repeats=1;
    const char* name() const { return "Temporal Memory v1 scalar CPU reference"; }
    Evaluation evaluate(const std::vector<Event>& events,bool full) {
        Evaluation result;
        const auto start=std::chrono::steady_clock::now();
        result.tree=reference_build(events);
        const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        result.timings.push_back({"cpu_build_wall",0,unsigned(events.size()),1,ms,ms,ms});
        verify_masks(result.tree);
        if (!full) return result;
        result.query=query_vector(events);
        for (unsigned l=0;l<=result.tree.levels.size();++l) {
            std::vector<uint16_t> distances;
            for (size_t i=0;i<level_count(result.tree,l);++i)
                distances.push_back(uint16_t(reference_distance(result.query.data(),level_vector(result.tree,l,i))));
            result.distances.push_back(std::move(distances));
        }
        for (unsigned p=0;p<PROPERTIES;++p) {
            result.searches[p]=exact_search(result.tree,p);
            std::sort(result.searches[p].regions.begin(),result.searches[p].regions.end());
        }
        return result;
    }
};
}
int main(int argc,char** argv) {
    try { temporal::CpuEngine engine; return temporal::run_experiment(argc,argv,engine); }
    catch (const std::exception& e) { std::cerr<<"Temporal CPU error: "<<e.what()<<'\n'; return 1; }
}

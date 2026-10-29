#include "experiment.h"

namespace sparse {
struct CpuEngine {
    unsigned repeats=1;
    const char* name() const { return "Temporal Memory v2 independent scalar CPU reference"; }
    Analysis analyze(const History& h,uint32_t target) {
        const auto start=std::chrono::steady_clock::now();
        auto a=reference_analyze(h,target);
        const auto middle=std::chrono::steady_clock::now();
        a.tree=reference_build(h.events); verify_masks(a.tree); a.fingerprints[0]=root(a.tree);
        const auto end=std::chrono::steady_clock::now();
        const auto sparse_ms=std::chrono::duration<double,std::milli>(middle-start).count();
        const auto tree_ms=std::chrono::duration<double,std::milli>(end-middle).count();
        a.timings.push_back({"cpu_sparse_wall",0,unsigned(h.events.size()),1,sparse_ms,sparse_ms,sparse_ms});
        a.timings.push_back({"cpu_v1_wall",0,unsigned(h.events.size()),1,tree_ms,tree_ms,tree_ms});
        return a;
    }
    Scores score(const std::vector<Analysis>& a,const std::vector<uint32_t>& roles) {
        Scores s; s.distances=reference_distances(a); s.ranks=reference_ranks(s.distances,roles); return s;
    }
};
}
int main(int argc,char** argv) {
    try { sparse::CpuEngine engine; return sparse::run(argc,argv,engine); }
    catch (const std::exception& e) { std::cerr<<"Sparse CPU error: "<<e.what()<<'\n'; return 1; }
}

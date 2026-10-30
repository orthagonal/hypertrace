#include "experiment.h"
namespace windowed {
struct CpuEngine {
    unsigned repeats=1; uint64_t peak_bytes=0;
    const char* name() const { return "Temporal Memory v3 independent scalar CPU reference"; }
    Matrix evaluate(const std::vector<Input>& inputs,const std::vector<uint32_t>& roles,unsigned width) {
        auto m=reference_plan(inputs,roles,width); peak_bytes=allocation_bytes(inputs,m); reference_retrieve(m,roles); return m;
    }
};
}
int main(int argc,char** argv) {
    try { windowed::CpuEngine engine; return windowed::run(argc,argv,engine); }
    catch (const std::exception& e) { std::cerr<<"V3 CPU error: "<<e.what()<<'\n'; return 1; }
}

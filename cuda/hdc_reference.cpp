// CPU mathematical reference and rule-based baseline. Compiles without CUDA.
#include "safety_data.h"
#include "read_dataset.h"
#include "report.h"
#include "diagnostics.h"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

static int pop32(uint32_t x) {
    int n=0;
    while(x) { x &= x-1; n++; }
    return n;
}
int main(int argc,char** argv) {
    try {
        const auto options = diagnostics::options(argc,argv,false);
        const std::vector<Record> data=load_dataset(argv[1]);
        const size_t n=data.size();
        if (!n) throw std::runtime_error("empty dataset");
        std::vector<uint32_t> vec(n*HDC_WORDS);
        for (size_t i=0;i<n;i++)
            for (unsigned w=0;w<HDC_WORDS;w++) vec[i*HDC_WORDS+w]=encode_word(data[i],w);
        uint32_t prototypes[2][HDC_WORDS] = {};
        for (unsigned bit=0;bit<HDC_DIM_BITS;bit++) {
            const unsigned word=bit/32,shift=bit%32;
            unsigned votes[2]={0,0}, totals[2]={0,0};
            for (size_t i=0;i<n;i++) if(data[i].split==0) {
                const unsigned cls=data[i].label;
                totals[cls]++;
                votes[cls]+=(vec[i*HDC_WORDS+word]>>shift)&1u;
            }
            for(unsigned cls=0;cls<2;cls++)
                if(totals[cls] && 2*votes[cls]>=totals[cls])
                    prototypes[cls][word]|=1u<<shift;
        }
        int delta=0;
        for(unsigned w=0;w<HDC_WORDS;w++) delta+=pop32(prototypes[0][w]^prototypes[1][w]);
        std::printf("CPU reference | %zu records | %u-bit HDC | class prototype separation %d bits\n",
            n,HDC_DIM_BITS,delta);
        PredictionComparison comparisons[4]{};
        std::vector<uint8_t> predictions(n);
        for(unsigned split=0;split<4;split++) {
            Metrics model{},baseline{};
            unsigned samples=0;
            for(size_t i=0;i<n;i++) if(data[i].split==split) {
                int distance0=0, distance1=0;
                for(unsigned w=0;w<HDC_WORDS;w++) {
                    distance0+=pop32(vec[i*HDC_WORDS+w]^prototypes[0][w]);
                    distance1+=pop32(vec[i*HDC_WORDS+w]^prototypes[1][w]);
                }
                const bool truth = data[i].label != 0;
                const bool hdc = distance1 < distance0;
                predictions[i] = static_cast<uint8_t>(hdc);
                const bool rules = basic_baseline(data[i]);
                model.add(truth,hdc);
                baseline.add(truth,rules);
                comparisons[split].add(truth,hdc,rules);
                samples++;
            }
            std::printf("\n%s (%u):\n",split_name(split),samples);
            model.print("CPU HDC prototype");
            baseline.print("simple rule baseline");
        }
        std::printf("\nPaired prediction comparison (same records; 1 = violation):\n");
        for (unsigned split=0;split<4;split++) {
            std::printf("\n%s:\n", split_name(split));
            comparisons[split].print();
        }
        diagnostics::run(data,prototypes[0],prototypes[1],predictions,options.directory);
        std::puts("\nUse CPU/GPU metrics as a sanity check; both share encoding equations.");
    } catch (const std::exception& ex) { std::fprintf(stderr,"Error: %s\n",ex.what());return 1; }
    return 0;
}

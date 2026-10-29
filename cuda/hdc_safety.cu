// GPU first experiment: event encoding -> per-class online-style HDC vote prototypes -> prediction.
// CUDA run is independent of Rust; it consumes the simulator's binary dataset.
#include "safety_data.h"
#include "read_dataset.h"
#include "report.h"
#include "diagnostics.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#define CUDA_OK(call) do { cudaError_t err=(call); if (err!=cudaSuccess) { \
    std::fprintf(stderr,"CUDA failure %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err)); \
    std::exit(2); } } while(0)

__global__ void encode_events(const Record* records, uint32_t* vectors, uint32_t n) {
    uint32_t item = blockIdx.x;
    uint32_t lane = threadIdx.x;
    if (item >= n || lane >= HDC_WORDS) return;
    vectors[item * HDC_WORDS + lane] = encode_word(records[item], lane);
}

// One GPU lane per HDC bit. Each lane accumulates votes across all training items.
// This deliberately simple first kernel provides a baseline to optimize later.
__global__ void learn_prototypes(const Record* records, const uint32_t* vectors,
                                 uint32_t* prototypes, uint32_t n) {
    uint32_t bit = blockIdx.x*blockDim.x + threadIdx.x;
    if (bit >= HDC_DIM_BITS) return;
    uint32_t word = bit / 32u, shift = bit % 32u;
    uint32_t votes[2] = {0, 0}, totals[2] = {0, 0};
    for (uint32_t i=0; i<n; ++i) {
        if (records[i].split != 0) continue;
        uint32_t cls = records[i].label;
        ++totals[cls];
        votes[cls] += (vectors[i*HDC_WORDS + word] >> shift) & 1u;
    }
    for (uint32_t cls=0; cls<2; ++cls) {
        if (totals[cls] && 2*votes[cls] >= totals[cls])
            atomicOr(prototypes + cls*HDC_WORDS + word, 1u << shift);
    }
}

__global__ void predict(const uint32_t* vectors, const uint32_t* prototypes,
                        uint8_t* predictions, int16_t* margins, uint32_t n) {
    uint32_t item = blockIdx.x;
    uint32_t lane = threadIdx.x;
    if (item >= n || lane >= HDC_WORDS) return;
    uint32_t hv = vectors[item*HDC_WORDS + lane];
    int bad = __popc(hv ^ prototypes[HDC_WORDS+lane]);
    int good = __popc(hv ^ prototypes[lane]);
    for (int mask=16; mask>0; mask >>= 1) {
        bad += __shfl_down_sync(0xffffffff, bad, mask);
        good += __shfl_down_sync(0xffffffff, good, mask);
    }
    if (lane == 0) {
        predictions[item] = static_cast<uint8_t>(bad < good);
        margins[item] = static_cast<int16_t>(good-bad);
    }
}

int main(int argc, char** argv) {
    try {
        const auto options = diagnostics::options(argc,argv,true);
        const bool verify = options.verify;
        auto data = load_dataset(argv[1]);
        const uint32_t n = static_cast<uint32_t>(data.size());
        if (!n) { std::fprintf(stderr, "empty dataset\n"); return 1; }
        cudaDeviceProp prop{};
        CUDA_OK(cudaGetDeviceProperties(&prop, 0));
        std::printf("GPU: %s | %u HDC bits | %u records\n", prop.name, HDC_DIM_BITS, n);
        Record* d_data=nullptr;
        uint32_t *d_vectors=nullptr, *d_protos=nullptr;
        uint8_t* d_pred=nullptr;
        int16_t* d_margin=nullptr;
        CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_data), size_t(n)*sizeof(Record)));
        CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_vectors), size_t(n)*HDC_WORDS*sizeof(uint32_t)));
        CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_protos), 2*HDC_WORDS*sizeof(uint32_t)));
        CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_pred), n));
        CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_margin), size_t(n)*sizeof(int16_t)));
        CUDA_OK(cudaMemcpy(d_data, data.data(), size_t(n)*sizeof(Record),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemset(d_protos,0,2*HDC_WORDS*sizeof(uint32_t)));
        cudaEvent_t start, stop;
        CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop));
        auto launch_timed = [&](auto launch, const char* name) {
            CUDA_OK(cudaEventRecord(start));
            launch();
            CUDA_OK(cudaGetLastError());
            CUDA_OK(cudaEventRecord(stop));
            CUDA_OK(cudaEventSynchronize(stop));
            float ms=0;
            CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
            std::printf("%-23s %.3f ms (single-run, not a benchmark)\n",name, ms);
        };
        // One warp per example; 4-thread variants become a later performance experiment.
        launch_timed([&]() { encode_events<<<n,32>>>(d_data,d_vectors,n); }, "encode_events");
        launch_timed([&]() { learn_prototypes<<<8,128>>>(d_data,d_vectors,d_protos,n); }, "learn_prototypes");
        launch_timed([&]() { predict<<<n,32>>>(d_vectors,d_protos,d_pred,d_margin,n); }, "predict");
        CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(stop));
        std::vector<uint8_t> pred(n);
        std::vector<int16_t> margin(n);
        uint32_t protos[2*HDC_WORDS];
        CUDA_OK(cudaMemcpy(pred.data(), d_pred, n, cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(margin.data(),d_margin,size_t(n)*sizeof(int16_t),cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(protos,d_protos,sizeof(protos),cudaMemcpyDeviceToHost));
        if (verify) {
            std::printf("Verifying GPU prototypes and predictions against CPU reference...\n");
            uint32_t expected[2][HDC_WORDS] = {};
            std::vector<uint32_t> cpu_vec(size_t(n)*HDC_WORDS);
            for (uint32_t i=0;i<n;i++)
                for(unsigned w=0;w<HDC_WORDS;w++) cpu_vec[size_t(i)*HDC_WORDS+w]=encode_word(data[i],w);
            for(unsigned bit=0;bit<HDC_DIM_BITS;bit++) {
                unsigned word=bit/32, shift=bit%32, totals[2]={0,0}, votes[2]={0,0};
                for(uint32_t i=0;i<n;i++) if(data[i].split==0) {
                    unsigned cls=data[i].label;
                    totals[cls]++; votes[cls]+=(cpu_vec[size_t(i)*HDC_WORDS+word]>>shift)&1u;
                }
                for(unsigned cls=0;cls<2;cls++)
                    if(totals[cls] && 2*votes[cls]>=totals[cls]) expected[cls][word]|=1u<<shift;
            }
            if (std::memcmp(expected,protos,sizeof(protos))!=0)
                throw std::runtime_error("GPU learned prototypes differ from CPU");
            std::vector<uint32_t> gpu_vec(size_t(n)*HDC_WORDS);
            CUDA_OK(cudaMemcpy(gpu_vec.data(),d_vectors,gpu_vec.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost));
            if(gpu_vec!=cpu_vec) throw std::runtime_error("GPU encode differs from CPU");
            for(uint32_t i=0;i<n;i++) {
                int distances[2]={0,0};
                for(unsigned w=0;w<HDC_WORDS;w++) {
                    uint32_t a=cpu_vec[size_t(i)*HDC_WORDS+w]^expected[0][w];
                    uint32_t b=cpu_vec[size_t(i)*HDC_WORDS+w]^expected[1][w];
                    while(a) { distances[0]++; a&=a-1; }
                    while(b) { distances[1]++; b&=b-1; }
                }
                if(pred[i]!=(distances[1]<distances[0]))
                    throw std::runtime_error("GPU prediction differs from CPU");
                if(margin[i]!=distances[0]-distances[1])
                    throw std::runtime_error("GPU prediction margin differs from CPU");
            }
            std::printf("PASS: GPU encoder, class prototypes, and every prediction match CPU.\n");
        }
        CUDA_OK(cudaFree(d_data)); CUDA_OK(cudaFree(d_vectors)); CUDA_OK(cudaFree(d_protos));
        CUDA_OK(cudaFree(d_pred)); CUDA_OK(cudaFree(d_margin));
        int gap=0; for(unsigned w=0;w<HDC_WORDS;w++) {
            uint32_t bits=protos[w]^protos[HDC_WORDS+w];
            while (bits) { ++gap; bits &= bits-1; }
        }
        std::printf("Class prototype separation: %d / 1024 bits\n",gap);
        PredictionComparison comparisons[4]{};
        for (unsigned split=0;split<4;split++) {
            Metrics gpu{},baseline{};
            for (uint32_t i=0;i<n;i++) if (data[i].split==split) {
                const bool truth = data[i].label != 0;
                const bool hdc = pred[i] != 0;
                const bool rules = basic_baseline(data[i]);
                gpu.add(truth,hdc);
                baseline.add(truth,rules);
                comparisons[split].add(truth,hdc,rules);
            }
            std::printf("\n%s:\n", split_name(split));
            gpu.print("GPU HDC prototype");
            baseline.print("simple rule baseline");
        }
        std::printf("\nPaired prediction comparison (same records; 1 = violation):\n");
        for (unsigned split=0;split<4;split++) {
            std::printf("\n%s:\n", split_name(split));
            comparisons[split].print();
        }
        diagnostics::run(data,protos,protos+HDC_WORDS,pred,options.directory,&margin);
        std::printf("\nNote: these are synthetic sequences, not evidence of real-world agent safety.\n");
    } catch (const std::exception& ex) { std::fprintf(stderr,"Error: %s\n",ex.what());return 1; }
    return 0;
}

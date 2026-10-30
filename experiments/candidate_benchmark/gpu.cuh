#pragma once
#include "core.h"
#include <cuda_runtime.h>

namespace candidate {
inline void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> struct Buffer {
    T* p=nullptr; size_t n;
    explicit Buffer(size_t count):n(count) { if(n) check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T))); }
    ~Buffer() { if(p) cudaFree(p); }
    Buffer(const Buffer&)=delete;
    void upload(const T* host,size_t count) { require(count<=n,"upload bounds"); if(count) check(cudaMemcpy(p,host,count*sizeof(T),cudaMemcpyHostToDevice)); }
    void download(T* host,size_t count) { require(count<=n,"download bounds"); if(count) check(cudaMemcpy(host,p,count*sizeof(T),cudaMemcpyDeviceToHost)); }
};
__device__ inline uint32_t hash_word(uint32_t x) {
    x=(x^(x>>16))*0x7feb352du; x=(x^(x>>15))*0x846ca68bu; return x^(x>>16);
}
__global__ void encode_contexts(const Context* contexts,uint32_t* vectors,unsigned count) {
    const unsigned i=blockIdx.x,w=threadIdx.x; if(i>=count || w>=32) return;
    const auto& c=contexts[i]; uint32_t bound[WIDTH];
    for(unsigned j=0;j<c.count;++j) {
        const Event e=c.events[j]; unsigned source=(w+j*7)%32; uint32_t tokens[11],word=0;
        for(unsigned t=0;t<11;++t) {
            auto value=t==0 ? e.action:t==1 ? e.object:t==2 ? e.source:(e.flags>>(t-3))&1u;
            tokens[t]=hash_word((t+1)*0x9e3779b9u^value^source*0x85ebca6bu^0xa5f152bdu);
        }
        for(unsigned bit=0;bit<32;++bit) { unsigned votes=0; for(auto token:tokens) votes+=(token>>bit)&1u;
            if(votes>=6) word|=1u<<bit; }
        if(j) word=(word<<j)|(word>>(32-j));
        bound[j]=word^hash_word(0x71a5b39du^j*0x9e3779b9u^w*0x85ebca6bu);
    }
    uint32_t word=0,tie=hash_word(0xe17a1465u^w*0xc2b2ae35u);
    if(c.count) for(unsigned bit=0;bit<32;++bit) { unsigned votes=0; for(unsigned j=0;j<c.count;++j) votes+=(bound[j]>>bit)&1u;
        if(2*votes>c.count || (2*votes==c.count && ((tie>>bit)&1u))) word|=1u<<bit; }
    vectors[i*32+w]=word;
}
// One warp per query context, four warps per candidate. Exactly w8_best:
// minimum Hamming distance among 4x8 contexts, scaled by 1024.
__global__ void compare_contexts(const uint32_t* queries,const uint32_t* gallery,const uint8_t* valid_g,
        const uint8_t* valid_q,const uint8_t* mask,uint32_t* scores,unsigned gcount) {
    unsigned g=blockIdx.x,q=blockIdx.y,lane=threadIdx.x%32,slot=threadIdx.x/32;
    if(g>=gcount) return;
    __shared__ unsigned best[4]; unsigned minimum=1024;
    if(valid_g[g] && valid_q[q] && (!mask || mask[q*gcount+g])) {
        auto x=queries[(q*QCONTEXTS+slot)*32+lane];
        for(unsigned s=0;s<GCONTEXTS;++s) {
            unsigned d=__popc(x^gallery[(g*GCONTEXTS+s)*32+lane]);
            for(unsigned delta=16;delta;delta/=2) d+=__shfl_down_sync(0xffffffffu,d,delta);
            if(!lane) minimum=min(minimum,d);
        }
    }
    if(!lane) best[slot]=minimum; __syncthreads();
    if(!threadIdx.x) {
        unsigned v=min(min(best[0],best[1]),min(best[2],best[3]))*1024;
        scores[q*gcount+g]=mask && !mask[q*gcount+g] ? NONE:v;
    }
}
__global__ void retrieve(const uint32_t* scores,unsigned* ranks,unsigned gcount,unsigned k) {
    unsigned q=blockIdx.x,t=threadIdx.x; __shared__ unsigned chosen[MAXK]; __shared__ unsigned long long keys[128];
    for(unsigned r=0;r<k;++r) {
        unsigned long long best=~0ull;
        for(unsigned g=t;g<gcount;g+=128) {
            bool used=false; for(unsigned j=0;j<r;++j) if(chosen[j]==g) used=true;
            if(!used && scores[q*gcount+g]!=NONE) best=min(best,(static_cast<unsigned long long>(scores[q*gcount+g])<<32)|g);
        }
        keys[t]=best; __syncthreads();
        for(unsigned d=64;d;d/=2) { if(t<d) keys[t]=min(keys[t],keys[t+d]); __syncthreads(); }
        if(!t) { chosen[r]=keys[0]==~0ull ? NONE:unsigned(keys[0]); ranks[q*k+r]=chosen[r]; } __syncthreads();
    }
}
struct SearchResult {
    std::vector<uint32_t> scores; std::vector<unsigned> ranks;
    double wall_ms=0,similarity_ms=0,topk_ms=0;
};
struct Gpu {
    unsigned gcount,qmax;
    Buffer<uint32_t> gallery,query,score;
    Buffer<unsigned> rank;
    Buffer<uint8_t> valid_g,valid_q,mask;
    Buffer<Context> raw_query;
    cudaEvent_t start,middle,stop;
    double build_wall_ms=0,build_kernel_ms=0;
    std::vector<Vector> gallery_host;
    Gpu(const std::vector<Context>& contexts,unsigned max_queries):gcount(unsigned(contexts.size()/GCONTEXTS)),qmax(max_queries),
        gallery(contexts.size()*32),query(max_queries*QCONTEXTS*32),score(size_t(max_queries)*gcount),rank(max_queries*MAXK),
        valid_g(gcount),valid_q(max_queries),mask(size_t(max_queries)*gcount),raw_query(max_queries*QCONTEXTS) {
        check(cudaEventCreate(&start)); check(cudaEventCreate(&middle)); check(cudaEventCreate(&stop));
        auto begin=Clock::now(); Buffer<Context> raw(contexts.size()); raw.upload(contexts.data(),contexts.size());
        std::vector<uint8_t> valid(gcount); for(unsigned g=0;g<gcount;++g) valid[g]=contexts[g*GCONTEXTS].count>0;
        valid_g.upload(valid.data(),valid.size()); check(cudaEventRecord(start));
        encode_contexts<<<unsigned(contexts.size()),32>>>(raw.p,gallery.p,unsigned(contexts.size()));
        check(cudaEventRecord(stop)); check(cudaEventSynchronize(stop)); float elapsed=0;
        check(cudaEventElapsedTime(&elapsed,start,stop)); build_kernel_ms=elapsed; build_wall_ms=ms(begin);
    }
    ~Gpu() { cudaEventDestroy(start); cudaEventDestroy(middle); cudaEventDestroy(stop); }
    uint64_t bytes() const { return (gallery.n+query.n+score.n)*4+rank.n*4+valid_g.n+valid_q.n+mask.n+raw_query.n*sizeof(Context); }
    void snapshot() { gallery_host.resize(gallery.n/32); gallery.download(reinterpret_cast<uint32_t*>(gallery_host.data()),gallery.n); }
    SearchResult search(const std::vector<Vector>& queries,const std::vector<uint8_t>& valid,const std::vector<uint8_t>& eligible={},const std::vector<Context>* raw=nullptr) {
        const unsigned nq=unsigned(queries.size()/QCONTEXTS); require(nq>0 && nq<=qmax,"query capacity");
        SearchResult out; out.scores.resize(size_t(nq)*gcount); out.ranks.resize(nq*MAXK);
        auto begin=Clock::now();
        if(raw) { raw_query.upload(raw->data(),raw->size()); encode_contexts<<<unsigned(raw->size()),32>>>(raw_query.p,query.p,unsigned(raw->size())); }
        else query.upload(reinterpret_cast<const uint32_t*>(queries.data()),queries.size()*32);
        valid_q.upload(valid.data(),valid.size()); if(!eligible.empty()) mask.upload(eligible.data(),eligible.size());
        check(cudaEventRecord(start));
        compare_contexts<<<dim3(gcount,nq),128>>>(query.p,gallery.p,valid_g.p,valid_q.p,eligible.empty() ? nullptr:mask.p,score.p,gcount);
        check(cudaEventRecord(middle)); retrieve<<<nq,128>>>(score.p,rank.p,gcount,MAXK);
        check(cudaEventRecord(stop)); score.download(out.scores.data(),out.scores.size()); rank.download(out.ranks.data(),out.ranks.size());
        out.wall_ms=ms(begin); float a=0,b=0;
        check(cudaEventElapsedTime(&a,start,middle)); check(cudaEventElapsedTime(&b,middle,stop)); out.similarity_ms=a; out.topk_ms=b;
        check(cudaGetLastError()); return out;
    }
};
inline std::vector<Vector> encode_queries(const std::vector<Context>& contexts) {
    Buffer<Context> raw(contexts.size()); Buffer<uint32_t> encoded(contexts.size()*32); raw.upload(contexts.data(),contexts.size());
    encode_contexts<<<unsigned(contexts.size()),32>>>(raw.p,encoded.p,unsigned(contexts.size()));
    std::vector<Vector> result(contexts.size()); encoded.download(reinterpret_cast<uint32_t*>(result.data()),contexts.size()*32);
    check(cudaGetLastError()); return result;
}
} // namespace candidate

#define main temporal_v1_embedded_entry_v3
#include "../temporal_v1/gpu.cu"
#undef main
#include "experiment.h"

namespace windowed {
__global__ void construct_windows(const Anchor* anchors,const uint32_t* anchor_begin,const uint32_t* anchor_count,
    const uint32_t* window_begin,const uint32_t* window_count,Window* windows,unsigned histories,unsigned width) {
    const unsigned id=blockIdx.x,w=threadIdx.x;
    unsigned h=0;
    while (h<histories && !(id>=window_begin[h] && id<window_begin[h]+window_count[h])) ++h;
    if (h==histories) return;
    const unsigned n=anchor_count[h],local=id-window_begin[h],stride=width/2;
    unsigned begin=0;
    if (n>=width) {
        const unsigned tail=n-width,insert_at=tail/stride+1;
        const bool inserted=tail%stride!=0;
        begin=inserted && local==insert_at ? tail : (local-unsigned(inserted && local>insert_at))*stride;
    }
    const unsigned count=min(width,n-begin),base=anchor_begin[h]+begin;
    uint32_t bound[32];
    for (unsigned i=0;i<count;++i) {
        const uint32_t content=anchors[base+i].hv[(w+(i*7u)%32)%32];
        const unsigned rotation=i%32;
        bound[i]=(rotation ? (content<<rotation)|(content>>(32-rotation)) : content) ^
            temporal::gpu_hash(0x71a5b39du ^ i*0x9e3779b9u ^ w*0x85ebca6bu);
    }
    const uint32_t tie=temporal::gpu_hash(0xe17a1465u ^ w*0xc2b2ae35u);
    uint32_t vector=0;
    for (unsigned bit=0;bit<32;++bit) {
        unsigned ones=0; for (unsigned i=0;i<count;++i) ones+=(bound[i]>>bit)&1u;
        if (2*ones>count || (2*ones==count && ((tie>>bit)&1u))) vector|=1u<<bit;
    }
    windows[id].hv[w]=vector;
    if (!w) {
        const auto& first=anchors[base]; const auto& last=anchors[base+count-1];
        windows[id].history_id=h; windows[id].width=width; windows[id].scope_begin=begin; windows[id].count=count;
        windows[id].first_step=first.event.step; windows[id].last_step=last.event.step;
        windows[id].first_anchor=first.ordinal; windows[id].last_anchor=last.ordinal;
        windows[id].object=first.event.object; windows[id].first_version=first.object_version;
        windows[id].last_version=last.object_version; windows[id].reserved=0;
    }
}
__global__ void compare_windows(const Window* windows,const uint32_t* queries,uint16_t* scores,unsigned count) {
    const unsigned c=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32,q=blockIdx.y;
    if (c>=count) return;
    if (queries[q]==MISSING) { if (!lane) scores[size_t(q)*count+c]=1024; return; }
    unsigned distance=__popc(windows[queries[q]].hv[lane]^windows[c].hv[lane]);
    for (unsigned step=16;step;step>>=1) distance+=__shfl_down_sync(0xffffffffu,distance,step);
    if (!lane) scores[size_t(q)*count+c]=uint16_t(distance);
}
__global__ void retrieve_windows(const Window* windows,const uint16_t* scores,const uint32_t* roles,
    uint32_t* output,unsigned count) {
    __shared__ unsigned long long reduce[128];
    __shared__ uint32_t chosen[WINDOW_TOPK];
    const unsigned q=blockIdx.x,t=threadIdx.x;
    for (unsigned k=0;k<WINDOW_TOPK;++k) {
        unsigned long long best=~0ull;
        for (unsigned c=t;c<count;c+=128) if (!roles[windows[c].history_id]) {
            bool used=false; for (unsigned prior=0;prior<k;++prior) used|=chosen[prior]==c;
            const unsigned long long key=(static_cast<unsigned long long>(scores[size_t(q)*count+c])<<32)|c;
            if (!used && key<best) best=key;
        }
        reduce[t]=best; __syncthreads();
        for (unsigned step=64;step;step>>=1) { if (t<step) reduce[t]=min(reduce[t],reduce[t+step]); __syncthreads(); }
        if (!t) { chosen[k]=reduce[0]==~0ull ? MISSING : uint32_t(reduce[0]); output[q*WINDOW_TOPK+k]=chosen[k]; }
        __syncthreads();
    }
}
__global__ void aggregate_histories(const uint16_t* distances,const uint32_t* support,const uint32_t* queries,
    uint32_t* output,unsigned windows,unsigned histories,unsigned groups) {
    const unsigned row=blockIdx.x*128+threadIdx.x;
    if (row>=AGGREGATIONS*groups*histories) return;
    const unsigned candidate=row%histories,group=(row/histories)%groups,strategy=row/(histories*groups);
    unsigned d[QUERY_SLOTS][SUPPORT_SLOTS];
    for (unsigned q=0;q<QUERY_SLOTS;++q) for (unsigned s=0;s<SUPPORT_SLOTS;++s) {
        const unsigned c=support[candidate*SUPPORT_SLOTS+s];
        d[q][s]=queries[group*QUERY_SLOTS+q]==MISSING || c==MISSING ? 1024 : distances[size_t(group*QUERY_SLOTS+q)*windows+c];
    }
    if (!strategy) {
        unsigned best=1024; for (unsigned q=0;q<QUERY_SLOTS;++q) for (unsigned s=0;s<SUPPORT_SLOTS;++s) best=min(best,d[q][s]);
        output[row]=best*1024; return;
    }
    if (strategy==1) {
        unsigned sum=0;
        for (unsigned q=0;q<QUERY_SLOTS;++q) {
            unsigned first=2048,second=2048;
            for (unsigned s=0;s<SUPPORT_SLOTS;++s) {
                const auto value=d[q][s];
                if (value<first) { second=first; first=value; } else if (value<second) second=value;
            }
            sum+=first+second;
        }
        output[row]=sum*128; return;
    }
    // Independently enumerate all increasing middle-slot pairs. The first and
    // last query contexts retain their extent; CPU uses dynamic programming.
    unsigned best=INF;
    for (unsigned a=1;a<7;++a) for (unsigned b=a+1;b<7;++b)
        best=min(best,d[0][0]+d[1][a]+d[2][b]+d[3][7]);
    output[row]=best*256;
}
__global__ void retrieve_histories(const uint32_t* scores,const uint32_t* roles,uint32_t* output,unsigned histories,unsigned groups) {
    __shared__ unsigned long long reduce[128];
    // At most 256 histories are admitted by the unchanged reader/driver contract.
    __shared__ uint32_t chosen[256];
    const unsigned row=blockIdx.x,t=threadIdx.x;
    for (unsigned k=0;k<histories;++k) {
        unsigned long long best=~0ull;
        for (unsigned c=t;c<histories;c+=128) if (!roles[c]) {
            bool used=false; for (unsigned p=0;p<k;++p) used|=chosen[p]==c;
            const unsigned long long key=(static_cast<unsigned long long>(scores[row*histories+c])<<32)|c;
            if (!used && key<best) best=key;
        }
        reduce[t]=best; __syncthreads();
        for (unsigned step=64;step;step>>=1) { if (t<step) reduce[t]=min(reduce[t],reduce[t+step]); __syncthreads(); }
        if (!t) { chosen[k]=reduce[0]==~0ull ? MISSING : uint32_t(reduce[0]); output[row*histories+k]=chosen[k]; }
        __syncthreads();
    }
}
struct GpuEngine {
    unsigned repeats=15; uint64_t peak_bytes=0;
    temporal::GpuEngine timer;
    std::string description="Temporal Memory v3 CUDA";
    const char* name() const { return description.c_str(); }
    Matrix evaluate(const std::vector<Input>& inputs,const std::vector<uint32_t>& roles,unsigned width) {
        timer.repeats=repeats;
        auto expected=reference_plan(inputs,roles,width);
        const unsigned n=unsigned(inputs.size()),nw=unsigned(expected.windows.size()),nq=unsigned(expected.queries.size()),groups=unsigned(expected.query_histories.size());
        std::vector<Anchor> flat; std::vector<uint32_t> begin,count;
        for (const auto& input:inputs) { begin.push_back(uint32_t(flat.size())); count.push_back(uint32_t(input.scoped.size())); flat.insert(flat.end(),input.scoped.begin(),input.scoped.end()); }
        temporal::Device<Anchor> anchors(flat.size()); anchors.upload(flat.data());
        temporal::Device<uint32_t> ab(n),ac(n),wb(n),wc(n),device_roles(n),support(expected.support.size()),queries(expected.queries.size());
        ab.upload(begin.data()); ac.upload(count.data()); wb.upload(expected.begins.data()); wc.upload(expected.counts.data());
        device_roles.upload(roles.data()); support.upload(expected.support.data()); queries.upload(expected.queries.data());
        temporal::Device<Window> windows(nw);
        temporal::Device<uint16_t> distances(size_t(nq)*nw);
        temporal::Device<uint32_t> raw_ranks(size_t(nq)*WINDOW_TOPK),scores(size_t(AGGREGATIONS)*groups*n),ranks(size_t(AGGREGATIONS)*groups*n);
        Matrix m=expected; m.timings.clear(); peak_bytes=allocation_bytes(inputs,m);
        if (nw) m.timings.push_back(timer.timed("construct_windows",0,nw,[&] {
            construct_windows<<<nw,32>>>(anchors.data,ab.data,ac.data,wb.data,wc.data,windows.data,n,width);
        }));
        windows.download(m.windows.data());
        if (nw) require(!std::memcmp(m.windows.data(),expected.windows.data(),nw*sizeof(Window)),"GPU window metadata/vector mismatch");
        auto compare=[&] { if (nq && nw) compare_windows<<<dim3((nw+3)/4,nq),128>>>(windows.data,queries.data,distances.data,nw); };
        auto raw_topk=[&] { if (nq) retrieve_windows<<<nq,128>>>(windows.data,distances.data,device_roles.data,raw_ranks.data,nw); };
        auto aggregate=[&] { if (groups) aggregate_histories<<<(AGGREGATIONS*groups*n+127)/128,128>>>(distances.data,support.data,queries.data,scores.data,nw,n,groups); };
        auto topk=[&] { if (groups) retrieve_histories<<<AGGREGATIONS*groups,128>>>(scores.data,device_roles.data,ranks.data,n,groups); };
        // Single batch query wall time includes launches and result transfers,
        // but excludes independent CPU verification and the repeated timing runs.
        const auto start=std::chrono::steady_clock::now();
        compare(); temporal::cuda_check(cudaGetLastError()); raw_topk(); temporal::cuda_check(cudaGetLastError());
        aggregate(); temporal::cuda_check(cudaGetLastError()); topk(); temporal::cuda_check(cudaGetLastError());
        m.distances.resize(size_t(nq)*nw); m.window_ranks.resize(size_t(nq)*WINDOW_TOPK);
        m.history_scores.resize(size_t(AGGREGATIONS)*groups*n); m.history_ranks.resize(size_t(AGGREGATIONS)*groups*n);
        distances.download(m.distances.data()); raw_ranks.download(m.window_ranks.data());
        scores.download(m.history_scores.data()); ranks.download(m.history_ranks.data());
        const auto wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        m.timings.push_back({"query_batch_wall",0,groups,1,wall,wall,wall});
        if (nq && nw) m.timings.push_back(timer.timed("compare_windows",0,nw,compare));
        if (nq) m.timings.push_back(timer.timed("window_topk",0,nw,raw_topk));
        if (groups) {
            m.timings.push_back(timer.timed("aggregate_histories",0,groups*n,aggregate));
            m.timings.push_back(timer.timed("history_topk",0,groups*n,topk));
        }
        reference_retrieve(expected,roles);
        require(m.distances==expected.distances,"GPU similarity matrix mismatch");
        require(m.window_ranks==expected.window_ranks,"GPU parallel window top-K mismatch");
        require(m.history_scores==expected.history_scores && m.history_ranks==expected.history_ranks,"GPU aggregation/history ranking mismatch");
        return m;
    }
};
}
int main(int argc,char** argv) {
    try {
        cudaDeviceProp device{}; temporal::cuda_check(cudaGetDeviceProperties(&device,0));
        windowed::GpuEngine engine; engine.description=std::string("Temporal Memory v3 CUDA | GPU: ")+device.name;
        return windowed::run(argc,argv,engine);
    } catch (const std::exception& e) { std::cerr<<"V3 CUDA error: "<<e.what()<<'\n'; return 1; }
}

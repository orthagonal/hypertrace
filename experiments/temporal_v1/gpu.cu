#include "experiment.h"
#include <cuda_runtime.h>
#include <memory>

namespace temporal {
inline void cuda_check(cudaError_t status) {
    if (status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> struct Device {
    T* data=nullptr;
    size_t size=0;
    explicit Device(size_t count):size(count) {
        if (count) cuda_check(cudaMalloc(reinterpret_cast<void**>(&data),count*sizeof(T)));
    }
    Device(const Device&)=delete;
    Device& operator=(const Device&)=delete;
    ~Device() { if (data) cudaFree(data); }
    void upload(const T* source) { if (size) cuda_check(cudaMemcpy(data,source,size*sizeof(T),cudaMemcpyHostToDevice)); }
    void download(T* destination) const { if (size) cuda_check(cudaMemcpy(destination,data,size*sizeof(T),cudaMemcpyDeviceToHost)); }
};
__device__ uint32_t gpu_hash(uint32_t x) {
    x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; return x^(x>>16);
}
__global__ void encode_history(const Event* events,Leaf* leaves,uint32_t count) {
    const uint32_t i=blockIdx.x, w=threadIdx.x;
    if (i>=count || w>=WORDS) return;
    const Event e=events[i];
    if (!w) leaves[i].event=e;
    uint32_t tokens[11];
    for (unsigned t=0;t<11;++t) {
        const uint32_t value=t==0 ? e.action : t==1 ? e.object : t==2 ? e.source : (e.flags>>(t-3))&1u;
        tokens[t]=gpu_hash((t+1)*0x9e3779b9u ^ value ^ w*0x85ebca6bu ^ 0xa5f152bdu);
    }
    uint32_t hv=0;
    for (unsigned bit=0;bit<32;++bit) {
        unsigned votes=0;
        for (auto token:tokens) votes+=(token>>bit)&1u;
        if (votes>=6) hv|=1u<<bit;
    }
    leaves[i].hv[w]=hv;
}
__global__ void build_temporal_level(const Leaf* leaves,const Node* children,Node* parents,
                                     uint32_t count,uint32_t level) {
    const uint32_t p=blockIdx.x,w=threadIdx.x,begin=p*32;
    if (begin>=count || w>=WORDS) return;
    const unsigned n=min(32u,count-begin);
    uint32_t bound[32];
    for (unsigned c=0;c<n;++c) {
        const unsigned source_word=(w+(c*7+level)%32)%32;
        const uint32_t raw=leaves ? leaves[begin+c].hv[source_word] : children[begin+c].hv[source_word];
        const uint32_t rotated=c ? (raw<<c)|(raw>>(32-c)) : raw;
        bound[c]=rotated ^ gpu_hash(0x51ed270bu ^ level*0x9e3779b9u ^ c*0x85ebca6bu ^ w*0xc2b2ae35u);
    }
    const uint32_t tie=gpu_hash(0xd1b54a35u ^ level*0x94d049bbu ^ w*0x369dea0fu);
    uint32_t hv=0;
    for (unsigned bit=0;bit<32;++bit) {
        unsigned votes=0;
        for (unsigned c=0;c<n;++c) votes+=(bound[c]>>bit)&1u;
        if (2*votes>n || (2*votes==n && ((tie>>bit)&1u))) hv|=1u<<bit;
    }
    parents[p].hv[w]=hv;
    if (!w) {
        parents[p].start_step=leaves ? leaves[begin].event.step : children[begin].start_step;
        parents[p].child_index=begin; parents[p].child_count=n;
        uint32_t properties=0;
        for (unsigned property=0;property<PROPERTIES;++property) {
            uint32_t mask=0;
            for (unsigned c=0;c<n;++c) {
                const uint32_t props=leaves ? leaves[begin+c].event.flags : children[begin+c].properties;
                if (props&(1u<<property)) mask|=1u<<c;
            }
            parents[p].masks[property]=mask;
            if (mask) properties|=1u<<property;
        }
        parents[p].properties=properties;
    }
}
__global__ void compare_query(const uint32_t* memory,unsigned stride,unsigned offset,
                               const uint32_t* query,uint16_t* distances,uint32_t count) {
    const unsigned i=blockIdx.x,lane=threadIdx.x;
    if (i>=count || lane>=32) return;
    unsigned distance=__popc(memory[size_t(i)*stride+offset+lane]^query[lane]);
    for (unsigned delta=16;delta;delta>>=1) distance+=__shfl_down_sync(0xffffffffu,distance,delta);
    if (!lane) distances[i]=uint16_t(distance);
}
// Exact navigation: these kernels never access an HDC vector or a similarity score.
__global__ void descend_masks(const Node* parents,const uint8_t* active,uint8_t* children,
                               unsigned property,uint32_t count) {
    const unsigned p=blockIdx.x,c=threadIdx.x;
    if (p>=count || c>=parents[p].child_count) return;
    children[parents[p].child_index+c]=uint8_t(active[p] && ((parents[p].masks[property]>>c)&1u));
}
__global__ void exact_leaf_filter(const Leaf* leaves,const uint8_t* active,uint8_t* result,
                                   unsigned property,uint32_t count) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<count) result[i]=uint8_t(active[i] && (leaves[i].event.flags&(1u<<property)));
}
// Correctness challenge after similarity checks: exact navigation must work even
// with arbitrary HDC contents. Exact fields/masks and archived host vectors stay intact.
__global__ void poison_vectors(uint32_t* memory,unsigned stride,unsigned offset,uint32_t count) {
    const unsigned i=blockIdx.x,lane=threadIdx.x;
    if (i<count && lane<32) memory[size_t(i)*stride+offset+lane]=0xdeadbeefu ^ lane;
}
struct Events {
    cudaEvent_t start{},stop{};
    Events() { cuda_check(cudaEventCreate(&start)); cuda_check(cudaEventCreate(&stop)); }
    ~Events() { cudaEventDestroy(start); cudaEventDestroy(stop); }
};
struct GpuEngine {
    unsigned repeats=15;
    std::string description="Temporal Memory v1 CUDA";
    const char* name() const { return description.c_str(); }
    template<class Launch> Timing timed(const std::string& name,unsigned level,unsigned count,Launch launch) {
        for (unsigned i=0;i<2;++i) { launch(); cuda_check(cudaGetLastError()); }
        cuda_check(cudaDeviceSynchronize());
        Events events;
        std::vector<float> samples;
        for (unsigned i=0;i<repeats;++i) {
            cuda_check(cudaEventRecord(events.start)); launch(); cuda_check(cudaGetLastError());
            cuda_check(cudaEventRecord(events.stop)); cuda_check(cudaEventSynchronize(events.stop));
            float ms=0; cuda_check(cudaEventElapsedTime(&ms,events.start,events.stop)); samples.push_back(ms);
        }
        std::sort(samples.begin(),samples.end());
        return {name,level,count,repeats,samples[samples.size()/2],samples.front(),samples.back()};
    }
    Evaluation evaluate(const std::vector<Event>& events,bool full) {
        Evaluation result;
        const unsigned count=unsigned(events.size());
        Device<Event> input(count); input.upload(events.data());
        Device<Leaf> leaves(count);
        std::vector<std::unique_ptr<Device<Node>>> levels;
        if (count) result.timings.push_back(timed("encode_history",0,count,[&] {
            encode_history<<<count,32>>>(input.data,leaves.data,count);
        }));
        unsigned children=count;
        while (children>1) {
            const unsigned parents=(children+31)/32,level=unsigned(levels.size()+1);
            auto next=std::make_unique<Device<Node>>(parents);
            result.timings.push_back(timed("build_temporal_level",level,parents,[&] {
                build_temporal_level<<<parents,32>>>(level==1 ? leaves.data : nullptr,
                    level==1 ? nullptr : levels.back()->data,next->data,children,level);
            }));
            levels.push_back(std::move(next)); children=parents;
        }
        result.tree.leaves.resize(count); leaves.download(result.tree.leaves.data());
        for (const auto& level:levels) {
            std::vector<Node> nodes(level->size); level->download(nodes.data());
            result.tree.levels.push_back(std::move(nodes));
        }
        const auto reference=reference_build(events);
        check_equal(result.tree,reference);
        if (!full) return result;
        result.query=query_vector(events);
        Device<uint32_t> query(WORDS); query.upload(result.query.data());
        for (unsigned l=0;l<=levels.size();++l) {
            const unsigned n=l ? unsigned(levels[l-1]->size) : count;
            Device<uint16_t> distances(n);
            if (n) result.timings.push_back(timed("compare_query",l,n,[&] {
                const uint32_t* base=l ? reinterpret_cast<uint32_t*>(levels[l-1]->data) : reinterpret_cast<uint32_t*>(leaves.data);
                compare_query<<<n,32>>>(base,l ? sizeof(Node)/4 : sizeof(Leaf)/4,l ? 0 : 8,query.data,distances.data,n);
            }));
            std::vector<uint16_t> host(n); distances.download(host.data());
            for (unsigned i=0;i<n;++i)
                require(host[i]==reference_distance(result.query.data(),level_vector(reference,l,i)),"GPU query Hamming distance mismatch");
            result.distances.push_back(std::move(host));
        }
        if (count) {
            poison_vectors<<<count,32>>>(reinterpret_cast<uint32_t*>(leaves.data),sizeof(Leaf)/4,8,count);
            cuda_check(cudaGetLastError());
            for (const auto& level:levels) {
                poison_vectors<<<unsigned(level->size),32>>>(reinterpret_cast<uint32_t*>(level->data),sizeof(Node)/4,0,unsigned(level->size));
                cuda_check(cudaGetLastError());
            }
            cuda_check(cudaDeviceSynchronize());
        }
        for (unsigned p=0;p<PROPERTIES;++p) {
            auto& found=result.searches[p];
            if (!count) continue;
            std::vector<std::unique_ptr<Device<uint8_t>>> active;
            active.push_back(std::make_unique<Device<uint8_t>>(count));
            for (const auto& level:levels) active.push_back(std::make_unique<Device<uint8_t>>(level->size));
            cuda_check(cudaMemset(active.back()->data,1,active.back()->size));
            for (unsigned l=unsigned(levels.size());l>0;--l) {
                std::vector<uint8_t> host_active(active[l]->size); active[l]->download(host_active.data());
                for (unsigned i=0;i<host_active.size();++i)
                    if (host_active[i] && result.tree.levels[l-1][i].masks[p]) found.regions.push_back({l,i});
                result.timings.push_back(timed(std::string("descend_masks_")+property_name(p),l,unsigned(levels[l-1]->size),[&] {
                    descend_masks<<<unsigned(levels[l-1]->size),32>>>(levels[l-1]->data,active[l]->data,active[l-1]->data,p,unsigned(levels[l-1]->size));
                }));
            }
            Device<uint8_t> matches(count);
            result.timings.push_back(timed(std::string("exact_leaf_filter_")+property_name(p),0,count,[&] {
                exact_leaf_filter<<<(count+255)/256,256>>>(leaves.data,active[0]->data,matches.data,p,count);
            }));
            std::vector<uint8_t> flags(count); matches.download(flags.data());
            for (unsigned i=0;i<count;++i) if (flags[i]) found.events.push_back(i);
            std::sort(found.regions.begin(),found.regions.end());
            auto expected=exact_search(reference,p);
            std::sort(expected.regions.begin(),expected.regions.end());
            require(found.events==expected.events && found.regions==expected.regions,"GPU exact navigation mismatch");
        }
        return result;
    }
};
} // namespace temporal
int main(int argc,char** argv) {
    try {
        cudaDeviceProp device{}; temporal::cuda_check(cudaGetDeviceProperties(&device,0));
        std::cout<<"Temporal GPU: "<<device.name<<'\n';
        temporal::GpuEngine engine;
        engine.description=std::string("Temporal Memory v1 CUDA | GPU: ")+device.name+
            " | all nodes verified against independent CPU";
        return temporal::run_experiment(argc,argv,engine);
    } catch (const std::exception& e) { std::cerr<<"Temporal CUDA error: "<<e.what()<<'\n'; return 1; }
}

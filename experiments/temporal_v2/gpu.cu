// Reuse the unchanged v1 GPU implementation, including its independent-node
// verification. Renaming only its entry point keeps v1 files byte-identical.
#define main temporal_v1_embedded_entry
#include "../temporal_v1/gpu.cu"
#undef main
#include "experiment.h"
#include <set>

namespace sparse {
__device__ bool gpu_write(unsigned a) {
    return a==1 || a==2 || a==3 || a==4 || a==8 || a==15 || a==16;
}
__global__ void mark_anchors(const Event* events,uint32_t* selected,unsigned n) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    const auto e=events[i];
    const bool read=(e.action==1 || e.action==2 || e.action==8 || (e.flags&1u)) && (e.flags&16u);
    selected[i]=uint32_t(read || e.action==3 || e.action==4 || e.action==16 || (e.flags&8u) ||
        e.action==5 || e.action==13 || e.action==14 || e.action==11 || e.action==12 || (e.flags&4u) || (e.flags&2u));
}
__global__ void scan_blocks(const uint32_t* input,uint32_t* prefix,uint32_t* sums,unsigned n) {
    __shared__ uint32_t values[256];
    const unsigned t=threadIdx.x,i=blockIdx.x*256+t;
    const uint32_t x=i<n ? input[i] : 0;
    values[t]=x; __syncthreads();
    for (unsigned offset=1;offset<256;offset<<=1) {
        const uint32_t add=t>=offset ? values[t-offset] : 0;
        __syncthreads(); values[t]+=add; __syncthreads();
    }
    if (i<n) prefix[i]=values[t]-x;
    if (t==255) sums[blockIdx.x]=values[t];
}
__global__ void add_scan_blocks(const uint32_t* local,uint32_t* prefix,const uint32_t* blocks,unsigned n) {
    const unsigned i=blockIdx.x*256+threadIdx.x;
    if (i<n) prefix[i]=local[i]+blocks[blockIdx.x];
}
__global__ void compute_versions(const Event* events,Version* versions,const uint32_t* objects,unsigned n) {
    // One block per observed object, independent of IDs' numeric range. Each
    // chunk performs a prefix MAX of writer indices; no quadratic backward search.
    __shared__ uint32_t writers[128],carry;
    const unsigned t=threadIdx.x,object=objects[blockIdx.x];
    if (!t) carry=0; __syncthreads();
    for (unsigned begin=0;begin<n;begin+=128) {
        const unsigned i=begin+t;
        Event e{}; if (i<n) e=events[i];
        writers[t]=i<n && e.object==object && gpu_write(e.action) ? i+1 : 0;
        __syncthreads();
        for (unsigned offset=1;offset<128;offset<<=1) {
            const uint32_t prior=t>=offset ? writers[t-offset] : 0;
            __syncthreads(); writers[t]=max(writers[t],prior); __syncthreads();
        }
        const uint32_t current=max(carry,writers[t]),before=max(carry,t ? writers[t-1] : 0);
        if (i<n && e.object==object) versions[i].object=current ? events[current-1].step : UNKNOWN;
        if (i<n && e.source==object) versions[i].source=before ? events[before-1].step : UNKNOWN;
        __syncthreads(); if (!t) carry=max(carry,writers[127]); __syncthreads();
    }
}
__global__ void encode_anchors(const Event* events,const Version* versions,const uint32_t* selected,
                                const uint32_t* prefix,Anchor* anchors,unsigned n) {
    const unsigned i=blockIdx.x,w=threadIdx.x;
    if (i>=n || !selected[i]) return;
    const Event e=events[i]; const unsigned out=prefix[i];
    if (!w) {
        anchors[out].event=e; anchors[out].event_index=i; anchors[out].ordinal=out;
        anchors[out].object_version=versions[i].object; anchors[out].source_version=versions[i].source;
    }
    uint32_t tokens[11];
    for (unsigned tag=1;tag<=11;++tag) {
        const uint32_t value=tag==1 ? e.action : tag==2 ? e.object : tag==3 ? e.source : (e.flags>>(tag-4))&1u;
        tokens[tag-1]=temporal::gpu_hash(tag*0x9e3779b9u ^ value ^ w*0x85ebca6bu ^ 0xa5f152bdu);
    }
    uint32_t word=0;
    for (unsigned bit=0;bit<32;++bit) {
        unsigned votes=0; for (auto token:tokens) votes+=(token>>bit)&1u;
        if (votes>=6) word|=1u<<bit;
    }
    anchors[out].hv[w]=word;
}
__global__ void mark_scope(const Anchor* anchors,uint32_t* selected,unsigned count,uint32_t target) {
    const unsigned i=blockIdx.x*256+threadIdx.x;
    if (i<count) selected[i]=uint32_t(anchors[i].event.object==target);
}
__global__ void accumulate_votes(const Anchor* anchors,const uint32_t* scoped_prefix,uint32_t* votes,
                                  unsigned count,unsigned target,unsigned tiles) {
    const unsigned tile=blockIdx.x,method=blockIdx.y/4,bit=(blockIdx.y%4)*256+threadIdx.x;
    const unsigned word=bit/32,shift=bit%32,begin=tile*256,end=min(count,begin+256);
    unsigned ones=0;
    for (unsigned i=begin;i<end;++i) if (!method || anchors[i].event.object==target) {
        const uint32_t position=method ? scoped_prefix[i] : i;
        const uint32_t content=anchors[i].hv[(word+(position*7u)%32)%32];
        const uint32_t rotation=position%32;
        const uint32_t bound=(rotation ? (content<<rotation)|(content>>(32-rotation)) : content) ^
            temporal::gpu_hash(0x71a5b39du ^ position*0x9e3779b9u ^ word*0x85ebca6bu);
        ones+=(bound>>shift)&1u;
    }
    votes[(method*tiles+tile)*1024+bit]=ones;
}
__global__ void finish_fingerprints(const uint32_t* votes,uint32_t* output,unsigned tiles,unsigned all_count,unsigned scoped_count) {
    const unsigned method=blockIdx.x,w=threadIdx.x,count=method ? scoped_count : all_count;
    uint32_t hv=0;
    if (count) for (unsigned b=0;b<32;++b) {
        unsigned ones=0;
        for (unsigned t=0;t<tiles;++t) ones+=votes[(method*tiles+t)*1024+w*32+b];
        const uint32_t tie=temporal::gpu_hash(0xe17a1465u ^ w*0xc2b2ae35u);
        if (2*ones>count || (2*ones==count && ((tie>>b)&1u))) hv|=1u<<b;
    }
    output[method*32+w]=hv;
}
__global__ void exact_pattern_key(const Event* events,const Version* versions,uint32_t* key,unsigned n,unsigned target) {
    if (threadIdx.x || blockIdx.x) return;
    uint32_t first=0,last=0,writer=UNKNOWN,approval=UNKNOWN,approved_version=UNKNOWN,stage=0,source=0,flags=0;
    for (unsigned i=0;i<n;++i) if (events[i].object==target) {
        const auto e=events[i];
        if (gpu_write(e.action)) { if (!first) first=e.action; last=e.action; source=e.source; writer=e.step; }
        if (e.action==11) { approval=e.step; approved_version=versions[i].object; }
        if (e.action==13 || e.action==14) { stage=e.action; flags=e.flags&(16u|64u|2u); }
    }
    key[0]=first; key[1]=last; key[2]=approval!=UNKNOWN && writer!=UNKNOWN ? uint32_t(approval>writer) : 2;
    key[3]=stage; key[4]=target; key[5]=source; key[6]=flags;
    key[7]=uint32_t(approval!=UNKNOWN && writer!=UNKNOWN && approved_version==writer);
}
__global__ void collection_distances(const uint32_t* vectors,const uint32_t* keys,uint16_t* output,unsigned n) {
    const unsigned c=blockIdx.x,q=blockIdx.y,method=blockIdx.z,lane=threadIdx.x;
    unsigned distance=method==3 ? (lane<8 ? uint32_t(keys[q*8+lane]!=keys[c*8+lane]) : 0) :
        __popc(vectors[(q*3+method)*32+lane]^vectors[(c*3+method)*32+lane]);
    for (unsigned offset=16;offset;offset>>=1) distance+=__shfl_down_sync(0xffffffffu,distance,offset);
    if (!lane) output[(method*n+q)*n+c]=uint16_t(method==3 ? (distance ? 1024 : 0) : distance);
}
__global__ void rank_collection(const uint16_t* distances,const uint32_t* roles,uint32_t* ranks,unsigned n) {
    const unsigned row=blockIdx.x*128+threadIdx.x;
    if (row>=METHODS*n) return;
    const unsigned method=row/n,q=row%n;
    for (unsigned k=0;k<TOPK;++k) ranks[row*TOPK+k]=UNKNOWN;
    if (!roles[q]) return;
    for (unsigned k=0;k<TOPK;++k) {
        uint32_t best=UNKNOWN; unsigned best_distance=0xffff;
        for (unsigned c=0;c<n;++c) if (!roles[c]) {
            bool used=false; for (unsigned j=0;j<k;++j) used|=ranks[row*TOPK+j]==c;
            const unsigned d=distances[(method*n+q)*n+c];
            if (!used && (d<best_distance || (d==best_distance && c<best))) { best=c; best_distance=d; }
        }
        ranks[row*TOPK+k]=best;
    }
}
struct GpuEngine {
    unsigned repeats=15;
    temporal::GpuEngine v1;
    std::string description="Temporal Memory v2 CUDA";
    const char* name() const { return description.c_str(); }
    void scan(temporal::Device<uint32_t>& input,temporal::Device<uint32_t>& prefix,unsigned n,
              std::vector<Timing>& times,unsigned depth=0) {
        if (!n) return;
        const unsigned blocks=(n+255)/256;
        temporal::Device<uint32_t> sums(blocks),offsets(blocks);
        times.push_back(v1.timed("sparse_prefix_scan",depth,n,[&] {
            scan_blocks<<<blocks,256>>>(input.data,prefix.data,sums.data,n);
        }));
        if (blocks>1) {
            scan(sums,offsets,blocks,times,depth+1);
            temporal::Device<uint32_t> local(n);
            temporal::cuda_check(cudaMemcpy(local.data,prefix.data,n*sizeof(uint32_t),cudaMemcpyDeviceToDevice));
            times.push_back(v1.timed("sparse_prefix_add",depth,n,[&] {
                add_scan_blocks<<<blocks,256>>>(local.data,prefix.data,offsets.data,n);
            }));
        }
    }
    unsigned count(const temporal::Device<uint32_t>& selected,const temporal::Device<uint32_t>& prefix) {
        if (!selected.size) return 0;
        uint32_t a=0,b=0;
        temporal::cuda_check(cudaMemcpy(&a,selected.data+selected.size-1,4,cudaMemcpyDeviceToHost));
        temporal::cuda_check(cudaMemcpy(&b,prefix.data+prefix.size-1,4,cudaMemcpyDeviceToHost));
        return a+b;
    }
    Analysis analyze(const History& h,uint32_t target) {
        v1.repeats=repeats;
        auto built=v1.evaluate(h.events,false);
        Analysis result;
        result.tree=std::move(built.tree); result.fingerprints[0]=root(result.tree);
        result.timings=std::move(built.timings);
        for (auto& t:result.timings) t.kernel="v1_"+t.kernel;
        // Keep a full, unchanged v1 tree resident while constructing the extra index.
        temporal::Device<Leaf> full_leaves(result.tree.leaves.size()); full_leaves.upload(result.tree.leaves.data());
        std::vector<std::unique_ptr<temporal::Device<Node>>> full_levels;
        for (const auto& level:result.tree.levels) {
            auto device=std::make_unique<temporal::Device<Node>>(level.size()); device->upload(level.data());
            full_levels.push_back(std::move(device));
        }
        const unsigned n=unsigned(h.events.size());
        temporal::Device<Event> events(n); events.upload(h.events.data());
        temporal::Device<uint32_t> selected(n),prefix(n);
        temporal::Device<Version> versions(n);
        if (n) temporal::cuda_check(cudaMemset(versions.data,0xff,n*sizeof(Version)));
        std::set<uint32_t> unique;
        for (const auto& e:h.events) { if (e.object) unique.insert(e.object); if (e.source) unique.insert(e.source); }
        const std::vector<uint32_t> object_ids(unique.begin(),unique.end());
        temporal::Device<uint32_t> objects(object_ids.size()); objects.upload(object_ids.data());
        if (n) {
            result.timings.push_back(v1.timed("sparse_select",0,n,[&] { mark_anchors<<<(n+255)/256,256>>>(events.data,selected.data,n); }));
            scan(selected,prefix,n,result.timings);
            if (!object_ids.empty()) result.timings.push_back(v1.timed("sparse_versions",0,n,[&] {
                compute_versions<<<unsigned(object_ids.size()),128>>>(events.data,versions.data,objects.data,n);
            }));
        }
        const unsigned m=count(selected,prefix);
        temporal::Device<Anchor> anchors(m);
        if (n) result.timings.push_back(v1.timed("sparse_encode",0,n,[&] {
            encode_anchors<<<n,32>>>(events.data,versions.data,selected.data,prefix.data,anchors.data,n);
        }));
        temporal::Device<uint32_t> scope(m),scope_prefix(m);
        if (m) {
            result.timings.push_back(v1.timed("sparse_scope_select",0,m,[&] { mark_scope<<<(m+255)/256,256>>>(anchors.data,scope.data,m,target); }));
            scan(scope,scope_prefix,m,result.timings);
        }
        result.scoped_count=count(scope,scope_prefix);
        const unsigned tiles=(m+255)/256;
        temporal::Device<uint32_t> votes(size_t(tiles)*2*1024),output(64),key(8);
        if (m) result.timings.push_back(v1.timed("sparse_accumulate",0,m,[&] {
            accumulate_votes<<<dim3(tiles,8),256>>>(anchors.data,scope_prefix.data,votes.data,m,target,tiles);
        }));
        result.timings.push_back(v1.timed("sparse_finish",0,m,[&] {
            finish_fingerprints<<<2,32>>>(votes.data,output.data,tiles,m,result.scoped_count);
        }));
        result.timings.push_back(v1.timed("exact_pattern_key",0,n,[&] { exact_pattern_key<<<1,1>>>(events.data,versions.data,key.data,n,target); }));
        std::array<uint32_t,64> fingerprints{}; output.download(fingerprints.data());
        std::copy_n(fingerprints.data(),32,result.fingerprints[1].begin());
        std::copy_n(fingerprints.data()+32,32,result.fingerprints[2].begin());
        key.download(result.pattern.data());
        result.anchors.resize(m); anchors.download(result.anchors.data());
        result.versions.resize(n); versions.download(result.versions.data());
        const auto expected=reference_analyze(h,target);
        require(expected.anchors.size()==m,"GPU anchor count mismatch");
        if (m) require(std::memcmp(result.anchors.data(),expected.anchors.data(),m*sizeof(Anchor))==0,"GPU anchor selection/order/versions/vectors mismatch");
        if (n) require(std::memcmp(result.versions.data(),expected.versions.data(),n*sizeof(Version))==0,"GPU version map mismatch");
        require(result.fingerprints[1]==expected.fingerprints[1] && result.fingerprints[2]==expected.fingerprints[2] &&
            result.pattern==expected.pattern && result.scoped_count==expected.scoped_count,"GPU fingerprint or exact pattern mismatch");
        std::vector<Leaf> check_leaves(result.tree.leaves.size()); full_leaves.download(check_leaves.data());
        if (n) require(std::memcmp(check_leaves.data(),result.tree.leaves.data(),n*sizeof(Leaf))==0,"sparse index changed v1 leaves");
        for (unsigned l=0;l<full_levels.size();++l) {
            std::vector<Node> check(result.tree.levels[l].size()); full_levels[l]->download(check.data());
            require(std::memcmp(check.data(),result.tree.levels[l].data(),check.size()*sizeof(Node))==0,"sparse index changed v1 nodes");
        }
        return result;
    }
    Scores score(const std::vector<Analysis>& analyses,const std::vector<uint32_t>& roles) {
        v1.repeats=repeats;
        const unsigned n=unsigned(analyses.size());
        std::vector<uint32_t> host_vectors,host_keys;
        for (const auto& a:analyses) {
            for (const auto& v:a.fingerprints) host_vectors.insert(host_vectors.end(),v.begin(),v.end());
            host_keys.insert(host_keys.end(),a.pattern.begin(),a.pattern.end());
        }
        temporal::Device<uint32_t> vectors(host_vectors.size()),keys(host_keys.size()),device_roles(n),ranks(METHODS*n*TOPK);
        temporal::Device<uint16_t> distances(METHODS*n*n);
        vectors.upload(host_vectors.data()); keys.upload(host_keys.data()); device_roles.upload(roles.data());
        Scores s;
        s.timings.push_back(v1.timed("collection_hamming_and_exact",0,n,[&] {
            collection_distances<<<dim3(n,n,METHODS),32>>>(vectors.data,keys.data,distances.data,n);
        }));
        s.timings.push_back(v1.timed("collection_topk",0,n,[&] {
            rank_collection<<<(METHODS*n+127)/128,128>>>(distances.data,device_roles.data,ranks.data,n);
        }));
        s.distances.resize(METHODS*n*n); s.ranks.resize(METHODS*n*TOPK);
        distances.download(s.distances.data()); ranks.download(s.ranks.data());
        const auto expected=reference_distances(analyses);
        require(s.distances==expected && s.ranks==reference_ranks(expected,roles),"GPU collection distances/top-K mismatch");
        return s;
    }
};
}
int main(int argc,char** argv) {
    try {
        cudaDeviceProp device{}; temporal::cuda_check(cudaGetDeviceProperties(&device,0));
        sparse::GpuEngine engine; engine.description=std::string("Temporal Memory v2 CUDA | GPU: ")+device.name;
        return sparse::run(argc,argv,engine);
    } catch (const std::exception& e) { std::cerr<<"Sparse CUDA error: "<<e.what()<<'\n'; return 1; }
}

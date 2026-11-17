#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
struct Edge { int a,b; double w; };
struct DSU {
    std::vector<int> p;
    DSU(int n):p(n) { std::iota(p.begin(),p.end(),0); }
    int find(int a) { while(p[a]!=a) { p[a]=p[p[a]]; a=p[a]; } return a; }
    void join(int a,int b) { a=find(a);b=find(b); if(a!=b)p[b]=a; }
};

// This build flag is emitted only by the header/compiler fingerprint check in
// tensor_harmonic.py. Other rebuilds retain the original allocator by default.
#if defined(R08_VERIFIED_ARENA) && R08_VERIFIED_ARENA == 1 && defined(__APPLE__) && defined(__arm64__) && _LIBCPP_VERSION == 170006 && _LIBCPP_ABI_VERSION == 1 && __clang_major__ == 15 && __clang_minor__ == 0 && __clang_patchlevel__ == 0
constexpr bool verified_arena = true;
#else
constexpr bool verified_arena = false;
#endif

struct AllocStats { size_t requests=0,system_allocations=0,peak_bytes=0; int fallbacks=0; };
thread_local AllocStats last_allocator_stats;

struct Arena {
    struct Block { Block* next; size_t capacity,used; };
    static constexpr size_t block_bytes=1024*1024, maximum_bytes=64*1024*1024;
    Block* head=nullptr;
    size_t total=0,cap;
    int64_t fail_after;
    AllocStats& stats;
    explicit Arena(AllocStats& s,size_t limit=maximum_bytes,int64_t fail=-1):cap(limit),fail_after(fail),stats(s) {}
    Arena(const Arena&)=delete;
    Arena& operator=(const Arena&)=delete;
    ~Arena() { while(head) { Block* next=head->next;std::free(head);head=next; } }
    void* allocate(size_t bytes,size_t alignment) {
        ++stats.requests;
        if(!alignment||(alignment&(alignment-1)))throw std::bad_alloc();
        const size_t max=std::numeric_limits<size_t>::max();
        if(bytes>max-(alignment-1)||bytes+alignment-1>max-sizeof(Block))throw std::bad_alloc();
        auto available=[&]() -> void* {
            if(!head)return nullptr;
            void* p=reinterpret_cast<unsigned char*>(head)+head->used;
            size_t space=head->capacity-head->used;
            if(!std::align(alignment,bytes,p,space))return nullptr;
            head->used=static_cast<unsigned char*>(p)-reinterpret_cast<unsigned char*>(head)+bytes;
            return p;
        };
        if(void* p=available())return p;
        const size_t capacity=std::max(block_bytes,sizeof(Block)+bytes+alignment-1);
        if(capacity>cap||total>cap-capacity)throw std::bad_alloc();
        if(fail_after>=0&&stats.system_allocations>=size_t(fail_after))throw std::bad_alloc();
        void* storage=std::malloc(capacity);if(!storage)throw std::bad_alloc();
        head=new(storage) Block{head,capacity,sizeof(Block)};
        total+=capacity;++stats.system_allocations;stats.peak_bytes=std::max(stats.peak_bytes,total);
        if(void* p=available())return p;
        throw std::bad_alloc();
    }
};

template<class T> struct ArenaAllocator {
    using value_type=T;
    Arena* arena;
    explicit ArenaAllocator(Arena* a=nullptr) noexcept:arena(a) {}
    template<class U> ArenaAllocator(const ArenaAllocator<U>& a) noexcept:arena(a.arena) {}
    template<class U> struct rebind { using other=ArenaAllocator<U>; };
    T* allocate(size_t n) {
        if(!arena||n>std::numeric_limits<size_t>::max()/sizeof(T))throw std::bad_alloc();
        return static_cast<T*>(arena->allocate(n*sizeof(T),alignof(T)));
    }
    void deallocate(T*,size_t) noexcept {}
    template<class U> bool operator==(const ArenaAllocator<U>& a) const noexcept { return arena==a.arena; }
    template<class U> bool operator!=(const ArenaAllocator<U>& a) const noexcept { return arena!=a.arena; }
};

#ifdef R08_TESTING
using TraceCallback=void(*)(const char*,const void*,size_t,void*);
struct Trace {
    TraceCallback callback=nullptr;void* user=nullptr;
    template<class T> void value(const char* name,const T& v) { if(callback)callback(name,&v,sizeof(T),user); }
    template<class T> void vector(const char* name,const std::vector<T>& v) { if(callback)callback(name,v.data(),v.size()*sizeof(T),user); }
};
thread_local Trace trace;
template<class T> struct CountingAllocator {
    using value_type=T;
    AllocStats* stats;
    size_t* current;
    explicit CountingAllocator(AllocStats* s=nullptr,size_t* c=nullptr):stats(s),current(c) {}
    template<class U> CountingAllocator(const CountingAllocator<U>& a):stats(a.stats),current(a.current) {}
    template<class U> struct rebind { using other=CountingAllocator<U>; };
    T* allocate(size_t n) {
        T* p=std::allocator<T>{}.allocate(n);++stats->requests;++stats->system_allocations;
        *current+=n*sizeof(T);stats->peak_bytes=std::max(stats->peak_bytes,*current);return p;
    }
    void deallocate(T* p,size_t n) { *current-=n*sizeof(T);std::allocator<T>{}.deallocate(p,n); }
    template<class U> bool operator==(const CountingAllocator<U>& a) const { return stats==a.stats; }
    template<class U> bool operator!=(const CountingAllocator<U>& a) const { return stats!=a.stats; }
};
#define TRACE_VALUE(v) trace.value(#v,v)
#define TRACE_VECTOR(v) trace.vector(#v,v)
#else
#define TRACE_VALUE(v) ((void)0)
#define TRACE_VECTOR(v) ((void)0)
#endif
}

// All arrays are current-frame inputs. Fixed endpoints enter the RHS, never
// the unknown connectivity graph. The hash combines all directed contributions.
template<class Allocator>
int solve_with_allocator(const uint8_t* image,const uint8_t* base,
    const uint8_t* unknown,const uint8_t* halo,const float* c,const float* q,
    const float* tx,const float* ty,int height,int width,uint8_t* output,double* stats,Allocator allocator) {
    const auto started=Clock::now(); const int size=height*width;
    std::copy(base,base+size*3,output);
    std::vector<int> index(size,-1),pixels;
    for(int p=0;p<size;p++) if(unknown[p]) { index[p]=pixels.size();pixels.push_back(p); }
    const int n=pixels.size();stats[0]=n; if(!n)return 0;
    std::unordered_map<uint64_t,double,std::hash<uint64_t>,std::equal_to<uint64_t>,Allocator> weights(0,std::hash<uint64_t>{},std::equal_to<uint64_t>{},allocator);
    weights.reserve(size_t(n)*12);
    auto add=[&](int p,int x,int y,double weight) {
      if(x<0||y<0||x>=width||y>=height||weight<=0)return;
      int z=y*width+x;if(p==z||(index[p]<0&&index[z]<0))return;
      const uint32_t a=std::min(p,z),b=std::max(p,z);
      weights[(uint64_t(a)<<32)|b]+=weight*.5;
    };
    for(int p=0;p<size;p++) if(halo[p]) {
      const int x=p%width,y=p/width;double axis=(1.-c[p]+.10)/4.;
      add(p,x+1,y,axis);add(p,x-1,y,axis);add(p,x,y+1,axis);add(p,x,y-1,axis);
      for(int sign:{-1,1}) {
        double px=x+sign*3.*tx[p],py=y+sign*3.*ty[p];
        int x0=std::floor(px),y0=std::floor(py);double dx=px-x0,dy=py-y0;
        const double s=c[p]/2.;
        add(p,x0,y0,s*(1-dx)*(1-dy));add(p,x0+1,y0,s*dx*(1-dy));
        add(p,x0,y0+1,s*(1-dx)*dy);add(p,x0+1,y0+1,s*dx*dy);
      }
    }
    std::vector<Edge> edges;edges.reserve(weights.size());DSU dsu(n);
    std::vector<double> degree(n),fixed(size_t(n)*3),lambda(n),diag(n),rhs(size_t(n)*3),u(size_t(n)*3);
    TRACE_VECTOR(index);TRACE_VECTOR(pixels);
#ifdef R08_TESTING
    if(trace.callback) {
        std::vector<uint64_t> map_keys;std::vector<double> map_weights;
        for(const auto& entry:weights){map_keys.push_back(entry.first);map_weights.push_back(entry.second);}
        TRACE_VECTOR(map_keys);TRACE_VECTOR(map_weights);
    }
#endif
    for(const auto& entry:weights) {
      int p=entry.first>>32,z=uint32_t(entry.first),a=index[p],b=index[z];double w=entry.second;
      if(a>=0)degree[a]+=w;if(b>=0)degree[b]+=w;
      if(a>=0&&b>=0) { edges.push_back({a,b,w});dsu.join(a,b); }
      else { if(a<0){std::swap(a,b);std::swap(p,z);}for(int ch=0;ch<3;ch++)fixed[3*a+ch]+=w*base[3*z+ch]; }
    }
    TRACE_VECTOR(degree);TRACE_VECTOR(fixed);TRACE_VECTOR(dsu.p);
    static_assert(sizeof(Edge)==2*sizeof(int)+sizeof(double),"Trace requires unpadded edges");
    TRACE_VECTOR(edges);
    stats[1]=weights.size();stats[2]=edges.size();weights.clear();weights.rehash(0);
    std::vector<int> groups(n),rootgroup(n,-1);int ng=0;
    for(int a=0;a<n;a++){int root=dsu.find(a);if(rootgroup[root]<0)rootgroup[root]=ng++;groups[a]=rootgroup[root];}
    std::vector<int> group_size(ng);for(int g:groups)group_size[g]++;
    stats[3]=ng;int limit=4+std::min(20,int(2000000/std::max<size_t>(edges.size(),1)));stats[4]=limit;
    TRACE_VECTOR(groups);TRACE_VECTOR(rootgroup);TRACE_VECTOR(group_size);TRACE_VALUE(limit);
    for(int a=0;a<n;a++) {
      double confidence=q[pixels[a]],ratio=(1-confidence)/std::max(confidence,.001);
      lambda[a]=degree[a]*(.05+ratio*ratio);diag[a]=degree[a]+lambda[a];
      if(!(diag[a]>0)||!std::isfinite(diag[a]))return 2;
      for(int ch=0;ch<3;ch++){int k=3*a+ch;u[k]=base[3*pixels[a]+ch];rhs[k]=lambda[a]*u[k]+fixed[k];}
    }
    stats[5]=seconds(started);const auto solving=Clock::now();
    TRACE_VECTOR(lambda);TRACE_VECTOR(diag);TRACE_VECTOR(rhs);TRACE_VECTOR(u);
    auto matvec=[&](const std::vector<double>& v,std::vector<double>& out) {
      for(int a=0;a<n;a++)for(int ch=0;ch<3;ch++)out[3*a+ch]=diag[a]*v[3*a+ch];
      for(const auto&e:edges)for(int ch=0;ch<3;ch++){out[3*e.a+ch]-=e.w*v[3*e.b+ch];out[3*e.b+ch]-=e.w*v[3*e.a+ch];}
    };
    std::vector<double> r(n*3),z(n*3),direction(n*3),ad(n*3),bu(n*3);
    std::vector<double> norm(ng*3),rz(ng*3),rr(ng*3),pap(ng*3),next(ng*3),alpha(ng*3),beta(ng*3);
    std::vector<int> iterations(ng*3);std::vector<uint8_t> live(ng*3,1);
    matvec(u,bu);double initial_energy=0;
    for(int a=0;a<n;a++)for(int ch=0;ch<3;ch++) {
      int k=3*a+ch,g=3*groups[a]+ch;initial_energy+=.5*u[k]*bu[k]-rhs[k]*u[k];
      if(group_size[groups[a]]==1){u[k]=rhs[k]/diag[a];bu[k]=rhs[k];}
      r[k]=rhs[k]-bu[k];z[k]=r[k]/diag[a];direction[k]=z[k];
      norm[g]+=rhs[k]*rhs[k];rz[g]+=r[k]*z[k];rr[g]+=r[k]*r[k];
    }
    for(int g=0;g<ng*3;g++)live[g]=std::sqrt(rr[g])/std::max(std::sqrt(norm[g]),1.)>1e-4;
    TRACE_VALUE(initial_energy);TRACE_VECTOR(bu);TRACE_VECTOR(r);TRACE_VECTOR(z);TRACE_VECTOR(direction);TRACE_VECTOR(norm);TRACE_VECTOR(rz);TRACE_VECTOR(rr);TRACE_VECTOR(live);
    for(int iter=0;iter<limit;iter++) {
      if(std::none_of(live.begin(),live.end(),[](uint8_t v){return v!=0;}))break;
      // Finished components have zero search direction and remain exact fixed results.
      for(int a=0;a<n;a++)for(int ch=0;ch<3;ch++)if(!live[3*groups[a]+ch])direction[3*a+ch]=0;
      matvec(direction,ad);std::fill(pap.begin(),pap.end(),0);std::fill(rr.begin(),rr.end(),0);std::fill(next.begin(),next.end(),0);
      for(int a=0;a<n;a++)for(int ch=0;ch<3;ch++)pap[3*groups[a]+ch]+=direction[3*a+ch]*ad[3*a+ch];
      for(int g=0;g<ng*3;g++)if(live[g]){if(!(pap[g]>0)||!std::isfinite(pap[g]))return 3;alpha[g]=rz[g]/pap[g];iterations[g]++;}
      for(int a=0;a<n;a++)for(int ch=0;ch<3;ch++) {
        int k=3*a+ch,g=3*groups[a]+ch;if(live[g]){u[k]+=alpha[g]*direction[k];r[k]-=alpha[g]*ad[k];}
        z[k]=r[k]/diag[a];rr[g]+=r[k]*r[k];next[g]+=r[k]*z[k];
      }
      for(int g=0;g<ng*3;g++)if(live[g]) {
        if(std::sqrt(rr[g])/std::max(std::sqrt(norm[g]),1.)<=1e-4)live[g]=0;
        else {beta[g]=next[g]/rz[g];rz[g]=next[g];}
      }
      for(int a=0;a<n;a++)for(int ch=0;ch<3;ch++){int k=3*a+ch,g=3*groups[a]+ch;direction[k]=live[g]?z[k]+beta[g]*direction[k]:0.;}
      TRACE_VALUE(iter);TRACE_VECTOR(u);TRACE_VECTOR(r);TRACE_VECTOR(z);TRACE_VECTOR(direction);TRACE_VECTOR(ad);TRACE_VECTOR(pap);TRACE_VECTOR(rr);TRACE_VECTOR(next);TRACE_VECTOR(alpha);TRACE_VECTOR(beta);TRACE_VECTOR(rz);TRACE_VECTOR(iterations);TRACE_VECTOR(live);
    }
    matvec(u,bu);double final_energy=0,maxres=0;std::fill(rr.begin(),rr.end(),0);
    for(int a=0;a<n;a++)for(int ch=0;ch<3;ch++) {
      int k=3*a+ch,g=3*groups[a]+ch;if(!std::isfinite(u[k]))return 4;
      final_energy+=.5*u[k]*bu[k]-rhs[k]*u[k];double residual=rhs[k]-bu[k];rr[g]+=residual*residual;
    }
    for(int g=0;g<ng*3;g++){double residual=std::sqrt(rr[g])/std::max(std::sqrt(norm[g]),1.);maxres=std::max(maxres,residual);stats[9]+=residual>1e-4;}
    stats[6]=seconds(solving);stats[7]=*std::max_element(iterations.begin(),iterations.end());stats[8]=maxres;
    stats[10]=initial_energy;stats[11]=final_energy;
    TRACE_VECTOR(u);TRACE_VECTOR(bu);TRACE_VECTOR(rr);TRACE_VECTOR(iterations);TRACE_VECTOR(live);TRACE_VALUE(final_energy);TRACE_VALUE(maxres);
    if(final_energy>initial_energy+1e-9*std::max(std::abs(initial_energy),1.))return 5;
    for(int a=0;a<n;a++)for(int ch=0;ch<3;ch++)output[3*pixels[a]+ch]=uint8_t(std::clamp(std::nearbyint(u[3*a+ch]),0.,255.));
    return 0;
}

namespace {
using Weight=std::pair<const uint64_t,double>;
#define SOLVE_ARGS image,base,unknown,halo,c,q,tx,ty,height,width,output,stats
int dispatch(const uint8_t* image,const uint8_t* base,const uint8_t* unknown,const uint8_t* halo,
    const float* c,const float* q,const float* tx,const float* ty,int height,int width,uint8_t* output,double* stats,
    int mode,size_t cap=Arena::maximum_bytes,int64_t fail_after=-1) {
    last_allocator_stats=AllocStats{};
    if(verified_arena&&mode!=0) {
        try {
#ifdef R08_TESTING
            trace.value("attempt",1);
#endif
            Arena arena(last_allocator_stats,cap,fail_after);
            return solve_with_allocator(SOLVE_ARGS,ArenaAllocator<Weight>(&arena));
        } catch(...) {
            if(mode==1)throw;
            ++last_allocator_stats.fallbacks;
            std::fill(stats,stats+12,0.);
            // All arena/map state has unwound before this complete original solve.
        }
    }
#ifdef R08_TESTING
    trace.value("attempt",0);
#endif
    return solve_with_allocator(SOLVE_ARGS,std::allocator<Weight>{});
}
}

extern "C" int tensor_solve(const uint8_t* image,const uint8_t* base,const uint8_t* unknown,const uint8_t* halo,
    const float* c,const float* q,const float* tx,const float* ty,int height,int width,uint8_t* output,double* stats) {
    try { return dispatch(SOLVE_ARGS,2); }
    catch(const std::bad_alloc&) { return 6; } catch(...) { return 7; }
}
extern "C" int tensor_allocator_enabled() { return verified_arena; }
extern "C" void tensor_allocator_last_stats(double* out) {
    out[0]=last_allocator_stats.requests;out[1]=last_allocator_stats.system_allocations;
    out[2]=last_allocator_stats.peak_bytes;out[3]=last_allocator_stats.fallbacks;out[4]=verified_arena;
}

#ifdef R08_TESTING
extern "C" int tensor_test_solve(const uint8_t* image,const uint8_t* base,const uint8_t* unknown,const uint8_t* halo,
    const float* c,const float* q,const float* tx,const float* ty,int height,int width,uint8_t* output,double* stats,
    int mode,size_t cap,int64_t fail_after,TraceCallback callback,void* user,double* alloc_stats) {
    trace=Trace{callback,user};int status;
    try {
        if(mode==3) { last_allocator_stats=AllocStats{};size_t current=0;
            status=solve_with_allocator(SOLVE_ARGS,CountingAllocator<Weight>(&last_allocator_stats,&current));
        } else status=dispatch(SOLVE_ARGS,mode,cap,fail_after);
    } catch(const std::bad_alloc&) { status=6; } catch(...) { status=7; }
    TRACE_VALUE(status);tensor_allocator_last_stats(alloc_stats);trace=Trace{};return status;
}
extern "C" int tensor_test_arena_boundaries() {
    try {
        AllocStats stats;Arena arena(stats);
        for(size_t alignment=1;alignment<=4096;alignment*=2) {
            void* p=arena.allocate(113,alignment);if(reinterpret_cast<uintptr_t>(p)%alignment)return 1;
        }
        void* p=arena.allocate(Arena::block_bytes,64);if(reinterpret_cast<uintptr_t>(p)%64)return 2;
        bool overflow=false;try { arena.allocate(std::numeric_limits<size_t>::max(),64); }catch(const std::bad_alloc&) { overflow=true; }
        if(!overflow)return 3;
        bool multiply=false;try { ArenaAllocator<uint64_t>(&arena).allocate(std::numeric_limits<size_t>::max()/8+1); }catch(const std::bad_alloc&) { multiply=true; }
        if(!multiply)return 4;
        bool invalid=false;try { arena.allocate(2,3); }catch(const std::bad_alloc&) { invalid=true; }
        if(!invalid)return 5;
        AllocStats bounded_stats;Arena bounded(bounded_stats,Arena::block_bytes);
        bounded.allocate(Arena::block_bytes-sizeof(Arena::Block),1);
        bool capped=false;try { bounded.allocate(1,1); }catch(const std::bad_alloc&) { capped=true; }
        if(!capped||bounded_stats.peak_bytes!=Arena::block_bytes)return 6;
        AllocStats fail_stats;Arena failure(fail_stats,Arena::maximum_bytes,0);
        bool failed=false;try { failure.allocate(1,1); }catch(const std::bad_alloc&) { failed=true; }
        if(!failed||fail_stats.system_allocations)return 7;
        AllocStats full_stats;Arena full(full_stats);
        for(int i=0;i<64;i++)full.allocate(Arena::block_bytes-sizeof(Arena::Block),1);
        bool full_cap=false;try { full.allocate(1,1); }catch(const std::bad_alloc&) { full_cap=true; }
        if(!full_cap||full_stats.peak_bytes!=Arena::maximum_bytes)return 8;
        return 0;
    } catch(...) { return 9; }
}
#endif

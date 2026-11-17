#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

// Preserve NumPy's separate lower/upper bincount sums and expression order.
// This library is compiled without contraction or fast-math.
extern "C" void edge_profile(const double* u, const double* colors,
    const double* weights, int64_t size, int bins, double* value, double* support) {
    std::vector<double> lo(bins), hi(bins), cross(bins), slo(bins), shi(bins);
    std::vector<double> rlo(bins*3), rhi(bins*3), d(bins), off(bins);
    for (int64_t i=0; i<size; ++i) {
        const int k=static_cast<int>(std::floor(u[i]));
        const double wt=weights[i];
        if (k<0 || k+1>=bins || !(wt>0)) continue;
        const double b=u[i]-k, a=1-b;
        const double wa=wt*a, wb=wt*b;
        lo[k]+=wa*a; hi[k+1]+=wb*b; cross[k]+=wa*b;
        slo[k]+=wa; shi[k+1]+=wb;
        for (int c=0; c<3; ++c) {
            rlo[k*3+c]+=wa*colors[i*3+c];
            rhi[(k+1)*3+c]+=wb*colors[i*3+c];
        }
    }
    for (int k=0; k<bins; ++k) {
        d[k]=((lo[k]+hi[k])+2*.06)+1e-5;
        off[k]=cross[k]-.06;
        support[k]=slo[k]+shi[k];
        for (int c=0; c<3; ++c) value[k*3+c]=rlo[k*3+c]+rhi[k*3+c];
    }
    for (int k=1; k<bins; ++k) {
        const double factor=off[k-1]/d[k-1];
        d[k]-=factor*off[k-1];
        for (int c=0; c<3; ++c) value[k*3+c]-=factor*value[(k-1)*3+c];
    }
    for (int c=0; c<3; ++c) value[(bins-1)*3+c]/=d[bins-1];
    for (int k=bins-2; k>=0; --k)
        for (int c=0; c<3; ++c)
            value[k*3+c]=(value[k*3+c]-off[k]*value[(k+1)*3+c])/d[k];
}

extern "C" int edge_direction_variance(const double* x, const double* y,
    const double* values, const double* weights, int64_t size,
    double cosine, double sine, double* variance, int capacity) {
    std::vector<double> coordinate(size), count(capacity), total(capacity), square(capacity);
    double minimum=std::numeric_limits<double>::infinity();
    for (int64_t i=0; i<size; ++i) {
        coordinate[i]=(x[i]*cosine+y[i]*sine)*2;
        minimum=std::min(minimum,coordinate[i]);
    }
    int bins=0;
    for (int64_t i=0; i<size; ++i) {
        const int k=static_cast<int>(std::nearbyint(coordinate[i]-minimum));
        if (k<0 || k>=capacity) return -1;
        bins=std::max(bins,k+1);
        count[k]+=weights[i];
        total[k]+=weights[i]*values[i];
        square[k]+=(weights[i]*values[i])*values[i];
    }
    for (int k=0; k<bins; ++k)
        variance[k]=square[k]-total[k]*total[k]/std::max(count[k],1e-10);
    return bins;
}

extern "C" void edge_sample(const double* profile, int bins,
    const double* u, int64_t size, double* prediction) {
    for (int64_t i=0; i<size; ++i) {
        const int low=std::clamp(static_cast<int>(std::floor(u[i])),0,bins-2);
        const double f=std::clamp(u[i]-low,0.,1.);
        for (int c=0; c<3; ++c)
            prediction[i*3+c]=profile[low*3+c]*(1-f)+profile[(low+1)*3+c]*f;
    }
}

extern "C" void edge_bracket(const int64_t* indices, const double* tangent,
    const uint8_t* observed, int64_t size, int bins, double* lower, double* upper) {
    std::fill(lower,lower+bins,std::numeric_limits<double>::infinity());
    std::fill(upper,upper+bins,-std::numeric_limits<double>::infinity());
    for (int64_t i=0; i<size; ++i) if (observed[i]) {
        const int k=static_cast<int>(indices[i]);
        lower[k]=std::min(lower[k],tangent[i]);
        lower[k+1]=std::min(lower[k+1],tangent[i]);
        upper[k]=std::max(upper[k],tangent[i]);
        upper[k+1]=std::max(upper[k+1],tangent[i]);
    }
}

extern "C" int edge_bounded(const uint8_t* original, const uint8_t* result,
    int64_t height, int64_t width, int64_t x0, int64_t y0, int64_t x1, int64_t y1,
    const uint8_t* mask) {
    if (!original || !result || !mask || height<=0 || width<=0 ||
        x0<0 || y0<0 || x1<x0 || y1<y0 || x1>width || y1>height) return -1;
    const size_t max_size=std::numeric_limits<size_t>::max();
    if (static_cast<uint64_t>(width)>max_size/3) return -1;
    const size_t row_bytes=static_cast<size_t>(width)*3;
    if (static_cast<uint64_t>(height)>max_size/row_bytes) return -1;
    const size_t top_bytes=static_cast<size_t>(y0)*row_bytes;
    if (std::memcmp(original,result,top_bytes)!=0) return 0;
    const size_t bottom_offset=static_cast<size_t>(y1)*row_bytes;
    const size_t bottom_bytes=static_cast<size_t>(height-y1)*row_bytes;
    if (std::memcmp(original+bottom_offset,result+bottom_offset,bottom_bytes)!=0) return 0;
    const size_t mask_width=static_cast<size_t>(x1-x0);
    const size_t left_bytes=static_cast<size_t>(x0)*3;
    const size_t right_offset=static_cast<size_t>(x1)*3;
    const size_t right_bytes=static_cast<size_t>(width-x1)*3;
    for (int64_t y=y0; y<y1; ++y) {
        const size_t offset=static_cast<size_t>(y)*row_bytes;
        const uint8_t* a=original+offset;
        const uint8_t* b=result+offset;
        if (std::memcmp(a,b,left_bytes)!=0 ||
            std::memcmp(a+right_offset,b+right_offset,right_bytes)!=0) return 0;
        const uint8_t* mask_row=mask+static_cast<size_t>(y-y0)*mask_width;
        for (size_t x=0; x<mask_width; ++x) if (mask_row[x]==0) {
            const size_t pixel=left_bytes+x*3;
            if (a[pixel]!=b[pixel] || a[pixel+1]!=b[pixel+1] || a[pixel+2]!=b[pixel+2])
                return 0;
        }
    }
    return 1;
}

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Moment = std::array<double, 18>;
double seconds(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}
struct Box { int x0, y0, x1, y1; };
Box window(int x, int y, int radius, int w, int h) {
    return {std::max(0, x-radius), std::max(0, y-radius),
            std::min(w, x+radius+1), std::min(h, y+radius+1)};
}
template<class T> double box_sum(const std::vector<T>& integral, int stride, Box b) {
    return double(integral[b.y1*stride+b.x1]) - integral[b.y0*stride+b.x1]
         - integral[b.y1*stride+b.x0] + integral[b.y0*stride+b.x0];
}
double count(const std::vector<Moment>& integral, int stride, Box b) {
    return integral[b.y1*stride+b.x1][0] - integral[b.y0*stride+b.x1][0]
         - integral[b.y1*stride+b.x0][0] + integral[b.y0*stride+b.x0][0];
}
bool fit(const std::vector<Moment>& integral, int stride, Box b, int x, int y,
         double radius, std::array<double, 3>& prediction, double& rms) {
    Moment s;
    for (int k=0; k<18; ++k)
        s[k] = integral[b.y1*stride+b.x1][k] - integral[b.y0*stride+b.x1][k]
             - integral[b.y1*stride+b.x0][k] + integral[b.y0*stride+b.x0][k];
    const double n=s[0];
    if (n < .30*(b.x1-b.x0)*(b.y1-b.y0) || n <= 0) return false;
    // Axis samples are useful to the fit but do not count as quadrant support.
    for (Box quadrant : {Box{b.x0,b.y0,x,y}, Box{x+1,b.y0,b.x1,y},
                         Box{b.x0,y+1,x,b.y1}, Box{x+1,y+1,b.x1,b.y1}})
        if (count(integral,stride,quadrant) < 8) return false;
    const double mx=s[1]/n, my=s[2]/n;
    const double xx=(s[3]-s[1]*mx)/(radius*radius);
    const double xy=(s[4]-s[1]*my)/(radius*radius);
    const double yy=(s[5]-s[2]*my)/(radius*radius);
    const double minimum=(xx+yy-std::hypot(xx-yy,2*xy))/(2*n);
    const double determinant=xx*yy-xy*xy;
    if (!std::isfinite(minimum) || minimum < .08 || !(determinant>0)) return false;
    rms=0;
    // This Schur-complement analytic solve is the unregularized 3x3 affine
    // normal equation. Centering the 2x2 block avoids a large-coordinate pivot.
    for (int c=0; c<3; ++c) {
        const double mean=s[6+c]/n;
        const double bx=(s[9+c]-mx*s[6+c])/radius;
        const double by=(s[12+c]-my*s[6+c])/radius;
        const double ax=(yy*bx-xy*by)/determinant;
        const double ay=(xx*by-xy*bx)/determinant;
        const double p=mean-ax*(mx-x)/radius-ay*(my-y)/radius;
        const double variance=std::max(0.,s[15+c]/n-mean*mean);
        const double residual=std::max(0.,s[15+c]-s[6+c]*mean-ax*bx-ay*by);
        const double error=std::sqrt(residual/n);
        if (!std::isfinite(p) || !std::isfinite(error) || error>2
            || p<mean-3*std::sqrt(variance)-2 || p>mean+3*std::sqrt(variance)+2)
            return false;
        prediction[c]=p;
        rms=std::max(rms,error);
    }
    return true;
}
}

extern "C" int affine_predict(const uint8_t* image, const uint8_t* exclusion,
                              const uint8_t* write, int h, int w, int radius,
                              double* prediction, double* alpha, double* stats) {
    if (h<=0 || w<=0 || radius<1) return 1;
    try {
        auto started=Clock::now();
        const int stride=w+1;
        const size_t size=size_t(h)*w, integral_size=size_t(h+1)*stride;
        std::fill(prediction,prediction+size*3,0.);
        std::fill(alpha,alpha+size,0.);
        std::fill(stats,stats+12,0.);
        std::vector<int> gradient(integral_size,0);
        for (int y=0; y<h; ++y) {
            int row=0;
            for (int x=0; x<w; ++x) {
                bool clean=x>0 && y>0 && x+1<w && y+1<h, strong=false;
                if (clean) {
                    for (int dy=-1;dy<=1 && clean;++dy)
                        for (int dx=-1;dx<=1;++dx)
                            if (exclusion[(y+dy)*w+x+dx]) { clean=false; break; }
                }
                if (clean) for (int c=0;c<3;++c) {
                    const double gx=(double(image[(y*w+x+1)*3+c])-image[(y*w+x-1)*3+c])*.5;
                    const double gy=(double(image[((y+1)*w+x)*3+c])-image[((y-1)*w+x)*3+c])*.5;
                    strong |= gx*gx+gy*gy > 36.;
                }
                row += strong;
                gradient[(y+1)*stride+x+1]=gradient[y*stride+x+1]+row;
                stats[1] += strong;
            }
        }
        std::vector<int> candidates;
        for (int y=0;y<h;++y) for (int x=0;x<w;++x) if (write[y*w+x]) {
            ++stats[0];
            if (box_sum(gradient,stride,window(x,y,2*radius,w,h))>0) ++stats[2];
            else candidates.push_back(y*w+x);
        }
        stats[8]=seconds(started);
        if (candidates.empty()) return 0;
        started=Clock::now();
        std::vector<Moment> integral(integral_size);
        for (int y=0;y<h;++y) {
            Moment row{};
            for (int x=0;x<w;++x) {
                if (!exclusion[y*w+x]) {
                    row[0]+=1; row[1]+=x; row[2]+=y;
                    row[3]+=double(x)*x; row[4]+=double(x)*y; row[5]+=double(y)*y;
                    for (int c=0;c<3;++c) {
                        const double color=image[(y*w+x)*3+c];
                        row[6+c]+=color; row[9+c]+=x*color;
                        row[12+c]+=y*color; row[15+c]+=color*color;
                    }
                }
                for (int k=0;k<18;++k)
                    integral[(y+1)*stride+x+1][k]=integral[y*stride+x+1][k]+row[k];
            }
        }
        stats[9]=seconds(started); started=Clock::now();
        for (int pixel:candidates) {
            const int x=pixel%w,y=pixel/w;
            std::array<double,3> small{},large{};
            double rms1=0,rms2=0;
            if (!fit(integral,stride,window(x,y,radius,w,h),x,y,radius,small,rms1)) {
                ++stats[3]; continue;
            }
            if (!fit(integral,stride,window(x,y,2*radius,w,h),x,y,radius,large,rms2)) {
                ++stats[4]; continue;
            }
            double difference=0;
            for (int c=0;c<3;++c) difference=std::max(difference,std::abs(small[c]-large[c]));
            if (!std::isfinite(difference) || difference>2) { ++stats[5]; continue; }
            const double weight=std::clamp(2-std::max(rms1,rms2),0.,1.)*
                                std::clamp(2-difference,0.,1.);
            for (int c=0;c<3;++c) prediction[pixel*3+c]=small[c];
            alpha[pixel]=weight;
            stats[6] += weight>0;
            stats[7] += weight==1;
        }
        stats[10]=seconds(started);
        return 0;
    } catch (...) { return 2; }
}

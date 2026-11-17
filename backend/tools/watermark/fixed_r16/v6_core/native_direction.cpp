#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

// Independent R12 kernel. Keep the old per-angle operation and sample order.
// Compile without contraction or fast-math. Each call owns its workspace.
extern "C" int direction_variance_batch(const double* x, const double* y,
    const double* values, const double* weights, int64_t size,
    const double* cosine, const double* sine, int angles,
    double* variance, int capacity, int32_t* occupied) noexcept {
    if (size < 1 || angles < 1 || capacity < 1) return -1;
    try {
        std::vector<double> coordinate(size), count(capacity), total(capacity), square(capacity);
        for (int angle=0; angle<angles; ++angle) {
            std::fill(count.begin(), count.end(), 0.);
            std::fill(total.begin(), total.end(), 0.);
            std::fill(square.begin(), square.end(), 0.);
            double minimum=std::numeric_limits<double>::infinity();
            for (int64_t i=0; i<size; ++i) {
                coordinate[i]=(x[i]*cosine[angle]+y[i]*sine[angle])*2;
                minimum=std::min(minimum,coordinate[i]);
            }
            int bins=0;
            for (int64_t i=0; i<size; ++i) {
                const double rounded=std::nearbyint(coordinate[i]-minimum);
                if (!(rounded>=0 && rounded<capacity)) return -2;
                const int k=static_cast<int>(rounded);
                bins=std::max(bins,k+1);
                count[k]+=weights[i];
                total[k]+=weights[i]*values[i];
                square[k]+=(weights[i]*values[i])*values[i];
            }
            for (int k=0; k<bins; ++k)
                variance[static_cast<int64_t>(angle)*capacity+k]=square[k]-total[k]*total[k]/std::max(count[k],1e-10);
            occupied[angle]=bins;
        }
        return 0;
    } catch (const std::bad_alloc&) {
        return -3;
    } catch (...) {
        return -4;
    }
}

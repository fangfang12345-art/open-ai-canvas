#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

// Use the same float32 BLAS products as NumPy's Accelerate backend. No fast-math
// or fused multiply-add contraction is permitted in the surrounding reductions.
static void multiply(const float* a, const float* b, int m, int n, int k,
                     float* output) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                1.f, a, k, b, n, 0.f, output, n);
}

static bool dynamic_fit(const std::vector<float>& matrix,
                        const std::vector<float>& colors, float* coefficients) {
    const int count = static_cast<int>(matrix.size()/3);
    float gram[9], rhs[9];
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 3, 3, count,
                1.f, matrix.data(), 3, matrix.data(), 3, 0.f, gram, 3);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 3, 3, count,
                1.f, matrix.data(), 3, colors.data(), 3, 0.f, rhs, 3);
    double svd_matrix[9], singular[3], workspace[128], unused_u=0, unused_vt=0;
    for (int i=0; i<9; ++i) svd_matrix[i] = gram[i];
    __CLPK_integer three=3, one=1, length=128, info=0, integers[24];
    char values_only='N';
    // Match np.linalg.cond's double SVD and float32 singular-value division,
    // including windows immediately next to its condition-number threshold.
    dgesdd_(&values_only, &three, &three, svd_matrix, &three, singular,
            &unused_u, &one, &unused_vt, &one, workspace, &length, integers, &info);
    if (info != 0 || !(singular[2] > 0) ||
            static_cast<float>(singular[0])/static_cast<float>(singular[2]) > 40.f)
        return false;
    // np.linalg.solve promotes float32 inputs to LAPACK double internally.
    double a[9], b[9];
    for (int row=0; row<3; ++row) for (int col=0; col<3; ++col) {
        a[col*3+row] = gram[row*3+col];
        b[col*3+row] = rhs[row*3+col];
    }
    __CLPK_integer pivots[3];
    dgesv_(&three, &three, a, &three, pivots, b, &three, &info);
    if (info != 0) return false;
    for (int row=0; row<3; ++row) for (int col=0; col<3; ++col)
        coefficients[row*3+col] = static_cast<float>(b[col*3+row]);
    return true;
}

// NumPy's float32 sum uses a pairwise reduction with eight lanes for contiguous
// arrays. Keeping its order matters for samples at the held-out acceptance gate.
static float pairwise_sum(const float* a, int64_t size) {
    if (size < 8) {
        float sum = -0.f;
        for (int64_t i=0; i<size; ++i) sum += a[i];
        return sum;
    }
    if (size <= 128) {
        float r[8];
        for (int i=0; i<8; ++i) r[i] = a[i];
        int64_t i=8;
        for (; i<size-(size%8); i+=8)
            for (int j=0; j<8; ++j) r[j] += a[i+j];
        float sum = ((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]));
        for (; i<size; ++i) sum += a[i];
        return sum;
    }
    int64_t middle = size/2;
    middle -= middle%8;
    return pairwise_sum(a, middle)+pairwise_sum(a+middle, size-middle);
}

extern "C" int64_t background_fit(const uint8_t* patch, const uint8_t* evidence,
    const uint8_t* write, int64_t pixels, int64_t window_count,
    const int64_t* metadata, const int64_t* samples, const float* designs,
    const uint8_t* training, const float* projectors, const int64_t* targets,
    const float* target_designs, const float* target_weights,
    uint8_t* result, uint8_t* accepted, float* uncertainty) {
    if (pixels <= 0 || window_count < 0) return -1;
    std::memcpy(result, patch, static_cast<size_t>(pixels)*3);
    std::fill(accepted, accepted+pixels, 0);
    std::fill(uncertainty, uncertainty+pixels, 0.f);
    std::vector<float> total(pixels*3), weights(pixels);
    std::vector<float> low(pixels*3, std::numeric_limits<float>::infinity());
    std::vector<float> high(pixels*3, -std::numeric_limits<float>::infinity());
    std::vector<float> colors, matrix, predicted, residual, square, target_prediction;
    int64_t accepted_windows=0;
    for (int64_t win=0; win<window_count; ++win) {
        const int64_t* meta=metadata+win*6;
        const int64_t offset=meta[0], size=meta[1], target_offset=meta[2];
        const int64_t target_count=meta[3], projector_offset=meta[4], training_count=meta[5];
        if (size > INT32_MAX || target_count > INT32_MAX || size < 0 || target_count < 0)
            return -1;
        colors.clear(); matrix.clear();
        int64_t test_count=0, usable_count=0;
        for (int64_t i=0; i<size; ++i) {
            const int64_t pixel=samples[offset+i];
            if (pixel<0 || pixel>=pixels) return -1;
            if (evidence[pixel]) continue;
            ++usable_count;
            if (!training[offset+i]) { ++test_count; continue; }
            for (int c=0; c<3; ++c) {
                colors.push_back(patch[pixel*3+c]);
                matrix.push_back(designs[(offset+i)*3+c]);
            }
        }
        const int64_t train_count=colors.size()/3;
        if (train_count<36 || test_count<12) continue;
        float coefficients[9];
        if (usable_count==size) {
            if (train_count!=training_count) return -1;
            multiply(projectors+projector_offset, colors.data(), 3, 3,
                     static_cast<int>(train_count), coefficients);
        } else if (!dynamic_fit(matrix, colors, coefficients)) continue;
        predicted.resize(size*3);
        multiply(designs+offset*3, coefficients, static_cast<int>(size), 3, 3,
                 predicted.data());
        residual.clear(); square.clear();
        bool rejected=false;
        for (int64_t i=0; i<size; ++i) {
            const int64_t pixel=samples[offset+i];
            if (evidence[pixel]) continue;
            float error=0.f;
            for (int c=0; c<3; ++c)
                error=std::max(error, std::abs(predicted[i*3+c]-patch[pixel*3+c]));
            if (error>7.f || !std::isfinite(error)) { rejected=true; break; }
            residual.push_back(error);
            if (!training[offset+i]) square.push_back(error*error);
        }
        if (rejected) continue;
        const float mse=pairwise_sum(square.data(), square.size())/static_cast<float>(square.size());
        if (mse>2.25f) continue;
        const double rank=(residual.size()-1)*.95;
        const size_t lower=static_cast<size_t>(rank), upper=static_cast<size_t>(std::ceil(rank));
        std::nth_element(residual.begin(), residual.begin()+lower, residual.end());
        const float a=residual[lower];
        float b=a;
        if (upper>lower) b=*std::min_element(residual.begin()+upper, residual.end());
        // np.percentile interpolates float32 endpoints with float64 weights.
        const double fraction=rank-lower, difference=static_cast<float>(b-a);
        const double percentile=fraction>=.5 ? b-difference*(1.-fraction)
                                             : a+difference*fraction;
        if (percentile>3.) continue;
        target_prediction.resize(target_count*3);
        multiply(target_designs+target_offset*3, coefficients,
                 static_cast<int>(target_count), 3, 3, target_prediction.data());
        for (float value: target_prediction)
            if (value<-.5f || value>255.5f || !std::isfinite(value)) { rejected=true; break; }
        if (rejected) continue;
        const float rmse=std::sqrt(mse);
        for (int64_t i=0; i<target_count; ++i) {
            const int64_t pixel=targets[target_offset+i];
            if (pixel<0 || pixel>=pixels) return -1;
            const float weight=target_weights[target_offset+i];
            weights[pixel]+=weight;
            uncertainty[pixel]+=rmse*weight;
            for (int c=0; c<3; ++c) {
                const float value=target_prediction[i*3+c];
                total[pixel*3+c]+=value*weight;
                low[pixel*3+c]=std::min(low[pixel*3+c], value);
                high[pixel*3+c]=std::max(high[pixel*3+c], value);
            }
        }
        ++accepted_windows;
    }
    for (int64_t pixel=0; pixel<pixels; ++pixel) {
        if (!write[pixel] || !(weights[pixel]>.05f)) continue;
        bool agreement=true;
        for (int c=0; c<3; ++c)
            if (high[pixel*3+c]-low[pixel*3+c]>3.f) agreement=false;
        if (!agreement) continue;
        accepted[pixel]=1;
        uncertainty[pixel]/=weights[pixel];
        for (int c=0; c<3; ++c)
            result[pixel*3+c]=static_cast<uint8_t>(std::clamp(
                std::nearbyint(total[pixel*3+c]/weights[pixel]), 0.f, 255.f));
    }
    return accepted_windows;
}

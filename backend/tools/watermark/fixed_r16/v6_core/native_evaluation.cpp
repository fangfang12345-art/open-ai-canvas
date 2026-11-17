// R11 on-demand evaluation. Every output is private until Python commits it.
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {
bool sample(const double* profile, int bins, double u, double* value) {
    if (!std::isfinite(u)) return false;
    const double floored = std::floor(u);
    const int low = static_cast<int>(std::clamp(floored, 0., double(bins-2)));
    const double f = std::clamp(u-low, 0., 1.);
    for (int c=0; c<3; ++c) {
        value[c] = profile[low*3+c]*(1-f)+profile[(low+1)*3+c]*f;
        if (!std::isfinite(value[c])) return false;
    }
    return true;
}

bool transport(const double* profile, const double* beta, int bins,
               double u, double tau, double* value) {
    if (!sample(profile, bins, u, value)) return false;
    for (int c=0; c<3; ++c) {
        value[c] = value[c] + tau*beta[c];
        if (!std::isfinite(value[c])) return false;
    }
    return true;
}
}

extern "C" int64_t evaluation_observed(
    const double* profile, int bins, const double* u, const double* tau,
    const double* colors, const uint8_t* observed, int64_t size,
    const double* beta, int use_beta, double* residual, int64_t capacity) {
    try {
        if (bins < 2 || size < 0 || capacity < 0) return -1;
        int64_t count = 0;
        for (int64_t i=0; i<size; ++i) if (observed[i]) {
            if (count >= capacity) return -2;
            double value[3];
            const bool valid = use_beta
                ? transport(profile, beta, bins, u[i], tau[i], value)
                : sample(profile, bins, u[i], value);
            if (!valid) return -3;
            double error = 0.;
            for (int c=0; c<3; ++c) {
                const double delta = std::abs(value[c]-colors[i*3+c]);
                if (!std::isfinite(delta)) return -3;
                error = std::max(error, delta);
            }
            residual[count++] = error;
        }
        return count;
    } catch (...) { return -4; }
}

extern "C" int64_t evaluation_targets(
    const double* u, const double* tau, const double* spatial,
    const int64_t* ui, const uint8_t* central, const uint8_t* common,
    const uint8_t* trusted, const double* disagreement,
    int old_valid, int new_valid, const double* p0,
    const double* profile, const double* beta,
    const double* left, const double* left_beta,
    const double* right, const double* right_beta,
    int64_t size, int bins, int64_t capacity,
    int64_t* indices, float* weights, float* contributions, int64_t* counts) {
    try {
        if (bins < 2 || size < 0 || capacity < 0) return -1;
        int64_t count=0, old_supported=0, old_count=0, new_count=0;
        for (int64_t i=0; i<size; ++i) if (central[i]) {
            const int64_t bin = ui[i];
            if (bin < 0 || bin >= bins-1) return -2;
            const bool active = old_valid && trusted[i];
            old_supported += active;
            bool new_active = false;
            double target_disagreement = 0.;
            if (new_valid && common[bin] && common[bin+1]) {
                double lv[3], rv[3];
                if (!transport(left, left_beta, bins, u[i], tau[i], lv)
                    || !transport(right, right_beta, bins, u[i], tau[i], rv)) return -3;
                for (int c=0; c<3; ++c) {
                    const double delta = std::abs(lv[c]-rv[c]);
                    if (!std::isfinite(delta)) return -3;
                    target_disagreement = std::max(target_disagreement, delta);
                }
                new_active = target_disagreement < 9;
            }
            if (!active && !new_active) continue;
            if (count >= capacity) return -2;
            double prediction[3];
            if (new_active) {
                if (!transport(profile, beta, bins, u[i], tau[i], prediction)) return -3;
                ++new_count;
            } else {
                if (!sample(p0, bins, u[i], prediction)) return -3;
                ++old_count;
            }
            const double delta = new_active ? target_disagreement : disagreement[bin];
            const double agreement_weight = std::clamp(1-delta/12, 0., 1.);
            const float weight = static_cast<float>((spatial[i]*agreement_weight)*1.);
            if (!std::isfinite(weight)) return -3;
            indices[count] = i;
            weights[count] = weight;
            for (int c=0; c<3; ++c) {
                const float color = static_cast<float>(prediction[c]);
                const float value = color*weight;
                if (!std::isfinite(value)) return -3;
                contributions[count*3+c] = value;
            }
            ++count;
        }
        counts[0] = old_supported;
        counts[1] = old_count;
        counts[2] = new_count;
        return count;
    } catch (...) { return -4; }
}

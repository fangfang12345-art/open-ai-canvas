#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
struct Edge {
    int p, z, i, j;
    double a, weight;
};
struct Interval {
    std::array<double, 3> lo{{255., 255., 255.}};
    std::array<double, 3> hi{{0., 0., 0.}};
    bool boundary = false;
};
struct Workspace {
    std::vector<int> pixels, index, group, queue, neighbor_offsets;
    std::vector<std::size_t> row_offsets, row_cursor, edge_indices;
    std::vector<Interval> intervals;
    std::vector<Edge> edges;
    std::vector<double> u, diagonal, rhs, residual, preconditioned, direction, applied,
                        row_weights;
    std::vector<uint8_t> quantized;
};
double difference(const Edge& edge, const double* colors) {
    const double db = colors[3*edge.p] - colors[3*edge.z];
    const double dg = colors[3*edge.p+1] - colors[3*edge.z+1];
    const double dr = colors[3*edge.p+2] - colors[3*edge.z+2];
    return ((db*db + dg*dg) + dr*dr)/3.;
}
double nearest_even(double value) {
    const double low = std::floor(value), fraction = value - low;
    return fraction < .5 ? low : (fraction > .5 ? low+1. :
                                 (std::fmod(low, 2.) == 0. ? low : low+1.));
}
double energy(const std::vector<Edge>& edges, const double* colors) {
    double value = 0.;
    for (const auto& edge : edges)
        value += edge.a*64.*std::log1p(difference(edge, colors)/64.);
    return value;
}
double quantized_energy(const std::vector<Edge>& edges, const uint8_t* colors) {
    double value = 0.;
    for (const auto& edge : edges) {
        double ds[3];
        for (int channel = 0; channel < 3; ++channel) {
            const double p = colors[3*edge.p+channel];
            const double z = colors[3*edge.z+channel];
            ds[channel] = p-z;
        }
        const double s = ((ds[0]*ds[0]+ds[1]*ds[1])+ds[2]*ds[2])/3.;
        value += edge.a*64.*std::log1p(s/64.);
    }
    return value;
}
void multiply(const std::vector<std::size_t>& row_offsets, const std::vector<int>& neighbor_offsets,
              const std::vector<double>& row_weights, const std::vector<double>& diagonal,
              const std::vector<double>& input, std::vector<double>& output) {
    for (std::size_t i = 0; i < diagonal.size(); ++i) {
        double b = diagonal[i]*input[3*i];
        double g = diagonal[i]*input[3*i+1];
        double r = diagonal[i]*input[3*i+2];
        for (std::size_t at = row_offsets[i]; at < row_offsets[i+1]; ++at) {
            const int neighbor = neighbor_offsets[at];
            const double weight = row_weights[at];
            b -= weight*input[neighbor];
            g -= weight*input[neighbor+1];
            r -= weight*input[neighbor+2];
        }
        output[3*i] = b; output[3*i+1] = g; output[3*i+2] = r;
    }
}
bool exceeds(double after, double before) {
    return after > before + 1e-9*std::max(std::abs(before), 1.);
}

int solve(const uint8_t* image, const uint8_t* base, const uint8_t* unknown,
          const double* c, const double* tx, const double* ty, int h, int w,
          double* output, double* summary, double* rounds, int fault) {
    if (fault == 1) throw std::bad_alloc();
    if (fault == 2) return 8;
    if (fault == 3) { output[0] = -777.; return 8; }
    if (h <= 0 || w <= 0 || h > std::numeric_limits<int>::max()/w) return 3;
    const int area = h*w;
    if (area > std::numeric_limits<int>::max()/3) return 3;
    for (int p = 0; p < area; ++p) {
        const double norm = std::sqrt(tx[p]*tx[p]+ty[p]*ty[p]);
        if (!std::isfinite(c[p]) || !std::isfinite(tx[p]) || !std::isfinite(ty[p]) ||
            c[p] < 0. || c[p] > 1. || norm < 1.-1e-5 || norm > 1.+1e-5) return 3;
    }
    auto started = Clock::now();
    // Calls on different worker threads have independent reusable scratch storage.
    thread_local Workspace workspace;
    auto& pixels = workspace.pixels; pixels.clear();
    auto& index = workspace.index; index.assign(area, -1);
    for (int p = 0; p < area; ++p) {
        for (int channel = 0; channel < 3; ++channel)
            output[3*p+channel] = image[3*p+channel];
        if (unknown[p]) { index[p] = static_cast<int>(pixels.size()); pixels.push_back(p); }
    }
    const int count = static_cast<int>(pixels.size());
    summary[0] = count;
    if (!count) return 0;
    auto& group = workspace.group; group.assign(count, -1);
    auto& queue = workspace.queue; queue.resize(count);
    auto& intervals = workspace.intervals; intervals.clear();
    for (int i = 0; i < count; ++i) {
        if (group[i] >= 0) continue;
        const int component = static_cast<int>(intervals.size());
        intervals.emplace_back();
        Interval& interval = intervals.back();
        int head = 0, tail = 1;
        queue[0] = i; group[i] = component;
        while (head < tail) {
            const int p = pixels[queue[head++]], y = p/w, x = p-y*w;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (!(dx || dy)) continue;
                    const int xx = x+dx, yy = y+dy;
                    if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
                    const int z = yy*w+xx, j = index[z];
                    if (j >= 0) {
                        if (group[j] < 0) { group[j] = component; queue[tail++] = j; }
                    } else {
                        interval.boundary = true; summary[16] += 1.;
                        for (int channel = 0; channel < 3; ++channel) {
                            const double value = image[3*z+channel];
                            interval.lo[channel] = std::min(interval.lo[channel], value);
                            interval.hi[channel] = std::max(interval.hi[channel], value);
                        }
                    }
                }
            }
        }
        if (!interval.boundary) { summary[3] = intervals.size(); summary[4] = seconds(started); return 2; }
    }
    summary[3] = intervals.size();
    auto& u = workspace.u; u.resize(3*count);
    auto& diagonal = workspace.diagonal; diagonal.resize(count);
    auto& rhs = workspace.rhs; rhs.resize(3*count);
    auto& residual = workspace.residual; residual.resize(3*count);
    auto& preconditioned = workspace.preconditioned; preconditioned.resize(3*count);
    auto& direction = workspace.direction; direction.resize(3*count);
    auto& applied = workspace.applied; applied.resize(3*count);
    for (int i = 0; i < count; ++i) {
        const auto& interval = intervals[group[i]];
        for (int channel = 0; channel < 3; ++channel) {
            const double initial = base[3*pixels[i]+channel];
            const double clipped = std::clamp(initial, interval.lo[channel], interval.hi[channel]);
            if (clipped != initial) {
                summary[6] += 1.; summary[7] = std::max(summary[7], std::abs(clipped-initial));
            }
            output[3*pixels[i]+channel] = u[3*i+channel] = clipped;
        }
    }
    summary[4] = seconds(started);

    started = Clock::now();
    auto& edges = workspace.edges; edges.clear();
    edges.reserve(std::min(static_cast<std::size_t>(area)*4,
                           static_cast<std::size_t>(count)*8));
    const int directions[4][2] = {{1,0}, {0,1}, {1,1}, {-1,1}};
    for (const auto& direction_pair : directions) {
        const int dx = direction_pair[0], dy = direction_pair[1];
        const double length2 = dx*dx+dy*dy;
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            const int xx = x+dx, yy = y+dy;
            if (xx < 0 || xx >= w || yy >= h) continue;
            const int p = y*w+x, z = yy*w+xx, i = index[p], j = index[z];
            if (i < 0 && j < 0) continue;
            const double dotp = -dx*ty[p]+dy*tx[p];
            const double dotz = dx*ty[z]-dy*tx[z];
            const double ap = (1.-c[p])+c[p]/(1.+64.*dotp*dotp/length2);
            const double az = (1.-c[z])+c[z]/(1.+64.*dotz*dotz/length2);
            const double a = (ap+az)/(2.*length2);
            if (!(a > 0.) || !std::isfinite(a)) return 4;
            edges.push_back({p,z,i,j,a,0.});
            if (i >= 0 && j >= 0) summary[2] += 1.;
        }
    }
    auto& row_offsets = workspace.row_offsets; row_offsets.assign(count+1, 0);
    for (const auto& edge : edges) {
        if (edge.i < 0 || edge.j < 0) continue;
        ++row_offsets[edge.i+1]; ++row_offsets[edge.j+1];
    }
    for (int i = 0; i < count; ++i) row_offsets[i+1] += row_offsets[i];
    auto& neighbor_offsets = workspace.neighbor_offsets; neighbor_offsets.resize(row_offsets.back());
    auto& edge_indices = workspace.edge_indices; edge_indices.resize(row_offsets.back());
    auto& row_weights = workspace.row_weights; row_weights.resize(row_offsets.back());
    auto& row_cursor = workspace.row_cursor;
    row_cursor.assign(row_offsets.begin(), row_offsets.end()-1);
    // Each row retains the exact subtraction order of the original edge scatter.
    for (std::size_t k = 0; k < edges.size(); ++k) {
        const auto& edge = edges[k];
        if (edge.i < 0 || edge.j < 0) continue;
        const std::size_t at_i = row_cursor[edge.i]++, at_j = row_cursor[edge.j]++;
        neighbor_offsets[at_i] = 3*edge.j; neighbor_offsets[at_j] = 3*edge.i;
        edge_indices[at_i] = edge_indices[at_j] = k;
    }
    summary[1] = edges.size(); summary[5] = seconds(started);
    double previous_energy = energy(edges, output);
    if (!std::isfinite(previous_energy)) return 4;
    summary[8] = previous_energy;

    for (int outer = 0; outer < 4; ++outer) {
        double* record = rounds+32*outer;
        started = Clock::now();
        std::fill(diagonal.begin(), diagonal.end(), 0.);
        std::fill(rhs.begin(), rhs.end(), 0.);
        double q_before = 0., weight_min = std::numeric_limits<double>::infinity(), weight_max = 0.;
        for (auto& edge : edges) {
            const double s = difference(edge, output);
            edge.weight = edge.a/(1.+s/64.);
            if (!(edge.weight > 0.) || !std::isfinite(edge.weight)) return 4;
            weight_min = std::min(weight_min, edge.weight); weight_max = std::max(weight_max, edge.weight);
            q_before += edge.weight*s;
            if (edge.i >= 0) {
                diagonal[edge.i] += edge.weight;
                if (edge.j < 0) for (int channel = 0; channel < 3; ++channel)
                    rhs[3*edge.i+channel] += edge.weight*output[3*edge.z+channel];
            }
            if (edge.j >= 0) {
                diagonal[edge.j] += edge.weight;
                if (edge.i < 0) for (int channel = 0; channel < 3; ++channel)
                    rhs[3*edge.j+channel] += edge.weight*output[3*edge.p+channel];
            }
        }
        for (std::size_t at = 0; at < row_weights.size(); ++at)
            row_weights[at] = edges[edge_indices[at]].weight;
        if (!std::isfinite(q_before)) return 4;
        for (double value : diagonal) if (!(value > 0.) || !std::isfinite(value)) return 5;
        record[0] = seconds(started); record[4] = q_before; record[6] = previous_energy;
        record[8] = weight_min; record[9] = weight_max;

        started = Clock::now();
        multiply(row_offsets, neighbor_offsets, row_weights, diagonal, u, applied);
        double norm[3] = {}, rz[3] = {}, squared[3] = {};
        for (int i = 0; i < count; ++i) for (int channel = 0; channel < 3; ++channel) {
            const int at = 3*i+channel;
            norm[channel] += rhs[at]*rhs[at];
            residual[at] = rhs[at]-applied[at];
            direction[at] = preconditioned[at] = residual[at]/diagonal[i];
            rz[channel] += residual[at]*preconditioned[at];
            squared[channel] += residual[at]*residual[at];
        }
        for (int channel = 0; channel < 3; ++channel) {
            norm[channel] = std::max(std::sqrt(norm[channel]), 1.);
            record[15+channel] = std::sqrt(squared[channel])/norm[channel];
            if (!std::isfinite(record[15+channel]) || !std::isfinite(rz[channel])) return 4;
        }
        if (fault == 5) return 5;
        for (int iteration = 0; iteration < 16; ++iteration) {
            bool live[3];
            for (int channel = 0; channel < 3; ++channel)
                live[channel] = std::sqrt(squared[channel])/norm[channel] > 1e-4;
            if (!(live[0] || live[1] || live[2])) break;
            for (int i = 0; i < count; ++i) for (int channel = 0; channel < 3; ++channel)
                if (!live[channel]) direction[3*i+channel] = 0.;
            multiply(row_offsets, neighbor_offsets, row_weights, diagonal, direction, applied);
            double denominator[3] = {}, alpha[3] = {}, next_rz[3] = {}, next_squared[3] = {};
            for (int i = 0; i < count; ++i) for (int channel = 0; channel < 3; ++channel)
                denominator[channel] += direction[3*i+channel]*applied[3*i+channel];
            for (int channel = 0; channel < 3; ++channel) if (live[channel]) {
                if (!(denominator[channel] > 0.) || !std::isfinite(denominator[channel]) ||
                    !(rz[channel] > 0.) || !std::isfinite(rz[channel])) return 5;
                alpha[channel] = rz[channel]/denominator[channel];
                if (!std::isfinite(alpha[channel])) return 4;
                record[12+channel] += 1.;
            }
            for (int i = 0; i < count; ++i) for (int channel = 0; channel < 3; ++channel) {
                const int at = 3*i+channel;
                if (live[channel]) {
                    u[at] += alpha[channel]*direction[at];
                    residual[at] -= alpha[channel]*applied[at];
                    preconditioned[at] = residual[at]/diagonal[i];
                }
                next_rz[channel] += residual[at]*preconditioned[at];
                next_squared[channel] += residual[at]*residual[at];
            }
            double beta[3] = {};
            for (int channel = 0; channel < 3; ++channel) {
                if (!std::isfinite(next_rz[channel]) || !std::isfinite(next_squared[channel])) return 4;
                if (live[channel]) beta[channel] = next_rz[channel]/rz[channel];
                if (!std::isfinite(beta[channel])) return 4;
            }
            for (int i = 0; i < count; ++i) for (int channel = 0; channel < 3; ++channel) {
                const int at = 3*i+channel;
                direction[at] = live[channel] ? preconditioned[at]+beta[channel]*direction[at] : 0.;
                if (!std::isfinite(direction[at])) return 4;
            }
            for (int channel = 0; channel < 3; ++channel) {
                rz[channel] = next_rz[channel]; squared[channel] = next_squared[channel];
            }
        }
        for (int channel = 0; channel < 3; ++channel)
            record[24+channel] = std::sqrt(squared[channel])/norm[channel];
        multiply(row_offsets, neighbor_offsets, row_weights, diagonal, u, applied);
        double solved_squared[3] = {};
        for (int i = 0; i < count; ++i) for (int channel = 0; channel < 3; ++channel) {
            const int at = 3*i+channel;
            const double residual_value = rhs[at]-applied[at];
            solved_squared[channel] += residual_value*residual_value;
        }
        for (int channel = 0; channel < 3; ++channel) {
            record[18+channel] = std::sqrt(solved_squared[channel])/norm[channel];
            record[21+channel] = record[18+channel] > 1e-4 ? 1. : 0.;
            if (!std::isfinite(record[18+channel])) return 4;
        }
        record[1] = seconds(started);

        started = Clock::now();
        for (int i = 0; i < count; ++i) {
            const auto& interval = intervals[group[i]];
            for (int channel = 0; channel < 3; ++channel) {
                const int at = 3*i+channel;
                if (!std::isfinite(u[at])) return 4;
                const double clipped = std::clamp(u[at], interval.lo[channel], interval.hi[channel]);
                if (clipped != u[at]) {
                    record[10] += 1.; record[11] = std::max(record[11], std::abs(clipped-u[at]));
                }
                u[at] = clipped; output[3*pixels[i]+channel] = clipped;
            }
        }
        record[2] = seconds(started);

        started = Clock::now();
        double true_squared[3] = {solved_squared[0], solved_squared[1], solved_squared[2]};
        if (record[10] != 0.) {
            multiply(row_offsets, neighbor_offsets, row_weights, diagonal, u, applied);
            std::fill(true_squared, true_squared+3, 0.);
            for (int i = 0; i < count; ++i) for (int channel = 0; channel < 3; ++channel) {
                const int at = 3*i+channel;
                const double residual_value = rhs[at]-applied[at];
                true_squared[channel] += residual_value*residual_value;
            }
        }
        for (int channel = 0; channel < 3; ++channel) {
            record[28+channel] = std::sqrt(true_squared[channel])/norm[channel];
            if (!std::isfinite(record[28+channel])) return 4;
        }
        double q_after = 0., next_energy = 0.;
        for (const auto& edge : edges) {
            const double s = difference(edge, output);
            q_after += edge.weight*s;
            next_energy += edge.a*64.*std::log1p(s/64.);
        }
        record[5] = q_after; record[7] = next_energy; record[3] = seconds(started);
        if (!std::isfinite(q_after) || !std::isfinite(next_energy)) return 4;
        if (fault == 6 || exceeds(q_after, q_before)) return 7;
        if (fault == 4 || exceeds(next_energy, previous_energy)) return 6;
        previous_energy = next_energy;
        record[27] = 1.;
    }
    summary[9] = previous_energy;
    started = Clock::now();
    auto& quantized = workspace.quantized;
    quantized.assign(image, image+3*area);
    for (const int p : pixels) for (int channel = 0; channel < 3; ++channel)
        quantized[3*p+channel] = static_cast<uint8_t>(
            std::clamp(nearest_even(output[3*p+channel]), 0., 255.));
    summary[10] = quantized_energy(edges, quantized.data());
    summary[11] = seconds(started);
    if (!std::isfinite(summary[10])) return 4;
    return 0;
}
}

extern "C" int robust_solve(const uint8_t* image, const uint8_t* base,
                            const uint8_t* unknown, const double* c,
                            const double* tx, const double* ty, int h, int w,
                            double* float_output, double* summary,
                            double* rounds, int fault) {
    const auto started = Clock::now();
    std::fill(summary, summary+24, 0.);
    std::fill(rounds, rounds+128, 0.);
    int status;
    try {
        status = solve(image, base, unknown, c, tx, ty, h, w, float_output, summary, rounds, fault);
    } catch (const std::bad_alloc&) { status = 1; }
      catch (...) { status = 8; }
    summary[12] = seconds(started);
    return status;
}

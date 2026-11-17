#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {
// Separate lower and upper sums preserve the original profile bit for bit.
// The additional right-hand side shares A's factorization without modifying p0.
struct System {
    int bins;
    std::vector<double> lo, hi, cross, slo, shi, rlo, rhi, qlo, qhi;
    double h = 0., z[3] = {0., 0., 0.};

    explicit System(int n) : bins(n), lo(n), hi(n), cross(n), slo(n), shi(n),
        rlo(n*3), rhi(n*3), qlo(n), qhi(n) {}

    void add(int k, double a, double b, double wt, double tau, const double* color) {
        const double wa=wt*a, wb=wt*b;
        lo[k]+=wa*a; hi[k+1]+=wb*b; cross[k]+=wa*b;
        slo[k]+=wa; shi[k+1]+=wb;
        for (int c=0; c<3; ++c) {
            rlo[k*3+c]+=wa*color[c];
            rhi[(k+1)*3+c]+=wb*color[c];
        }
        qlo[k]+=wa*tau; qhi[k+1]+=wb*tau;
        const double wtau=wt*tau;
        h+=wtau*tau;
        for (int c=0; c<3; ++c) z[c]+=wtau*color[c];
    }

    void solve(double* output) {
        double* p0=output;
        double* support=p0+bins*3;
        double* profile=support+bins;
        double* beta=profile+bins*3;
        std::vector<double> d(bins), off(bins), q(bins), v(bins);
        for (int k=0; k<bins; ++k) {
            d[k]=((lo[k]+hi[k])+2*.06)+1e-5;
            off[k]=cross[k]-.06;
            support[k]=slo[k]+shi[k];
            for (int c=0; c<3; ++c) p0[k*3+c]=rlo[k*3+c]+rhi[k*3+c];
            q[k]=qlo[k]+qhi[k];
            v[k]=q[k];
        }
        for (int k=1; k<bins; ++k) {
            const double factor=off[k-1]/d[k-1];
            d[k]-=factor*off[k-1];
            for (int c=0; c<3; ++c) p0[k*3+c]-=factor*p0[(k-1)*3+c];
            v[k]-=factor*v[k-1];
        }
        for (int c=0; c<3; ++c) p0[(bins-1)*3+c]/=d[bins-1];
        v[bins-1]/=d[bins-1];
        for (int k=bins-2; k>=0; --k) {
            for (int c=0; c<3; ++c)
                p0[k*3+c]=(p0[k*3+c]-off[k]*p0[(k+1)*3+c])/d[k];
            v[k]=(v[k]-off[k]*v[k+1])/d[k];
        }
        double qv=0., qp[3]={0.,0.,0.};
        for (int k=0; k<bins; ++k) {
            qv+=q[k]*v[k];
            for (int c=0; c<3; ++c) qp[c]+=q[k]*p0[k*3+c];
        }
        const double regularized_h=h+1e-5;
        const double schur=regularized_h-qv;
        const double ratio=schur/regularized_h;
        bool valid=std::isfinite(ratio) && ratio>=.02;
        for (int c=0; c<3; ++c) {
            beta[c]=(z[c]-qp[c])/schur;
            valid=valid && std::isfinite(beta[c]) && std::abs(beta[c])<=12.;
        }
        for (int k=0; k<bins; ++k) {
            valid=valid && std::isfinite(support[k]);
            for (int c=0; c<3; ++c) {
                profile[k*3+c]=p0[k*3+c]-v[k]*beta[c];
                valid=valid && std::isfinite(p0[k*3+c]) && std::isfinite(profile[k*3+c]);
            }
        }
        beta[3]=ratio;
        beta[4]=valid ? 1. : 0.;
    }
};
}

extern "C" void transport_profile(const double* u, const double* colors,
    const double* weights, const double* tau, int64_t size, int bins, double* output) {
    System system(bins);
    for (int64_t i=0; i<size; ++i) {
        const int k=static_cast<int>(std::floor(u[i]));
        const double wt=weights[i];
        if (k<0 || k+1>=bins || !(wt>0)) continue;
        const double b=u[i]-k, a=1-b;
        system.add(k,a,b,wt,tau[i],colors+i*3);
    }
    system.solve(output);
}

extern "C" void transport_paired(const double* u, const double* colors,
    const double* weights, const double* tangent, int64_t size, int bins,
    double cutoff, double radius, double* output) {
    System left(bins), right(bins);
    for (int64_t i=0; i<size; ++i) {
        // Each side sees precisely its original row-major accumulation order.
        System* side=tangent[i]<-cutoff ? &left : (tangent[i]>cutoff ? &right : nullptr);
        if (!side) continue;
        const int k=static_cast<int>(std::floor(u[i]));
        const double wt=weights[i];
        if (k<0 || k+1>=bins || !(wt>0)) continue;
        const double b=u[i]-k, a=1-b;
        side->add(k,a,b,wt,tangent[i]/radius,colors+i*3);
    }
    left.solve(output);
    right.solve(output+bins*7+5);
}

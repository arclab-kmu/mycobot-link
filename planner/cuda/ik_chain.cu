// ik_chain.cu: the planner's per-candidate IK chains on the GPU, one thread per candidate.
//
// Same algorithm as plan77.ik / plan77.ik_walk (kin.py DH model, degrees and mm, damped least squares on
// [mm, 100 * rad], damping 1e-2, tol 1e-3, 200 iterations; the walk splits the move into <= step mm / <= 6 deg
// steps). The CPU code differentiates its error numerically (seven FK calls per iteration); here the same
// derivative is taken analytically from one FK.
// A thread walks its candidate's chain of target poses in order: the first pose is reached by a walk from its
// seed, each later one by IK from the previous solution, and a pose whose error stays above `fail_e` gets one
// fallback walk (from its seed, or from the previous solution when `fb_from_prev`); a pose that still fails ends
// the chain (bad[c] = k), as the CPU loop breaks.
//
// Why a hand-written kernel: on the AGX Orin (2026-11-26) the same batched solve written as torch ops was slower
// than numpy (55.6 s vs 25.6 s for 150 walks): each iteration launched ~100 tiny kernels. Here a whole chain is
// one thread's loop and the whole candidate set is one launch.
//
// Precision: `real` is float unless built with -DIK_FP64. The Orin's GPU runs FP64 at a small fraction of its
// FP32 rate, and the IK tolerance (1e-3 on mm / 100*rad) is well above float rounding at arm scale (~1e-4).
//
// Build (AGX Orin, sm_87): planner/cuda/build.sh -> planner/cuda/libikchain.so, loaded by planner/ik_cuda.py.

#include <cuda_runtime.h>
#include <math.h>

#ifdef IK_FP64
typedef double real;
__device__ inline void sc(real x, real* s, real* c) { sincos(x, s, c); }
#else
typedef float real;
__device__ inline void sc(real x, real* s, real* c) { sincosf(x, s, c); }
#endif
#define R_(x) ((real)(x))

__constant__ real DH[24];  // per joint: theta_offset, d, a, alpha

__device__ const real DEG = R_(3.14159265358979323846 / 180.0);

// FK; also each joint's axis (z of the frame before it) and origin when z / o are given.
__device__ void fk(const real* q, real* T, real* z = 0, real* o = 0) {
    real M[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    for (int i = 0; i < 6; i++) {
        if (z) {
            z[3 * i] = M[2]; z[3 * i + 1] = M[6]; z[3 * i + 2] = M[10];
            o[3 * i] = M[3]; o[3 * i + 1] = M[7]; o[3 * i + 2] = M[11];
        }
        const real th = q[i] * DEG + DH[4 * i];
        const real d = DH[4 * i + 1], a = DH[4 * i + 2], al = DH[4 * i + 3];
        real st, ct, sa, ca;
        sc(th, &st, &ct);
        sc(al, &sa, &ca);
        const real A[16] = {ct, -st * ca, st * sa, a * ct, st, ct * ca, -ct * sa, a * st, 0, sa, ca, d, 0, 0, 0, 1};
        real Rm[16];
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++) {
                real s = 0;
                for (int k = 0; k < 4; k++) s += M[4 * r + k] * A[4 * k + c];
                Rm[4 * r + c] = s;
            }
        for (int j = 0; j < 16; j++) M[j] = Rm[j];
    }
    for (int j = 0; j < 16; j++) T[j] = M[j];
}

// M = Tg_R T_R^T (3x3)
__device__ void rel(const real* T, const real* Tg, real* M) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            real s = 0;
            for (int k = 0; k < 3; k++) s += Tg[4 * i + k] * T[4 * j + k];
            M[3 * i + j] = s;
        }
}

// plan77._err: [Tg_p - T_p, 0.5 * vee(M) * 100]
__device__ void err(const real* T, const real* Tg, real* e) {
    real M[9];
    rel(T, Tg, M);
    e[0] = Tg[3] - T[3];
    e[1] = Tg[7] - T[7];
    e[2] = Tg[11] - T[11];
    e[3] = R_(50.0) * (M[7] - M[5]);
    e[4] = R_(50.0) * (M[2] - M[6]);
    e[5] = R_(50.0) * (M[3] - M[1]);
}

__device__ real norm6(const real* e) {
    real s = 0;
    for (int i = 0; i < 6; i++) s += e[i] * e[i];
    return sqrt(s);
}

// Solve A x = b (6x6), Gaussian elimination with partial pivoting.
__device__ void solve6(real* A, real* b, real* x) {
    for (int k = 0; k < 6; k++) {
        int p = k;
        for (int i = k + 1; i < 6; i++)
            if (fabs(A[6 * i + k]) > fabs(A[6 * p + k])) p = i;
        if (p != k) {
            for (int j = 0; j < 6; j++) {
                const real t = A[6 * k + j];
                A[6 * k + j] = A[6 * p + j];
                A[6 * p + j] = t;
            }
            const real t = b[k];
            b[k] = b[p];
            b[p] = t;
        }
        for (int i = k + 1; i < 6; i++) {
            const real f = A[6 * i + k] / A[6 * k + k];
            for (int j = k; j < 6; j++) A[6 * i + j] -= f * A[6 * k + j];
            b[i] -= f * b[k];
        }
    }
    for (int i = 5; i >= 0; i--) {
        real s = b[i];
        for (int j = i + 1; j < 6; j++) s -= A[6 * i + j] * x[j];
        x[i] = s / A[6 * i + i];
    }
}

// plan77.ik: q in/out (deg); returns the final error norm. Column i of J = d(-e)/dq_i per degree:
// position z_i x (p - o_i); rotation 50 * vee(M K_i) with M = Tg_R T_R^T and K_i = [z_i]x.
__device__ real ik(real* q, const real* Tg) {
    real T[16], e[6], J[36], A[36], b[6], dq[6], z[18], o[18], M[9];
    for (int it = 0; it < 200; it++) {
        fk(q, T, z, o);
        err(T, Tg, e);
        if (norm6(e) < R_(1e-3)) break;
        rel(T, Tg, M);
        for (int i = 0; i < 6; i++) {
            const real* zi = z + 3 * i;
            const real r0 = T[3] - o[3 * i], r1 = T[7] - o[3 * i + 1], r2 = T[11] - o[3 * i + 2];
            J[0 + i] = DEG * (zi[1] * r2 - zi[2] * r1);
            J[6 + i] = DEG * (zi[2] * r0 - zi[0] * r2);
            J[12 + i] = DEG * (zi[0] * r1 - zi[1] * r0);
            const real K[9] = {0, -zi[2], zi[1], zi[2], 0, -zi[0], -zi[1], zi[0], 0};
            real MK[9];
            for (int a = 0; a < 3; a++)
                for (int c = 0; c < 3; c++) {
                    real s = 0;
                    for (int m = 0; m < 3; m++) s += M[3 * a + m] * K[3 * m + c];
                    MK[3 * a + c] = s;
                }
            J[18 + i] = R_(50.0) * DEG * (MK[7] - MK[5]);
            J[24 + i] = R_(50.0) * DEG * (MK[2] - MK[6]);
            J[30 + i] = R_(50.0) * DEG * (MK[3] - MK[1]);
        }
        for (int i = 0; i < 6; i++) {
            real s = 0;
            for (int r = 0; r < 6; r++) s += J[6 * r + i] * e[r];
            b[i] = s;
            for (int j = 0; j < 6; j++) {
                real t = 0;
                for (int r = 0; r < 6; r++) t += J[6 * r + i] * J[6 * r + j];
                A[6 * i + j] = t + (i == j ? R_(1e-2) : R_(0.0));
            }
        }
        solve6(A, b, dq);
        for (int j = 0; j < 6; j++) q[j] += dq[j];
    }
    fk(q, T);
    err(T, Tg, e);
    return norm6(e);
}

// plan77.ik_walk: from fk(q0) to Tg in steps; q out; returns the last error norm.
__device__ real ik_walk(const real* q0, const real* Tg, real step_mm, real* q) {
    real T0[16], Rr[9];
    fk(q0, T0);
    const real dp[3] = {Tg[3] - T0[3], Tg[7] - T0[7], Tg[11] - T0[11]};
    rel(T0, Tg, Rr);
    const real tr = Rr[0] + Rr[4] + Rr[8];
    const real ang = acos(fmin(R_(1.0), fmax(R_(-1.0), (tr - R_(1.0)) / R_(2.0)))) / DEG;
    const real dn = sqrt(dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2]);
    int n = (int)ceil(fmax(dn / step_mm, ang / R_(6.0)));
    if (n < 1) n = 1;
    real ax[3] = {Rr[7] - Rr[5], Rr[2] - Rr[6], Rr[3] - Rr[1]};
    const real an = sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]);
    if (an > R_(1e-9)) {
        ax[0] /= an; ax[1] /= an; ax[2] /= an;
    } else {
        ax[0] = 0; ax[1] = 0; ax[2] = 1;
    }
    for (int j = 0; j < 6; j++) q[j] = q0[j];
    real e = 0;
    const real K[9] = {0, -ax[2], ax[1], ax[2], 0, -ax[0], -ax[1], ax[0], 0};
    real KK[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            real s = 0;
            for (int m = 0; m < 3; m++) s += K[3 * i + m] * K[3 * m + j];
            KK[3 * i + j] = s;
        }
    for (int k = 1; k <= n; k++) {
        const real f = (real)k / (real)n;
        real s, c;
        sc(f * ang * DEG, &s, &c);
        real Rk[9];
        for (int i = 0; i < 9; i++) Rk[i] = (i % 4 == 0 ? R_(1.0) : R_(0.0)) + s * K[i] + (R_(1.0) - c) * KK[i];
        real Tk[16] = {0, 0, 0, T0[3] + f * dp[0], 0, 0, 0, T0[7] + f * dp[1], 0, 0, 0, T0[11] + f * dp[2], 0, 0, 0, 1};
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                real v = 0;
                for (int m = 0; m < 3; m++) v += Rk[3 * i + m] * T0[4 * m + j];
                Tk[4 * i + j] = v;
            }
        e = ik(q, Tk);
    }
    return e;
}

__global__ void chain_kernel(const double* T, const double* seed, int C, int K, real step0, real step_fb,
                             int fb_from_prev, real fail_e, double* qout, double* eout, int* bad) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    real prev[6], q[6], Tg[16], sd[6], e;
    int have_prev = 0;
    bad[c] = -1;
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < 16; j++) Tg[j] = (real)T[(size_t)(c * K + k) * 16 + j];
        for (int j = 0; j < 6; j++) sd[j] = (real)seed[(size_t)(c * K + k) * 6 + j];
        if (!have_prev) {
            e = ik_walk(sd, Tg, step0, q);
        } else {
            for (int j = 0; j < 6; j++) q[j] = prev[j];
            e = ik(q, Tg);
        }
        if (e > fail_e) e = ik_walk((fb_from_prev && have_prev) ? prev : sd, Tg, step_fb, q);
        for (int j = 0; j < 6; j++) qout[(size_t)(c * K + k) * 6 + j] = (double)q[j];
        eout[c * K + k] = (double)e;
        if (e > fail_e) {
            bad[c] = k;
            return;
        }
        for (int j = 0; j < 6; j++) prev[j] = q[j];
        have_prev = 1;
    }
}

extern "C" int ik_chain_fp64(void) {
#ifdef IK_FP64
    return 1;
#else
    return 0;
#endif
}

extern "C" int ik_chain(const double* T, const double* seed, int C, int K, double step0, double step_fb,
                        int fb_from_prev, double fail_e, const double* dh, double* qout, double* eout, int* bad) {
    real dhr[24];
    for (int i = 0; i < 24; i++) dhr[i] = (real)dh[i];
    cudaError_t rc = cudaMemcpyToSymbol(DH, dhr, sizeof(dhr));
    if (rc != cudaSuccess) return (int)rc;
    double *dT = 0, *dS = 0, *dQ = 0, *dE = 0;
    int* dB = 0;
    const size_t nT = (size_t)C * K * 16, nS = (size_t)C * K * 6, nE = (size_t)C * K;
    if ((rc = cudaMalloc(&dT, nT * sizeof(double))) != cudaSuccess) goto done;
    if ((rc = cudaMalloc(&dS, nS * sizeof(double))) != cudaSuccess) goto done;
    if ((rc = cudaMalloc(&dQ, nS * sizeof(double))) != cudaSuccess) goto done;
    if ((rc = cudaMalloc(&dE, nE * sizeof(double))) != cudaSuccess) goto done;
    if ((rc = cudaMalloc(&dB, C * sizeof(int))) != cudaSuccess) goto done;
    cudaMemcpy(dT, T, nT * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dS, seed, nS * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(dQ, 0, nS * sizeof(double));
    cudaMemset(dE, 0, nE * sizeof(double));
    chain_kernel<<<(C + 31) / 32, 32>>>(dT, dS, C, K, (real)step0, (real)step_fb, fb_from_prev, (real)fail_e, dQ, dE, dB);
    if ((rc = cudaGetLastError()) != cudaSuccess) goto done;
    if ((rc = cudaDeviceSynchronize()) != cudaSuccess) goto done;
    cudaMemcpy(qout, dQ, nS * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(eout, dE, nE * sizeof(double), cudaMemcpyDeviceToHost);
    rc = cudaMemcpy(bad, dB, C * sizeof(int), cudaMemcpyDeviceToHost);
done:
    cudaFree(dT);
    cudaFree(dS);
    cudaFree(dQ);
    cudaFree(dE);
    cudaFree(dB);
    return (int)rc;
}

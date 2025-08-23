#include "winograd.cuh"
#include <cublas_v2.h>
#include <cassert>

////////////////////////////////////////////////////////////////////////////////
// Constants: F(2x2, 3x3)
__constant__ float G[4][3] = {
    {1.0f, 0.0f, 0.0f},
    {0.5f, 0.5f, 0.5f},
    {0.5f, -0.5f, 0.5f},
    {0.0f, 0.0f, 1.0f}
};

__constant__ float B_T[4][4] = {
    {1.0f, 0.0f, -1.0f, 0.0f},
    {0.0f, 1.0f,  1.0f, 0.0f},
    {0.0f,-1.0f,  1.0f, 0.0f},
    {0.0f, 1.0f,  0.0f,-1.0f}
};

__constant__ float B_[4][4] = {
    { 1.0f,  0.0f,  0.0f,  0.0f},
    { 0.0f,  1.0f, -1.0f,  1.0f},
    {-1.0f,  1.0f,  1.0f,  0.0f},
    { 0.0f,  0.0f,  0.0f, -1.0f}
};

__constant__ float A_T[2][4] = {
    {1.0f, 1.0f,  1.0f,  0.0f},
    {0.0f, 1.0f, -1.0f, -1.0f}
};

////////////////////////////////////////////////////////////////////////////////
// Helpers: indexers for column-major faces (t = i*4 + j)
__device__ __host__ inline int idx_U(int t, int k, int c, int K, int C) {
    // U_face is (K x C) column-major: elem(r=k, col=c) at k + c*K
    return t * (K * C) + c * K + k;
}
__device__ __host__ inline int idx_V(int t, int c, long b, int C, long B) {
    // V_face is (C x B) column-major: elem(r=c, col=b) at c + b*C
    return t * (C * B) + b * C + c;
}
__device__ __host__ inline int idx_M(int t, int k, long b, int K, long B) {
    // M_face is (K x B) column-major: elem(r=k, col=b) at k + b*K
    return t * (K * B) + b * K + k;
}

////////////////////////////////////////////////////////////////////////////////
// 1) Filter transform: U = G g G^T  (for all k,c)
__global__ void filter_transform_kernel(const float* __restrict__ filter,  // [K*C*3*3]
                                        float* __restrict__ U,             // [16 * K * C] faces, col-major per face
                                        int K, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * C;
    if (idx >= total) return;

    int c = idx % C;
    int k = idx / C;

    const float* g = filter + (k * C + c) * 9; // [3x3], row-major

    // temp_g = G * g (4x3)
    float temp_g[4][3];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            temp_g[i][j] = G[i][0] * g[0 * 3 + j]
                         + G[i][1] * g[1 * 3 + j]
                         + G[i][2] * g[2 * 3 + j];
        }
    }

    // u = temp_g * G^T (4x4)
    float u[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        // Column 0,3 are straight copies from temp_g[][0 or 2] by the given G
        u[i][0] = temp_g[i][0];
        u[i][1] = 0.5f * (temp_g[i][0] + temp_g[i][1] + temp_g[i][2]);
        u[i][2] = 0.5f * (temp_g[i][0] - temp_g[i][1] + temp_g[i][2]);
        u[i][3] = temp_g[i][2];
    }

    // Scatter to 16 faces, each face is a scalar at (xi,nu) for all (k,c)
#pragma unroll
    for (int xi = 0; xi < 4; ++xi) {
#pragma unroll
        for (int nu = 0; nu < 4; ++nu) {
            int t = xi * 4 + nu;
            U[idx_U(t, k, c, K, C)] = u[xi][nu];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// 2) Input transform: V = B^T d B  (for all n,c,tiles)
__global__ void input_transform_kernel(const float* __restrict__ image, // [N*C*H*W]
                                       float* __restrict__ V,           // [16 * C * B] faces, col-major per face
                                       int N, int C, int H, int W,
                                       int outH, int outW) {
    const int tileH = outH / 2;
    const int tileW = outW / 2;
    const long Btiles = (long)N * tileH * tileW; // B dimension in GEMM

    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)C * Btiles;
    if (idx >= total) return;

    int c = (int)(idx % C);
    long b = idx / C; // 0..Btiles-1

    // Decode b -> (n, tile_y, tile_x)
    int n   = (int)(b / (tileH * tileW));
    int rem = (int)(b % (tileH * tileW));
    int tile_y = rem / tileW;
    int tile_x = rem % tileW;

    int h_start = tile_y * 2;
    int w_start = tile_x * 2;

    // Load 4x4 patch d
    float d[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int h = h_start + i;
            int w = w_start + j;
            d[i][j] = image[((n * C + c) * H + h) * W + w];
        }
    }

    // temp = B_T * d   (4x4)
    float tmp[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            tmp[i][j] = B_T[i][0] * d[0][j]
                      + B_T[i][1] * d[1][j]
                      + B_T[i][2] * d[2][j]
                      + B_T[i][3] * d[3][j];
        }
    }

    // v = tmp * B   (4x4)
    float v[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[i][j] = tmp[i][0] * B_[0][j]
                    + tmp[i][1] * B_[1][j]
                    + tmp[i][2] * B_[2][j]
                    + tmp[i][3] * B_[3][j];
        }
    }

    // Scatter to 16 faces: V_face is (C x B) column-major → elem(c,b)
#pragma unroll
    for (int xi = 0; xi < 4; ++xi) {
#pragma unroll
        for (int nu = 0; nu < 4; ++nu) {
            int t = xi * 4 + nu;
            V[idx_V(t, c, b, C, Btiles)] = v[xi][nu];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// 4) Inverse transform: Y = A^T M A  (for all k,tiles)
__global__ void inverse_transform_kernel(const float* __restrict__ M,   // [16 * K * B], col-major per face
                                         float* __restrict__ out,       // [N*K*outH*outW]
                                         int N, int K, int H, int W,    // H,W are input sizes (for indexing), not used directly
                                         int outH, int outW) {
    const int tileH = outH / 2;
    const int tileW = outW / 2;
    const long Btiles = (long)N * tileH * tileW;

    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)K * Btiles;
    if (idx >= total) return;

    int k = (int)(idx % K);
    long b = idx / K;

    // Decode b -> (n, tile_y, tile_x)
    int n   = (int)(b / (tileH * tileW));
    int rem = (int)(b % (tileH * tileW));
    int tile_y = rem / tileW;
    int tile_x = rem % tileW;

    // Gather 16 faces into m[4][4] (each is scalar at (xi,nu))
    float m[4][4];
#pragma unroll
    for (int xi = 0; xi < 4; ++xi) {
#pragma unroll
        for (int nu = 0; nu < 4; ++nu) {
            int t = xi * 4 + nu;
            m[xi][nu] = M[idx_M(t, k, b, K, Btiles)];
        }
    }

    // temp = A_T * m   (2x4)
    float tmp[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            tmp[i][j] = A_T[i][0] * m[0][j]
                      + A_T[i][1] * m[1][j]
                      + A_T[i][2] * m[2][j]
                      + A_T[i][3] * m[3][j];
        }
    }

    // y = tmp * A   (2x2) ; A (implicit) per F(2x2) gives:
    float Y00 = tmp[0][0] + tmp[0][1] + tmp[0][2];
    float Y01 = tmp[0][1] - tmp[0][2] - tmp[0][3];
    float Y10 = tmp[1][0] + tmp[1][1] + tmp[1][2];
    float Y11 = tmp[1][1] - tmp[1][2] - tmp[1][3];

    // Write back to output
    int h0 = tile_y * 2;
    int w0 = tile_x * 2;
    // out layout: [(n*K + k) * outH + h] * outW + w
    out[((n * K + k) * outH + (h0 + 0)) * outW + (w0 + 0)] = Y00;
    out[((n * K + k) * outH + (h0 + 0)) * outW + (w0 + 1)] = Y01;
    out[((n * K + k) * outH + (h0 + 1)) * outW + (w0 + 0)] = Y10;
    out[((n * K + k) * outH + (h0 + 1)) * outW + (w0 + 1)] = Y11;
}

////////////////////////////////////////////////////////////////////////////////
// Top-level: Orchestrate  U/V transforms -> batched GEMM -> inverse transform
void winograd_conv(thrust::device_vector<float>& image,
                   thrust::device_vector<float>& filter,
                   thrust::device_vector<float>& out,
                   thrust::device_vector<float>& U,
                   thrust::device_vector<float>& V,
                   thrust::device_vector<float>& M,
                   int H, int W, int C, int K, int N) {
    const int outH = H - 2;
    const int outW = W - 2;
    const int tileH = outH / 2;
    const int tileW = outW / 2;
    const long Btiles = (long)N * tileH * tileW; // B dimension in GEMM
    const int faces = 16;

    // 1) Filter transform
    {
        int threads = 256;
        int blocks = (K * C + threads - 1) / threads;
        filter_transform_kernel<<<blocks, threads>>>(
            filter.data().get(),
            U.data().get(),
            K, C
        );
    }

    // 2) Input transform
    {
        int threads = 256;
        long work = (long)C * Btiles;
        int blocks = (int)((work + threads - 1) / threads);
        input_transform_kernel<<<blocks, threads>>>(
            image.data().get(),
            V.data().get(),
            N, C, H, W,
            outH, outW
        );
    }

    // 3) 16-face batched GEMM: M^{(t)} = U^{(t)}(KxC) * V^{(t)}(CxB) -> (KxB)
    //    All matrices are stored column-major per face.
    {
        cublasHandle_t handle;
        cublasCreate(&handle);

        const float alpha = 1.0f, beta = 0.0f;

        // Per-face pointers (as a strided batch)
        // A (U): K x C, lda = K, strideA = K*C
        // B (V): C x B, ldb = C, strideB = C*B
        // C (M): K x B, ldc = K, strideC = K*B
        int m = K;          // rows of C
        long n = Btiles;    // cols of C
        int k = C;          // inner dim

        int lda = K;
        int ldb = C;
        int ldc = K;

        long strideA = (long)K * C;
        long strideB = (long)C * Btiles;
        long strideC = (long)K * Btiles;

        // cublas uses int for sizes; n could be large but still fits int for typical cases
        // If your max Btiles might exceed INT_MAX, you'd need cublasLt with 64-bit strides.
        // Here we assume config fits int.
        cublasStatus_t st = cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, (int)n, k,
            &alpha,
            U.data().get(), lda, strideA,
            V.data().get(), ldb, strideB,
            &beta,
            M.data().get(), ldc, strideC,
            faces
        );
        assert(st == CUBLAS_STATUS_SUCCESS);
        cublasDestroy(handle);
    }

    // 4) Inverse transform & writeback
    {
        int threads = 256;
        long work = (long)K * Btiles;
        int blocks = (int)((work + threads - 1) / threads);
        inverse_transform_kernel<<<blocks, threads>>>(
            M.data().get(),
            out.data().get(),
            N, K, H, W, outH, outW
        );
    }

    cudaDeviceSynchronize();
}
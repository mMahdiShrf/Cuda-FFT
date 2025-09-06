// ONLY MODIFY THIS FILE!
// YOU CAN MODIFY EVERYTHING IN THIS FILE!
// This code created by Mohammad H Najafi in June 2023

#include "fft.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

__global__ void rearange_radix2(float* x_r_d, float* x_i_d, const unsigned int N, unsigned int M);
__global__ void fft_kernel_radix2(float* x_r_d, float* x_i_d, const unsigned int N, unsigned int M);
__device__ __inline__ unsigned int digit_reverse_radix2(unsigned int thrds, unsigned int M);
__device__ __inline__ void twiddle_radix2(const unsigned int M, const unsigned int N, const unsigned int idx, float* w);
__device__ __inline__ void computeButterfly_radix2(const float* w, float x_i_d[], float x_r_d[], unsigned int index[]);

__global__ void rearange_radix4(float* x_r_d, float* x_i_d, const unsigned int N, unsigned int M);
__global__ void fft_kernel_radix4(float* x_r_d, float* x_i_d, const unsigned int N, const unsigned int M);
__device__ __inline__ unsigned int digit_reverse_radix4(unsigned int thrds, unsigned int M);
__device__ __inline__ void twiddle_radix4(const unsigned int M, const unsigned int idx, float* w_r);
__device__ __inline__ void computeButterfly_radix4(const float* w,
    float x_i_d[],
    float x_r_d[],
    const unsigned int index[]);

//-----------------------------------------------------------------------------
// This is the main function that performs the FFT on the input signal using the specified radix.
void gpuKernel(float* x_r_d, float* x_i_d, /*float* X_r_d, float* X_i_d,*/ const unsigned int N, const unsigned int M)
{
    // Perform bit-reversal permutation for the radix-2 or radix-4 algorithm
    if ((M % 2 == 1)) {
        // Perform the radix-2 FFT
        rearange_radix2<<<dim3(N / (1024 * 512), 512, 1), 1024>>>(x_r_d, x_i_d, N, M);
        for (int k = 1; k < N; k *= 2)
            fft_kernel_radix2<<<dim3(N / (1024 * 512), 32, 32), 256>>>(x_r_d, x_i_d, N, k);

    } else {
        // Perform the radix-4 FFT
        rearange_radix4<<<dim3(N / (512 * 512), 32, 32), 256>>>(x_r_d, x_i_d, N, M);

        for (int k = 1; k < N; k *= 4)
            fft_kernel_radix4<<<dim3(N / (1024 * 1024), 32, 32), 256>>>(x_r_d, x_i_d, N, k);
    }
}

// This kernel performs the bit-reversal permutation necessary for the radix-2 algorithm.
__global__ void rearange_radix2(float* x_r_d, float* x_i_d, const unsigned int N, unsigned int M)
{
    // Compute global thread index in 1D
    int idx = (gridDim.x * gridDim.y * blockIdx.z
                  + gridDim.x * blockIdx.y + blockIdx.x)
            * blockDim.x
        + threadIdx.x;

    // Make sure we don't go out-of-bounds
    if (idx >= (int)N) {
        return;
    }

    // Use the M-bit bit-reversal function
    unsigned int reversed = digit_reverse_radix2(idx, M);

    if (idx < (int)reversed) {
        float R_temp[2];
        float I_temp[2];

        // Swap elements in the input signal
        I_temp[0] = x_i_d[reversed];
        R_temp[0] = x_r_d[reversed];
        I_temp[1] = x_i_d[idx];
        R_temp[1] = x_r_d[idx];

        x_i_d[idx] = I_temp[0];
        x_r_d[idx] = R_temp[0];
        x_i_d[reversed] = I_temp[1];
        x_r_d[reversed] = R_temp[1];
    }
}

__device__ __inline__ unsigned int digit_reverse_radix2(unsigned int idx, unsigned int M)
{
    unsigned int reversed = 0;
    // Only reverse the lower M bits
    for (unsigned int k = 0; k < M; k++) {
        reversed <<= 1;
        reversed |= (idx & 1);
        idx >>= 1;
    }
    return reversed;
}

// This kernel performs the butterfly operations for each stage of the radix-2 FFT.
__global__ void fft_kernel_radix2(float* x_r_d, float* x_i_d, const unsigned int N, unsigned int M)
{
    int idx = (gridDim.x * gridDim.y * bz + gridDim.x * by + bx) * blockDim.x + tx;

    unsigned int index[2];

    index[0] = idx + (idx / M) * M;
    index[1] = idx + (idx / M) * M + M;

    // Compute the angle for the butterfly operation
    float w[2];
    twiddle_radix2(M, N, idx, w);
    computeButterfly_radix2(w, x_i_d, x_r_d, index);
}
__device__ __inline__ void twiddle_radix2(const unsigned int M, const unsigned int N, const unsigned int idx, float* w)
{
    const float angle = -2 * PI * ((N / (M * 2)) * idx - (N / 2) * (idx / M)) / N;
    w[0] = cos(angle);
    w[1] = sin(angle);
}

__device__ __inline__ void computeButterfly_radix2(const float* w, float x_i_d[], float x_r_d[], unsigned int index[])
{

    float R_temp[2] { x_r_d[index[0]], x_r_d[index[1]] };
    float I_temp[2] { x_i_d[index[0]], x_i_d[index[1]] };
    x_i_d[index[0]] = I_temp[0] + (R_temp[1] * w[1]) + (I_temp[1] * w[0]);
    x_i_d[index[1]] = I_temp[0] - (R_temp[1] * w[1]) - (I_temp[1] * w[0]);
    x_r_d[index[0]] = R_temp[0] + (R_temp[1] * w[0]) - (I_temp[1] * w[1]);
    x_r_d[index[1]] = R_temp[0] - (R_temp[1] * w[0]) + (I_temp[1] * w[1]);
}

// This kernel performs the bit-reversal permutation necessary for the radix-4 algorithm.
__global__ void rearange_radix4(float* x_r_d, float* x_i_d,
    const unsigned int N, unsigned int M)
{
    int thrds = (gridDim.x * gridDim.y * bz + gridDim.x * by + bx) * blockDim.x + tx;
    if (thrds >= (int)N) {
        return; // boundary check
    }

    // Use our new base-4 digit-reversal function
    unsigned int reversed = digit_reverse_radix4(thrds, M);

    // If thrds < ic, swap
    if (thrds < (int)reversed) {
        float R_temp[2];
        float I_temp[2];

        I_temp[0] = x_i_d[reversed];
        R_temp[0] = x_r_d[reversed];
        I_temp[1] = x_i_d[thrds];
        R_temp[1] = x_r_d[thrds];

        x_i_d[thrds] = I_temp[0];
        x_r_d[thrds] = R_temp[0];
        x_i_d[reversed] = I_temp[1];
        x_r_d[reversed] = R_temp[1];
    }
}

__device__ __inline__ unsigned int digit_reverse_radix4(unsigned int thrds, unsigned int M)
{
    // Each base-4 digit is 2 bits, so we have M/2 base-4 digits.
    // We'll collect those digits in 'reversed' in reverse order.
    unsigned int reversed = 0;
    // Number of base-4 digits
    unsigned int half_digits = M >> 1; // M/2

    for (unsigned int i = 0; i < half_digits; i++) {
        reversed <<= 2; // make room for 2 bits
        reversed |= (thrds & 0x3); // read 2 bits from thrds
        thrds >>= 2; // move to the next base-4 digit
    }
    return reversed;
}

// This kernel performs the butterfly operations for each stage of the radix-4 FFT.
__global__ void fft_kernel_radix4(float* x_r_d, float* x_i_d, const unsigned int N, const unsigned int M)
{
    int thrds = (gridDim.x * gridDim.y * bz + gridDim.x * by + bx) * blockDim.x + tx;

    unsigned int index[4];
    index[0] = (thrds / M) * 4 * M + thrds % M + 0 * M;
    index[1] = (thrds / M) * 4 * M + thrds % M + 1 * M;
    index[2] = (thrds / M) * 4 * M + thrds % M + 2 * M;
    index[3] = (thrds / M) * 4 * M + thrds % M + 3 * M;

    // Compute the angle for the butterfly operations
    float w[6];
    twiddle_radix4(M, thrds, w);
    computeButterfly_radix4(w, x_i_d, x_r_d, index);
}

__device__ __inline__ void twiddle_radix4(const unsigned int M, const unsigned int idx, float* w)
{
    const float angle = -2 * PI * (idx % M) / (4 * M);
    w[0] = cos(angle);
    w[1] = cos(angle * 2);
    w[2] = cos(angle * 3);
    w[3] = sin(angle);
    w[4] = sin(angle * 2);
    w[5] = sin(angle * 3);
}
__device__ __inline__ void computeButterfly_radix4(const float* w,
    float x_i_d[],
    float x_r_d[],
    const unsigned int index[])
{
    // Temporary arrays for real and imaginary parts
    float R_temp[4] { x_r_d[index[0]], x_r_d[index[1]],
        x_r_d[index[2]], x_r_d[index[3]] };
    float I_temp[4] { x_i_d[index[0]], x_i_d[index[1]],
        x_i_d[index[2]], x_i_d[index[3]] };

    // Compute x_i_d[...] and x_r_d[...] in one step for each index

    // index[0]
    x_i_d[index[0]] = (I_temp[0])
        + (R_temp[1] * w[3] + I_temp[1] * w[0])
        + (R_temp[2] * w[4] + I_temp[2] * w[1])
        + (R_temp[3] * w[5] + I_temp[3] * w[2]);

    x_r_d[index[0]] = (R_temp[0])
        + (R_temp[1] * w[0] - I_temp[1] * w[3])
        + (R_temp[2] * w[1] - I_temp[2] * w[4])
        + (R_temp[3] * w[2] - I_temp[3] * w[5]);

    // index[1]
    x_i_d[index[1]] = (I_temp[0])
        - (R_temp[1] * w[0] - I_temp[1] * w[3])
        - (R_temp[2] * w[4] + I_temp[2] * w[1])
        + (R_temp[3] * w[2] - I_temp[3] * w[5]);

    x_r_d[index[1]] = (R_temp[0])
        + (R_temp[1] * w[3] + I_temp[1] * w[0])
        - (R_temp[2] * w[1] - I_temp[2] * w[4])
        - (R_temp[3] * w[5] + I_temp[3] * w[2]);

    // index[2]
    x_i_d[index[2]] = (I_temp[0])
        - (R_temp[1] * w[3] + I_temp[1] * w[0])
        + (R_temp[2] * w[4] + I_temp[2] * w[1])
        - (R_temp[3] * w[5] + I_temp[3] * w[2]);

    x_r_d[index[2]] = (R_temp[0])
        - (R_temp[1] * w[0] - I_temp[1] * w[3])
        + (R_temp[2] * w[1] - I_temp[2] * w[4])
        - (R_temp[3] * w[2] - I_temp[3] * w[5]);

    // index[3]
    x_i_d[index[3]] = (I_temp[0])
        + (R_temp[1] * w[0] - I_temp[1] * w[3])
        - (R_temp[2] * w[4] + I_temp[2] * w[1])
        - (R_temp[3] * w[2] - I_temp[3] * w[5]);

    x_r_d[index[3]] = (R_temp[0])
        - (R_temp[1] * w[3] + I_temp[1] * w[0])
        - (R_temp[2] * w[1] - I_temp[2] * w[4])
        + (R_temp[3] * w[5] + I_temp[3] * w[2]);
}

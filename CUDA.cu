#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

#define SIZE  320  // Define image size as 320x320
#define B 256;     // Define block size
#define TH  256;   // Define the number of threads
#define TOTAL 256*256  // Define the total number of threads

// Declare GPU and CPU convolution functions
__global__ void CNN_GPU(int m1[], unsigned char m2[], int kernel[], int size_, int num, int big);
void CNN_CPU(int* m1, unsigned char *m2, int kernel[], int size_, int num, int big);

int main(int argc, char **argv) {
    // Define Sobel and Gaussian convolution kernels
    int kernel1[9] = { 1,0,-1,2,0,-2,3,0,-3 };
    int kernel2[25] = {1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1};	

    clock_t  start, end, S_t, G_t;  // Timing variables
    FILE* bin;
    bin = fopen("/home/a1075501/convolution/1280.bin", "rb"); // Open binary image file
    unsigned char c;
    
    // Allocate memory for image data
    int* matrix = (int*)malloc(sizeof(int) * SIZE * SIZE);
    unsigned char* s_result = (unsigned char*)malloc(sizeof(int) * (SIZE-2) * (SIZE-2));
    unsigned char* g_result = (unsigned char*)malloc(sizeof(int) * (SIZE-4) * (SIZE-4));
    
    // Read image data into matrix
    if (bin) {
        int i = 0;
        while ((c = fgetc(bin)) != EOF) {
            matrix[i++] = int(c);
            if (i == SIZE * SIZE)
                break;
        }
    }
    fclose(bin);
    
    // Perform Sobel convolution on CPU
    start = clock();
    CNN_CPU(matrix, s_result, kernel1, SIZE, 1, 3);
    end = clock();
    S_t = end - start;
    printf("CPU_sobel: %lf s \n", (double) S_t / CLOCKS_PER_SEC);
    free(s_result);
    
    // Perform Gaussian convolution on CPU
    start = clock();
    CNN_CPU(matrix, g_result, kernel2, SIZE-2, 256, 5);
    end = clock();
    G_t = end - start;
    printf("CPU_gaussian: %lf s \n", (double) G_t / CLOCKS_PER_SEC);
    free(g_result);

    // Initialize CUDA timers
    cudaEvent_t c_start, c_stop;
    cudaEventCreate(&c_start);
    cudaEventCreate(&c_stop);

    // Allocate memory on GPU
    int *matrix_g;
    unsigned char *G_s_result, *G_g_result;
    int *kernel1_G, *kernel2_G;
    float GTime, STime = 0;

    cudaMalloc((void**)&matrix_g, sizeof(int) * SIZE * SIZE);
    cudaMalloc((void**)&G_s_result, sizeof(int) * (SIZE-2) * (SIZE-2));
    cudaMalloc((void**)&G_g_result, sizeof(int) * (SIZE-4) * (SIZE-4));
    cudaMalloc((void**)&kernel1_G, sizeof(int) * 9);
    cudaMalloc((void**)&kernel2_G, sizeof(int) * 25);

    // Allocate memory on CPU
    s_result = (unsigned char*)malloc(sizeof(int) * (SIZE-2) * (SIZE-2));
    g_result = (unsigned char*)malloc(sizeof(int) * (SIZE-4) * (SIZE-4));

    // Copy data to GPU
    cudaMemcpy(matrix_g, matrix, sizeof(int) * SIZE * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(kernel1_G, kernel1, sizeof(int) * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(kernel2_G, kernel2, sizeof(int) * 25, cudaMemcpyHostToDevice);

    // Perform Sobel convolution on GPU
    cudaEventRecord(c_start, 0);
    CNN_GPU<<<256, 256>>>(matrix_g, G_s_result, kernel1_G, SIZE, 1, 9);
    cudaEventRecord(c_stop, 0);
    cudaEventSynchronize(c_stop);
    cudaEventElapsedTime(&STime, c_start, c_stop);
    cudaMemcpy(s_result, G_s_result, sizeof(int) * (SIZE-2) * (SIZE-2), cudaMemcpyDeviceToHost);
    printf("GPU_sobel: %lf s \n", (double) STime / 1000);
    free(s_result);
    cudaFree(kernel1_G);
    cudaFree(G_s_result);

    // Perform Gaussian convolution on GPU
    cudaEventRecord(c_start, 0);
    CNN_GPU<<<256, 256>>>(matrix_g, G_g_result, kernel2_G, SIZE-2, 256, 25);
    cudaEventRecord(c_stop, 0);
    cudaEventSynchronize(c_stop);
    cudaEventElapsedTime(&GTime, c_start, c_stop);
    cudaMemcpy(g_result, G_g_result, sizeof(int) * (SIZE-2) * (SIZE-2), cudaMemcpyDeviceToHost);
    printf("GPU_gaussian %lf s \n", (double) GTime / 1000);

    // Free memory
    free(g_result);
    cudaFree(G_g_result);
    cudaFree(kernel2_G);
    cudaFree(matrix_g);
    free(matrix);

    return 0;
}

// CPU convolution function
void CNN_CPU(int *m1, unsigned char *m2, int kernel[], int size_, int num, int big) {
    for (int i = 0; i < size_ - 2; i++) {
        for (int j = 0; j < size_ - 2; j++) {
            int t = 0;
            for (int k = 0; k < big * big; k++) {
                t = t + kernel[k] * m1[(k / big + i) * SIZE + k % big + j];
            }
            t /= num;
            if (t < 0) t = 0;
            else if (t > 255) t = 255;
            m2[i * (size_ - 2) + j] = (unsigned char)t;
        }
    }
}

// GPU convolution kernel
__global__ void CNN_GPU(int m1[], unsigned char m2[], int kernel[], int size_, int num, int big) {
    int bias, temp_B;
    if (big == 9) {
        bias = size_;
        temp_B = 3;
    } else if (big == 25) {
        bias = size_ - 2;
        temp_B = 5;
    }

    int bb = bias * bias;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = 0; i < bb; i += TOTAL) {
        if (i + j < bb) {
            int t = 0;
            for (int k = 0; k < big; k++) {
                t += kernel[k] * m1[(k / temp_B + (i + j) / bias) * size_ + k % temp_B + (i + j) % bias];
            }
            t /= num;
            if (t < 0) t = 0;
            else if (t > 255) t = 255;
            m2[(i + j) / bias * bias + (i + j) % bias] = (unsigned char)t;
        }
    }
}

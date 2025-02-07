#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<unistd.h>
#include<sys/time.h>
#include <omp.h>
#define SIZE 320

// Kernel definitions for Sobel and Gaussian filtering
int kernel1[9] = { 1,0,-1,2,0,-2,3,0,-3 }; // Sobel kernel
int kernel2[25] = {1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1}; // Gaussian kernel

// Global variables for storing results and matrix data
unsigned char * s_result;
unsigned char * g_result;
int* matrix; 

// Function prototype for convolution operation
void CNN_CPU(unsigned char *m2, int kernel[], int size_, int num, int kernel_size, int threads_num);

int main(int argc, char *argv[])
{
    struct timeval start, end;
    FILE* bin;
    bin = fopen("320.bin", "rb"); // Open binary file
    unsigned char c;
    double total_time_s = 0, total_time_g = 0;
    
    // Allocate memory for results and matrix
    s_result = (unsigned char*)malloc(sizeof(int) * (SIZE-2) * (SIZE-2));
    g_result = (unsigned char*)malloc(sizeof(int) * (SIZE-4) * (SIZE-4));
    matrix = (int*)malloc(sizeof(int) * SIZE * SIZE);

    // Read file into matrix
    if (bin)
    {
        int i = 0;
        while ((c = fgetc(bin)) != EOF)
        {
            matrix[i++] = (int)c;
            if (i == SIZE * SIZE)
                break;
        }
    }
    fclose(bin);
    
    // Sobel filtering with varying thread counts
    for (int threads_num = 1; threads_num < 9; threads_num++)
    {
        gettimeofday(&start, NULL);
        CNN_CPU(s_result, kernel1, SIZE, 1, 3, threads_num);
        gettimeofday(&end, NULL);
        total_time_s = (double)(1000000*(end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)) / 1000000;
        printf("Number of Thread(s) : %d  ==> Time used : %lf s \n", threads_num, total_time_s);
    }
    
    // Write Sobel result to file
    FILE* write_ptr1 = fopen("output_s.bin", "wb");
    fwrite(s_result, sizeof(unsigned char), (SIZE-2)*(SIZE-2), write_ptr1);
    fclose(write_ptr1);
    
    // Gaussian filtering with varying thread counts
    for (int threads_num = 1; threads_num < 9; threads_num++)
    {
        gettimeofday(&start, NULL);
        CNN_CPU(g_result, kernel2, SIZE-2, 256, 5, threads_num);
        gettimeofday(&end, NULL);
        total_time_g = (double)(1000000*(end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)) / 1000000;
        printf("Number of Thread(g) : %d  ==> Time used : %lf s \n", threads_num, total_time_g);
    }
    
    // Write Gaussian result to file
    FILE* write_ptr2 = fopen("output_g.bin", "wb");
    fwrite(g_result, sizeof(unsigned char), (SIZE-4)*(SIZE-4), write_ptr2);
    fclose(write_ptr2);
    
    // Free allocated memory
    free(s_result);
    free(g_result);
    free(matrix);
    
    return 0;
}

// Convolution function applying the given kernel to the input matrix
void CNN_CPU(unsigned char *m2, int kernel[], int size_, int num, int kernel_size, int threadNum)
{
    int t, i, k;
    int bias = size_ - 2;
    
    // Parallelize loop using OpenMP
    #pragma omp parallel for num_threads(threadNum) shared(m2, kernel, size_, num, kernel_size) private(i, t, k)
    for (i = 0; i < bias * bias; i++)
    {
        t = 0;
        
        // Apply kernel to the matrix
        for (k = 0; k < kernel_size * kernel_size; k++)
        {
            t += kernel[k] * matrix[(k/kernel_size + i/bias) * SIZE + k%kernel_size + i%bias];
        }
        
        t /= num; // Normalize value
        
        // Ensure values are within valid range (0-255)
        if (t < 0) t = 0;
        else if (t > 255) t = 255;
        
        m2[i / bias * bias + i % bias] = (unsigned char)t;
    }
}

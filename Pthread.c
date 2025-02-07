#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<pthread.h>
#include<unistd.h>
#include<sys/time.h>

#define SIZE  320  // Define the image size (320x320)

// Sobel and Gaussian kernels for image processing
int kernel1[9] = { 1,0,-1,2,0,-2,3,0,-3 };  // Sobel kernel
int kernel2[25] = {1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1}; // Gaussian kernel

// Global variables to hold image data
unsigned char * s_result;  // Sobel filter result
unsigned char * g_result;  // Gaussian filter result
int* matrix;  // Original image matrix

// Struct to define thread processing boundaries
typedef struct bounding
{
	int upperbound;
	int lowerbound;
}*arg;

// Function prototypes for convolution operations
void *CNN_CPU_s(void *);
void *CNN_CPU_g(void *);

int main(int argc, char *argv[])
{
	struct timeval start, end;
	FILE* bin;
	bin = fopen("320.bin", "rb"); // Open image file in binary mode
	unsigned char c;
	double total_time_s = 0, total_time_g = 0;
	
	// Allocate memory for image processing
	s_result = (unsigned char*)malloc(sizeof(int) * (SIZE-2) * (SIZE-2));
	g_result = (unsigned char*)malloc(sizeof(int) * (SIZE-4) * (SIZE-4));
	matrix = (int*)malloc(sizeof(int) * SIZE*SIZE);
	
	// Read the binary file and store pixel values into matrix
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
	
	// Perform Sobel filtering using multithreading
	for (int threads_num = 1; threads_num < 9; threads_num++)
	{
		struct bounding bound_s[threads_num];
		for (int i = 0; i < threads_num; i++)
		{
			bound_s[i].lowerbound = i * ((SIZE-2)*(SIZE-2) / threads_num);
			bound_s[i].upperbound = (i == (threads_num-1)) ? (SIZE-2)*(SIZE-2) : (i+1) * ((SIZE-2)*(SIZE-2) / threads_num);
		}
		gettimeofday(&start, NULL);
		pthread_t threads_s[threads_num];
		for (int i = 0; i < threads_num; i++)
			pthread_create(&threads_s[i], NULL, CNN_CPU_s, (void *)&bound_s[i]);
		for (int i = 0; i < threads_num; i++)
			pthread_join(threads_s[i], NULL);
		gettimeofday(&end, NULL);
		total_time_s = (double)(1000000*(end.tv_sec-start.tv_sec)+(end.tv_usec-start.tv_usec))/1000000;
		printf("Number of Thread(s) : %d  ==> Time used : %lf s \n", threads_num, total_time_s);
	}
	
	// Save Sobel filter output
	FILE* write_ptr1 = fopen("output_s.bin", "wb");
	fwrite(s_result, sizeof(unsigned char), (SIZE-2)*(SIZE-2), write_ptr1);
	fclose(write_ptr1);
	
	// Perform Gaussian filtering using multithreading
	for (int threads_num = 1; threads_num < 9; threads_num++)
	{
		struct bounding bound_g[threads_num];
		for (int i = 0; i < threads_num; i++)
		{
			bound_g[i].lowerbound = i * ((SIZE-4)*(SIZE-4) / threads_num);
			bound_g[i].upperbound = (i == (threads_num-1)) ? (SIZE-4)*(SIZE-4) : (i+1) * ((SIZE-4)*(SIZE-4) / threads_num);
		}
		gettimeofday(&start, NULL);
		pthread_t threads_g[threads_num];
		for (int i = 0; i < threads_num; i++)
			pthread_create(&threads_g[i], NULL, CNN_CPU_g, (void *)&bound_g[i]);
		for (int i = 0; i < threads_num; i++)
			pthread_join(threads_g[i], NULL);
		gettimeofday(&end, NULL);
		total_time_g = (double)(1000000*(end.tv_sec-start.tv_sec)+(end.tv_usec-start.tv_usec))/1000000;
		printf("Number of Thread(g) : %d  ==> Time used : %lf s \n", threads_num, total_time_g);
	}
	
	// Save Gaussian filter output
	FILE* write_ptr2 = fopen("output_g.bin", "wb");
	fwrite(g_result, sizeof(unsigned char), 316*316, write_ptr2);
	fclose(write_ptr2);

	// Free allocated memory
	free(s_result);
	free(g_result);
	free(matrix);

	return 0;
}

// Sobel filter convolution function (executed by threads)
void *CNN_CPU_s(void *s)
{
	struct bounding b = *((struct bounding *)s);
	int bias = SIZE - 2;
	for (int i = b.lowerbound; i < b.upperbound; i++)
	{
		int t = 0;
		for (int k = 0; k < 9; k++)
			t += kernel1[k] * matrix[(k / 3 + i/bias)*SIZE + k % 3 + i%bias];
		t = (t < 0) ? 0 : (t > 255) ? 255 : t;
		s_result[i/ bias * bias + i % bias] = (unsigned char)t;
	}
	pthread_exit(NULL);
} 

// Gaussian filter convolution function (executed by threads)
void *CNN_CPU_g(void *s)
{
	struct bounding b = *((struct bounding *)s);
	int bias = SIZE - 4;
	for (int i = b.lowerbound; i < b.upperbound; i++)
	{
		int t = 0;
		for (int k = 0; k < 25; k++)
			t += kernel2[k] * matrix[(k / 5 + i/bias)*SIZE + k % 5 + i%bias];
		t /= 256;
		t = (t < 0) ? 0 : (t > 255) ? 255 : t;
		g_result[i/ bias * bias + i % bias] = (unsigned char)t;
	}
	pthread_exit(NULL);
}

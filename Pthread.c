#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<pthread.h>
#include<unistd.h>
#include<sys/time.h>
#define SIZE  320


// Sobel and Gaussian kernels for image processing
int kernel1[9] = { 1,0,-1,2,0,-2,3,0,-3 };
int kernel2[25] = {1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1};
// Global variables to hold image data
unsigned char * s_result;
unsigned char * g_result;
int* matrix; // Original image matrix

// Struct to define thread processing boundaries
typedef struct bounding
{
	int upperbound;
	int lowerbound;
}*arg;

void *CNN_CPU_s(void *);
void *CNN_CPU_g(void *);

int main(int argc, char *argv[])
{
	struct timeval  start , end ;
	FILE* bin;
	bin = fopen("320.bin", "rb"); // Open image file in binary mode
	unsigned char c;
	double total_time_s = 0, total_time_g = 0;
	// Allocate memory for image processing
	s_result = (unsigned char*)malloc(sizeof(int) * (SIZE-2) * (SIZE-2));
	g_result = (unsigned char*)malloc(sizeof(int) * (SIZE-4) * (SIZE-4));
	matrix = (int*)malloc(sizeof(int) * SIZE*SIZE);

/////////////////////////////////////////////////////////////////////READ FILE	
	if (bin)
	{
		int i = 0;
		while ((c = fgetc(bin)) != EOF)
		{
			
			matrix[i++] = (int) c;
			if (i == SIZE * SIZE)
				break;
		}
	}
	
	fclose(bin);
	
	
///////////////////////////////////////////////////////////////////////SOBEL	
	
	for ( int threads_num = 1 ; threads_num < 9 ; threads_num++ )
	{
		
		struct bounding bound_s[threads_num];
		for (int i = 0 ; i < threads_num ; i++ )
        	{
	    		bound_s[i].lowerbound = i*((SIZE-2)*(SIZE-2)/threads_num);
	    		if ( i == (threads_num-1))
	    			bound_s[i].upperbound = (SIZE-2)*(SIZE-2);
	    		else
	    			bound_s[i].upperbound = (i+1)*((SIZE-2)*(SIZE-2)/threads_num);
		}
		gettimeofday(&start,NULL);
		pthread_t threads_s[threads_num];
        	for ( int i = 0 ; i < threads_num ; i++ )
		{	
			
			pthread_create(&threads_s[i] , NULL , CNN_CPU_s , (void *)&bound_s[i] );
			
		}
		for ( int i = 0 ; i < threads_num ; i++)
			pthread_join(threads_s[i],NULL);
		gettimeofday(&end,NULL);	
		total_time_s = (double)(1000000*(end.tv_sec-start.tv_sec)+(end.tv_usec-start.tv_usec))/1000000;
		printf("Number of Thread(s) : %d  ==> Time used : %lf s \n" , threads_num , total_time_s);
		
	}

	// Save Sobel filter output
	FILE* write_ptr1;
	write_ptr1 = fopen("output_s.bin", "wb");
	fwrite(s_result, sizeof(unsigned char), (SIZE-2)*(SIZE-2), write_ptr1);
	fclose(write_ptr1);
	
//////////////////////////////////////////////////////////////////////////GAUSSIAN
	
	for ( int threads_num = 1 ; threads_num < 9 ; threads_num++ )
	{
		
		struct bounding bound_g[threads_num];
		for (int i = 0 ; i < threads_num ; i++ )
        	{
	    		bound_g[i].lowerbound = i*((SIZE-4)*(SIZE-4)/threads_num);
	    		if ( i == (threads_num-1))
	    			bound_g[i].upperbound = (SIZE-4)*(SIZE-4);
	    		else
	    			bound_g[i].upperbound = (i+1)*((SIZE-4)*(SIZE-4)/threads_num);
		}
		gettimeofday(&start,NULL);
		pthread_t threads_g[threads_num];
		for ( int i = 0 ; i < threads_num ; i++ )
		{	
			
			pthread_create(&threads_g[i] , NULL , CNN_CPU_g , (void *)&bound_g[i] );
			
		}

		for ( int i = 0 ; i < threads_num ; i++)
			pthread_join(threads_g[i],NULL);
		gettimeofday(&end,NULL);	
		total_time_g = (double)(1000000*(end.tv_sec-start.tv_sec)+(end.tv_usec-start.tv_usec))/1000000;
		printf("Number of Thread(g) : %d  ==> Time used : %lf s \n" , threads_num , total_time_g);
	}
	
//////////////////////////////////////////////////////////////////////////////WRITE FILE
	FILE* write_ptr2;

	write_ptr2 = fopen("output_g.bin", "wb");

	fwrite(g_result, sizeof(unsigned char), 316*316, write_ptr2);
	fclose(write_ptr2);
	// Free allocated memory
	free(s_result);
	free(g_result);
	free(matrix);

	return 0;

}


//Sobel CPU   
void *CNN_CPU_s(void *s)
{
	struct bounding b = *((struct bounding *)s);

	int bias = SIZE - 2;
	//printf("%d ~ %d\n", b.lowerbound, b.upperbound);
	for (int i = b.lowerbound ; i < b.upperbound; i++)
	{
		
	   		int t = 0;
	
	   		for (int k = 0; k < 9; k++)
	   		{
	    			t = t + kernel1[k] * matrix[(k / 3 + i/bias)*SIZE + k % 3+i%bias];
	   		}
			if (t < 0)
			{
				t = 0;
			}
		    	else if (t > 255)
			{
				t = 255;
			}
			s_result[i/ bias * bias + i % bias] = (unsigned char)t;
 	}
 	pthread_exit(NULL);
} 

//Gaussian CPU   
void *CNN_CPU_g(void *s)
{
	struct bounding b = *((struct bounding *)s);

	int bias = SIZE - 4;
	for (int i = b.lowerbound ; i < b.upperbound; i++)
	{
	   		int t = 0;
	
	   		for (int k = 0; k < 25; k++)
	   		{
	    			t = t + kernel2[k] * matrix[(k / 5 + i/bias)*SIZE + k % 5+i%bias];
	   		}
			t/=256;
			if (t < 0)
			{
				t = 0;
			}
		    	else if (t > 255)
			{
				t = 255;
			}
			g_result[i/ bias * bias + i % bias] = (unsigned char)t;
	  		
 	}
 	pthread_exit(NULL);
} 





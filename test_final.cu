/*
 * Parallel bitonic sort using CUDA.
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 */ 
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <omp.h>

#include <curand.h>
#include <curand_kernel.h>

/* Every thread gets exactly one value in the unsorted array. */
#define THREADS 1024L
#define BLOCKS 8192L

#define NUM_VALS ((unsigned long int)THREADS*BLOCKS*512)


void array_print(int *arr, unsigned long int length)
{
  unsigned long int i;
  for (i = 0; i  < 200; ++i) {
    printf("%d ",  arr[i]);
  }
  
  printf("\n");
  for (i = length-1; i  >length-200; --i) {
    printf("%d ",  arr[i]);
  }
  printf("\n");
}

__global__ void init(int seed, curandState_t* states) {

	unsigned long int blockId_2D;
	blockId_2D = blockIdx.x + blockIdx.y*gridDim.x;
    unsigned long int id = threadIdx.x + blockDim.x*blockId_2D;
	
  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              id, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[id]);
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void randoms(curandState_t* states,int* numbers) {
  /* curand works like rand - except that it takes a state as a parameter */
  unsigned long int blockId_2D;
	blockId_2D = blockIdx.x + blockIdx.y*gridDim.x;
    unsigned long int id = threadIdx.x + blockDim.x*blockId_2D;
  
  numbers[id] = curand(&states[id]) % 10000;
}

/*void array_fill(int *arr, unsigned long int length)
{
  srand(time(NULL));
  unsigned long int i;
  #pragma omp parallel
  for (i = 0; i < length; ++i) {
    arr[i] = rand()%10000+1;
  }
}*/


__global__ void bitonic_sort_step(int *dev_values, unsigned long int j, unsigned long int k)
{
  unsigned long int i,blockId_2D; /* Sorting partners: i and ixj */
  unsigned long long int ixj;
  //i = threadIdx.x + blockDim.x * blockIdx.x;
  blockId_2D = blockIdx.x + blockIdx.y*gridDim.x;
  i = threadIdx.x + blockDim.x*blockId_2D;
  
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
        /* inline exchange(i,ixj); */
        int temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        /* inline exchange(i,ixj); */
        int temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}


double get_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  return elapsed;
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(int *values)
{ 
  clock_t start1, stop1;
  /*int *dev_values;
  unsigned long int size = NUM_VALS * sizeof(int);
  
  start1 = clock();
  cudaMalloc((void**) &dev_values, size);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  stop1 = clock();
  double elapsed = get_elapsed(start1, stop1);
  printf("copy from host 2 device time: %.6fs\n",elapsed);*/
  
  
  // Flat memory layout:
  // [T,T,T,T...,T] [T,T,T,T...,T] [T,T,T,T...,T] [T,T,T,T...,T]
  // \___________/  \___________/  \___________/  \___________/
  //    Block 1        Block 2        Block 3        Block 4

  dim3 blocks(BLOCKS,512,1);    /* Number of blocks   */
  dim3 threads(THREADS,1,1);  /* Number of threads  */

  unsigned long int j, k;
  /* Major step */
  //#pragma omp parallel for
 
  start1 = clock();
  #pragma omp parallel for private( j )
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
	//#pragma omp parallel for
    for (j=k>>1; j>0; j=j>>1) {
      //printf("Major step: k=%d, Minor step: j=%d\n", k, j);
      bitonic_sort_step<<<blocks, threads>>>(values, j, k);
    }
  }
  cudaDeviceSynchronize();
  stop1 = clock();
  double elapsed = get_elapsed(start1, stop1);
  printf("only bitonic sort time: %.6fs\n",elapsed);
  
  /*start1 = clock();
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaFree(dev_values); 
  stop1 = clock();
  elapsed = get_elapsed(start1, stop1);
  printf("after copy and free array time: %.6fs\n",elapsed);*/
}




void merge_sort(int *values, int *values1, int*values2, int *sum)
{
	unsigned int lo = 0, hi = NUM_VALS-1;
	unsigned int i = lo, j = lo, l = lo;

    //然后按照规则将数据从辅助数组中拷贝回原始的array中
	#pragma omp parallel for
    for (unsigned int k = lo; k < hi*2; k++)
    {
		if(l>NUM_VALS/2-1){
			if(values[i]<values1[j]){
				if(values[i]<values2[l]){
					sum[k] = values[i++];
				}else{
					sum[k] = values[l++];
				}
			}else{
				if(values[j]<values2[l]){
					sum[k] = values[j++];
				}else{
					sum[k] = values[l++];
				}
			}
				
		}else{
        //如果左边元素没了， 直接将右边的剩余元素都合并到到原数组中
        if (i > hi)
        {
            sum[k] = values1[j++];
        }//如果右边元素没有了，直接将所有左边剩余元素都合并到原数组中
        else if (j > hi)
        {
            sum[k] = values[i++];
        }//如果左边右边小，则将左边的元素拷贝到原数组中
        //else if (values[i].CompareTo(values1[j]) < 0)
		else if (values[i]<values1[j])
        {
            sum[k] = values[i++];
        }
        else
        {
            sum[k] = values1[j++];
        }
		}
    }
}


int main(void)
{
  clock_t start, stop, stop1;
  
  
  int *values;
  int *values1;
  int *values2;
  
  int *sum1 = (int*) malloc( NUM_VALS * sizeof(int)*2.5);
  //int *sum2 = (int*) malloc( NUM_VALS * sizeof(int)*2.5);
  
  cudaMallocManaged( (void**)&values, NUM_VALS * sizeof(int), cudaHostAllocMapped ) ;
  cudaMallocManaged( (void**)&values1, NUM_VALS * sizeof(int) , cudaHostAllocMapped ) ;
  cudaMallocManaged( (void**)&values2, NUM_VALS * sizeof(int)/2, cudaHostAllocMapped  ) ;

 
  /*array_fill(values, NUM_VALS);
  array_fill(values1, NUM_VALS);
  array_fill(values2, NUM_VALS);*/
  
  
  
  dim3 blocks(BLOCKS,512,1);    /* Number of blocks   */
  dim3 threads(THREADS,1,1);  /* Number of threads  */
  
  curandState* devStates;
  
  
  
  cudaHostAlloc ( &devStates, NUM_VALS*sizeof( curandState ) , cudaHostAllocDefault );

  start = clock();
  init<<<blocks, threads >>>(time(0), devStates);
  
  randoms<<<blocks, threads >>>(devStates,values);
  randoms<<<blocks, threads >>>(devStates,values1);
  randoms<<<blocks, threads >>>(devStates,values2);
  /*int* N1;
    cudaMalloc((void**) &N1, sizeof(int)*NUM_VALS);
	randoms<<<blocks, threads >>>(devStates,N1);
	cudaMemcpy(values, N1, sizeof(float)*NUM_VALS, cudaMemcpyDeviceToHost);
	cudaFree(N1);
	
	int* N2;
    cudaMalloc((void**) &N2, sizeof(int)*NUM_VALS);
	randoms<<<blocks, threads >>>(devStates,N2);
	cudaMemcpy(values1, N2, sizeof(float)*NUM_VALS, cudaMemcpyDeviceToHost);
	cudaFree(N2);
	
	int* N3;
    cudaMalloc((void**) &N3, sizeof(int)*NUM_VALS);
	randoms<<<blocks, threads >>>(devStates,N3);
	cudaMemcpy(values2, N3, sizeof(float)*NUM_VALS, cudaMemcpyDeviceToHost);
	cudaFree(N3);*/
  
  /* invoke the GPU to initialize all of the random states */
  


  /*randoms<<<blocks, threads >>>(devStates,values);
  randoms<<<blocks, threads >>>(devStates,values1);
  randoms<<<blocks, threads >>>(devStates,values2);*/

  /* invoke the kernel to get some random numbers */
  
  cudaDeviceSynchronize();
  
  stop = clock();
  double elapsed = get_elapsed(start, stop);
  printf("make and fill arrays time: %.6fs\n",elapsed);
  
  cudaFree(devStates);

  
  start = clock();
  bitonic_sort(values);
  bitonic_sort(values1);
  bitonic_sort(values2);
  stop1 = clock();
  elapsed = get_elapsed(start, stop1);
  printf("all bitonic sort time: %.6fs\n",elapsed);
  
  start = clock();
  
  merge_sort(values,values1,values2,sum1);
  
  stop = clock();
  elapsed += get_elapsed(start, stop);
  
  printf("Elements: %lu (%lu MB)  totally sort time: %.6fs\n", NUM_VALS*5/2, (NUM_VALS *5/2 * sizeof(int))/(1024*1024),elapsed);
  
  array_print(values, NUM_VALS);
  //int padding = nextPow2(NUM_VALS) - NUM_VALS;
  //printf("Elements: %d\n", NUM_VALS);
  //printf("Padding: %d\n", padding);
  
  cudaFreeHost( values );
  cudaFreeHost( values1 );
  cudaFreeHost( values2 );
  /*cudaFreeHost( sum1 );
  cudaFreeHost( sum2 );*/
  
  
}
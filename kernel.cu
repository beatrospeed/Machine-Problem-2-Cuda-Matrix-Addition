
#include "cuda_runtime.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <vector>
#include <stdio.h>
#include <random>
#include <algorithm>
#include <chrono>
#include <map>


#define BLOCK_WIDTH 16

__global__ void MatrixSumElement(float* A, float* B, float* C, int w) {
	 int row = blockIdx.x * blockDim.x + threadIdx.x;
	 int col = blockIdx.y * blockDim.y + threadIdx.y;
	 int idx = col * w + row;
	if (row < w && col < w)
	{
		C[idx] = A[idx] + B[idx];
		//printf("Element Compairason:  matrix N: %f------ matrix M: %f matrix R:------ %f \n", A[idx], B[idx], C[idx]);
	}
}

__global__ void perRow(float* A, float* B, float* C, int w) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < w)
	{
		int rowIdx = row * w;

		for (int i = 0; i < w; i++)
		{
			const int currentIdx = rowIdx + i;
			C[currentIdx] = A[currentIdx] + B[currentIdx];
		}
	}
}

__global__ void perCol(float* A, float* B, float* C, int w) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < w)
	{
		for (int i = 0; i < w; i++)
		{
			const int Idx = i * w + col;
			C[Idx] = A[Idx] + B[Idx];
		}
	}



}
void sumMatrixOnHost(float* A, float* B, float* C, int w)
{
	float* N = A;
	float* M = B;
	float* R = C;
	for (int k = 0; k < w; k++)
	{
		for (int p = 0; p < w; p++)
		{
			R[p] = N[p] + M[p];
		}
		N += w;
		M += w;
		R += w;
	}
	return;
}

void checkResult(float* CPU, float* GPU, const int dim) {
	double Margin = 1.0E-8;
	for (int i = 0; i < dim; i++)
	{
		if (abs(CPU[i] - GPU[i]) > Margin)
		{
			printf("CPU %f GPU %f ", CPU[i], GPU[i]);
			printf("Matricies do not match.\n\n");
			break;
		}



	}
	printf("Test PASSED\n\n");
}

void initialData(float* Matrix, const int dim)
{
	int i;
	for (i = 0; i < dim; i++)
	{
		Matrix[i] = (float)(rand() & 0xFF) / 10.0f;
	}

}

void computeMatrix(int S) {

	float* H_N, *H_M, *H_R, *H_R1;
	//Size of matrix dimension i.e. 1024


	// Multiply each dimension to get the matrix and then multiply by size of int to get the value in bytes

	size_t sizeInFloats = S * S * sizeof(float);
	//input host vector N

	H_N = (float*)malloc(sizeInFloats);
	H_M = (float*)malloc(sizeInFloats);
	H_R = (float*)malloc(sizeInFloats);
	H_R1 = (float*)malloc(sizeInFloats);

	initialData(H_N, S * S);
	initialData(H_M, S * S);

	memset(H_R, 0, S);
	memset(H_R1, 0, S);
	auto t1 = std::chrono::high_resolution_clock::now();
	sumMatrixOnHost(H_N, H_M, H_R, S);

	auto t2 = std::chrono::high_resolution_clock::now();
	float* D_N, *D_M, *D_R, *D_R1, *D_R2;
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	printf("The CPU took %d to complete the computation.\n\n", duration);

	cudaMalloc((void**)&D_N, sizeInFloats);
	cudaMalloc((void**)&D_M, sizeInFloats);
	cudaMalloc((void**)&D_R, sizeInFloats);
	cudaMalloc((void**)&D_R1, sizeInFloats);
	cudaMalloc((void**)&D_R2, sizeInFloats);




	cudaMemcpy(D_N, H_N, sizeInFloats, cudaMemcpyHostToDevice);
	cudaMemcpy(D_M, H_M, sizeInFloats, cudaMemcpyHostToDevice);

	int NumBlocks = (S + BLOCK_WIDTH - 1) / BLOCK_WIDTH;

	dim3 thread(NumBlocks, NumBlocks);
	dim3 block(BLOCK_WIDTH, BLOCK_WIDTH);

	float time;

	cudaEvent_t start, stop, start1, stop1, start2, stop2;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	MatrixSumElement << <thread, block >> > (D_N, D_M, D_R, S);
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	

	printf("The GPU took %f microseconds to complete the computation with one thread per element.\n\n", time*1000);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	time = 0;
	cudaMemcpy(H_R1, D_R, sizeInFloats, cudaMemcpyDeviceToHost);
	checkResult(H_R, H_R1, S * S);


	NumBlocks = (S + BLOCK_WIDTH - 1) / BLOCK_WIDTH;

	dim3 thread1(NumBlocks);
	dim3 block1(BLOCK_WIDTH);

	
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1);

	perRow << <thread1, block1 >> > (D_N, D_M, D_R1, S);
	
	cudaEventRecord(stop1);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&time, start1, stop1);

	printf("The GPU took %f microseconds to complete the computation with one thread per Row.\n\n", time*1000);

	time = 0; 

	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);

	cudaMemcpy(H_R1, D_R1, sizeInFloats, cudaMemcpyDeviceToHost);
	checkResult(H_R, H_R1, S * S);


	NumBlocks = (S + BLOCK_WIDTH - 1) / BLOCK_WIDTH;

	dim3 thread2(NumBlocks);
	dim3 block2(BLOCK_WIDTH);

	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventRecord(start2);

	perCol << <thread2, block2 >> > (D_N, D_M, D_R2, S);
	
	cudaEventRecord(stop2);
	cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&time, start2, stop2);

	printf("The GPU took %f microseconds to complete the computation with one thread per Column.\n\n", time*1000);

	time = 0;

	cudaEventDestroy(start2);
	cudaEventDestroy(stop2);


	cudaMemcpy(H_R1, D_R2, sizeInFloats, cudaMemcpyDeviceToHost);
	checkResult(H_R, H_R1, S * S);

	cudaFree(D_N);
	cudaFree(D_M);
	cudaFree(D_R);
	cudaFree(D_R1);
	cudaFree(D_R2);
	free(H_N);
	free(H_M);
	free(H_R);
	free(H_R1);
	// reset device

	cudaDeviceReset();


}

int main()
{

	printf("100 x 100 matrix addition.\n\n");
	computeMatrix(100);
	printf("200 x 200 matrix addition.\n\n");
	computeMatrix(200);
	printf("500 x 500 matrix addition.\n\n");
	computeMatrix(500);
	printf("1000 x 1000 matrix addition.\n\n");
	computeMatrix(1000);
	printf("1500 x 1500 matrix addition.\n\n");
	computeMatrix(1500);
	printf("5000 x 5000 matrix addition.\n\n");
	computeMatrix(5000);

	return 0;


}


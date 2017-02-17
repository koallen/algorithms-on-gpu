#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iostream>

#include "bfs.cuh"

#define BLOCK_SIZE 256

__global__ void cudaBfsKernel(int *, int *, int *, int *, int *, int);

int emptyFrontier(int *F, int vertexCount)
{
	for (int i = 0; i < vertexCount; ++i)
		if (F[i] == 1)
			return 0;
	return 1;
}

void cudaBfs(int *V, int *E, int *C, int vertexCount, int edgeCount, int source)
{
	// allocate frontier array
	int *F = (int *)malloc(sizeof(int) * vertexCount);
	memset(F, 0, sizeof(int) * vertexCount);

	// allocate visited array
	int *X = (int *)malloc(sizeof(int) * vertexCount);
	memset(X, 0, sizeof(int) * vertexCount);

	// update arrays for source
	F[source] = 1;

	// setup array on device
	int *V_d, *E_d, *F_d, *X_d, *C_d;
	cudaMalloc(&V_d, sizeof(int) * (vertexCount + 1));
	cudaMalloc(&E_d, sizeof(int) * edgeCount);
	cudaMalloc(&F_d, sizeof(int) * vertexCount);
	cudaMalloc(&X_d, sizeof(int) * vertexCount);
	cudaMalloc(&C_d, sizeof(int) * vertexCount);
	cudaMemcpy(V_d, V, sizeof(int) * (vertexCount + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(E_d, E, sizeof(int) * edgeCount, cudaMemcpyHostToDevice);
	cudaMemcpy(F_d, F, sizeof(int) * vertexCount, cudaMemcpyHostToDevice);
	cudaMemcpy(X_d, X, sizeof(int) * vertexCount, cudaMemcpyHostToDevice);
	cudaMemcpy(C_d, C, sizeof(int) * vertexCount, cudaMemcpyHostToDevice);

	// run the kernel
	dim3 grid((vertexCount+BLOCK_SIZE)/(BLOCK_SIZE), 1, 1);
	dim3 block(BLOCK_SIZE, 1, 1);
	do {
		cudaBfsKernel<<<grid, block>>>(V_d, E_d, F_d, X_d, C_d, vertexCount);
		cudaDeviceSynchronize();
		cudaMemcpy(F, F_d, sizeof(int) * vertexCount, cudaMemcpyDeviceToHost);
	} while (!emptyFrontier(F, vertexCount));

	cudaMemcpy(C, C_d, sizeof(int) * vertexCount, cudaMemcpyDeviceToHost);

	cudaFree(V_d);
	cudaFree(E_d);
	cudaFree(F_d);
	cudaFree(X_d);
	cudaFree(C_d);
}

__global__ void cudaBfsKernel(int *V, int *E, int *F, int *X, int *C, int VCount)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= VCount) return;
	if (F[tid])
	{
		F[tid] = 0;
		X[tid] = 1;
		for (int i = V[tid]; i < V[tid+1]; ++i)
		{
			int nid = E[i];
			if (X[nid] != 1)
			{
				C[nid] = C[tid] + 1;
				F[nid] = 1;
			}
		}
	}
}

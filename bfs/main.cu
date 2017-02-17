#include <cuda_runtime.h>
#include <iostream>

#include "bfs.cuh"

using namespace std;

int main()
{
	// test data
	int V[] = {0, 1, 2, 3, 5, 6, 7, 8, 9}; // the last one is not a vetex
	int E[] = {1, 3, 1, 2, 4, 5, 7, 4, 6};
	int C[] = {0, INF, INF, INF, INF, INF, INF, INF};

	cudaBfs(V, E, C, 8, 9, 0);

	cout << "Shortest distances are:" << endl;
	for (int i = 0; i < 8; ++i)
		cout << i << ": " << C[i] << endl;

	return 0;
}

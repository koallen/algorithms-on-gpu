#include <limits>

#define MAX_INT std::numeric_limits<int>::max()
#define BLOCK_SIZE 256

int emptyFrontier(int *, int);
void cudaBfs(int *, int *, int *, int, int, int);
__global__ void cudaBfsKernel(int *, int *, int *, int *, int *, int);

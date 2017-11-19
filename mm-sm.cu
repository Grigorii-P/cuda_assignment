/**
 * 
 * Matrix Multiplication - CUDA for GPUs
 *
 * CS3210
 *
 **/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>

// Thread block size
#define BLOCK_SIZE 32
#define STRIDE BLOCK_SIZE

int size;

typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;


__device__ float GetElement(const Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];
}

__device__ void SetElement(Matrix A, int row, int col, float value) {
    A.elements[row * A.stride + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
Matrix Asub;
Asub.width = BLOCK_SIZE;
Asub.height = BLOCK_SIZE;
Asub.stride = A.stride;
Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
return Asub;
}


long long wall_clock_time()
{
#ifdef __linux__
	struct timespec tp;
	clock_gettime(CLOCK_REALTIME, &tp);
	return (long long)(tp.tv_nsec + (long long)tp.tv_sec * 1000000000ll);
#else
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (long long)(tv.tv_usec * 1000 + (long long)tv.tv_sec * 1000000000ll);
#endif
}

void allocate_matrix(Matrix* m)
{

	m->elements = (float*)malloc(size * size * sizeof(float));


	// int i;
	// m->elements = (float**)malloc(sizeof(float*) * size);
	// if (m->elements == NULL)
	// {
	// 	fprintf(stderr, "Out of memory\n");
	// 	exit(1);
	// }
	
	// // allocate an array for each row of the matrix
	// for (i = 0; i < size; i++)
	// {
	// 	m->elements[i] = (float*)malloc(sizeof(float) * size);
	// 	if (m->elements[i] == NULL)
	// 	{
	// 		fprintf(stderr, "Out of memory\n");
	// 		exit(1);
	// 	}
	// }
}

/**
 * Free the memory allocated for a matrix.
 **/
void free_matrix(Matrix* m) {
	int i;
	for (i = 0; i < size; i++)
		cudaFree(m->elements[i]);
	cudaFree(m->elements);
}

/**
 * Initializes the elements of the matrix with
 * random values between 0 and 9
 **/
void init_matrix(Matrix m)
{
	m.stride = STRIDE;
	for (i = 0; i < size*size; i++) {
		m.elements[i] = rand() % 10;
	}

	// int i, j;
	
	// for (i = 0; i < size; i++)
	// 	for (j = 0; j < size; j++)
	// 	{
	// 		m.elements[i][j] = rand() % 10;
	// 	}
	
}


__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.stride = A.width;
    d_A.width = d_A.stride; 
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.stride = B.width;
    d_B.width = d_B.stride; 
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    
    // Allocate C in device memory
    Matrix d_C;
    d_C.stride = C.width;
    d_C.width = d_C.stride; 
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);
    
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    
    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}


// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
	}	
    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}


void work()
{
	Matrix a, b, result1, result2;
	long long before, after;
	int correct, i, j, dim;
	cudaError_t rc;

	// Allocate memory for matrices
	allocate_matrix(&a);
	allocate_matrix(&b);
	allocate_matrix(&result1);
	allocate_matrix(&result2);	

	// Initialize matrix elements
	init_matrix(a);
	init_matrix(b);

	// Perform CUDA matrix  multiplication
	before = wall_clock_time();
	MatMul(a, b, result2);
	cudaDeviceSynchronize();
	after = wall_clock_time();
	fprintf(stderr, "Matrix multiplication on GPU took %1.2f seconds\n", ((float)(after - before))/1000000000);

	// was there any error?
        rc = cudaGetLastError();
        if (rc != cudaSuccess)
                printf("Last CUDA error %s\n", cudaGetErrorString(rc));

	free_matrix(&a);
	free_matrix(&b);
	free_matrix(&result1);
	free_matrix(&result2);
}


int main(int argc, char ** argv)
{
	srand(0); 

	printf("Usage: %s <size>\n", argv[0]);
    
	if (argc >= 2)
		size = atoi(argv[1]);
	else
		size = 1024;
		
	fprintf(stderr,"Sequential matrix multiplication of size %d\n", size);
    
	// Multiply the matrices
	work();

	return 0;
}

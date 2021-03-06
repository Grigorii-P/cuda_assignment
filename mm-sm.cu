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

int input_size;

typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

typedef struct
{
	float ** element;
} matrix;


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
	m->elements = (float*)malloc(input_size * input_size * sizeof(float));
}

void allocate_matrix_seq(matrix* m)
{
	int i;
	
	// allocate array for all the rows
	m->element = (float**)malloc(sizeof(float*) * input_size);
	if (m->element == NULL)
	{
		fprintf(stderr, "Out of memory\n");
		exit(1);
	}
	
	// allocate an array for each row of the matrix
	for (i = 0; i < input_size; i++)
	{
		m->element[i] = (float*)malloc(sizeof(float) * input_size);
		if (m->element[i] == NULL)
		{
			fprintf(stderr, "Out of memory\n");
			exit(1);
		}
	}
}

/**
 * Free the memory allocated for a matrix.
 **/
// void free_matrix(Matrix* m) {
// 	int i;
// 	for (i = 0; i < size*size; i++)
// 		cudaFree(m->elements[i]);

// 	// int i;
// 	// for (i = 0; i < size; i++)
// 	// 	cudaFree(m->elements[i]);
// 	// cudaFree(m->elements);
// }

/**
 * Initializes the elements of the matrix with
 * random values between 0 and 9
 **/
void init_matrix(Matrix m)
{
	m.stride = STRIDE;

	int i;
	for (i = 0; i < input_size*input_size; i++) {
		m.elements[i] = rand() % 10;
	}
	
}

void init_matrix_seq(matrix m)
{
	int i, j;
	
	for (i = 0; i < input_size; i++)
		for (j = 0; j < input_size; j++)
		{
			m.element[i][j] = rand() % 10;
		}
}

void mm(matrix a, matrix b, matrix result)
{
	int i, j, k;

	// Do the multiplication
	for (i = 0; i < input_size; i++)
		for (j = 0; j < input_size; j++)
			for(k = 0; k < input_size; k++)
				result.element[i][j] += a.element[i][k] * b.element[k][j];    
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
	Matrix a, b, result2;
	matrix a_seq, b_seq, result1;
	long long before, after;
	int correct, i, j, dim;
	cudaError_t rc;

	// Allocate memory for matrices
	allocate_matrix(&a);
	allocate_matrix(&b);
	allocate_matrix(&result2);

	allocate_matrix_seq(&a_seq);
	allocate_matrix_seq(&b_seq);
	allocate_matrix_seq(&result1);

	// Initialize matrix elements
	init_matrix(a);
	init_matrix(b);

	init_matrix_seq(a_seq);
	init_matrix_seq(b_seq);

	// Perform SEQ matrix  multiplication
	before = wall_clock_time();
	mm(a_seq, b_seq, result1);
	after = wall_clock_time();
        fprintf(stderr, "Sequential matrix multiplication took %1.2f seconds\n", ((float)(after - before))/1000000000);

	// Perform CUDA matrix  multiplication
	before = wall_clock_time();
	MatMul(a, b, result2);
	cudaDeviceSynchronize();
	after = wall_clock_time();
	fprintf(stderr, "CUDA matrix multiplication on GPU took %1.2f seconds\n", ((float)(after - before))/1000000000);

	// was there any error?
        rc = cudaGetLastError();
        if (rc != cudaSuccess)
                printf("Last CUDA error %s\n", cudaGetErrorString(rc));

    // Compare the results
    int v = 0;
	correct = 1;
	for (i = 0; correct && i < input_size; i++)
		for (j = 0; j < input_size; j++) {
			if (result1.element[i][j] != result2.elements[v]) {
				correct = 0;
				break;
			}
			v++;
		}

	if (correct)
		printf("The result matrices are identical!\n");
	else
		printf("Difference in result matrices at element (%d, %d)!\n", i, j);

	// free_matrix(&a);
	// free_matrix(&b);
	// free_matrix(&result1);
	// free_matrix(&result2);
}


int main(int argc, char ** argv)
{
	srand(0); 

	printf("Usage: %s <size>\n", argv[0]);
    
	if (argc >= 2)
		input_size = atoi(argv[1]);
	else
		input_size = 1024;
		
	fprintf(stderr,"Sequential matrix multiplication of size %d\n", input_size);
    
	// Multiply the matrices
	work();

	return 0;
}

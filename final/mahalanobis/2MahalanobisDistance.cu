#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sequence.h>

#include <stdio.h>
#include <iostream>
#include <math.h> 

#include "Utilities.cuh"
#include "TimingGPU.cuh"

float computeRangeBetweenVectors(float* a, float* b, float*  inverse_covariance_matrix,  int n)
{
    thrust::device_vector<float> d_a(n);
    thrust::device_vector<float> d_b(n);
    thrust::device_vector<float> d_inv_cov(n*n);
    thrust::device_vector<float> d_tmp_result(n);
    thrust::device_vector<float> d_distance;
	
    for (size_t i = 0; i < n; i++)   d_a[i] = a[i];
    for (size_t i = 0; i < n; i++)   d_b[i] = b[i];
    for (size_t i = 0; i < n*n; i++) d_inv_cov[i] = inverse_covariance_matrix[i];

    float alpha = 1.f;
    float beta = 0.f;
   
    // --- cuBLAS handle creation
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

    cublasSafeCall(
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 1, n, n,
			    &alpha,
		            thrust::raw_pointer_cast(d_a.data()), n,
		            thrust::raw_pointer_cast(d_inv_cov.data()), n,
			    &beta,
		            thrust::raw_pointer_cast(d_tmp_result.data()), 1));

     cublasSafeCall(
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, n, 1, 
		            &alpha,
		            thrust::raw_pointer_cast(d_tmp_result.data()), 1,
		            thrust::raw_pointer_cast(d_inv_cov.data()), n,
			    &beta,
		            thrust::raw_pointer_cast(d_distance.data()), 1));

   float result = sqrt(d_distance[0]);
   return result;
}


__global__ void inv_kernel(float *a_i, float *c_o, int n)
{ 
    int p[3], info[1], batch;
    cublasHandle_t hdl;
    cublasStatus_t status = cublasCreate(&hdl);
    printf("handle %d n = %d\n", status, n);

    info[0] = 0;
    batch = 1;
    float *a[] = {a_i};
    const float *aconst[] = {a_i};
    float *c[] = {c_o};

    status = cublasSgetrfBatched(hdl, n, a, n, p, info, batch);
    __syncthreads();
    printf("rf %d info %d\n", status, info[0]);
    status = cublasSgetriBatched(hdl, n, aconst, n, p,
      c, n, info, batch);
    __syncthreads();
    printf("ri %d info %d\n", status, info[0]);

   // cublasDestroy(hdl);
    printf("done\n");
}

static void run_inv(float *in, float *out, int n)
{
  float *a_d, *c_d;

  cudaMalloc(&a_d, n*n*sizeof(float));
  cudaMalloc(&c_d, n*n*sizeof(float));
  cudaMemcpy(a_d, in, n*n*sizeof(float), cudaMemcpyHostToDevice);

  inv_kernel<<<1, 1>>>(a_d, c_d, n);

  cudaDeviceSynchronize();

  cudaMemcpy(out, c_d, n*n*sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(a_d);
  cudaFree(c_d);
}

/*************************************/
/* CONVERT LINEAR INDEX TO ROW INDEX */
/*************************************/
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T> {
	
	T Ncols; // --- Number of columns
  
	__host__ __device__ linear_index_to_row_index(T Ncols) : Ncols(Ncols) {}

	__host__ __device__ T operator()(T i) { return i / Ncols; }
};

//thrust::device_vector<float> computeCovarianceMatrix(int Nsamples, int NX, thrust::device_vector<float> d_X)
float* computeCovarianceMatrix(int Nsamples, int NX, thrust::device_vector<float> d_X)
{	
    // --- cuBLAS handle creation
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

	/*************************************************/
    /* CALCULATING THE MEANS OF THE RANDOM VARIABLES */
	/*************************************************/
    // --- Array containing the means multiplied by Nsamples
	thrust::device_vector<float> d_means(NX);

	thrust::device_vector<float> d_ones(Nsamples, 1.f);

    float alpha = 1.f / (float)Nsamples;
    float beta  = 0.f;
    cublasSafeCall(
	cublasSgemv(handle, CUBLAS_OP_T, Nsamples, NX, 
			    &alpha, 
			    thrust::raw_pointer_cast(d_X.data()), Nsamples, 
                            thrust::raw_pointer_cast(d_ones.data()), 1,
 			    &beta,
			    thrust::raw_pointer_cast(d_means.data()), 1));
	
	/**********************************************/
    /* SUBTRACTING THE MEANS FROM THE MATRIX ROWS */
	/**********************************************/
	thrust::transform(
                d_X.begin(), d_X.end(),
                thrust::make_permutation_iterator(
                        d_means.begin(),
                        thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Nsamples))),
                d_X.begin(),
				thrust::minus<float>());	
	
	/*************************************/
    /* CALCULATING THE COVARIANCE MATRIX */
	/*************************************/
    thrust::device_vector<float> d_cov(NX * NX);

    alpha = 1.f;
    cublasSafeCall(
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, NX, NX, Nsamples,
			    &alpha,
		            thrust::raw_pointer_cast(d_X.data()), Nsamples,
			    thrust::raw_pointer_cast(d_X.data()), Nsamples,
			    &beta, 
			    thrust::raw_pointer_cast(d_cov.data()), NX));

	// --- Final normalization by Nsamples - 1
	thrust::transform(
                d_cov.begin(), d_cov.end(),
                thrust::make_constant_iterator((float)(Nsamples-1)),
                d_cov.begin(),
				thrust::divides<float>());	

	//return d_cov;
    float *result;
    result = (float *)malloc(NX*sizeof(float));
    thrust::copy(d_cov.begin(), d_cov.end(), result);
    return result;

}


/********/
/* MAIN */
/********/
int main()
{
    const int Nsamples = 3;		// --- Number of realizations for each random variable (number of rows of the X matrix)
    const int NX	= 2;		// --- Number of random variables (number of columns of the X matrix)

	// --- Random uniform integer distribution between 10 and 99
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(10, 99);

    // --- Matrix allocation and initialization
    thrust::device_vector<float> d_X(Nsamples * NX);

//    for (size_t i = 0; i < d_X.size(); i++) d_X[i] = (float)dist(rng);

    d_X[0]=1.0;
    d_X[1]=2.0;
    d_X[2]=2.0;
    d_X[3]=1.0;
    d_X[4]=4.0;
    d_X[5]=4.0;

    //thrust::device_vector<float> d_cov;
    thrust::device_vector<float> d_inv_cov;
    float* d_cov = computeCovarianceMatrix(3,2, d_X);
    float* inv_cov;
    run_inv(d_cov, inv_cov, 2);
                
    for(int i = 0; i < NX * NX; i++) std::cout << d_inv_cov[i] << "\n";

    return 0;
}



#include <cublas_v2.h>
#include <Eigen>

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

using std::vector;
using thrust::device_vector;

#define CUDA_CALL(res, str) { if (res != cudaSuccess) { printf("CUDA Error : %s : %s %d : ERR %s\n", str, __FILE__, __LINE__, cudaGetErrorName(res)); } }
#define CUBLAS_CALL(res, str) { if (res != CUBLAS_STATUS_SUCCESS) { printf("CUBLAS Error : %s : %s %d : ERR %d\n", str, __FILE__, __LINE__, int(res)); } }

struct square_root : public thrust::unary_function<float,float>
{
  __host__ __device__
  float operator()(float x) const
  {
    return sqrtf(x);
  }
};

vector<float> computeRangeBetweenVectors
       (vector<float> base, 
	vector<float> vectors,
	float* inverse_covariance_matrix,
        int dim, int n)
{
    device_vector<float> d_a(n*dim);
    device_vector<float> d_b(dim);
    device_vector<float> d_inv_cov(inverse_covariance_matrix, inverse_covariance_matrix + dim*dim);
    device_vector<float> d_tmp_result(n*dim);
    device_vector<float> d_distances(n);
    
    d_a = vectors;
    d_b = base;
    //d_inv_cov = inverse_covariance_matrix;
    float alpha = 1.f;
    float beta = 0.f;
   
    // --- cuBLAS handle creation
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

    cublasSafeCall(
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, dim, dim,
			    &alpha,
		            thrust::raw_pointer_cast(d_a.data()), n,
		            thrust::raw_pointer_cast(d_inv_cov.data()), dim,
			    &beta,
		            thrust::raw_pointer_cast(d_tmp_result.data()), n));

     cublasSafeCall(
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, 1, dim, 
		            &alpha,
		            thrust::raw_pointer_cast(d_tmp_result.data()), n,
		            thrust::raw_pointer_cast(d_b.data()), dim,
			    &beta,
		            thrust::raw_pointer_cast(d_distances.data()), 1));

                                                                                         
     typedef device_vector<float>::iterator FloatIterator;
     thrust::transform_iterator<square_root, FloatIterator> iter
				(d_distances.begin(), square_root());
     
     // copy a device_vector into an STL vector
    vector<float> stl_vector(d_distances.size());
    thrust::copy(d_distances.begin(), d_distances.end(), stl_vector.begin());
    return stl_vector;
}



thrust::device_vector<float> computeInverseMatrix(float* mat, int n){
//float* computeInverseMatrix(float* mat, int n){

    thrust::device_vector<float> d_matrix(n*n);
    for (size_t i = 0; i < n*n; i++) d_matrix[i] = mat[i];

    // --- cuBLAS handle creation
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));
 
    thrust::device_vector<int> d_pivot_vector(n);
    thrust::device_vector<int> d_info(1);
    thrust::device_vector<float> d_inv_matrix(n*n);

    float* ptr_matrix = thrust::raw_pointer_cast(d_matrix.data());
    int* ptr_pivot = thrust::raw_pointer_cast(d_pivot_vector.data());
    int* ptr_info = thrust::raw_pointer_cast(d_info.data());
    float* ptr_inv_matrix = thrust::raw_pointer_cast(d_inv_matrix.data());

    printf("Starting Matrix Inversion");

    cublasSafeCall(
	cublasSgetrfBatched(handle, n, &ptr_matrix, n, ptr_pivot, ptr_info, 1));
    cudaDeviceSynchronize();
    cublasSafeCall(
	cublasSgetriBatched(handle, n,(const float **) &ptr_matrix, n, ptr_pivot, &ptr_inv_matrix, n, ptr_info, 1));
    cudaDeviceSynchronize();

    return d_inv_matrix;
//    float *result;
//    result = (float *)malloc(n*sizeof(float));
//    thrust::copy(d_inv_matrix.begin(), d_inv_matrix.end(), result);
//    return result;
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
device_vector<float> d_computeCovarianceMatrix(int Nsamples, int NX, thrust::device_vector<float> d_X)
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

	return d_cov;
}

vector<float> computeCovarianceMatrix(vector<float> data, int n, int m)
{
    device_vector<float> d_X(n*m);
    d_X = data;

    device_vector<float> d_res = d_computeCovarianceMatrix(n, m, d_X);

    vector<float> res(d_res.size());
    thrust::copy(d_res.begin(), d_res.end(), res.begin());

    return res;
}



int test()
{
    const int Nsamples = 3;		// --- Number of realizations for each random variable (number of rows of the X matrix)
    const int NX	= 2;		// --- Number of random variables (number of columns of the X matrix)

	// --- Random uniform integer distribution between 10 and 99
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(10, 99);

    // --- Matrix allocation and initialization
    thrust::device_vector<float> d_X(Nsamples * NX);

    float t[] = {1,2,2,1,4,4};
    vector<float> m;
    m.assign(t, t+6);

    vector<float> d_cov = computeCovarianceMatrix(m, 3,2);
    for(int i = 0; i < NX * NX; i++) std::cout << d_cov.at(i) << "\n";
    
    double identity[] = {0.333333f, 1.0f, 1.0f, 3.0f};
    double cov_double[NX*NX];
    
    for(int i = 0; i < NX * NX; i++){
	 cov_double[i] = static_cast<double> (identity[i]);
    }
    Eigen::MatrixXd eigenX = Eigen::Map<Eigen::MatrixXd>(cov_double, 2, 2); 
    std::cout << eigenX << std::endl;
    std::cout << eigenX.inverse() << std::endl;
                
    //for(int i = 0; i < NX * NX; i++) std::cout << d_inv_cov[i] << "\n";

    return 0;
}


#pragma once

#include "clVirtualDevice.h"
#include <cinttypes>

namespace kernels
{

	void init_kernels_skmd();
	void init_kernels_mkmd();

	const static std::string task_MM =
	R"(
		kernel void mat_mul_no_opt(global T* restrict C, const global T* restrict A, const global T* restrict B,
		const int mat_width, const int mat_height)
		{
			//Naive (non optimized) data-parallel mat_mul impl.
			//C[NxM] = A[KxM]*B[NxK] 

			const int K = mat_width;
			const int M = mat_height;
			const int N = M;

			const int x = get_global_id(0);
			const int y = get_global_id(1);

			float acc = 0.0f;
			for (int k = 0; k < K; k++)
				acc += A[k * M + x] * B[y * K + k];

			C[y * M + x] = acc;
		}

		#ifndef BLOCK_SIZE
		#define BLOCK_SIZE 16	
		#endif
		
		#define AS(i, j) As[j + i * BLOCK_SIZE]
		#define BS(i, j) Bs[j + i * BLOCK_SIZE]

		///////////////////////////////////////////////////////////////////////////////
		//! Matrix multiplication on the device: C = A * B
		//! uiWA is A's width and uiWB is B's width
		////////////////////////////////////////////////////////////////////////////////
		kernel void matrixMul( global float* C, global float* A, global float* B, 
			   local float* As, local float* Bs, const int uiWA, const int uiWB, const int trueLocalSize1)
		{

			//GPU-optimized, data-parallel mat_transpose impl. (SOURCE NVIDIA CUDA_SDK)

			// Block index
			const int bx = get_group_id(0);
			const int by = get_group_id(1);

			// Thread index
			const int tx = get_local_id(0);
			const int ty = get_local_id(1);

			// Index of the first sub-matrix of A processed by the block
			int aBegin = uiWA * BLOCK_SIZE * by;

			// Index of the last sub-matrix of A processed by the block
			int aEnd   = aBegin + uiWA - 1;

			// Step size used to iterate through the sub-matrices of A
			int aStep  = BLOCK_SIZE;

			// Index of the first sub-matrix of B processed by the block
			int bBegin = BLOCK_SIZE * bx;

			// Step size used to iterate through the sub-matrices of B
			int bStep  = BLOCK_SIZE * uiWB;

			// Csub is used to store the element of the block sub-matrix
			// that is computed by the thread
			float Csub = 0.0f;

			// Loop over all the sub-matrices of A and B
			// required to compute the block sub-matrix
			for (int a = aBegin, b = bBegin;
					 a <= aEnd;
					 a += aStep, b += bStep) {

				// Load the matrices from device memory
				// to shared memory; each thread loads
				// one element of each matrix
				AS(ty, tx) = A[a + uiWA * ty + tx];
				BS(ty, tx) = B[b + uiWB * ty + tx];
	
				// Synchronize to make sure the matrices are loaded
				barrier(CLK_LOCAL_MEM_FENCE);

				// Multiply the two matrices together;
				// each thread computes one element
				// of the block sub-matrix        
				#pragma unroll
				for (int k = 0; k < BLOCK_SIZE; ++k)
					Csub += AS(ty, k) * BS(k, tx);

				// Synchronize to make sure that the preceding
				// computation is done before loading two new
				// sub-matrices of A and B in the next iteration
				barrier(CLK_LOCAL_MEM_FENCE);
			}

			if (get_global_id(1) < trueLocalSize1)
			// Write the block sub-matrix to device memory;
			// each thread writes one element
			C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = Csub;
		}

	)";


	const static std::string task_MT =
    R"(

    #ifndef BLOCK_DIM
    #define BLOCK_DIM 16
    #endif

    kernel void mat_transpose(global T * restrict odata, const global T * restrict  idata,
		const int offset, const int width, const int height, __local T * block)
	{
		//GPU-optimized, data-parallel mat_transpose impl. (SOURCE NVIDIA CUDA_SDK)

		// read the matrix tile into shared memory
		int xIndex = get_global_id(0);
		int yIndex = get_global_id(1);

		if ((xIndex + offset < width) && (yIndex < height))
		{
			const int index_in = yIndex * width + xIndex + offset;
			block[get_local_id(1) * (BLOCK_DIM + 1) + get_local_id(0)] = idata[index_in];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// write the transposed matrix tile to global memory
		xIndex = get_group_id(1) * BLOCK_DIM + get_local_id(0);
		yIndex = get_group_id(0) * BLOCK_DIM + get_local_id(1);
		if ((xIndex < height) && (yIndex + offset < width))
		{
			const int index_out = yIndex * height + xIndex;
			odata[index_out] = block[get_local_id(0) * (BLOCK_DIM + 1) + get_local_id(1)];
		}

		//const int index_in = yIndex * width + xIndex + offset;
		//const int index_out = yIndex * height + xIndex;
		//odata[index_out] = idata[index_in];
	})";

	const static std::string task_MERGE =
		R"(
		kernel void mat_merge(global int* OUT, const global int* IN, 
		const int mat_width,const int mat_height)
		{
			const int tidx = get_global_id(0);
			const int tidy = get_global_id(1);
			const int tid = tidy*mat_width+tidx;
			OUT[tid] += IN[tid];
		})";
	
	const static std::string task_ADD = R"(
		kernel void mat_add(global T* restrict OUT,
		const global T* restrict IN_a,const global T* restrict IN_b, const global T* restrict IN_c,
		const int mat_width,const int mat_height)
		{
			const int tidx = get_global_id(0);
			const int tidy = get_global_id(1);
			const int tid = tidy*mat_width+tidx;
			OUT[tid] = IN_a[tid]+IN_b[tid]+IN_c[tid];
		})";

	const static std::string task_ADD_SUB = R"(
		kernel void mat_add_sub(global T* restrict OUT,
		const global T* restrict IN_a,const global T* restrict IN_b, const global T* restrict IN_c,
		const int mat_width,const int mat_height)
		{
			const int tidx = get_global_id(0);
			const int tidy = get_global_id(1);
			const int tid = tidy*mat_width+tidx;
			OUT[tid] = IN_a[tid]+IN_b[tid]-IN_c[tid];
		})";
}

constexpr auto SCHEDULE_DEFAULT = 0;
constexpr auto SCHEDULE_COARSE = 1;
constexpr auto SCHEDULE_FINE = 2;


struct mkmd_input
{
	int _matrix_width{ 0 };
	int _matrix_height{ 0 };
	int _schedule{ SCHEDULE_DEFAULT };
	int _iterations{ 10 };
	std::uint8_t _gpu_id{ 0 };
	virtual_device* _ptr_device;

	bool _is_obj_061{ 0 };
	bool _is_obj_119{ 0 };
	bool _is_obj_129{ 0 };

	mkmd_input(virtual_device& device,
		const int mw, const int mh,
		const std::uint8_t gpu_id = 0,
		const int schedule = SCHEDULE_DEFAULT,
		const int iterations = 10)
	{
		_matrix_width = mw;
		_matrix_height = mh;
		_schedule = schedule;
		_iterations = iterations;
		_gpu_id = gpu_id;
		_ptr_device = &device;

		if(device.cnt_gpus()==1) _is_obj_061 = 1;
		else if (device.cnt_gpus() == 2)_is_obj_129 = 1;
		else if (device.cnt_gpus() == 3) _is_obj_119 = 1;		

		std::cout << "Matrix (width,height): (" << _matrix_width << "," << _matrix_height << ") \n";
		std::cout << "Matrix size:\t" << (_matrix_width*_matrix_height * sizeof(float))*1e-6 << " MB\n";
	}

	bool is_obj_061()const { return _is_obj_061; }
	bool is_obj_119()const { return _is_obj_119; }
	bool is_obj_129()const { return _is_obj_129; }

	int items()const { return _matrix_height * _matrix_width; }
	int items_out()const { return _matrix_height * _matrix_height; }

	int schedule()const { return _schedule; }
	int iterations()const { return _iterations; }
	auto gpu_id()const { return _gpu_id; }
	auto device() { return _ptr_device; }

	auto width()const { return _matrix_width; }
	auto height()const { return _matrix_height; }
};

namespace mkmd
{
	int Algebraic_Bernoulli_ABE(mkmd_input matrix_data);
	int Generalized_Algebraic_Bernoulli_GABE(mkmd_input matrix_data);
	int SVD(mkmd_input matrix_data);
	int Continuous_Lyapunov(mkmd_input matrix_data);
	int Matrix_equation(mkmd_input matrix_data);	
};

namespace skmd 
{
	int Benchmark_obj119(mkmd_input matrix_data);
	int Benchmark_obj061(mkmd_input matrix_data);
	int Benchmark_obj129(mkmd_input matrix_data);

	int Benchmark_BS(mkmd_input matrix_data);
	int Benchmark_NB(mkmd_input matrix_data);
	int Benchmark_BL(mkmd_input matrix_data);

	int Benchmark_MM(mkmd_input matrix_data);
	int Benchmark_MT(mkmd_input matrix_data);
	int Benchmark_MA(mkmd_input matrix_data);
	int Benchmark_MV(mkmd_input matrix_data, const bool apply_vector_vector = false);
	int Benchmark_VV(mkmd_input matrix_data);

	int Benchmark_D2D(mkmd_input matrix_data);

};


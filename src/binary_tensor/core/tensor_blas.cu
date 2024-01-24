/*
Copyright 2024 TensorArray-Creators

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>
#include <cassert>
#include <cstring>
#ifndef TENSOR_CONTENT
#define TENSOR_CONTENT
#include "tensor.hh"
#undef TENSOR_CONTENT
#endif // !TENSOR_CONTENT

namespace binary_tensor
{
    namespace value
    {
        constexpr int
            BMMA_M = 8,
            BMMA_N = 8,
            BMMA_K = 128;

        __global__ void kernel_matmul_binary_GEMM(unsigned int batch, int M, int N, int K, void* c, const void* a, const void* b)
        {
            int lda = M;
            int ldb = N;
            int ldc = M;
            __shared__ unsigned int shared_a[128];
            __shared__ unsigned int shared_b[128];
            __shared__ int shared_c[64];
            unsigned int warpBatch = (blockIdx.x * blockDim.x + threadIdx.x);
            unsigned int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
            unsigned int warpN = (blockIdx.z * blockDim.z + threadIdx.z);
#if __CUDA_ARCH__ >= 800
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, BMMA_M, BMMA_N, BMMA_K, nvcuda::wmma::experimental::precision::b1, nvcuda::wmma::row_major> a_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, BMMA_M, BMMA_N, BMMA_K, nvcuda::wmma::experimental::precision::b1, nvcuda::wmma::col_major> b_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, BMMA_M, BMMA_N, BMMA_K, int> acc_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, BMMA_M, BMMA_N, BMMA_K, int> c_frag;
            // Initialize the output to zero
            nvcuda::wmma::fill_fragment(acc_frag, 0);
#endif

            for (unsigned i = 0; i < K; i += (BMMA_K/8))
            {
                for (unsigned j = 0; j < (BMMA_K/32); j++)
                {
                    int aRow = warpM * (BMMA_K/32) + j;
                    int aCol = i;
                    int bRow = i;
                    int bCol = warpN * (BMMA_K/32) + j;
                    // Bounds checking
                    if (aRow < M && aCol < K && bRow < K && bCol < N)
                    {
                        shared_a[threadIdx.y*(BMMA_K/32) + j] = static_cast<const unsigned char*>(a)[(warpBatch*M*K)+(aRow * lda + aCol)];
                        shared_b[threadIdx.z*(BMMA_K/32) + j] = static_cast<const unsigned char*>(b)[(warpBatch*K*N)+(bRow + bCol * ldb)];
                    }
                    else
                    {
                        shared_a[threadIdx.y*(BMMA_K/32) + j] = 0;
                        shared_b[threadIdx.z*(BMMA_K/32) + j] = 0;
                    }
                }

#if __CUDA_ARCH__ >= 800
                // Load the inputs
                nvcuda::wmma::load_matrix_sync(a_frag, shared_a, BMMA_K);
                nvcuda::wmma::load_matrix_sync(b_frag, shared_b, BMMA_K);
                // Perform the matrix multiplication
                nvcuda::wmma::bmma_sync(acc_frag, a_frag, b_frag, acc_frag, nvcuda::wmma::experimental::bmmaBitOpAND);
#endif
            }
#if __CUDA_ARCH__ >= 800
            // Store the output
            nvcuda::wmma::store_matrix_sync(shared_c, acc_frag, M, nvcuda::wmma::mem_row_major);
#endif
            for (unsigned i = 0; i < (BMMA_K/32); i++)
            {
                int cRow = warpM * (BMMA_K/32) + i;
                int cCol = warpN * (BMMA_K/32) + i;
                if (cRow < M && cCol < N)
                {
                    static_cast<unsigned char*>(c)[(warpBatch*M*N)+(cRow * ldc + cCol)];
                }
            }
        }

        template <typename T>
        __global__ void kernel_matmul_GEMM(unsigned int batch, unsigned int M, unsigned int N, unsigned int K, T* c, const T* a, const T* b)
        {
            unsigned int wrapBatch = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int warpM = blockIdx.y * blockDim.y + threadIdx.y;
            unsigned int warpN = blockIdx.z * blockDim.z + threadIdx.z;
            T Cvalue = 0;
            if (wrapBatch < batch && warpM < M && warpN < N)
                for (unsigned int i = 0; i < K; i++)
                    Cvalue =
                        a[wrapBatch * M * K + warpM * K + i] +
                        b[wrapBatch * K * N + i * N + warpN];
            c[wrapBatch * M * N + warpM * N + warpN] = Cvalue;
        }

        Tensor batchedmatmul(const Tensor& a, const Tensor& b, bool is_derive, const DataBuffer&)
		{
			cudaError cudaStat;
			devices::Device this_cuda{ devices::CUDA };
			cudaStat = cudaGetDevice(&this_cuda.index);
			cudaDeviceProp cu_dev_prop;
			cudaStat = cudaGetDeviceProperties(&cu_dev_prop, this_cuda.index);
			TensorBase base_a = a.get_buffer().change_device(this_cuda);
			TensorBase base_b = b.get_buffer().change_device(this_cuda);
			const std::initializer_list<unsigned int> shape_a = base_a.shape();
			const std::initializer_list<unsigned int> shape_b = base_b.shape();
			assert(shape_a.size() == shape_b.size() && std::memcmp(shape_a.begin(), shape_b.begin(), std::min(shape_a.size(), shape_b.size()) - 2) && shape_a.end()[-1] == shape_b.end()[-2]);
			std::vector<std::pair<Tensor, Derivation>> temp;
			if (is_derive)
			{
				temp.push_back(std::make_pair(a, Derivation(b.transpose(shape_b.size() - 2, shape_b.size() - 1, false), batchedmatmul, false)));
				temp.push_back(std::make_pair(b, Derivation(a.transpose(shape_a.size() - 2, shape_a.size() - 1, false), batchedmatmul, true)));
			}
			void* c_ptr;

			unsigned int batch_size = 1U;
			for (std::size_t i = 0; i < shape_a.size() - 2; i++)
				batch_size = shape_a.begin()[i];

			cudaStat = cudaMalloc(&c_ptr, batch_size * shape_a.end()[-2] * shape_b.end()[-1]);
			cudaStat = cudaMemset(c_ptr, 0, batch_size * shape_a.end()[-2] * shape_b.end()[-1]);

            constexpr unsigned int thread_value_x = 1U;
            constexpr unsigned int thread_value_y = 32U;
            constexpr unsigned int thread_value_z = 32U;
            dim3 block_dim(thread_value_x, thread_value_y, thread_value_z);
			dim3 grid_dim
            (
                batch_size / block_dim.x + (batch_size % block_dim.x  ? 1U : 0U),
                shape_a.end()[-2] / block_dim.y + (shape_a.end()[-2] % block_dim.y ? 1U : 0U),
                shape_b.end()[-1] / block_dim.z + (shape_b.end()[-1] % block_dim.z ? 1U : 0U)
            );
            kernel_matmul_GEMM<<<grid_dim, block_dim>>>
            (
                batch_size,
                shape_a.end()[-2], shape_b.end()[-1], shape_a.end()[-1],
                static_cast<uint1_t_x8*>(c_ptr),
                static_cast<const uint1_t_x8*>(base_a.data()),
                static_cast<const uint1_t_x8*>(base_b.data())
            );

			TensorBase value_buf({ batch_size, shape_a.end()[-2] , shape_b.end()[-1] }, c_ptr, this_cuda);
			cudaStat = cudaFree(c_ptr);
			return Tensor(std::move(value_buf), std::move(temp));
		}
    }
}

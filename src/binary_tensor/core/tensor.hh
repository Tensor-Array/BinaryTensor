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

#include <algorithm>
#include <mutex>
#include "tensorbase.hh"

#pragma once

#ifdef __WIN32__
#ifdef CUDA_ML_EXPORTS
#define CUDA_ML_API __declspec(dllexport)
#else
#define CUDA_ML_API __declspec(dllimport)
#endif
#else
#define CUDA_ML_API
#endif

namespace binary_tensor
{
	namespace value
	{
        extern CUDA_ML_API bool use_grad;

#ifdef TENSOR_CONTENT
        void* create_mem_101(std::size_t s, const void* dat);
        class DataBuffer
        {
        public:
            template<typename T>
            constexpr DataBuffer(const T(&data)) :
                data(create_mem_101(sizeof(T), &data)),
                data_size(sizeof(T))
            {
                static_assert(std::is_trivially_copyable_v<T>, "Requied default constructor");
            }
            template<typename T>
            constexpr DataBuffer(const std::initializer_list<T> &data) :
                data(create_mem_101(sizeof(T) * data.size(), data.begin())),
                data_size(sizeof(T) * data.size())
            {
                static_assert(std::is_trivially_copyable_v<T>, "Requied default constructor");
            }
            DataBuffer();
            DataBuffer(std::nullptr_t);
            DataBuffer(const DataBuffer&);
            ~DataBuffer();
            const void* const& get_data() const;
            const std::size_t& get_data_size() const;
            DataBuffer& operator=(const DataBuffer&);
            friend bool operator==(const DataBuffer&, const DataBuffer&);
        private:
            const void* data;
            std::size_t data_size;
        };

        class Derivation;
#endif

        struct dimension
        {
            unsigned int x = 1U, y = 1U, z = 1U;
        };

        struct ConvolutionParameter
        {
            dimension
                input,
                kernel,
                strides,
                dilation;
        };

        /**
         * \brief Dynamic derivative tensor.
         * \brief This class use to calculate the tensor.
         */
        class CUDA_ML_API Tensor
        {
        public:
            /**
             * \brief Create an empty tensor.
             */
            constexpr Tensor() = default;

            /**
             * \brief Base Tensor copy.
             */
            Tensor(const TensorBase&);

            /**
             * \brief Base Tensor move.
             */
            Tensor(TensorBase&&);

            ~Tensor();

            friend class WeakTensor;
            friend struct std::hash<binary_tensor::value::Tensor>;
            friend struct std::equal_to<binary_tensor::value::Tensor>;

            /**
             * \brief This class can iterate copy child tensor by index and derivate to parent tensor,
             */
            class CUDA_ML_API Iterator
            {
            public:
                using iterator_category = std::forward_iterator_tag;
                using difference_type = std::ptrdiff_t;
                using value_type = Tensor;
                using reference = value_type;
                using reference_left = const value_type&;
                Iterator(reference_left, unsigned int);
                reference operator*() const;
                Iterator& operator++();
                Iterator& operator--();
                Iterator operator++(int);
                Iterator operator--(int);
                friend bool CUDA_ML_API operator==(const Iterator&, const Iterator&);
                friend bool CUDA_ML_API operator!=(const Iterator&, const Iterator&);
            private:
                unsigned long long index;
                reference_left ref;
            };
            struct Slice
            {
                int begin = 0;
                int end = -1;
                int strides = 1;
            };
            Tensor clone() const;
            void save(const char*) const;
            int real_index(int) const;
            Slice correct_slice(const Slice&) const;
            bool& multithread_derive();
            long tensor_use_count();
            void calc_grad();
            Iterator begin() const;
            Iterator end() const;
            Iterator cbegin() const;
            Iterator cend() const;
            const TensorBase& get_buffer() const;
            Tensor padding(unsigned int);
            Tensor loss(const Tensor&) const;
            Tensor value_scalar() const;
            Tensor get_grad() const;
            Tensor expand(unsigned char, unsigned int);
            Tensor reshape(const std::initializer_list<unsigned int>&) const;
            Tensor reshape(const std::vector<unsigned int>&) const;
            Tensor conv_padding(const dimension&) const;
#ifdef TENSOR_CONTENT
            friend Tensor derive_transpose(const Tensor&, const Tensor&, bool, const DataBuffer&);

            friend Tensor derive_reshape_cast(const Tensor&, const Tensor&, bool, const DataBuffer&);
#endif
            Tensor transpose(unsigned char, unsigned char) const;
            std::pair<Tensor, Tensor> max(unsigned char = 0) const;
            std::pair<Tensor, Tensor> min(unsigned char = 0) const;
            friend std::pair<Tensor, Tensor> tensor_broadcasting(const Tensor&, const Tensor&, unsigned char, unsigned char);
#ifdef TENSOR_CONTENT
            friend CUDA_ML_API Tensor add_dim(const std::vector<Tensor>&);
#endif
            bool has_tensor() const;
            template<typename T>
            operator T () const;


            Tensor unslice(const std::initializer_list<unsigned int>&, const std::initializer_list<Slice>&) const;

            /**
             * \brief Array Operator.
             * You can chain tensor array operator to a scalar.
             * \param pos Position of this tensor.
             * \return
             * Tensor
             */
            Tensor operator[](unsigned int) const;

            Tensor operator[](const std::initializer_list<Slice>&) const;

            Tensor operator+() const;

            Tensor operator-() const;

            Tensor& operator+=(const Tensor&);

            Tensor& operator-=(const Tensor&);

            Tensor& operator*=(const Tensor&);
            
            Tensor reduce_sum(unsigned char) const;
#ifdef TENSOR_CONTENT
            friend Tensor tensor_rand(const std::initializer_list<unsigned int>&, unsigned int);
            
            friend Tensor add(const Tensor&, const Tensor&, bool);

            friend Tensor multiply(const Tensor&, const Tensor&, bool, const DataBuffer&);

            friend Tensor dot(const Tensor&, const Tensor&, bool, const DataBuffer&);

            friend Tensor condition(const Tensor&, const Tensor&, const Tensor&, bool);

            friend Tensor derive_convolution_padding(const Tensor&, const Tensor&, bool, const DataBuffer&);
            friend Tensor convolution_padding(const Tensor&, const Tensor&, bool, const DataBuffer&);
            friend Tensor convolution_col2im(const Tensor&, const Tensor&, bool, const DataBuffer&);
            friend Tensor convolution_im2col(const Tensor&, const Tensor&, bool, const DataBuffer&);
            friend Tensor matmul(const Tensor&, const Tensor&, bool, const DataBuffer&);
            friend Tensor batchedmatmul(const Tensor&, const Tensor&, bool, const DataBuffer&);
#endif
            friend CUDA_ML_API std::ostream& operator<<(std::ostream&, const Tensor&);

        private:
#ifdef TENSOR_CONTENT
            friend class TensorContentDerivation;
            friend class TensorContentGradient;
            friend struct Derivation;
            Tensor slice(const std::initializer_list<Slice>&, bool) const;
            Tensor reshape(const std::initializer_list<unsigned int>&, bool) const;
            Tensor transpose(unsigned char, unsigned char, bool) const;
            Tensor convolution_convert(const ConvolutionParameter&);
            Tensor(const TensorBase&, const std::vector<std::pair<Tensor, Derivation>>&);
            Tensor(TensorBase&&, std::vector<std::pair<Tensor, Derivation>>&&);
#endif // TENSOR_CONTENT
            struct TensorContent;
            Tensor(const std::shared_ptr<TensorContent>&);
            Tensor(std::shared_ptr<TensorContent>&&);
            std::shared_ptr<TensorContent> tensor_data;
        };

        class CUDA_ML_API WeakTensor
        {
        public:
            WeakTensor(const Tensor&);
            bool is_tensor_expired();
            Tensor to_tensor();
        private:
            std::weak_ptr<Tensor::TensorContent> tensor_data;
        };

        CUDA_ML_API dimension operator+(const dimension&, const dimension&);

        CUDA_ML_API dimension operator-(const dimension&, const dimension&);

        CUDA_ML_API dimension operator*(const dimension&, const dimension&);

        CUDA_ML_API dimension operator/(const dimension&, const dimension&);

        /**
         * \brief Plus 2 n-d tensors.
         * \param other The tensor that plus with this.
         * \return
         * Tensor
         */
        CUDA_ML_API Tensor operator+(const Tensor&, const Tensor&);

        CUDA_ML_API Tensor operator-(const Tensor&, const Tensor&);

        /**
         * \brief Multiply 2 n-d tensors.
         * \param other The tensor that multiply with this.
         * \return
         * Tensor
         */
        CUDA_ML_API Tensor operator*(const Tensor&, const Tensor&);

        CUDA_ML_API Tensor operator/(const Tensor&, const Tensor&);
        CUDA_ML_API Tensor tensor_file_load(const char*);
        CUDA_ML_API Tensor add(const Tensor&, const Tensor&);
        CUDA_ML_API Tensor multiply(const Tensor&, const Tensor&);
        CUDA_ML_API Tensor dot(const Tensor&, const Tensor&);
        /**
         * \brief Matrix multiplication 2 matrices.
         * \param a Matrix/Tensor that has size (batch*)m*k.
         * \param b Matrix/Tensor that has size (batch*)k*n.
         * \return Tensor - Matrix that has size (batch*)m*n.
         * \exception a.col != b.row 
         */
        CUDA_ML_API Tensor matmul(const Tensor&, const Tensor&);
        CUDA_ML_API Tensor condition(const Tensor&, const Tensor&, const Tensor&);
        /**
         * \brief Convolution
         * \brief Only suport 1D, 2D, 3D convolution
         * \param input Tensor (N, C, ...).
         * \param kernel Tensor (C, ..., K).
         * \param strides dimension.
         * \param dilation dimension.
         * \return
         * Tensor (N, K, ...)
         */
        CUDA_ML_API Tensor convolution(const Tensor&, const Tensor&, const dimension& = value::dimension(), const dimension& = value::dimension());
        CUDA_ML_API std::pair<Tensor, Tensor> tensor_broadcasting(const Tensor&, const Tensor&, unsigned char = 0, unsigned char = 0);
        CUDA_ML_API Tensor tensor_rand(const std::initializer_list<unsigned int>&, unsigned int = std::rand());
        CUDA_ML_API Tensor values(const std::initializer_list<unsigned int>&, uint1_t_x8);
#ifndef TENSOR_CONTENT
        CUDA_ML_API Tensor add_dim(const std::vector<Tensor>&);
#endif
        CUDA_ML_API Tensor tensor_rand(const std::vector<unsigned int>&, unsigned int = std::rand());

#ifdef TENSOR_CONTENT
        class Derivation
        {
        private:
            typedef Tensor(*multiply_type)(const Tensor&, const Tensor&, bool, const DataBuffer&);
            Tensor derive_value;
            multiply_type multi;
            bool is_value_before;
            DataBuffer option;
        public:
            Derivation(const Tensor&, const multiply_type, bool = false, const DataBuffer & = DataBuffer());
            Tensor calc_grad_temp(const Tensor&) const;
            friend std::vector<Derivation> check_derive_data(const std::vector<Derivation>&);
        };
#endif

        template<typename T>
        Tensor::operator T () const
        {
            const TensorBase& base = this->get_buffer();
            if (base.shape().size() != 0 && base.type() != typeid(T)) throw 0;
            return *static_cast<const T*>(base.data());
        }

        inline Tensor values(const std::vector<unsigned int>& shape_vector, uint1_t_x8 value)
        {
            return values(wrapper::initializer_wrapper<unsigned int>(shape_vector.begin().operator->(), shape_vector.end().operator->()), value);
        }

        inline Tensor zeros(const std::initializer_list<unsigned int>& shape_list)
        {
            return values(shape_list, (unsigned char)(0));
        }

        inline Tensor zeros(const std::vector<unsigned int>& shape_vector)
        {
            return values(shape_vector, (unsigned char)(0));
        }

        inline Tensor ones(const std::initializer_list<unsigned int>& shape_list)
        {
            return values(shape_list, (unsigned char)(-1));
        }

        inline Tensor ones(const std::vector<unsigned int>& shape_vector)
        {
            return values(shape_vector, (unsigned char)(-1));
        }
}
}

template<>
struct std::hash<binary_tensor::value::Tensor>
{
    inline std::size_t operator()(const binary_tensor::value::Tensor& t) const
    {
        return std::hash<std::shared_ptr<binary_tensor::value::Tensor::TensorContent>>()(t.tensor_data);
    }
};

template<>
struct std::equal_to<binary_tensor::value::Tensor>
{
    inline std::size_t operator()(const binary_tensor::value::Tensor& a, const binary_tensor::value::Tensor& b) const
    {
        return std::equal_to<std::shared_ptr<binary_tensor::value::Tensor::TensorContent>>()(a.tensor_data, b.tensor_data);
    }
};

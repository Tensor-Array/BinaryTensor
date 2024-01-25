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

#include <cstring>
#include "uint1_t_x8.cuh"

__host__ __device__ uint1_t_x8::uint1_t_x8(unsigned char values)
{
    std::memcpy(this, &values, sizeof(uint1_t_x8));
}

__host__ __device__ uint1_t_x8& uint1_t_x8::operator=(unsigned char values)
{
    std::memcpy(this, &values, sizeof(uint1_t_x8));
}

__host__ __device__ uint1_t_x8::operator unsigned char() const
{
    return *reinterpret_cast<const unsigned char*>(this);
}

__device__ uint1_t_x8 atomicAdd(uint1_t_x8*, uint1_t_x8);

__host__ __device__ uint1_t_x8 operator&(const uint1_t_x8& a, const uint1_t_x8& b)
{
    return static_cast<const unsigned char&>(a) & static_cast<const unsigned char&>(b);
}

__host__ __device__ uint1_t_x8 operator|(const uint1_t_x8& a, const uint1_t_x8& b)
{
    return static_cast<const unsigned char&>(a) | static_cast<const unsigned char&>(b);
}

__host__ __device__ uint1_t_x8 operator^(const uint1_t_x8& a, const uint1_t_x8& b)
{
    return static_cast<const unsigned char&>(a) ^ static_cast<const unsigned char&>(b);
}

__host__ __device__ uint1_t_x8 operator~(const uint1_t_x8& values)
{
    return ~static_cast<const unsigned char&>(values);
}

__host__ __device__ uint1_t_x8 operator+(const uint1_t_x8& a, const uint1_t_x8& b)
{
    return a ^ b;
}

__host__ __device__ uint1_t_x8 operator-(const uint1_t_x8& a, const uint1_t_x8& b)
{
    return a ^ b;
}

__host__ __device__ uint1_t_x8 operator*(const uint1_t_x8& a, const uint1_t_x8& b)
{
    return a & b;
}

__host__ __device__ uint1_t_x8 operator/(const uint1_t_x8& a, const uint1_t_x8& b);

__host__ __device__ uint1_t_x8& operator&=(uint1_t_x8& a, const uint1_t_x8& b)
{
    return (a = a & b);
}

__host__ __device__ uint1_t_x8& operator|=(uint1_t_x8& a, const uint1_t_x8& b)
{
    return (a = a | b);
}

__host__ __device__ uint1_t_x8& operator^=(uint1_t_x8& a, const uint1_t_x8& b)
{
    return (a = a ^ b);
}

__host__ __device__ uint1_t_x8& operator+=(uint1_t_x8& a, const uint1_t_x8& b)
{
    return (a = a + b);
}

__host__ __device__ uint1_t_x8& operator-=(uint1_t_x8& a, const uint1_t_x8& b)
{
    return (a = a - b);
}

__host__ __device__ uint1_t_x8& operator*=(uint1_t_x8& a, const uint1_t_x8& b)
{
    return (a = a * b);
}

__host__ __device__ uint1_t_x8& operator/=(uint1_t_x8& a, const uint1_t_x8& b);

namespace binary_tensor
{
    namespace dtype
    {

    }
}

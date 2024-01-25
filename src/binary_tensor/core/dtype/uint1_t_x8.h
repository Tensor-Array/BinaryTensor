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

#ifdef __CUDACC__
#include <device_launch_parameters.h>
#else
#endif
#pragma once

#if defined(__cplusplus)
extern "C" struct uint1_t_x8;
#endif

#ifdef __CUDACC__
#define __binary_tensor_DEVICE_CODE__ __host__ __device__
#else
#define __binary_tensor_DEVICE_CODE__
#endif

struct uint1_t_x8
{
    unsigned char bit_0:1;
    unsigned char bit_1:1;
    unsigned char bit_2:1;
    unsigned char bit_3:1;
    unsigned char bit_4:1;
    unsigned char bit_5:1;
    unsigned char bit_6:1;
    unsigned char bit_7:1;
#if defined(__cplusplus)
    uint1_t_x8() = default;
    __binary_tensor_DEVICE_CODE__ uint1_t_x8(unsigned char);
    __binary_tensor_DEVICE_CODE__ uint1_t_x8& operator=(unsigned char);
    __binary_tensor_DEVICE_CODE__ operator unsigned char() const;
#endif
};

#if defined(__cplusplus)
#include <iostream>

template <typename _CharT, typename _Traits>
inline std::basic_ostream<_CharT, _Traits>& operator<<(std::basic_ostream<_CharT, _Traits>& stream, uint1_t_x8 value)
{
    unsigned int temp;
    temp = value.bit_0;
    stream << temp;
    temp = value.bit_1;
    stream << temp;
    temp = value.bit_2;
    stream << temp;
    temp = value.bit_3;
    stream << temp;
    temp = value.bit_4;
    stream << temp;
    temp = value.bit_5;
    stream << temp;
    temp = value.bit_6;
    stream << temp;
    temp = value.bit_7;
    stream << temp;
    return stream;
}

template <typename _CharT, typename _Traits>
inline std::basic_istream<_CharT, _Traits>& operator>>(std::basic_istream<_CharT, _Traits>& stream, uint1_t_x8 value)
{
    unsigned int temp;
    stream >> temp;
    value.bit_0 = temp;
    stream >> temp;
    value.bit_1 = temp;
    stream >> temp;
    value.bit_2 = temp;
    stream >> temp;
    value.bit_3 = temp;
    stream >> temp;
    value.bit_4 = temp;
    stream >> temp;
    value.bit_5 = temp;
    stream >> temp;
    value.bit_6 = temp;
    stream >> temp;
    value.bit_7 = temp;
    return stream;
}

__binary_tensor_DEVICE_CODE__ uint1_t_x8 operator+(const uint1_t_x8&, const uint1_t_x8&);
__binary_tensor_DEVICE_CODE__ uint1_t_x8 operator-(const uint1_t_x8&, const uint1_t_x8&);
__binary_tensor_DEVICE_CODE__ uint1_t_x8 operator*(const uint1_t_x8&, const uint1_t_x8&);
__binary_tensor_DEVICE_CODE__ uint1_t_x8 operator/(const uint1_t_x8&, const uint1_t_x8&);
__binary_tensor_DEVICE_CODE__ uint1_t_x8 operator&(const uint1_t_x8&, const uint1_t_x8&);
__binary_tensor_DEVICE_CODE__ uint1_t_x8 operator|(const uint1_t_x8&, const uint1_t_x8&);
__binary_tensor_DEVICE_CODE__ uint1_t_x8 operator^(const uint1_t_x8&, const uint1_t_x8&);
__binary_tensor_DEVICE_CODE__ uint1_t_x8& operator+=(uint1_t_x8&, const uint1_t_x8&);
__binary_tensor_DEVICE_CODE__ uint1_t_x8& operator-=(uint1_t_x8&, const uint1_t_x8&);
__binary_tensor_DEVICE_CODE__ uint1_t_x8& operator*=(uint1_t_x8&, const uint1_t_x8&);
__binary_tensor_DEVICE_CODE__ uint1_t_x8& operator/=(uint1_t_x8&, const uint1_t_x8&);
__binary_tensor_DEVICE_CODE__ uint1_t_x8& operator&=(uint1_t_x8&, const uint1_t_x8&);
__binary_tensor_DEVICE_CODE__ uint1_t_x8& operator|=(uint1_t_x8&, const uint1_t_x8&);
__binary_tensor_DEVICE_CODE__ uint1_t_x8& operator^=(uint1_t_x8&, const uint1_t_x8&);
__binary_tensor_DEVICE_CODE__ uint1_t_x8 operator~(const uint1_t_x8&);

namespace binary_tensor
{
    namespace dtype
    {
        using ::uint1_t_x8;
    }
    
}
#endif

#undef __binary_tensor_DEVICE_CODE__

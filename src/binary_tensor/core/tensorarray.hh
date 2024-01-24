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

#include <typeinfo>
#include "dtype/uint1_t_x8.h"

namespace binary_tensor
{
	namespace value
	{
        /**
         * \brief Static tensor.
         * \brief Not use for calulate
         */
        template <unsigned int ...>
        struct TensorArray;

        template <unsigned int sz0, unsigned int ... sz>
        struct TensorArray<sz0, sz...>
        {
            static_assert(sz0 != 0U, "A dimension must have a value.");
            static_assert(sizeof...(sz) <= 255UL, "Max dimension is 255");
            using value_type = TensorArray<sz...>;
            using pointer = TensorArray<sz...>*;
            using reference = TensorArray<sz...>&;
            value_type data[sz0];
            reference operator[](unsigned int index)
            {
                return data[index];
            }

            constexpr value_type operator[](unsigned int index) const
            {
                return data[index];
            }

            pointer begin()
            {
                return &data[0];
            }

            pointer end()
            {
                return &data[sz0];
            }

            const pointer cbegin() const
            {
                return &data[0];
            }

            const pointer cend() const
            {
                return &data[sz0];
            }

            pointer rbegin()
            {
                return &data[sz0 - 1];
            }

            pointer rend()
            {
                return &data[-1];
            }

            const pointer crbegin() const
            {
                return &data[sz0 - 1];
            }

            const pointer crend() const
            {
                return &data[-1];
            }
        };

        template <>
        struct TensorArray<>
        {
            using value_type = dtype::uint1_t_x8;
            using pointer = dtype::uint1_t_x8*;
            using reference = dtype::uint1_t_x8&;
            value_type data;

            operator reference()
            {
                return data;
            }

            constexpr operator value_type() const
            {
                return data;
            }
        };
    }
}
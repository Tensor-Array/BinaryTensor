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

#include "tensorarray.hh"
#include <vector>
#include <type_traits>
#include <memory>
#include "devices.hh"
#include "util/initializer_wrapper.hh"
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
        /**
         * \brief This class look like std::any but it tensor.
         */
        class CUDA_ML_API TensorBase
        {
        private:
            struct TensorStorage
            {
                virtual ~TensorStorage() = default;
                virtual std::unique_ptr<TensorStorage> clone() const = 0;
                virtual std::initializer_list<unsigned int> dim_sizes() const = 0;
                virtual const dtype::uint1_t_x8* data() const = 0;
                virtual size_t data_size() const = 0;
                virtual std::unique_ptr<TensorStorage> child_create(unsigned int) const = 0;
                virtual const devices::Device& get_device() const = 0;
            };

            friend class TensorBuf;

            template <unsigned int ... sz>
            class TensorArrayStorage;

            template <unsigned int sz0, unsigned int ... sz>
            class TensorArrayStorage<sz0, sz...> final : public TensorStorage
            {
            private:
                static constexpr const std::array<unsigned int, sizeof...(sz) + 1ULL> dim_size_array{ sz0, sz... };
                const TensorArray<sz0, sz...> arr_data;
            public:
                constexpr TensorArrayStorage(const TensorArray<sz0, sz...>& arr_data) :
                    arr_data(arr_data)
                {}

                constexpr TensorArrayStorage(TensorArray<sz0, sz...>&& arr_data) :
                    arr_data(std::forward<TensorArray<sz0, sz...>>(arr_data))
                {}

                inline std::unique_ptr<TensorStorage> clone() const override
                {
                    return std::make_unique<TensorArrayStorage<sz0, sz...>>(this->arr_data);
                }

                inline std::initializer_list<unsigned int> dim_sizes() const override
                {
                    return wrapper::initializer_wrapper<unsigned int>(dim_size_array.data(), dim_size_array.data() + sizeof...(sz) + 1ULL);
                }

                inline const dtype::uint1_t_x8* data() const override
                {
                    return reinterpret_cast<const dtype::uint1_t_x8*>(&arr_data);
                }

                inline size_t data_size() const override
                {
                    return sizeof arr_data;
                }

                inline std::unique_ptr<TensorStorage> child_create(unsigned int index) const override
                {
                    return std::make_unique<TensorArrayStorage<sz...>>(this->arr_data[index]);
                }

                inline const devices::Device& get_device() const override
                {
                    return devices::DEVICE_CPU_0;
                }
            };

            std::unique_ptr<TensorStorage> instance;
        public:
            constexpr TensorBase() = default;

            /**
             * \brief Copying TensorArray into TensorBase.
             * \param arr
             */
            template <unsigned int ... sz>
            constexpr TensorBase(const TensorArray<sz...>& arr):
                instance(std::make_unique<TensorArrayStorage<sz...>>(arr))
            {
            }

            /**
             * \brief Forwarding TensorArray into TensorBase.
             * \param arr
             */
            template <unsigned int ... sz>
            constexpr TensorBase(TensorArray<sz...>&& arr) :
                instance(std::make_unique<TensorArrayStorage<sz...>>(std::forward<TensorArray<sz...>>(arr)))
            {
            }

            TensorBase(const std::initializer_list<unsigned int>&, const void* = nullptr, const devices::Device & = devices::local_device(), const devices::Device & = devices::local_device());
            TensorBase(const std::vector<unsigned int>&, const void* = nullptr, const devices::Device& = devices::local_device(), const devices::Device& = devices::local_device());
            TensorBase(const TensorBase&);
            TensorBase(const TensorBase&, const devices::Device&);
            TensorBase(TensorBase&&) = default;
            TensorBase& operator=(const TensorBase&);
            TensorBase operator[](unsigned int) const;
            const std::type_info& type() const;
            std::initializer_list<unsigned int> shape() const;
            const void* data() const;
            size_t data_size() const;
            const devices::Device& get_device() const;
            TensorBase change_device(const devices::Device&) const;
            bool has_tensor() const;
            void swap(TensorBase&);
            void save(const char* dir) const;
        };

        std::size_t get_sizeof_type(const std::type_info&);

        template <>
        class TensorBase::TensorArrayStorage<> final : public TensorBase::TensorStorage
        {
        private:
            const TensorArray<> arr_data;
        public:
            constexpr TensorArrayStorage(const TensorArray<>& arr_data) :
                arr_data(arr_data)
            {}

            constexpr TensorArrayStorage(TensorArray<>&& arr_data) :
                arr_data(std::forward<TensorArray<>>(arr_data))
            {}

            inline std::unique_ptr<TensorStorage> clone() const override
            {
                return std::make_unique<TensorArrayStorage<>>(this->arr_data);
            }

            inline std::initializer_list<unsigned int> dim_sizes() const override
            {
                return std::initializer_list<unsigned int>();
            }

            inline const dtype::uint1_t_x8* data() const override
            {
                return reinterpret_cast<const dtype::uint1_t_x8*>(&this->arr_data);
            }

            inline size_t data_size() const override
            {
                return sizeof arr_data;
            }

            inline std::unique_ptr<TensorStorage> child_create(unsigned int index) const override
            {
                throw 0;
            }

            inline const devices::Device& get_device() const override
            {
                return devices::DEVICE_CPU_0;
            }
        };
    }
}

#undef CUDA_ML_API
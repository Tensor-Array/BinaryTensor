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

#include <initializer_list>

namespace binary_tensor
{
    namespace wrapper
    {
        template<class _E>
        class initializer_wrapper
        {
        public:
            typedef _E 		value_type;
            typedef const _E& 	reference;
            typedef const _E& 	const_reference;
            typedef size_t 		size_type;
            typedef const _E* 	iterator;
            typedef const _E* 	const_iterator;

        private:
#ifdef __GNUC__
            iterator			_M_array;
            size_type			_M_len;
#endif

        public:
            constexpr initializer_wrapper(const_iterator __a, size_type __l):
#ifdef __GNUC__
            _M_array(__a), _M_len(__l)
#endif
            {}

            constexpr initializer_wrapper(const_iterator __begin, const_iterator __end):
#ifdef __GNUC__
            _M_array(__begin), _M_len(__end - __begin)
#endif
            {}
            
            constexpr initializer_wrapper() noexcept:
#ifdef __GNUC__
            _M_array(0), _M_len(0)
#endif
            {}
            
            // Number of elements.
            constexpr size_type
            size() const noexcept
            {
#ifdef __GNUC__
                return _M_len;
#endif
            }
            
            // First element.
            constexpr const_iterator
            begin() const noexcept
            {
#ifdef __GNUC__
                return _M_array;
#endif
            }
            
            // One past the last element.
            constexpr const_iterator
            end() const noexcept
            {
#ifdef __GNUC__
                return begin() + size();
#endif
            }

            constexpr operator std::initializer_list<_E>() const { return reinterpret_cast<const std::initializer_list<_E>&>(*this); }
        };
    }
}
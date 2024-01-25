#include <binary_tensor/core/tensor.hh>

using namespace binary_tensor::value;
using namespace binary_tensor::dtype;

int main(int argc, char const *argv[])
{
    /* code */
    TensorArray<2, 2> a1 =
    {{
        {{
            1
        }},
        {{
            1
        }}
    }};
    TensorArray<2, 2> a2 =
    {{
        {{
            1, 1
        }}
    }};
    Tensor a01 = Tensor(a1);
    Tensor a02 = Tensor(a2);
    auto b = a01 + a02;
    b.calc_grad(ones(b.get_buffer().shape()));
    std::cout << b << std::endl <<
        a01.get_grad() << std::endl <<
        sizeof a1 << std::endl;
    return 0;
}


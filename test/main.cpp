#include <binary_tensor/core/tensorarray.hh>

using namespace binary_tensor::value;

int main(int argc, char const *argv[])
{
    /* code */
    TensorArray<2, 2> a =
    {{
        {{
            1
        }}
    }};
    sizeof a;
    std::cout << a[0][0].data << std::endl;
    return 0;
}


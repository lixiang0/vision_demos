#include <torch/torch.h>
#include <iostream>

int main(){
    torch::Tensor tensor = torch::rand({2,3});
    std::cout << tensor << std::endl;
}

// cmake -DCMAKE_PREFIX_PATH=/home/ruben/workspace/vision_demos/test_libtorch/libtorchgpu/libtorch ..
// cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
// CUDACXX=/usr/local/cuda-11.0/bin/nvcc
// cmake --build . --config Realease
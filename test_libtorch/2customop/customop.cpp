#include <torch/extension.h>



torch::Tensor add(torch::Tensor x,torch::Tensor y) {
  auto s = x+y;
  return s;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", &add, "customop add");
}
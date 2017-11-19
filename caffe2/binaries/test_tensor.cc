#include <Eigen/Dense>
#include <iostream>
#include <stdint.h>
#include <bitset>
#include "unsupported/Eigen/CXX11/Tensor"

using namespace std;
using namespace Eigen;
const int input_size = 3;
const int filter_size = 2;
const int output_size = input_size -filter_size + 1;

void print8(uint8_t* data_ptr, int total, int line_width, string name) {
  std::cout << "Tensor name: " << name << std::endl;
  for (int i = 0; i < total; ++i) {
    std::cout << std::bitset<8>(data_ptr[i]) << " ";
    if ((i + 1) % line_width == 0) {
      std::cout << std::endl;
    }
  }
  return;
}

uint8_t bitcount(uint8_t a) {
  uint8_t res = 0;
  while (a != 0) {
    a = a & (a -  1);
    res++;
  }
  return res;
}

class xnor_uint8_t {
 public:
  uint8_t val;
  xnor_uint8_t() { val = 0; }
  xnor_uint8_t(uint8_t v) : val(v) {}
  xnor_uint8_t operator* (const xnor_uint8_t b) const {
    //return bitcount(!(val^(b.val)));
    std::cout << bitset<8>(val) << "  *  " << bitset<8>(b.val) << " = " << bitset<8>(bitcount(val^(b.val)))  << endl;
    return bitcount(val^(b.val));
  }
  xnor_uint8_t operator+ (const xnor_uint8_t b) const {
    return b.val + val;
  }
  xnor_uint8_t operator+= (const xnor_uint8_t b) {
    val = val + b.val;
    return val;
  }
};

namespace Eigen {
template<> struct NumTraits<xnor_uint8_t> {
  typedef uint8_t Nested;
  typedef uint8_t Real;
  typedef uint8_t Literal;
  typedef uint8_t NonInteger;
  enum {
    IsComplex = 0,
    IsInteger = 1,
    IsSigned = 0,
    RequireInitialization = 0,
    ReadCost = 1,
    AddCost = 2,
    MulCost = 2
  };
  static Real digits10() { return 1; }
  static Real epsilon() { return 1; }
};
}
void test1() {
  uint8_t* input_ptr = (uint8_t*)malloc(input_size * input_size * sizeof(uint8_t));
  uint8_t* filter_ptr = (uint8_t*)malloc(filter_size * filter_size * sizeof(uint8_t));
  uint8_t* output_ptr = (uint8_t*)malloc(output_size * output_size * sizeof(uint8_t));
  for (size_t i = 0; i < input_size * input_size; ++i) {
    input_ptr[i] = 16 + i;
  }
  for (size_t i = 0; i < filter_size * filter_size; ++i) {
    filter_ptr[i] = i;
  }
  for (size_t i = 0; i < output_size * output_size; ++i) {
    output_ptr[i] = 255;
  }
  print8(input_ptr, input_size * input_size, input_size, "input_tensor");
  print8(filter_ptr, filter_size * filter_size, filter_size, "filter_tensor");
  print8(output_ptr, output_size * output_size, output_size, "output_tensor");

  Eigen::Array<int, 4, 1> filter_shuffles({2, 3, 1, 0});
  Eigen::Array<int, 4, 1> input_shuffles({0, 2, 3, 1});
  
  Eigen::Tensor<xnor_uint8_t, 4, Eigen::RowMajor> input_tensor = 
    Eigen::TensorMap<Eigen::Tensor<xnor_uint8_t, 4, Eigen::RowMajor>>(
        (reinterpret_cast<xnor_uint8_t*>(input_ptr)),
        1, 1, input_size, input_size).shuffle(input_shuffles);

  Eigen::Tensor<xnor_uint8_t, 4, Eigen::RowMajor> filter_tensor = 
    Eigen::TensorMap<Eigen::Tensor<xnor_uint8_t, 4, Eigen::RowMajor>>(
        (reinterpret_cast<xnor_uint8_t*>(filter_ptr)),
        1, 1, filter_size, filter_size).shuffle(filter_shuffles);

  typedef Eigen::internal::traits<
    Eigen::Tensor<xnor_uint8_t, 4, Eigen::RowMajor>>::Index TensorIndex; 
  Eigen::array<Eigen::IndexPair<TensorIndex>, 1> contract_dims;
  contract_dims[0] = Eigen::IndexPair<TensorIndex>(1, 0);

  Eigen::DSizes<TensorIndex, 2> pre_contract_dims;
  pre_contract_dims[1] = filter_size * filter_size * 1;
  pre_contract_dims[0] = output_size * output_size / 1;

  Eigen::DSizes<TensorIndex, 2> filter_dims;
  filter_dims[0] = filter_size * filter_size;
  filter_dims[1] = 1;

  Eigen::Tensor<xnor_uint8_t, 4, Eigen::RowMajor> Y_tensor(1, output_size, output_size, 1);
  Y_tensor = input_tensor
              .extract_image_patches(
                  filter_size, filter_size,
                  1, 1,
                  1, 1,
                  1, 1,
                  0, 0, 0, 0, 0)
              .reshape(pre_contract_dims)
              .contract(filter_tensor.reshape(filter_dims), contract_dims)
              .reshape(Y_tensor.dimensions());

  Eigen::Array<int, 4, 1> output_shuffles({0, 3, 1, 2});
  Eigen::TensorMap<Eigen::Tensor<xnor_uint8_t, 4, Eigen::RowMajor>>(
        reinterpret_cast<xnor_uint8_t*>(output_ptr), 1, 1, output_size, output_size) = Y_tensor.shuffle(output_shuffles);

  print8(input_ptr, input_size * input_size, input_size, "input_tensor");
  print8(filter_ptr, filter_size * filter_size, filter_size, "filter_tensor");
  print8(output_ptr, output_size * output_size, output_size, "output_tensor");
}
 int main() {
  test1();
  return 0;
 }

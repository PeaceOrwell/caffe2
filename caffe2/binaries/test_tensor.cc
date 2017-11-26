#include <Eigen/Dense>
#include <iostream>
#include <stdint.h>
#include <bitset>
#include "unsupported/Eigen/CXX11/Tensor"

using namespace std;
using namespace Eigen;
const int input_size = 4;
const int filter_size = 3;
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
void print_float(float* data_ptr, int total, int line_width, string name) {
  std::cout << "Tensor name : " << name << std::endl;
  for (int i = 0 ; i < total; ++i) {
    std::cout << data_ptr[i] << " ";
    if ((i + 1) % line_width == 0) {
      std::cout << std::endl;
    }
  }
}

namespace Eigen {
class xnor_uint8_t {
 public:
  uint8_t val;
  xnor_uint8_t() { val = 0; }
  xnor_uint8_t(float v) {
    val = (v > 0.0? 1 : 0);
  }
  xnor_uint8_t(int v) : val(v) {}
  xnor_uint8_t(uint8_t v) : val(v) {}
  xnor_uint8_t operator* (const xnor_uint8_t b) const {
    uint8_t res = (b.val == 0 ? (val ^ 1) : val);
    std::cout << bitset<8>(val) << "  *  " << bitset<8>(b.val) << " = " << bitset<8>(res)  << endl;
    return res;
  }
  xnor_uint8_t operator+ (const xnor_uint8_t b) const {
    //std::cout << bitset<8>(val) << "  +  " << bitset<8>(b.val) << " = " << bitset<8>(val + b.val)  << endl;
    return b.val + val;
  }
  xnor_uint8_t operator+= (const xnor_uint8_t b) {
    //std::cout << bitset<8>(val) << "  +=  " << bitset<8>(b.val) << " = " << bitset<8>(val + b.val)  << endl;
    val = val + b.val;
    return val;
  }
  friend ostream& operator<<(ostream& os, const xnor_uint8_t& m);
};
ostream& operator<<(ostream& os, const xnor_uint8_t& m) { os << bitset<8>(m.val); return os; }

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
    input_ptr[i] = i;
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

  Eigen::DSizes<TensorIndex, 1> filter_dims;
  filter_dims[0] = filter_size * filter_size;
  //filter_dims[1] = 1;

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
void test2() {
    std::cout << "===============test 2===============" << std::endl;
    Eigen::Tensor<double, 3> tensor(2, 2, 2);
    tensor.setValues({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    Eigen::Tensor<double, 2> tensor2(2, 2);
    tensor2.setValues({{1, 2}, {3, 4}});
    Eigen::Tensor<double, 1> tensor1;

    std::array<Eigen::IndexPair<int>, 2> product_dims;
    product_dims[0] =  { IndexPair<int>(0, 0) };
    product_dims[1] =  { IndexPair<int>(2, 1) };
    auto vv = tensor.contract(tensor2, product_dims);
    cerr<<" value: "<<vv<<endl;
    tensor1 = vv;
}
void test3() {
    std::cout << "===============test 3===============" << std::endl;
    Eigen::Tensor<double, 2> tensor(2, 2);
    tensor.setValues({{1, 2}, {3, 4}});
    Eigen::Tensor<double, 2> tensor2(2, 2);
    tensor2.setValues({{1, 2}, {3, 4}});

    std::array<Eigen::IndexPair<int>, 1> product_dims;
    product_dims[0] =  { IndexPair<int>(1, 1) };
    //product_dims[1] =  { IndexPair<int>(0, 1) };
    auto vv = tensor.contract(tensor2, product_dims);
    cerr<<" value: "<<vv<<endl;
}
void test4() {
  std::cout << "===============test 4=============" << std::endl;
  // set the shape of input, filter and output
  int ni = 1, hi = 3, wi = 3;
  int co = 3, ci  = 3, kh = 2, kw = 2;
  int no = ni, ho = (hi - kh) + 1, wo = ho;; 
  int input_size = ni * ci * hi * wi;
  int filter_size = co * ci * kh * kw;
  int output_size = no * co * ho * wo;

  float* input_data_ptr = (float*)malloc(input_size * sizeof(float));
  float* filter_data_ptr = (float*)malloc(filter_size * sizeof(float));
  float* output_data_ptr = (float*)malloc(output_size * sizeof(float));
  uint8_t* output_data_uint8_ptr = (uint8_t*)malloc(output_size * sizeof(uint8_t));

  for (int i = 0; i < input_size; ++i) {
    input_data_ptr[i] = (rand() % 10 -5) / 10.0;
  }
  for (int i = 0; i < filter_size; ++i) {
    filter_data_ptr[i] = (rand() % 10 -5) / 10.0;
  }
  for (int i = 0; i < output_size; ++i) {
    output_data_ptr[i] = (rand() % 10 - 5) / 10.0;
  }
  Eigen::Array<int, 4, 1> filter_shuffles({2, 3, 1, 0});
  Eigen::Array<int, 4, 1> input_shuffles({0, 2, 3, 1});

  Eigen::Tensor<float, 4, Eigen::RowMajor> input_tensor_float = 
    Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>>(
        input_data_ptr, ni, ci, hi, wi).shuffle(input_shuffles);

  Eigen::Tensor<float, 4, Eigen::RowMajor> filter_tensor_float = 
    Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>>(
        filter_data_ptr, co, ci, kh, kw).shuffle(filter_shuffles);
  
  print_float(input_data_ptr, input_size, wi, "input tensor");
  print_float(filter_data_ptr, filter_size, kw, "filter tensor");
  print_float(output_data_ptr, output_size, wo, "output tensor");
    
  // get the mean val across channels
  Eigen::Tensor<float, 4, Eigen::RowMajor> input_tensor_mean(ni, hi, wi, 1);
  for (int i = 0; i < ni; ++i) {
    for (int j = 0; j < hi; ++j) {
      for (int k = 0; k < wi; ++k) {
        float temp = 0.0;
        for (int w = 0; w < ci; ++w) {
          temp += input_tensor_float(i, j, k, w);
        }
        input_tensor_mean(i, j, k, 0) = temp / ci;
      }
    }
  }
  typedef Eigen::internal::traits<
    Eigen::Tensor<float, 4, Eigen::RowMajor>>::Index MeanIndex; 
  Eigen::array<Eigen::IndexPair<MeanIndex>, 1> mean_contract_dims;
  mean_contract_dims[0] = Eigen::IndexPair<MeanIndex>(1, 0);

  Eigen::DSizes<MeanIndex, 2> mean_pre_contract_dims;
  mean_pre_contract_dims[1] = kh * kw;
  mean_pre_contract_dims[0] = ho * wo * no;

   Eigen::Tensor<float, 2, Eigen::RowMajor> mean_filter(kh * kw, 1);
   mean_filter.setConstant(1.0/(kw * kh));
   std::cout << mean_filter << std::endl;
   
   Eigen::Array<int, 4, 1> output_mean_shuffles({0, 3, 1, 2});
   Eigen::Tensor<float, 4, Eigen::RowMajor> mean_output_tensor(no, ho, wo, 1);
   mean_output_tensor = input_tensor_mean 
               .extract_image_patches(
                   kw, kh,
                   1, 1,
                   1, 1,
                   1, 1,
                   0, 0, 0, 0, 0)
                .reshape(mean_pre_contract_dims)
                .contract(mean_filter, mean_contract_dims)
                .reshape(mean_output_tensor.dimensions())
                .shuffle(output_mean_shuffles);
  std::cout << mean_output_tensor << std::endl; 
  Eigen::Tensor<xnor_uint8_t, 4, Eigen::RowMajor> input_tensor_xnor(ni, hi, wi, ci);
  for (int i = 0; i < ni; ++i) {
    for (int j = 0; j < hi; ++j) {
      for (int k = 0; k < wi; ++k) {
        for (int w = 0; w < ci; ++w) {
          input_tensor_xnor(i, j, k, w) = input_tensor_float(i, j, k, w);
        }
      }
    }
  }
  // get mean of filter
  float filter_mean = 0.0;
  for (int i = 0; i < co; ++i) {
    for (int j = 0; j < ci; ++j) {
      for (int k = 0; k < kh; ++k) {
        for (int w = 0; w < kw; ++w) {
          filter_mean += filter_tensor_float(i, j, k, w); 
        }
      }
    }
  }
  filter_mean /= filter_size;
  //float filter_mean = filter_tensor_float.mean();
  Eigen::Tensor<xnor_uint8_t, 4, Eigen::RowMajor> filter_tensor_xnor(kh, kw, ci, co);
  for (int i = 0; i < kh; ++i) {
    for (int j = 0; j < kw; ++j) {
      for (int k = 0; k < ci; ++k) {
        for (int w = 0; w < co; ++w) {
          filter_tensor_xnor(i, j, k, w)  = filter_tensor_float(i, j, k, w);
        }
      }
    }
  }

  typedef Eigen::internal::traits<
    Eigen::Tensor<xnor_uint8_t, 4, Eigen::RowMajor>>::Index TensorIndex; 
  Eigen::array<Eigen::IndexPair<TensorIndex>, 1> contract_dims;
  contract_dims[0] = Eigen::IndexPair<TensorIndex>(1, 0);

  Eigen::DSizes<TensorIndex, 2> pre_contract_dims;
  pre_contract_dims[1] = kh * kw * ci;
  pre_contract_dims[0] = ho * wo * no;

  Eigen::DSizes<TensorIndex, 2> filter_dims;
  filter_dims[0] = kh * kw * ci;
  filter_dims[1] = co;

  Eigen::Array<int, 4, 1> output_shuffles({0, 3, 1, 2});
  Eigen::Tensor<xnor_uint8_t, 4, Eigen::RowMajor> Y_tensor(1, ho, wo, co);
  Y_tensor = input_tensor_xnor
              .extract_image_patches(
                  kw, kh,
                  1, 1,
                  1, 1,
                  1, 1,
                  0, 0, 0, 0, 0)
              .reshape(pre_contract_dims)
              .contract(filter_tensor_xnor.reshape(filter_dims), contract_dims)
              .reshape(Y_tensor.dimensions())
              .shuffle(output_shuffles);

  //Eigen::TensorMap<Eigen::Tensor<xnor_uint8_t, 4, Eigen::RowMajor>>(
  //      reinterpret_cast<xnor_uint8_t*>(output_data_uint8_ptr), no, co, ho, wo) = Y_tensor.shuffle(output_shuffles);

  //print8(input_ptr, input_size * input_size, input_size, "input_tensor");
  //print8(filter_ptr, filter_size * filter_size, filter_size, "filter_tensor");
  //print8(output_data_uint8_ptr, output_size, wo, "output_tensor");
  for (int i = 0; i < no; ++i) {
    for (int j = 0; j < co; ++j) {
      for (int k = 0; k < ho; ++k) {
        for (int w = 0; w < wo; ++w) {
          output_data_ptr[i * co * ho * wo + j * ho * wo + k * wo + w] = Y_tensor(i, j, k, w).val * filter_mean;          
        }
      }
    }
  }
  using EigenArrayMap = Eigen::Map<Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>;
  using EigenVectorArrayMap = Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1> >;
   EigenArrayMap Y_arr(
         output_data_ptr, co, no * ho * wo);
   EigenVectorArrayMap scale_arr(mean_output_tensor.data(), ho * wo );
//   Y_arr = Y_arr.colwise() * scale_arr;

  print_float(output_data_ptr, output_size, wo, "output tensor");
  free(input_data_ptr);
  free(filter_data_ptr);
  free(output_data_ptr);
}

int main() {
 test1();
 test2();
 test3();
 test4();
 return 0;
} 

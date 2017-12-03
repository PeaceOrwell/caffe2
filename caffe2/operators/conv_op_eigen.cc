#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/core/timer.h"

#include <stdint.h>
#include <bitset>
#include <iostream>

#include "unsupported/Eigen/CXX11/Tensor"

using namespace Eigen;

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
  inline xnor_uint8_t operator* (const xnor_uint8_t b) const {
    //uint8_t res = (b.val == 0 ? (val ^ 1) : val);
    return (b.val == 0 ? (val ^ 1) : val);
    //return val;
  }
  inline xnor_uint8_t operator+ (const xnor_uint8_t b) const {
    return b.val + val;
  }
  xnor_uint8_t operator+= (const xnor_uint8_t b) {
    val = val + b.val;
    return val;
  }
  friend std::ostream& operator<<(std::ostream& os, const xnor_uint8_t& m);
};
std::ostream& operator<<(std::ostream& os, const xnor_uint8_t& m) { os << std::bitset<8>(m.val); return os; }

template<> struct NumTraits<xnor_uint8_t> {
  typedef xnor_uint8_t Nested;
  typedef uint8_t Real;
  typedef xnor_uint8_t Literal;
  typedef xnor_uint8_t NonInteger;
  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 0,
    RequireInitialization = 0,
    ReadCost = 1,
    AddCost = 1,
    MulCost = 1
  };
  static Real digits10() { return 1; }
  static Real epsilon() { return 1; }
};
}

namespace caffe2 {

template <typename T>
class EigenConvOp final : public ConvPoolOpBase<CPUContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CPUContext);
  EigenConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(group_ == 1, "Group convolution not supported yet.");
  }
  ~EigenConvOp() {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 private:
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

// The NCHW implementation: we do explicit transposes before and after, which
// are not ideal but provides a compatible path instead of throwing the error.
template <typename T>
bool EigenConvOp<T>::RunOnDeviceWithOrderNCHW() {
  std::cout << "go into conv" << std::endl;
  auto time1 = new Timer();
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  auto* Y = Output(0);
  const int N = X.dim32(0), C = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
  CAFFE_ENFORCE(4 == filter.ndim());
  const int M = filter.dim32(0);
  CAFFE_ENFORCE(filter.dim32(1) == C);
  CAFFE_ENFORCE(filter.dim32(2) == kernel_h());
  CAFFE_ENFORCE(filter.dim32(3) == kernel_w());
  ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, filter.dim32(0));
  Eigen::array<TIndex, 4> kernel_shuffles
      { {TIndex(2), TIndex(3), TIndex(1), TIndex(0)} };
  Eigen::array<TIndex, 4> input_shuffles
      { {TIndex(0), TIndex(2), TIndex(3), TIndex(1)} };
  auto time_shu1 = new Timer(); 
  Eigen::Tensor<T, 4, Eigen::RowMajor> filter_tensor =
      Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>(
          const_cast<T*>(filter.template data<T>()),
          M,
          C,
          kernel_h(),
          kernel_w())
          .shuffle(kernel_shuffles);
  std::cout << "time_shuffle 1 : " << time_shu1->MilliSeconds() << std::endl;
  auto time_shu2 = new Timer(); 
  Eigen::Tensor<T, 4, Eigen::RowMajor> X_tensor =
      Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>(
          const_cast<T*>(X.template data<T>()), N, C, H, W)
          .shuffle(input_shuffles);
  std::cout << "time_shuffle 2 : " << time_shu2->MilliSeconds() << std::endl;
  std::cout << "time1 : " << time1->MilliSeconds() << std::endl;
  delete(time1);
  auto time2 = new Timer();
  auto time2_con = new Timer();
//  the mean tensor across all channels
  Eigen::Tensor<float, 4, Eigen::RowMajor>  input_tensor_mean(N, H, W, 1);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < H; ++j) {
      for (int k = 0; k < W; ++k) {
        float temp = 0.0;
        for (int w = 0; w < C; ++w) {
          temp += X_tensor(i, j, k, w);
        }
        input_tensor_mean(i, j, k, 0) = temp / C;
      }
    }
  }

  std::cout << "time2 get mean float1 : " << time2_con->MilliSeconds() << std::endl;
  auto time2_con2 = new Timer();
  typedef Eigen::internal::traits<
    Eigen::Tensor<float, 4, Eigen::RowMajor>>::Index MeanIndex; 
  Eigen::array<Eigen::IndexPair<MeanIndex>, 1> mean_contract_dims;
  mean_contract_dims[0] = Eigen::IndexPair<MeanIndex>(1, 0);

  Eigen::DSizes<MeanIndex, 2> mean_pre_contract_dims;
  mean_pre_contract_dims[1] = kernel_h() * kernel_w();
  mean_pre_contract_dims[0] = Y->dim32(3) * Y->dim32(2) * N;
  std::cout << "X->dim32: " << N << " " << C << " " << H << " " << W << std::endl;
  std::cout << "filter->dim32: " << M << " " << C << " " << kernel_h() << " " << kernel_w() << std::endl;
  std::cout << "Y->dim32: " << Y->dim32(0) << " " << Y->dim32(1) << " " << Y->dim32(2) << " " << Y->dim32(3) << std::endl;

   Eigen::Tensor<float, 2, Eigen::RowMajor> mean_filter(kernel_h() * kernel_w(), 1);
   mean_filter.setConstant(1.0/(kernel_h() * kernel_w()));
   
   Eigen::Array<int, 4, 1> output_mean_shuffles({0, 3, 1, 2});
   Eigen::Tensor<float, 4, Eigen::RowMajor> mean_output_tensor(N, Y->dim32(3), Y->dim32(2), 1);
   mean_output_tensor = input_tensor_mean 
               .extract_image_patches(
                   kernel_w(), kernel_h(),
                   1, 1,
                   1, 1,
                   1, 1,
                   0, 0, 0, 0, 0)
                .reshape(mean_pre_contract_dims)
                .contract(mean_filter, mean_contract_dims)
                .reshape(mean_output_tensor.dimensions());

  std::cout << "time2 get mean float2 : " << time2_con2->MilliSeconds() << std::endl;
  auto time2_input_xnor = new Timer();
  Eigen::Tensor<xnor_uint8_t, 4, Eigen::RowMajor> input_tensor_xnor(N, H, W, C);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < H; ++j) {
      for (int k = 0; k < W; ++k) {
        for (int w = 0; w < C; ++w) {
          input_tensor_xnor(i, j, k, w) = X_tensor(i, j, k, w);
        }
      }
    }
  }

  std::cout << "time2 get input xnor: " << time2_input_xnor->MilliSeconds() << std::endl;
  //  get mean of filter
  float filter_mean = 0.0;
  for (int i = 0; i < kernel_h(); ++i) {
    for (int j = 0; j < kernel_w(); ++j) {
      for (int k = 0; k < C; ++k) {
        for (int w = 0; w < M; ++w) {
          filter_mean += filter_tensor(i, j, k, w); 
        }
      }
    }
  }
  filter_mean /= (M * C * kernel_h() * kernel_w());;
  //  get xnor filter
  Eigen::Tensor<xnor_uint8_t, 4, Eigen::RowMajor> filter_tensor_xnor(kernel_h(), kernel_w(), C, M);
  for (int i = 0; i < kernel_h(); ++i) {
    for (int j = 0; j < kernel_w(); ++j) {
      for (int k = 0; k < C; ++k) {
        for (int w = 0; w < M; ++w) {
          filter_tensor_xnor(i, j, k, w)  = filter_tensor(i, j, k, w);
        }
      }
    }
  }
  std::cout << "time2 : " << time2->MilliSeconds() << std::endl;
  delete(time2);
  auto time3 = new Timer();
  typedef Eigen::internal::traits<
    Eigen::Tensor<xnor_uint8_t, 4, Eigen::RowMajor>>::Index TensorIndex; 
  Eigen::array<Eigen::IndexPair<TensorIndex>, 1> contract_dims;
  contract_dims[0] = Eigen::IndexPair<TensorIndex>(1, 0);

  Eigen::DSizes<TensorIndex, 2> pre_contract_dims;
  pre_contract_dims[1] = kernel_h() * kernel_w() * C;
  pre_contract_dims[0] = Y->dim32(3) * Y->dim32(2) * N;

  Eigen::DSizes<TensorIndex, 2> filter_dims;
  filter_dims[0] = kernel_h() * kernel_w() * C;
  filter_dims[1] = M;

  Eigen::Tensor<xnor_uint8_t, 4, Eigen::RowMajor> Y_tensor(Y->dim32(0), Y->dim32(2), Y->dim32(3), Y->dim32(1));
  Y_tensor = input_tensor_xnor
              .extract_image_patches(
                  kernel_w(), kernel_h(),
                  1, 1,
                  1, 1,
                  1, 1,
                  0, 0, 0, 0, 0)
              .reshape(pre_contract_dims)
              .contract(filter_tensor_xnor.reshape(filter_dims), contract_dims)
              .reshape(Y_tensor.dimensions());

  std::cout << "xnor compute time " << time3->MilliSeconds() << std::endl;
  delete(time3);
  auto time4 = new Timer();
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < Y->dim32(2); ++j) {
      for (int k = 0; k < Y->dim32(3); ++k) {
        for (int w = 0; w < M; ++w) {
          Y->template mutable_data<T>()[i * M * Y->dim32(3) * Y->dim32(2) + j * M * Y->dim32(2) + k * M + w] = Y_tensor(i, j, k, w).val * filter_mean;          
        }
      }
    }
  }

  //  if (InputSize() == 3) {
  //    auto& bias = Input(BIAS);
  //    CAFFE_ENFORCE(1 == bias.ndim());
  //    CAFFE_ENFORCE(bias.dim32(0) == M);
  //    // It seems that the bias broadcast is still slower so let's do the
  //    // following for now.
  //    EigenArrayMap<T> Y_arr(
  //        Y_tensor.data(), static_cast<TIndex>(M), Y->size() / M);
  //    ConstEigenVectorArrayMap<T> bias_arr(bias.template data<T>(), M);
  //    Y_arr = Y_arr.colwise() + bias_arr;
  //  }

   EigenArrayMap<T> Y_arr(
         Y->template mutable_data<T>(), M, N * Y->dim32(2) * Y->dim32(3));
   EigenVectorArrayMap<T> scale_arr(mean_output_tensor.data(), N * Y->dim32(3) * Y->dim32(2));
   Y_arr = Y_arr.colwise() * scale_arr;
  std::cout << "get result time " << time4->MilliSeconds() << std::endl;
  // Do a last transpose.
  //Eigen::array<TIndex, 4> output_shuffles
  //    { {TIndex(0), TIndex(3), TIndex(1), TIndex(2) } };

  //Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>(
  //    Y->template mutable_data<T>(), N, M, Y->dim32(2), Y->dim32(3)) =
  //    Y_tensor.shuffle(output_shuffles);
   std::cout << "finish conv" << std::endl;
  return true;
}

template <typename T>
bool EigenConvOp<T>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  auto* Y = Output(0);
  const int N = X.dim32(0), H = X.dim32(1), W = X.dim32(2), C = X.dim32(3);
  CAFFE_ENFORCE(4 == filter.ndim());
  const int M = filter.dim32(0);
  CAFFE_ENFORCE(filter.dim32(1) == kernel_h());
  CAFFE_ENFORCE(filter.dim32(2) == kernel_w());
  CAFFE_ENFORCE(filter.dim32(3) == C);
  ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, filter.dim32(0));
  // Eigen expects filter to be of shape (kernel_h, kernel_w, C, M) for
  // optimization purposes, so we will create a temp one.
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> temp_filter(
      M, kernel_h() * kernel_w() * C);
  temp_filter = ConstEigenArrayMap<T>(
                    filter.template data<T>(), kernel_h() * kernel_w() * C, M)
                    .transpose();

  // Create tensor maps, and call spatial convolution.
  // TODO(jiayq): right now we const cast away the const pointer, but we will
  // need to figure out how to properly do a const tensormap.
  Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> X_tensor(
      const_cast<T*>(X.template data<T>()), N, H, W, C);
  Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> Y_tensor(
      Y->template mutable_data<T>(), N, Y->dim32(1), Y->dim32(2), M);
  Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> filter_tensor(
      const_cast<T*>(temp_filter.data()), kernel_h(), kernel_w(), C, M);

  // For Eigen, the definition of row and col actually correspond to width
  // and height instead of the other way round, so notice how we pass the
  // stride, pad and dilation values.
  typedef typename Eigen::internal::traits<
      Eigen::Tensor<T, 4, Eigen::RowMajor>>::Index TensorIndex;
  Eigen::array<Eigen::IndexPair<TensorIndex>, 1> contract_dims;
  contract_dims[0] = Eigen::IndexPair<TensorIndex>(1, 0);

  Eigen::DSizes<TensorIndex, 2> pre_contract_dims;
  pre_contract_dims[1] = kernel_h() * kernel_w() * C;
  pre_contract_dims[0] = Y->size() / M;

  Eigen::DSizes<TensorIndex, 2> kernel_dims;
  kernel_dims[0] = kernel_h() * kernel_w() * C;
  kernel_dims[1] = M;

  Eigen::array<TensorIndex, 4> bcast_dims;
  bcast_dims[0] = N;
  bcast_dims[1] = Y->dim32(1);
  bcast_dims[2] = Y->dim32(2);
  bcast_dims[3] = 1;

  Y_tensor = X_tensor
                 .extract_image_patches(
                     kernel_w(),
                     kernel_h(),
                     stride_w(),
                     stride_h(),
                     dilation_w(),
                     dilation_h(),
                     1,
                     1,
                     pad_l(),
                     pad_r(),
                     pad_t(),
                     pad_b(),
                     0)
                 .reshape(pre_contract_dims)
                 .contract(filter_tensor.reshape(kernel_dims), contract_dims)
                 .reshape(Y_tensor.dimensions());

  if (InputSize() == 3) {
    auto& bias = Input(BIAS);
    CAFFE_ENFORCE(1 == bias.ndim());
    CAFFE_ENFORCE(bias.dim32(0) == M);
    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> bias_tensor(
        const_cast<T*>(bias.template data<T>()), 1, 1, 1, M);
    // It seems that the bias broadcast is still slower so let's do the
    // following for now.
    EigenArrayMap<T> Y_arr(
        Y->template mutable_data<T>(), static_cast<TIndex>(M), Y->size() / M);
    ConstEigenVectorArrayMap<T> bias_arr(bias.template data<T>(), M);
    Y_arr = Y_arr.colwise() + bias_arr;
  }
  return true;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(Conv, EIGEN, EigenConvOp<float>);

} // namespace caffe2

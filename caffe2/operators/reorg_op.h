#ifndef OPERATOR_REORG_OP_H
#define OPERATOR_REORG_OP_H

#include "caffe2/core/operator.h"

namespace caffe2 {

template<typename Dtype>
void reorg_cpu(const Dtype *bottom_data, const int b_w, const int b_h, const int b_c, const int b_n, const int stride, const bool forward, Dtype *top_data) {
  int t_c = b_c / (stride * stride);
  int t_w = b_w * stride;
  int t_h = b_h * stride;
  for (int n = 0; n < b_n; n++) {
    for (int c = 0; c < b_c; c++) {
      for (int h = 0; h < b_h; h++) {
        for (int w = 0; w < b_w; w++) {
          int bottom_index = w + b_w * (h + b_h * (c + b_c * n));
          int c2 = c % t_c;
          int offset = c / t_c;
          int w2 = w * stride + offset % stride;
          int h2 = h * stride + offset / stride;
          int top_index = w2 + t_w * (h2 + t_h * (c2 + t_c * n));
          if (forward) { 
            top_data[top_index] = bottom_data[bottom_index];
          } else {
            top_data[bottom_index] = bottom_data[top_index];
          }
        }
      }
    }
  }
}

template <typename T, class Context>
class ReorgOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ReorgOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
       stride_(OperatorBase::GetSingleArgument<int32_t>("stride", 1)),
       reverse_(OperatorBase::GetSingleArgument<bool>("reverse", false)){}
  ~ReorgOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* Y = Output(0);
    CAFFE_ENFORCE(X.ndim() == 4, X.ndim());
    const int N = X.dim32(0);
    const int C = X.dim32(1);
    const int H = X.dim32(2);
    const int W = X.dim32(3);
    auto reorg_shape = X.dims();
    if (reverse_) {
      reorg_shape[1] /= (stride_ * stride_);
      reorg_shape[2] *= stride_;
      reorg_shape[3] *= stride_;
    } else {
      reorg_shape[1] *= (stride_ * stride_);
      reorg_shape[2] /= stride_;
      reorg_shape[3] /= stride_;
    }
   Y->Resize(reorg_shape);
    reorg_cpu<T>(X.template data<T>(), W, H, C, N, stride_, reverse_, Y->template mutable_data<T>());
    return true;
  }
 protected:
  int32_t stride_;
  bool reverse_;
};

}  // namespace caffe2

#endif  // OPERATOR_REOEG_OP_H

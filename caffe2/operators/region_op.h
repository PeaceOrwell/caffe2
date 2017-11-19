#ifndef OPERATOR_REGION_OP_H
#define OPERATOR_REGION_OP_H

#include "caffe2/core/operator.h"
#include <cmath>

namespace caffe2 {
template<typename Dtype>
void flatten(Dtype *out, int area, int channel, int batch, bool forward) {
  Dtype *swap = (Dtype *) malloc((size_t) (area * channel * batch) * sizeof(Dtype));
  int i, c, b;
  for (b = 0; b < batch; ++b) {
    for (c = 0; c < channel; ++c) {
      for (i = 0; i < area; ++i) {
        int i1 = b * channel * area + c * area + i;
        int i2 = b * channel * area + i * channel + c;
        if (forward) {
          swap[i2] = out[i1];
        } else {
          swap[i1] = out[i2];
        }
      }
    }
  }
  memcpy(out, swap, sizeof(Dtype) * area * channel * batch);
  free(swap);
}
template<typename Dtype>
void do_softmax(Dtype *input, int n, Dtype temp, Dtype *output) {
  int i;
  Dtype sum = 0;
  Dtype largest = input[0];
  for (i = 0; i < n; ++i) {
    if (input[i] > largest) largest = input[i];
  }
  for (i = 0; i < n; ++i) {
    Dtype e = exp(input[i] / temp - largest / temp);
    sum += e;
    output[i] = e;
  }
  for (i = 0; i < n; ++i) {
    output[i] /= sum;
  }
}
float logistic_activate(float x) { return (float) (1. / (1 + exp(-x))); }
double logistic_activate(double x) { return (double) (1. / (1 + exp(-x))); }

template <typename T, class Context>
class RegionOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RegionOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
       classes_(OperatorBase::GetSingleArgument<int32_t>("classes", 20)),
       coords_(OperatorBase::GetSingleArgument<int32_t>("coords", 4)),
       softmax_(OperatorBase::GetSingleArgument<bool>("softmax", false)),
       boxes_of_each_grid_(OperatorBase::GetSingleArgument<int32_t>("boxes_of_each_grid", 5)){}
  ~RegionOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* Y = Output(0);
    CAFFE_ENFORCE(X.ndim() == 4, X.ndim());
    const int N = X.dim32(0);
    const int C = X.dim32(1);
    const int H = X.dim32(2);
    const int W = X.dim32(3);
    Y->ResizeLike(X);
    const auto* x_data = X.template data<T>();
    auto* y_data = Y->template mutable_data<T>();
    memcpy(y_data, x_data, sizeof(T) * N * C * H * W);
    int size = coords_ + classes_ + 1;
    flatten<T>(y_data, W * H, boxes_of_each_grid_ * size, N, true);
    // logistic_activate the scale value
    for (int b = 0; b < N; ++b) {
      for (int i = 0; i < H * W *boxes_of_each_grid_; ++i) {
        int index = size * i + b * H * W * C;
        y_data[index + 4] = logistic_activate(y_data[index + 4]); 
      }
    }
    // softmax the problity of all the classes
    if (softmax_) {
      for (int b = 0; b < N; ++b) {
        std::cout << "do softmax" << std::endl;
        for (int i = 0; i < H * W * boxes_of_each_grid_; ++i) {
          int index = size * i + b * H * W * C;
          do_softmax<T>(y_data + index + 5, classes_, (T) 1, y_data + index + 5);
        }
      }
    }
    return true;
  }
 protected:
  int32_t classes_;
  int32_t coords_;
  int32_t boxes_of_each_grid_;
  bool softmax_;
};

}  // namespace caffe2

#endif  // OPERATOR_REOEG_OP_H

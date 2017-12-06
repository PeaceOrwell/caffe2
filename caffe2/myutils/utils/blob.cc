#include "caffe2/myutils/utils/blob.h"
#include "caffe2/myutils/utils/tensor.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

TensorCPU BlobUtil::Get() {
#ifdef WITH_CUDA
  if (blob_.IsType<TensorCUDA>()) {
    return TensorCPU(blob_.Get<TensorCUDA>());
  }
#endif
  return blob_.Get<TensorCPU>();
}

void BlobUtil::Set(const TensorCPU &value, bool force_cuda) {
#ifdef WITH_CUDA
  if (force_cuda || blob_.IsType<TensorCUDA>()) {
    auto tensor = blob_.GetMutable<TensorCUDA>();
    tensor->ResizeLike(value);
    tensor->ShareData(value);
    return;
  }
#endif
  auto tensor = blob_.GetMutable<TensorCPU>();
  tensor->ResizeLike(value);
  tensor->ShareData(value);
}

void BlobUtil::Print(const std::string &name, int max) {
  auto tensor = Get();
  TensorUtil(tensor).Print(name, max);
}

}  // namespace caffe2

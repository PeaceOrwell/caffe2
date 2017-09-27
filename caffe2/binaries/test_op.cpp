#include "caffe2/core/init.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/operator_gradient.h"

#include "caffe2/utils/myutil/print.h"
#include "caffe2/myutils/utils/net.h"

namespace caffe2 {

void run() {
  Workspace workspace;
  DeviceOption option;
  option.set_device_type(CUDA);
  option.set_cuda_gpu_id(0);
  CUDAContext* context = new CUDAContext(option);

  std::vector<float> data(2);
  for (auto& v: data) {
    v = 0.0;
  }
  std::vector<float> data_grad(2);
  for (auto& v: data_grad) {
    v = 10.0;
  }

  std::vector<float> one(1);
  for (auto& v: one) {
    v = 1.0;
  }

  {
    auto tensor = workspace.CreateBlob("data")->GetMutable<TensorCUDA>();
    auto value = TensorCUDA({2}, data, context);
    tensor->ResizeLike(value);
    tensor->ShareData(value);
  }
  
  {
    auto tensor = workspace.CreateBlob("one")->GetMutable<TensorCUDA>();
    auto value = TensorCUDA({1}, one, context);
    tensor->ResizeLike(value);
    tensor->ShareData(value);
  }
  {
    auto tensor = workspace.CreateBlob("lr")->GetMutable<TensorCUDA>();
    auto value = TensorCUDA({1}, one, context);
    tensor->ResizeLike(value);
    tensor->ShareData(value);
  }
  {
    auto tensor = workspace.CreateBlob("data_grad")->GetMutable<TensorCUDA>();
    auto value = TensorCUDA({2}, data_grad, context);
    tensor->ResizeLike(value);
    tensor->ShareData(value);
  }

  NetDef predictModel;

  NetUtil* net_util = new NetUtil(predictModel);
  net_util->AddWeightedSumOp({"data", "one", "data_grad", "lr"}, {"data"});
  net_util->SetDeviceCUDA();
  net_util->Print();


  auto predictNet = CreateNet(predictModel, &workspace);
  for (int i = 0 ; i < 100; ++i) {
    predictNet->Run();
    print(Tensor<CPUContext>((workspace.GetBlob("data")->Get<Tensor<CUDAContext>>())));
    std::cout << std::endl;
  }
}
}  // namespace caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}

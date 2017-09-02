#include "caffe2/core/init.h"
#include "caffe2/core/operator_gradient.h"

#include "caffe2/utils/myutil/print.h"
#include "caffe2/myutils/utils/net.h"

namespace caffe2 {

void run() {
  Workspace workspace;

  std::vector<float> data(5);
  for (auto& v: data) {
    v = 100.0;
  }
  std::vector<float> data_grad(5);
  for (auto& v: data_grad) {
    v = 10.0;
  }

  std::vector<float> one(1);
  for (auto& v: one) {
    v = 1.0;
  }

  {
    auto tensor = workspace.CreateBlob("data")->GetMutable<TensorCPU>();
    auto value = TensorCPU({5}, data, NULL);
    tensor->ResizeLike(value);
    tensor->ShareData(value);
  }
  {
    auto tensor = workspace.CreateBlob("one")->GetMutable<TensorCPU>();
    auto value = TensorCPU({1}, one, NULL);
    tensor->ResizeLike(value);
    tensor->ShareData(value);
  }
  {
    auto tensor = workspace.CreateBlob("lr")->GetMutable<TensorCPU>();
    auto value = TensorCPU({1}, one, NULL);
    tensor->ResizeLike(value);
    tensor->ShareData(value);
  }
  {
    auto tensor = workspace.CreateBlob("data_grad")->GetMutable<TensorCPU>();
    auto value = TensorCPU({5}, data_grad, NULL);
    tensor->ResizeLike(value);
    tensor->ShareData(value);
  }

  NetDef predictModel;

  NetUtil* net_util = new NetUtil(predictModel);
  net_util->AddWeightedSumOp({"data", "one", "data_grad", "lr"}, {"data"});
  net_util->Print();


  auto predictNet = CreateNet(predictModel, &workspace);
  for (int i = 0 ; i < 10; ++i) {
    predictNet->Run();
    print(*(workspace.GetBlob("data")));
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

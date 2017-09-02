#include "caffe2/core/init.h"
#include "caffe2/core/operator_gradient.h"

#include "caffe2/utils/myutil/print.h"
#include "caffe2/myutils/zoo/keeper.h"
#include "caffe2/myutils/utils/model.h"

namespace caffe2 {

void run() {
  Workspace workspace;

  std::vector<int> data(3 * 224 * 224);
  for (auto& v: data) {
    v = 100 * rand() / RAND_MAX;
  }

  std::vector<int> label(1);
  for (auto& v: label) {
    v = 1000 * rand() / RAND_MAX;
  }

  {
    auto tensor = workspace.CreateBlob("data")->GetMutable<TensorCPU>();
    auto value = TensorCPU({ 1, 3, 224, 224 }, data, NULL);
    tensor->ResizeLike(value);
    tensor->ShareData(value);
  }

  // >>> workspace.FeedBlob("label", label)
  {
    auto tensor = workspace.CreateBlob("label")->GetMutable<TensorCPU>();
    auto value = TensorCPU({1}, label, NULL);
    tensor->ResizeLike(value);
    tensor->ShareData(value);
  }

  // >>> m = model_helper.ModelHelper(name="my first net")
  NetDef initModel;
  NetDef predictModel;

  Keeper* keeper = new Keeper("darknet19");
  keeper->AddModel(initModel, predictModel, false);
  //ModelUtil* model_util = new ModelUtil(initModel, predictModel);
  NetUtil* net_util = new NetUtil(predictModel);
  net_util->Print();

  auto initNet = CreateNet(initModel, &workspace);
  initNet->Run();
  std::cout << "init model run finish!" << std::endl;

  // >>> workspace.CreateNet(m.net)
  auto predictNet = CreateNet(predictModel, &workspace);

  // >>> for j in range(0, 100):
  for (auto i = 0; i < 10; i++) {

    std::vector<float> data(3 * 224 * 224);
    for (auto& v: data) {
      v = (float)rand() * 255 / RAND_MAX;
    }

    std::vector<int> label(1);
    for (auto& v: label) {
      v = 1000 * rand() / RAND_MAX;
    }

    {
      auto tensor = workspace.GetBlob("data")->GetMutable<TensorCPU>();
      auto value = TensorCPU({ 1, 3, 224, 224}, data, NULL);
      tensor->ResizeLike(value);
      tensor->ShareData(value);
    }

    {
      auto tensor = workspace.GetBlob("label")->GetMutable<TensorCPU>();
      auto value = TensorCPU({1}, label, NULL);
      tensor->ResizeLike(value);
      tensor->ShareData(value);
    }

    // >>> workspace.RunNet(m.name, 10)   # run for 10 times
    for (auto j = 0; j < 10; j++) {
      LOG(INFO) << "predict model run " << j << " times" << std::endl;
      predictNet->Run();
      std::cout << "step: " << i << " loss: ";
      print(*(workspace.GetBlob("conv19_grad")));
      std::cout << " ";
      print(*(workspace.GetBlob("conv19_w")));
      std::cout << " ";
      print(*(workspace.GetBlob("loss1/loss")));
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;
  }

}  // namespace caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}

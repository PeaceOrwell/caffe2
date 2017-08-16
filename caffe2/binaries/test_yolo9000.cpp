#include "caffe2/core/init.h"
#include "caffe2/core/operator_gradient.h"

#include "caffe2/utils/myutil/print.h"
#include "caffe2/utils/myutil/models.h"

namespace caffe2 {

void run() {
  Workspace workspace;
  
  // >>> data = np.random.rand(16, 100).astype(np.float32)
  std::vector<float> data(3 * 416 * 416);
  for (auto& v: data) {
    v = (float)rand() / RAND_MAX;
  }

  // >>> label = (np.random.rand(16) * 10).astype(np.int32)
  std::vector<int> label(16);
  for (auto& v: label) {
    v = 10 * rand() / RAND_MAX;
  }

  // >>> workspace.FeedBlob("data", data)
  {
    auto tensor = workspace.CreateBlob("data")->GetMutable<TensorCPU>();
    auto value = TensorCPU({ 1, 3, 416, 416 }, data, NULL);
    tensor->ResizeLike(value);
    tensor->ShareData(value);
  }

  // >>> workspace.FeedBlob("label", label)
  {
    auto tensor = workspace.CreateBlob("label")->GetMutable<TensorCPU>();
    auto value = TensorCPU({ 16 }, label, NULL);
    tensor->ResizeLike(value);
    tensor->ShareData(value);
  }

  // >>> m = model_helper.ModelHelper(name="my first net")
  NetDef initModel;
  NetDef predictModel;
  add_yolo9000_model(initModel, predictModel);
  print(predictModel);
  
  auto initNet = CreateNet(initModel, &workspace);
  initNet->Run();
  std::cout << "init model run finish!" << std::endl;

  // >>> workspace.CreateNet(m.net)
  auto predictNet = CreateNet(predictModel, &workspace);

  // >>> for j in range(0, 100):
  for (auto i = 0; i < 1; i++) {

    // >>> data = np.random.rand(16, 100).astype(np.float32)
    std::vector<float> data(3 * 416 * 416);
    for (auto& v: data) {
      v = (float)rand() / RAND_MAX;
    }

    // >>> label = (np.random.rand(16) * 10).astype(np.int32)
    std::vector<int> label(16);
    for (auto& v: label) {
      v = 10 * rand() / RAND_MAX;
    }

    // >>> workspace.FeedBlob("data", data)
    {
      auto tensor = workspace.GetBlob("data")->GetMutable<TensorCPU>();
      auto value = TensorCPU({ 1, 3, 416, 416 }, data, NULL);
      tensor->ShareData(value);
    }

    // >>> workspace.FeedBlob("label", label)
    {
      auto tensor = workspace.GetBlob("label")->GetMutable<TensorCPU>();
      auto value = TensorCPU({ 16 }, label, NULL);
      tensor->ShareData(value);
    }

    // >>> workspace.RunNet(m.name, 10)   # run for 10 times
    for (auto j = 0; j < 1; j++) {
      LOG(INFO) << "predict model run " << j << " times" << std::endl;
      predictNet->Run();
      std::cout << "step: " << i << " regression: ";
      print(*(workspace.GetBlob("regression")));
      std::cout << std::endl;
    }
  }

  std::cout << std::endl;

  // >>> print(workspace.FetchBlob("softmax"))
  // print(*(workspace.GetBlob("softmax")), "softmax");

  std::cout << std::endl;

  // >>> print(workspace.FetchBlob("loss"))
  // print(*(workspace.GetBlob("loss")), "loss");
}

}  // namespace caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}

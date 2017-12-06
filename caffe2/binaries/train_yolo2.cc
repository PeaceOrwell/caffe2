#include "caffe2/core/init.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator_gradient.h"

#include "caffe2/myutils/zoo/keeper.h"
#include "caffe2/utils/myutil/print.h"
#include "caffe2/myutils/utils/model.h"

#include <iostream>
#include <string>
#include <sstream>

CAFFE2_DEFINE_bool(finetune, false, "finetue or not, default is false");
CAFFE2_DEFINE_string(init_net_pb, "init_net.pb", "the name of net to init from");
namespace caffe2 {

void dump(int idx, NetDef predict_net, Workspace* workspace) {
  NetUtil* net_util = new NetUtil(predict_net); 
  NetDef param_net;
  for (auto blob_name : net_util->CollectParams()) {
    auto tensor = new Tensor<CPUContext>(workspace->GetBlob(blob_name)->template Get<Tensor<CUDAContext>>()); 
    auto tensor_data = (float*)tensor->raw_data();
    std::cout << blob_name << std::endl;
    LOG(INFO) << "blob name: " << blob_name << " tensor " << tensor->ndim();
    auto op = param_net.add_op();
    op->set_name(blob_name);
    op->add_output(blob_name);
    op->set_type("GivenTensorFill");
    auto arg = op->add_arg();
    arg->set_name("shape");
    for (int i = 0 ; i < tensor->ndim(); ++i) {
      arg->add_ints(tensor->dim32(i));
    }
    arg = op->add_arg();
    arg->set_name("values");
    for (int i = 0; i < tensor->size(); ++i) {
      arg->add_floats(tensor_data[i]);
    }
  }
  std::stringstream ss;
  string name;
  ss << "init_net_" << idx << ".pb"; 
  ss >> name;
  WriteProtoToTextFile(param_net, name); 
}

void run() {
  Workspace workspace;
  // init CUDAContext
  DeviceOption option;
  option.set_device_type(CUDA);
  option.set_cuda_gpu_id(2);
  CUDAContext* context = new CUDAContext(option);

  NetDef init_net, predict_net;
  Keeper* keeper = new Keeper("darknet19");
  keeper->AddModel(init_net, predict_net, false);
  // print predict_net to screen
  NetUtil* net_util = new NetUtil(predict_net);
  net_util->Print();
  
  // init the net parameters
  auto initNet = CreateNet(init_net, &workspace);
  initNet->Run();
  if (FLAGS_finetune == true) {
    NetDef trained_init_net;
    CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_init_net_pb, &trained_init_net));
    trained_init_net.mutable_device_option()->set_device_type(CUDA);
    auto trained_initNet = CreateNet(trained_init_net, &workspace);
    trained_initNet->Run(); 
  }

  auto predictNet = CreateNet(predict_net, &workspace);
  int epoch = 10, iteration = 5000;
  for (auto i = 0; i < epoch; ++i) {
    LOG(INFO) << "start the " << i << "th epoch forward";
    for (auto j = 0; j < iteration; ++j) {
      predictNet->Run();    
      std::cout << "epoch " << i << " , iter " << j << " , loss : ";
      print(Tensor<CPUContext>((workspace.GetBlob("loss1/loss"))->template Get<Tensor<CUDAContext>>()));
      std::cout << " lr : ";
      print(Tensor<CPUContext>((workspace.GetBlob("lr"))->template Get<Tensor<CUDAContext>>()));
      std::cout << std::endl;
    }
  }
  dump(epoch * iteration, predict_net, &workspace);
  LOG(INFO) << "Net param is inited down!";
}

} // namespace caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::ShowLogInfoToStderr();
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}


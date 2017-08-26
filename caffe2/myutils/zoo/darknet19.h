#ifndef ZOO_DARKNET19_H
#define ZOO_DARKNET19_H

#include "caffe2/myutils/utils/net.h"
#include "caffe2/myutils/utils/model.h"
using std::string;
using std::vector;

namespace caffe2 {

class Darknet19Model : public ModelUtil {
 public:
  Darknet19Model(NetDef &init_net, NetDef &predict_net) 
      : ModelUtil(init_net, predict_net) {}

  OperatorDef* AddConvBnRelu(const std::string &prefix, const std::string &input, int in_size, int out_size, int kernel, int padding, int stride, bool group=false) {
    auto output = "conv" + prefix;
    init_.AddXavierFillOp({out_size, in_size, kernel, kernel}, output + "_w");
    predict_.AddInput(output + "_w");
    init_. AddConstantFillOp({out_size}, output + "_b");
    predict_.AddInput(output + "_b");
    auto conv = predict_.AddConvOp(input, output + "_w", output + "_b", output, stride, padding, kernel);
    if (group) {
      auto arg = conv->add_arg();
      arg->set_name("group");
      arg->set_i(2);
    }
    vector<string> bn_inputs({output, output + "_bn_scale_in", output + "_bn_bias_in", output + "_bn_mean", output + "_bn_var"});
    vector<string> bn_outputs({output + "_bn", output + "_bn_mean", output + "_bn_var", output + "_saved_mean", output + "_saved_var"});
    predict_.AddInput(output + "_bn_scale_in");
    predict_.AddInput(output + "_bn_bias_in");
    predict_.AddInput(output + "_bn_mean");
    predict_.AddInput(output + "_bn_var");
    predict_.AddBnOp(bn_inputs, bn_outputs);
    return predict_.AddLeakyReluOp(output + "_bn", output + "_bn", 0.1);
  }
  OperatorDef* AddTrain(const string& prefix, const string& input) {
    string output = "loss" + prefix + "/";
    string layer = input;
    predict_.AddInput("label");
    layer = predict_.AddSoftmaxOp(layer, output + "softmax")->output(0);
    layer = predict_.AddLabelCrossEntropyOp(layer, "label", output + "xent")->output(0);
    return predict_.AddAveragedLossOp(layer, output + "loss");
  }
  void AddGradientOp(string input) {
    auto op = predict_.AddOp("ConstantFill", {input}, {input + "_grad"});
    auto arg = op->add_arg();
    arg->set_name("value");
    arg->set_f(1.0);
    op->set_is_gradient_op("true");
  }

  void Add(int out_size = 1000) {
    predict_.SetName("Darknet19");
    string layer = "data";
    predict_.AddInput("data");
    layer = AddConvBnRelu("1", layer, 3, 32, 3, 1, 1)->output(0);
    layer = predict_.AddMaxPoolOp(layer, "pool1", 2, 0 ,2)->output(0); 
    layer = AddConvBnRelu("2", layer, 32, 64, 3, 1, 1)->output(0);
    layer = predict_.AddMaxPoolOp(layer, "pool2", 2, 0 ,2)->output(0); 
    layer = AddConvBnRelu("3", layer, 64, 128, 3, 1, 1)->output(0);
    layer = AddConvBnRelu("4", layer, 128, 64, 1, 0, 1)->output(0);
    layer = AddConvBnRelu("5", layer, 64, 128, 3, 1, 1)->output(0);
    layer = predict_.AddMaxPoolOp(layer, "pool5", 2, 0 ,2)->output(0); 
    layer = AddConvBnRelu("6", layer, 128, 256, 3, 1, 1)->output(0);
    layer = AddConvBnRelu("7", layer, 256, 128, 1, 0, 1)->output(0);
    layer = AddConvBnRelu("8", layer, 128, 256, 3, 1, 1)->output(0);
    layer = predict_.AddMaxPoolOp(layer, "pool8", 2, 0 ,2)->output(0); 
    layer = AddConvBnRelu("9", layer, 256, 512, 3, 1, 1)->output(0);
    layer = AddConvBnRelu("10", layer, 512, 256, 1, 0, 1)->output(0);
    layer = AddConvBnRelu("11", layer, 256, 512, 3, 1, 1)->output(0);
    layer = AddConvBnRelu("12", layer, 512, 256, 1, 0, 1)->output(0);
    layer = AddConvBnRelu("13", layer, 256, 512, 3, 1, 1)->output(0);
    layer = predict_.AddMaxPoolOp(layer, "pool13", 2, 0 ,2)->output(0); 
    layer = AddConvBnRelu("14", layer, 512, 1024, 3, 1, 1)->output(0);
    layer = AddConvBnRelu("15", layer, 1024, 512, 1, 0, 1)->output(0);
    layer = AddConvBnRelu("16", layer, 512, 1024, 3, 1, 1)->output(0);
    layer = AddConvBnRelu("17", layer, 1024, 512, 1, 0, 1)->output(0);
    layer = AddConvBnRelu("18", layer, 512, 1024, 3, 1, 1)->output(0);
    layer = AddConvBnRelu("19", layer, 1024, 1000, 1, 0, 1)->output(0);
    layer = predict_.AddAveragePoolOp(layer, "pool19", 7, 0, 7)->output(0); 
    layer = AddTrain("1", layer)->output(0);
    AddGradientOp(layer);
    predict_.AddGradientOps();
    AddIterLrOps(0.1);
    string opt = "sgd";
    AddOptimizerOps(opt);
  }
};
}
#endif

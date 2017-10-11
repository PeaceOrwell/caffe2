#include "caffe2/core/init.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/core/logging.h"

CAFFE2_DEFINE_bool(topb, false, "the file to store the readable pb.");
CAFFE2_DEFINE_string(txt, "", "the file to store the readable pb.");
CAFFE2_DEFINE_string(net, "", "the file to store the readable pb.");

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  std::unique_ptr<caffe2::Workspace> workspace(new caffe2::Workspace());
  caffe2::NetDef net_def;
  if (caffe2::FLAGS_topb) { 
    CAFFE_ENFORCE(ReadProtoFromFile(caffe2::FLAGS_txt, &net_def));
    WriteProtoToBinaryFile(net_def, caffe2::FLAGS_net);
  } else {
    CAFFE_ENFORCE(ReadProtoFromFile(caffe2::FLAGS_net, &net_def));
    WriteProtoToTextFile(net_def, caffe2::FLAGS_txt);
  }
  return 0;
}

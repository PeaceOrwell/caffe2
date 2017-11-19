#include "caffe2/core/init.h"
#include "caffe2/utils/proto_utils.h"

using namespace caffe2;

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << "Usage: transfer_pb src_pb dst_pb 0/1" << std::endl 
              << " 0 : to bin" 
              << " 1 : to txt" << std::endl;
    return 0;
  }
  NetDef net_model;
  CAFFE_ENFORCE(ReadProtoFromFile(argv[1], &net_model));
  if (strcmp(argv[3], "0") == 0) { 
    WriteProtoToBinaryFile(net_model, argv[2]);
    std::cout << "write into " << argv[2] << std::endl;
  } else if (strcmp(argv[3], "1") == 0) {
    WriteProtoToTextFile(net_model, argv[2]); 
    std::cout << "write into " << argv[2] << std::endl;
  } else {
    std::cout << "wrong mode!!" << std::endl;
  }
  return 0;
}

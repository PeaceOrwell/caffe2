#include "caffe2/operators/region_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Region, RegionOp<float, CPUContext>);

OPERATOR_SCHEMA(Region)
    .NumInputs(1)
    .NumOutputs(1)
    .ScalarType(TensorProto::FLOAT)
    .SetDoc("simlar to reshape, but change the stored data index.")
    .Input(0, "X",
           "a 4-D tensor");

SHOULD_NOT_DO_GRADIENT(Region);

}  // namespace caffe2

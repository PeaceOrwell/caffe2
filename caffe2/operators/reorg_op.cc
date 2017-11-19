#include "caffe2/operators/reorg_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Reorg, ReorgOp<float, CPUContext>);

OPERATOR_SCHEMA(Reorg)
    .NumInputs(1)
    .NumOutputs(1)
    .ScalarType(TensorProto::FLOAT)
    .SetDoc("simlar to reshape, but change the stored data index.")
    .Input(0, "X",
           "a 4-D tensor");

SHOULD_NOT_DO_GRADIENT(Reorg);

}  // namespace caffe2

#include <nan.h>
#include "src/buffer.h"
#include "src/dtype.h"
#include "src/graph.h"
#include "src/library.h"
#include "src/operation.h"
#include "src/session.h"
#include "src/tensor.h"

void InitModule(Handle<Object> target) {
  TensorflowNode::DType::Init(target);
  TensorflowNode::Buffer::Init(target);
  TensorflowNode::Tensor::Init(target);
  TensorflowNode::Graph::Init(target);
  TensorflowNode::Operation::Init(target);
  TensorflowNode::Session::Init(target);
  TensorflowNode::Library::Init(target);
}

NODE_MODULE(tensorflow, InitModule);
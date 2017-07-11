#include "src/graph.h"
#include "src/internal.h"

using namespace v8;

namespace TensorflowNode {

NAN_MODULE_INIT(Graph::Init) {
  Local<FunctionTemplate> tmpl = Nan::New<FunctionTemplate>(New);
  tmpl->SetClassName(Nan::New("Graph").ToLocalChecked());
  tmpl->InstanceTemplate()->SetInternalFieldCount(1);

  Nan::SetPrototypeMethod(tmpl, "destory", Destory);
  Nan::SetPrototypeMethod(tmpl, "setShape", SetShape);
  Nan::SetPrototypeMethod(tmpl, "getShape", GetShape);
  Nan::SetPrototypeMethod(tmpl, "getNumOfDims", GetNumOfDims);

  Nan::Set(target, 
    Nan::New("Graph").ToLocalChecked(), 
    Nan::GetFunction(tmpl).ToLocalChecked());
}

NAN_METHOD(Graph::New) {
  TensorflowNode::Graph* graph = new TensorflowNode::Graph();
  graph->Wrap(info.This());
  info.GetReturnValue().Set(info.This());
}

NAN_METHOD(Graph::Destory) {
  TensorflowNode::Graph* graph = ObjectWrap::Unwrap<TensorflowNode::Graph>(info.This());
  delete graph;
}

NAN_METHOD(Graph::SetShape) {
  // TensorflowNode::Graph* graph = ObjectWrap::Unwrap<TensorflowNode::Graph>(info.This());
  info.GetReturnValue().Set(info.This());
}

NAN_METHOD(Graph::GetShape) {
  // TensorflowNode::Graph* graph = ObjectWrap::Unwrap<TensorflowNode::Graph>(info.This());
  info.GetReturnValue().Set(info.This());
}

NAN_METHOD(Graph::GetNumOfDims) {
  // TensorflowNode::Graph* graph = ObjectWrap::Unwrap<TensorflowNode::Graph>(info.This());
  info.GetReturnValue().Set(info.This());
}

Graph::Graph() {
  _graph = TF_NewGraph();
}

Graph::~Graph() {
  if (_graph != NULL) {
    TF_DeleteGraph(_graph);
  }
}

}
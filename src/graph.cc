#include "src/graph.h"
#include "src/operation.h"
#include "src/internal.h"
#include "tensorflow/c/c_api.h"

using namespace v8;

namespace TensorflowNode {

NAN_MODULE_INIT(Graph::Init) {
  Local<FunctionTemplate> tmpl = Nan::New<FunctionTemplate>(New);
  tmpl->SetClassName(Nan::New("Graph").ToLocalChecked());
  tmpl->InstanceTemplate()->SetInternalFieldCount(1);

  Nan::SetPrototypeMethod(tmpl, "_import", Import);
  Nan::SetPrototypeMethod(tmpl, "destory", Destory);
  Nan::SetPrototypeMethod(tmpl, "setShape", SetShape);
  Nan::SetPrototypeMethod(tmpl, "getShape", GetShape);
  Nan::SetPrototypeMethod(tmpl, "getNumOfDims", GetNumOfDims);
  Nan::SetPrototypeMethod(tmpl, "getGraphDef", GetGraphDef);

  Local<Function> func = Nan::GetFunction(tmpl).ToLocalChecked();
  // static method
  Nan::SetMethod(func, "_getAllOpList", Graph::GetAllOpList);
  Nan::Set(target, Nan::New("Graph").ToLocalChecked(), func);
}

NAN_METHOD(Graph::New) {
  TensorflowNode::Graph* graph = new TensorflowNode::Graph();
  graph->Wrap(info.This());
  info.GetReturnValue().Set(info.This());
}

NAN_METHOD(Graph::Import) {
  TensorflowNode::Graph* graph = ObjectWrap::Unwrap<TensorflowNode::Graph>(info.This());
  TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();

  ArrayBuffer::Contents data = Local<ArrayBuffer>::Cast(info[0])->GetContents();
  TF_Buffer* proto = TF_NewBufferFromString(data.Data(), data.ByteLength());

  Local<Object> v8options = info[1]->ToObject();
  Local<String> prefixKey = Nan::New("prefix").ToLocalChecked();
  if (Nan::Has(v8options, prefixKey).FromJust()) {
    const char* prefix = *String::Utf8Value(Nan::Get(v8options, prefixKey).ToLocalChecked());
    TF_ImportGraphDefOptionsSetPrefix(opts, prefix);
  }

  TF_GraphImportGraphDef(graph->_graph, proto, opts, status);
  if (TF_GetCode(status) != TF_OK) {
    ThrowStatusError();
    return;
  }

  Local<Function> iterator = Local<Function>::Cast(info[1]);
  size_t pos = 0;
  TF_Operation* oper;
  while ((oper = TF_GraphNextOperation(graph->_graph, &pos)) != nullptr) {
    Local<Value> argv[1] = { TensorflowNode::Operation::NewFromOperation(info.This(), oper) };
    Nan::Call(iterator, info.This(), 1, argv);
  }

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

NAN_METHOD(Graph::GetGraphDef) {
  TensorflowNode::Graph* graph = ObjectWrap::Unwrap<TensorflowNode::Graph>(info.This());
  TF_Buffer* graphdef = TF_NewBuffer();
  TF_GraphToGraphDef(graph->_graph, graphdef, status);
  if (TF_GetCode(status) != TF_OK) {
    ThrowStatusError();
    return;
  }
  info.GetReturnValue().Set(
    Nan::NewBuffer((char*)graphdef->data, graphdef->length).ToLocalChecked());
}

NAN_METHOD(Graph::GetAllOpList) {
  TF_Buffer* list = TF_GetAllOpList();
  info.GetReturnValue().Set(
    Nan::CopyBuffer((char *)list->data, list->length).ToLocalChecked());
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
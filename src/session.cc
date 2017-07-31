#include "src/session.h"
#include "src/internal.h"
#include "src/graph.h"
#include "src/operation.h"
#include "src/tensor.h"

using namespace v8;

namespace TensorflowNode {

NAN_MODULE_INIT(Session::Init) {
  Local<FunctionTemplate> tmpl = Nan::New<FunctionTemplate>(New);
  tmpl->SetClassName(Nan::New("Session").ToLocalChecked());
  tmpl->InstanceTemplate()->SetInternalFieldCount(1);

  Nan::SetPrototypeMethod(tmpl, "close", Close);
  Nan::SetPrototypeMethod(tmpl, "destory", Destory);
  Nan::SetPrototypeMethod(tmpl, "_run", Run);

  Local<Function> func = Nan::GetFunction(tmpl).ToLocalChecked();
  // static methods

  constructor().Reset(Nan::GetFunction(tmpl).ToLocalChecked());
  Nan::Set(target, Nan::New("Session").ToLocalChecked(), func);
}

NAN_METHOD(Session::New) {
  if (info.Length() == 0) {
    Nan::ThrowTypeError("graph are required");
    return;
  }

  ArrayBuffer::Contents options;
  TensorflowNode::Session* session;
  TensorflowNode::Graph* graph = ObjectWrap::Unwrap<TensorflowNode::Graph>(info[0]->ToObject());

  // check for target parameter
  const char* target = *String::Utf8Value(info[1]);

  // check for options parameter
  if (info[2]->IsArrayBuffer()) {
    Local<ArrayBuffer> optionsBuffer = Local<ArrayBuffer>::Cast(info[2]);
    options = optionsBuffer->GetContents();
  }

  if (info[3]->IsString()) {
    if (!info[4]->IsArray()) {
      return Nan::ThrowTypeError("tags should be an array.");
    }
    Local<Array> tagsArr = Local<Array>::Cast(info[4]);
    uint32_t numOfTags = tagsArr->Length();

    String::Utf8Value dirstr(info[3]->ToString());
    const char* dir = *dirstr;
    const char* tags[numOfTags];

    for (uint32_t i = 0; i < numOfTags; i++) {
      tags[i] = *String::Utf8Value(Nan::Get(tagsArr, i).ToLocalChecked());
    }

    printf("foobar: %s\n", dir);
    session = new TensorflowNode::Session(graph->_graph, target, options, dir, tags, numOfTags);
  } else {
    session = new TensorflowNode::Session(graph->_graph, target, options);
  }
  session->Wrap(info.This());

  // this._graph = graph
  Nan::Set(info.This(), Nan::New<String>("_graph").ToLocalChecked(), info[0]);
  info.GetReturnValue().Set(info.This());
}

inline Nan::Persistent<v8::Function>& 
Session::constructor() {
  static Nan::Persistent<v8::Function> my_constructor;
  return my_constructor;
}

NAN_METHOD(Session::Close) {
  TensorflowNode::Session* session = ObjectWrap::Unwrap<TensorflowNode::Session>(info.This());
  TF_CloseSession(session->_session, status);
}

NAN_METHOD(Session::Destory) {
  TensorflowNode::Session* session = ObjectWrap::Unwrap<TensorflowNode::Session>(info.This());
  TF_DeleteSession(session->_session, status);
}

NAN_METHOD(Session::Run) {
  TensorflowNode::Session* session = ObjectWrap::Unwrap<TensorflowNode::Session>(info.This());
  TensorflowNode::Operation* placeholder = ObjectWrap::Unwrap<TensorflowNode::Operation>(info[0]->ToObject());

  session->SetOutputs({placeholder->_oper});
  // session->SetTargets({operation->_oper});

  if (info[1]->IsArray()) {
    Local<Array> feeds = Local<Array>::Cast(info[1]);
    size_t lengthOfFeeds = feeds->Length();
    std::vector<std::pair<TF_Operation*, TF_Tensor*>> inputs;
    for (size_t i = 0; i < lengthOfFeeds; i++) {
      Local<Object> feed = Nan::Get(feeds, i).ToLocalChecked()->ToObject();
      TensorflowNode::Operation* op = Nan::ObjectWrap::Unwrap<TensorflowNode::Operation>(Nan::Get(feed, 0).ToLocalChecked()->ToObject());
      TensorflowNode::Tensor* tensor = Nan::ObjectWrap::Unwrap<TensorflowNode::Tensor>(Nan::Get(feed, 1).ToLocalChecked()->ToObject());
      inputs.push_back({op->_oper, tensor->_tensor});
    }
    session->SetInputs(inputs);
  }

  const TF_Output* inputs_ptr;
  TF_Tensor* const* input_values_ptr;
  if (session->inputs_.empty()) {
    inputs_ptr = nullptr;
    input_values_ptr = nullptr;
  } else {
    inputs_ptr = &session->inputs_[0];
    input_values_ptr = &session->input_values_[0];
  }

  const TF_Output* outputs_ptr;
  TF_Tensor** output_values_ptr;
  if (session->outputs_.empty()) {
    outputs_ptr = nullptr;
    output_values_ptr = nullptr;
  } else {
    outputs_ptr = &session->outputs_[0];
    output_values_ptr = &session->output_values_[0];
  }
  TF_Operation* const* targets_ptr = session->targets_.empty() ? nullptr : &session->targets_[0];

  TF_SessionRun(
    session->_session,
    nullptr,
    inputs_ptr, input_values_ptr, session->inputs_.size(),
    outputs_ptr, output_values_ptr, session->outputs_.size(),
    targets_ptr, session->targets_.size(),
    nullptr, 
    status);

  if (TF_GetCode(status) != TF_OK) {
    Nan::ThrowError(TF_Message(status));
    return;
  }

  size_t outputSize = session->output_values_.size();
  Local<Array> result = Nan::New<Array>(outputSize);

  for (size_t i = 0; i < outputSize; i++) {
    TF_Tensor* item = session->output_values_[i];
    Local<Object> obj = TensorflowNode::Tensor::NewFromTensor(item);
    Nan::Set(result, i, obj);
  }
  info.GetReturnValue().Set(result);
}

Session::Session(TF_Graph* graph, const char* target, ArrayBuffer::Contents options) {
  TF_SessionOptions* opts = TF_NewSessionOptions();
  if (target) {
    // TODO
    // TF_SetTarget(opts, target);
  }
  if (options.ByteLength() != 0) {
    TF_SetConfig(opts, options.Data(), options.ByteLength(), status);
    if (TF_GetCode(status) != TF_OK) {
      Nan::ThrowError(TF_Message(status));
      return;
    }
  }

  _session = TF_NewSession(graph, opts, status);
  TF_DeleteSessionOptions(opts);
  if (TF_GetCode(status) != TF_OK) {
    Nan::ThrowError(TF_Message(status));
    return;
  }
}

Session::Session(TF_Graph* graph, const char* target, ArrayBuffer::Contents options, 
  const char* exportDir, const char* const* tags, int tagsLen) {
  TF_SessionOptions* opts = TF_NewSessionOptions();
  if (target) {
    // TODO
    // TF_SetTarget(opts, target);
  }
  if (options.ByteLength() != 0) {
    TF_SetConfig(opts, options.Data(), options.ByteLength(), status);
    if (TF_GetCode(status) != TF_OK) {
      Nan::ThrowError(TF_Message(status));
      return;
    }
  }

  _session = TF_LoadSessionFromSavedModel(opts, 
                                          nullptr, 
                                          exportDir, 
                                          tags, tagsLen, 
                                          graph, 
                                          nullptr, 
                                          status);
  TF_DeleteSessionOptions(opts);
  if (TF_GetCode(status) != TF_OK) {
    Nan::ThrowError(TF_Message(status));
    return;
  }
}

Session::~Session() {
  DeleteInputValues();
  DeleteOutputValues();
  if (_session != NULL) {
    TF_CloseSession(_session, status);
    TF_DeleteSession(_session, status);
    _session = NULL;
  }
}

void 
Session::DeleteInputValues() {
  for (size_t i = 0; i < input_values_.size(); ++i) {
    if (input_values_[i] != NULL)
      TF_DeleteTensor(input_values_[i]);
  }
  input_values_.clear();
}

void 
Session::DeleteOutputValues() {
  for (size_t i = 0; i < output_values_.size(); ++i) {
    if (output_values_[i] != NULL) {
      TF_DeleteTensor(output_values_[i]);
    }
  }
  output_values_.clear();
}

void 
Session::SetInputs(std::vector<std::pair<TF_Operation*, TF_Tensor*>> inputs) {
  DeleteInputValues();
  inputs_.clear();
  for (const auto& p : inputs) {
    inputs_.emplace_back(TF_Output{p.first, 0});
    input_values_.emplace_back(p.second);
  }
}

void 
Session::SetOutputs(std::initializer_list<TF_Operation*> outputs) {
  DeleteOutputValues();
  outputs_.clear();
  for (TF_Operation* o : outputs) {
    outputs_.emplace_back(TF_Output{o, 0});
  }
  output_values_.resize(outputs_.size());
}

void 
Session::SetTargets(std::initializer_list<TF_Operation*> targets) {
  targets_.clear();
  for (TF_Operation* t : targets) {
    targets_.emplace_back(t);
  }
}


}
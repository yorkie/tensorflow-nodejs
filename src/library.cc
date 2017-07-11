#include "src/library.h"
#include "src/internal.h"

using namespace v8;

namespace TensorflowNode {

NAN_MODULE_INIT(Library::Init) {
  Nan::HandleScope scope;
  Local<FunctionTemplate> tmpl = Nan::New<FunctionTemplate>(New);
  Local<String> name = Nan::New<String>("Library").ToLocalChecked();
  tmpl->InstanceTemplate()->SetInternalFieldCount(1);
  Nan::Set(target, name, tmpl->GetFunction());
}

NAN_METHOD(TensorflowNode::Library::New) {
  Nan::HandleScope scope;
  info.GetReturnValue().Set(info.This());
}

}
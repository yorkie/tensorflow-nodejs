#include "src/buffer.h"
#include "src/internal.h"

using namespace v8;

namespace TensorflowNode {

NAN_MODULE_INIT(Buffer::Init) {
  Local<FunctionTemplate> tmpl = Nan::New<FunctionTemplate>(New);
  Local<String> name = Nan::New<String>("Buffer").ToLocalChecked();
  tmpl->InstanceTemplate()->SetInternalFieldCount(1);
  Nan::Set(target, name, tmpl->GetFunction());
}

NAN_METHOD(Buffer::New) {
  info.GetReturnValue().Set(info.This());
}

}
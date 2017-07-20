#include "src/library.h"
#include "src/internal.h"

using namespace v8;

namespace TensorflowNode {

NAN_MODULE_INIT(Library::Init) {
  Nan::HandleScope scope;
  Local<FunctionTemplate> tmpl = Nan::New<FunctionTemplate>(New);
  Local<String> name = Nan::New<String>("Library").ToLocalChecked();
  tmpl->InstanceTemplate()->SetInternalFieldCount(1);

  Nan::SetPrototypeMethod(tmpl, "getOperations", GetOperations);
  Nan::Set(target, name, tmpl->GetFunction());
}

NAN_METHOD(Library::New) {
  V8_STRING_TO_CSTR(filename, info[0]);
  TensorflowNode::Library* library = new TensorflowNode::Library(filename);
  library->Wrap(info.This());
  info.GetReturnValue().Set(info.This());
}

NAN_METHOD(Library::GetOperations) {
  TensorflowNode::Library* library = Nan::ObjectWrap::Unwrap<TensorflowNode::Library>(info.This());
  TF_Buffer list = TF_GetOpList(library->_handle);
  info.GetReturnValue().Set(
    Nan::CopyBuffer((const char*)list.data, list.length).ToLocalChecked());
}

Library::Library(const char* filename) {
  _handle = TF_LoadLibrary(filename, status);
}

Library::~Library() {
  TF_DeleteLibraryHandle(_handle);
}

}
#pragma once

#include <stdio.h>
#include <nan.h>

using namespace v8;

#define V8_STRING_TO_CSTR(NAME, ARGV)                   \
  String::Utf8Value v8_ ## NAME(ARGV);                  \
  const char* NAME = *v8_ ## NAME;                      \

#define V8_STRING(NAME)                                 \
  Nan::New<String>(#NAME).ToLocalChecked()              \

#define V8_GET_PROPERTY(TARGET, NAME)                   \
  Nan::Get(TARGET, V8_STRING(NAME)).ToLocalChecked()    \

#define V8_HAS_PROPERTY(TARGET, NAME)                   \
  Nan::Has(TARGET, V8_STRING(NAME)).FromJust()          \

namespace TensorflowNode {

static TF_Status* status = TF_NewStatus();

inline void ThrowStatusError() {
  Local<Value> err = Nan::Error(TF_Message(status));
  Nan::Set(err->ToObject(), 
    Nan::New<String>("code").ToLocalChecked(), 
    Nan::New<Number>(TF_GetCode(status)));
  Nan::ThrowError(err);
}

}
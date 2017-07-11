#pragma once

#include <stdio.h>
#include <node.h>
#include <nan.h>
#include "tensorflow/c/c_api.h"

using namespace v8;

namespace TensorflowNode {

/**
 * @class Buffer
 * @extends Nan::ObjectWrap
 */
class Buffer : public Nan::ObjectWrap {
public:
  static void Init(Handle<Object> target);
  static NAN_METHOD(New);
  static NAN_METHOD(NewFromString);
  static NAN_METHOD(GetData);
  static NAN_METHOD(Destory);
  TF_Buffer* _buffer;
private:
};

}
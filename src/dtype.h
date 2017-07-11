#pragma once

#include <stdio.h>
#include <node.h>
#include <nan.h>
#include "tensorflow/c/c_api.h"

using namespace v8;

namespace TensorflowNode {

/**
 * @class DType
 * @extends Nan::ObjectWrap
 */
class DType : public Nan::ObjectWrap {
public:
  static NAN_MODULE_INIT(Init);
  static NAN_METHOD(GetSize);
};

}
#pragma once

#include <stdio.h>
#include <node.h>
#include <nan.h>
#include "tensorflow/c/c_api.h"

using namespace v8;

namespace TensorflowNode {

/**
 * @class Library
 * @extends Nan::ObjectWrap
 */
class Library : public Nan::ObjectWrap {
public:
  static NAN_MODULE_INIT(Init);

  /**
   * @constructor
   */
  static NAN_METHOD(New);

  /**
   * @method getOperationList
   */
  static NAN_METHOD(GetOperations);
  
  TF_Library* _handle;

private:

  Library(const char* filename);
  ~Library();
};

}
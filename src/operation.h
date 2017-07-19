#pragma once

#include <stdio.h>
#include <node.h>
#include <nan.h>
#include "tensorflow/c/c_api.h"
#include "src/graph.h"

using namespace v8;

namespace TensorflowNode {

/**
 * @class Operation
 * @extends Nan::ObjectWrap
 */
class Operation : public Nan::ObjectWrap {
public:
  static NAN_MODULE_INIT(Init);

  /**
   * @constructor
   * @param {Graph} graph
   * @param {String} type
   * @param {String} name
   */
  static NAN_METHOD(New);

  /**
   * @property name
   */
  static NAN_PROPERTY_GETTER(NameGetter);

  /**
   * @property type
   */
  static NAN_PROPERTY_GETTER(TypeGetter);

  /**
   * @property device
   */
  static NAN_PROPERTY_GETTER(DeviceGetter);

  /**
   * @property outputs
   */
  static NAN_PROPERTY_GETTER(OutputsGetter);

  /**
   * @method setAttrType
   * @param {String} name - the attribute name
   * @param {Number} type - the type, tf.dtype
   */
  static NAN_METHOD(SetAttrType);

  /**
   * @method SetAttrBool
   * @param {String} name - the attribute name
   * @param {Number} value - the value
   */
  static NAN_METHOD(SetAttrBool);

  /**
   * @method SetAttrInt
   * @param {String} name - the attribute name
   * @param {Number} value - the value
   */
  static NAN_METHOD(SetAttrInt);

  /**
   * @method SetAttrFloat
   * @param {String} name - the attribute name
   * @param {Number} value - the value, float
   */
  static NAN_METHOD(SetAttrFloat);

  /**
   * @method setAttrString
   * @param {String} name - the attribute name
   * @param {String} value - the attribute value
   */
  static NAN_METHOD(SetAttrString);

  /**
   * @method setAttrShape
   * @param {String} name - the attribute name.
   * @param {Number} shape - the shape/dims.
   */
  static NAN_METHOD(SetAttrShape);

  /**
   * @method setAttrTensor
   * @param {String} name
   * @param {Tensor} tensor
   */
  static NAN_METHOD(SetAttrTensor);

  /**
   * @method addInput
   * @param {Object} input
   * @param {Number} input.index
   * @param {Operation} input.op
   */
  static NAN_METHOD(AddInput);

  /**
   * @method addInputList
   * @param {Array} inputs
   */
  static NAN_METHOD(AddInputList);

  /**
   * @method addControlInput
   * @param {Object} input
   * @param {Number} input.index
   * @param {Operation} input.op
   */
  static NAN_METHOD(AddControlInput);

  /**
   * @method finish
   */
  static NAN_METHOD(Finish);

  /**
   * @method NewFromOperation
   */
  static Local<Object> NewFromOperation(Local<Object> graph, TF_Operation* oper);

  TF_Operation* _oper;
  TF_OperationDescription* _description;

private:
  explicit Operation();
  explicit Operation(TF_Graph* graph, const char* type, const char* name);
  ~Operation();

  static inline Nan::Persistent<v8::Function>& constructor();
};

}
#pragma once

#include <stdio.h>
#include <node.h>
#include <nan.h>
#include "tensorflow/c/c_api.h"

using namespace v8;

namespace TensorflowNode {

/**
 * @class Tensor
 * @extends Nan::ObjectWrap
 */
class Tensor : public Nan::ObjectWrap {
public:
  static NAN_MODULE_INIT(Init);

  /**
   * @property type
   */
  static NAN_PROPERTY_GETTER(TypeGetter);

  /**
   * @property data
   */
  static NAN_PROPERTY_GETTER(DataGetter);

  /**
   * @property shape
   */
  static NAN_PROPERTY_GETTER(ShapeGetter);

  /**
   * @constructor
   * @param {TF_DataType} type - the data type
   * @param {Array} dims - the dims array
   * @param {Object} options - the tensor options
   * @param {Buffer} options.data - the data
   * @param {Number} options.size - the data size
   */
  static NAN_METHOD(New);

  /**
   * @method GetDimByIndex
   * @param {Number} index - the index
   * @return {Number} the number
   */
  static NAN_METHOD(GetDimByIndex);

  /**
   * @method MaybeMove
   * @return {Tensor} return new moved tensor, or null if not
   */
  static NAN_METHOD(MaybeMove);

  /**
   * destroy the tensor object
   * @method Destroy
   */
  static NAN_METHOD(Destory);

  /**
   * @static
   * @method _stringEncode
   */
  static NAN_METHOD(StringEncode);

  /**
   * @static
   * @method _stringDecode
   */
  static NAN_METHOD(StringDecode);

  /**
   * @method NewFromTensor
   */
  static Local<Object> NewFromTensor(TF_Tensor* data);

  /**
   * interner TF_Tensor reference
   */
  TF_Tensor* _tensor;

private:
  explicit Tensor();
  explicit Tensor(TF_DataType type, const int64_t* dims, int numOfDims, void* data, size_t len);
  explicit Tensor(TF_DataType type, const int64_t* dims, int numOfDims, size_t len);
  ~Tensor();

  static void Deallocate(void* data, size_t len, void* arg);
  static inline Nan::Persistent<v8::Function>& constructor();
};


}
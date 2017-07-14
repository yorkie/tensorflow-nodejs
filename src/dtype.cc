#include "src/dtype.h"
#include "src/internal.h"

using namespace v8;

namespace TensorflowNode {

NAN_MODULE_INIT(DType::Init) {
  Local<Object> obj = Nan::New<Object>();
  // define types
  Nan::Set(obj, 
    Nan::New<String>("float16").ToLocalChecked(), 
    Nan::New<Number>(TF_HALF));
  Nan::Set(obj, 
    Nan::New<String>("float32").ToLocalChecked(), 
    Nan::New<Number>(TF_FLOAT));
  Nan::Set(obj, 
    Nan::New<String>("float64").ToLocalChecked(), 
    Nan::New<Number>(TF_DOUBLE));
  Nan::Set(obj, 
    Nan::New<String>("bfloat16").ToLocalChecked(), 
    Nan::New<Number>(TF_BFLOAT16));
  Nan::Set(obj, 
    Nan::New<String>("complex64").ToLocalChecked(), 
    Nan::New<Number>(TF_COMPLEX64));
  Nan::Set(obj, 
    Nan::New<String>("complex128").ToLocalChecked(), 
    Nan::New<Number>(TF_COMPLEX128));
  Nan::Set(obj, 
    Nan::New<String>("int8").ToLocalChecked(), 
    Nan::New<Number>(TF_INT8));
  Nan::Set(obj, 
    Nan::New<String>("int16").ToLocalChecked(), 
    Nan::New<Number>(TF_INT16));
  Nan::Set(obj, 
    Nan::New<String>("int32").ToLocalChecked(), 
    Nan::New<Number>(TF_INT32));
  Nan::Set(obj, 
    Nan::New<String>("int64").ToLocalChecked(), 
    Nan::New<Number>(TF_INT64));
  Nan::Set(obj, 
    Nan::New<String>("uint8").ToLocalChecked(), 
    Nan::New<Number>(TF_UINT8));
  Nan::Set(obj, 
    Nan::New<String>("uint16").ToLocalChecked(), 
    Nan::New<Number>(TF_UINT16));
  Nan::Set(obj, 
    Nan::New<String>("bool").ToLocalChecked(),
    Nan::New<Number>(TF_BOOL));
  Nan::Set(obj, 
    Nan::New<String>("string").ToLocalChecked(), 
    Nan::New<Number>(TF_STRING));
  Nan::Set(obj, 
    Nan::New<String>("qint8").ToLocalChecked(), 
    Nan::New<Number>(TF_QINT8));
  Nan::Set(obj, 
    Nan::New<String>("qint16").ToLocalChecked(), 
    Nan::New<Number>(TF_QINT16));
  Nan::Set(obj, 
    Nan::New<String>("qint32").ToLocalChecked(), 
    Nan::New<Number>(TF_QINT32));
  Nan::Set(obj, 
    Nan::New<String>("quint8").ToLocalChecked(), 
    Nan::New<Number>(TF_QUINT8));
  Nan::Set(obj, 
    Nan::New<String>("quint16").ToLocalChecked(), 
    Nan::New<Number>(TF_QUINT16));
  Nan::Set(obj, 
    Nan::New<String>("resource").ToLocalChecked(), 
    Nan::New<Number>(TF_RESOURCE));
  // define util methods
  Nan::Set(obj, Nan::New<String>("sizeOf").ToLocalChecked(), Nan::New<Function>(GetSize));
  Nan::Set(target, Nan::New<String>("dtype").ToLocalChecked(), obj);
}

NAN_METHOD(DType::GetSize) {
  if (!info[0]->IsNumber()) {
    return Nan::ThrowTypeError("First argument must be number.");
  }
  TF_DataType type = (TF_DataType)info[0]->Uint32Value();
  size_t size = TF_DataTypeSize(type);
  info.GetReturnValue().Set(Nan::New<Number>(size));
}

}
#include "src/tensor.h"
#include "src/internal.h"

using namespace v8;

namespace TensorflowNode {

NAN_MODULE_INIT(Tensor::Init) {
  Local<FunctionTemplate> tmpl = Nan::New<FunctionTemplate>(New);
  tmpl->SetClassName(Nan::New("Tensor").ToLocalChecked());
  tmpl->InstanceTemplate()->SetInternalFieldCount(1);

  Nan::SetAccessor(tmpl->InstanceTemplate(),
    Nan::New<String>("type").ToLocalChecked(), TypeGetter);
  Nan::SetAccessor(tmpl->InstanceTemplate(), 
    Nan::New<String>("data").ToLocalChecked(), DataGetter);

  Nan::SetPrototypeMethod(tmpl, "getDimByIndex", GetDimByIndex);
  Nan::SetPrototypeMethod(tmpl, "maybeMove", MaybeMove);
  Nan::SetPrototypeMethod(tmpl, "destroy", Destory);

  constructor().Reset(Nan::GetFunction(tmpl).ToLocalChecked());
  Nan::Set(target,
    Nan::New("Tensor").ToLocalChecked(), 
    Nan::GetFunction(tmpl).ToLocalChecked());
}

NAN_PROPERTY_GETTER(Tensor::TypeGetter) {
  TensorflowNode::Tensor* tensor = ObjectWrap::Unwrap<TensorflowNode::Tensor>(info.This());
  TF_DataType type = TF_TensorType(tensor->_tensor);
  info.GetReturnValue().Set(Nan::New<Number>(type));
}

NAN_PROPERTY_GETTER(Tensor::NumOfDimGetter) {
  TensorflowNode::Tensor* tensor = ObjectWrap::Unwrap<TensorflowNode::Tensor>(info.This());
  int num = TF_NumDims(tensor->_tensor);
  info.GetReturnValue().Set(Nan::New<Number>(num));
}

NAN_PROPERTY_GETTER(Tensor::DataGetter) {
  TensorflowNode::Tensor* tensor = ObjectWrap::Unwrap<TensorflowNode::Tensor>(info.This());
  void* data = TF_TensorData(tensor->_tensor);
  size_t len = TF_TensorByteSize(tensor->_tensor);
  Local<Object> buffer = Nan::NewBuffer((char*)data, len).ToLocalChecked();
  info.GetReturnValue().Set(buffer);
}

NAN_METHOD(Tensor::New) {
  TensorflowNode::Tensor *tensor;

  // Tensorflow::Tensor accepts zero arguments instance for C++ level to create
  // an instance of Tensor(), and set TF_Tensor directly by setTensorData.
  if (info.Length() == 0) {
    tensor = new TensorflowNode::Tensor();
  } else {
    // type: the type could be tf.dtype.*
    TF_DataType type = (TF_DataType)info[0]->Uint32Value();
    // dims: the dimensions of this tensor.
    Local<Object> maybeDims = info[1]->ToObject();

    Local<ArrayBuffer> maybeData;
    bool hasData = false;
    size_t len = 0;

    if (!maybeDims->IsArray()) {
      Nan::ThrowError("The second parameter `dims` should be an array");
      return;
    }
    Local<String> lenstr = Nan::New("length").ToLocalChecked();
    size_t numOfDims = Nan::Get(maybeDims, lenstr).ToLocalChecked()->Uint32Value();
    int64_t dims[numOfDims];

    for (size_t i = 0; i < numOfDims; i++) {
      dims[i] = Nan::Get(maybeDims, i).ToLocalChecked()->Int32Value();
    }

    if (info.Length() == 3 && info[2]->IsObject()) {
      Local<Object> options = info[2]->ToObject();
      Local<String> dataKey = Nan::New<String>("data").ToLocalChecked()->ToString();
      if (Nan::Has(options, dataKey).FromJust()) {
        // The `options.data` are a type of `Buffer`, and we could get the 
        // typed array by `data.buffer`.
        Local<String> bufferKey = Nan::New<String>("buffer").ToLocalChecked()->ToString();
        Local<Object> data = Nan::Get(options, dataKey).ToLocalChecked()->ToObject();
        maybeData = Local<ArrayBuffer>::Cast(Nan::Get(data, bufferKey).ToLocalChecked()->ToObject());
        hasData = true;
      } else {
        Local<String> sizeKey = Nan::New<String>("size").ToLocalChecked()->ToString();
        len = Nan::Get(options, sizeKey).ToLocalChecked()->Int32Value();
      }
    }

    if (hasData) {
      void* data = maybeData->GetContents().Data();
      len = maybeData->ByteLength();
      tensor = new TensorflowNode::Tensor(type, dims, numOfDims, data, len);
    } else {
      tensor = new TensorflowNode::Tensor(type, dims, numOfDims, len);
    }
  }
  
  tensor->Wrap(info.This());
  info.GetReturnValue().Set(info.This());
}

NAN_METHOD(Tensor::GetDimByIndex) {
  TensorflowNode::Tensor* tensor = ObjectWrap::Unwrap<TensorflowNode::Tensor>(info.This());
  int index = info.Length() > 0 ? info[0]->Int32Value() : 0;
  int64_t dim = TF_Dim(tensor->_tensor, index);
  info.GetReturnValue().Set(Nan::New<Number>(dim));
}

NAN_METHOD(Tensor::MaybeMove) {
  TensorflowNode::Tensor* tensor = ObjectWrap::Unwrap<TensorflowNode::Tensor>(info.This());
  TF_Tensor* movedTensor = TF_TensorMaybeMove(tensor->_tensor);
  if (movedTensor != nullptr) {
    tensor->_tensor = movedTensor;
    info.GetReturnValue().Set(info.This());
  } else {
    info.GetReturnValue().Set(false);
  }
}

NAN_METHOD(Tensor::Destory) {
  TensorflowNode::Tensor* tensor = ObjectWrap::Unwrap<TensorflowNode::Tensor>(info.This());
  delete tensor;
  info.GetReturnValue().Set(info.This());
}

Local<Object> 
Tensor::NewFromTensor(TF_Tensor* data) {
  Nan::EscapableHandleScope scope;
  Local<Function> tmpl = Nan::New(constructor());
  Local<Object> target = Nan::NewInstance(tmpl).ToLocalChecked();
  TensorflowNode::Tensor *tensor = ObjectWrap::Unwrap<TensorflowNode::Tensor>(target);
  tensor->_tensor = data;
  return scope.Escape(target);
}

Tensor::Tensor() {
  // Placeholder
}

Tensor::Tensor(TF_DataType type, const int64_t* dims, int numOfDims, void* data, size_t len) {
  _tensor = TF_NewTensor(type, dims, numOfDims, data, len, &Deallocate, nullptr);
}

Tensor::Tensor(TF_DataType type, const int64_t* dims, int numOfDims, size_t len) {
  _tensor = TF_AllocateTensor(type, dims, numOfDims, len);
}

Tensor::~Tensor() {
  if (_tensor != nullptr) {
    TF_DeleteTensor(_tensor);
  }
}

void 
Tensor::Deallocate(void* data, size_t len, void* arg) {
  // TODO
}

inline Nan::Persistent<v8::Function>& 
Tensor::constructor() {
  static Nan::Persistent<v8::Function> my_constructor;
  return my_constructor;
}

}
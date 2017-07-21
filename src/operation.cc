#include "src/operation.h"
#include "src/internal.h"
#include "src/graph.h"
#include "src/tensor.h"

using namespace v8;

namespace TensorflowNode {

#define OP_PROPERTY_GETTER(NAME)                                                    \
  TensorflowNode::Operation* operation =                                            \
    ObjectWrap::Unwrap<TensorflowNode::Operation>(info.This());                     \
  if (operation->_oper == NULL)                                                     \
    return Nan::ThrowTypeError("operation is not created yet.");                    \
  Local<String> val =                                                               \
    Nan::New<String>(TF_Operation ## NAME(operation->_oper)).ToLocalChecked();      \
  info.GetReturnValue().Set(val);                                                   \


NAN_MODULE_INIT(Operation::Init) {
  Local<FunctionTemplate> tmpl = Nan::New<FunctionTemplate>(New);
  tmpl->SetClassName(Nan::New("Operation").ToLocalChecked());
  tmpl->InstanceTemplate()->SetInternalFieldCount(1);

  Nan::SetAccessor(tmpl->InstanceTemplate(), 
    Nan::New<String>("name").ToLocalChecked(), NameGetter);
  Nan::SetAccessor(tmpl->InstanceTemplate(), 
    Nan::New<String>("type").ToLocalChecked(), TypeGetter);
  Nan::SetAccessor(tmpl->InstanceTemplate(), 
    Nan::New<String>("device").ToLocalChecked(), DeviceGetter);
  Nan::SetAccessor(tmpl->InstanceTemplate(),
    Nan::New<String>("outputs").ToLocalChecked(), OutputsGetter);

  Nan::SetPrototypeMethod(tmpl, "setAttrType", SetAttrType);
  Nan::SetPrototypeMethod(tmpl, "SetAttrBool", SetAttrBool);
  Nan::SetPrototypeMethod(tmpl, "SetAttrInt", SetAttrInt);
  Nan::SetPrototypeMethod(tmpl, "SetAttrFloat", SetAttrFloat);
  Nan::SetPrototypeMethod(tmpl, "setAttrString", SetAttrString);
  Nan::SetPrototypeMethod(tmpl, "setAttrShape", SetAttrShape);
  Nan::SetPrototypeMethod(tmpl, "setAttrTensor", SetAttrTensor);
  Nan::SetPrototypeMethod(tmpl, "addInput", AddInput);
  Nan::SetPrototypeMethod(tmpl, "addInputList", AddInputList);
  Nan::SetPrototypeMethod(tmpl, "addControlInput", AddControlInput);
  Nan::SetPrototypeMethod(tmpl, "finish", Finish);

  Local<Function> func = Nan::GetFunction(tmpl).ToLocalChecked();
  constructor().Reset(func);
  Nan::Set(target, Nan::New("Operation").ToLocalChecked(), func);
}

NAN_METHOD(Operation::New) {
  if (info.Length() == 0) {
    return Nan::ThrowTypeError("should call with graph, type and name.");
  }
  TensorflowNode::Graph* graph = ObjectWrap::Unwrap<TensorflowNode::Graph>(info[0]->ToObject());
  TensorflowNode::Operation* operation;

  if (info.Length() >= 3) {
    V8_STRING_TO_CSTR(type, info[1]);
    V8_STRING_TO_CSTR(name, info[2]);
    operation = new TensorflowNode::Operation(graph->_graph, type, name);
  } else {
    operation = new TensorflowNode::Operation();
  }
  operation->Wrap(info.This());

  // this._graph = graph
  Nan::Set(info.This(), Nan::New<String>("_graph").ToLocalChecked(), info[0]);
  info.GetReturnValue().Set(info.This());
}

NAN_PROPERTY_GETTER(Operation::NameGetter) {
  OP_PROPERTY_GETTER(Name);
}

NAN_PROPERTY_GETTER(Operation::TypeGetter) {
  OP_PROPERTY_GETTER(OpType);
}

NAN_PROPERTY_GETTER(Operation::DeviceGetter) {
  OP_PROPERTY_GETTER(Device);
}

NAN_PROPERTY_GETTER(Operation::OutputsGetter) {
  Local<Object> graphObj = Nan::Get(info.This(), Nan::New<String>("_graph").ToLocalChecked()).ToLocalChecked()->ToObject();
  TensorflowNode::Graph* graph = ObjectWrap::Unwrap<TensorflowNode::Graph>(graphObj);
  TensorflowNode::Operation* operation = ObjectWrap::Unwrap<TensorflowNode::Operation>(info.This());
  if (operation->_oper == NULL)
    return Nan::ThrowTypeError("operation is not created yet.");
  int numOfOutputs = TF_OperationNumOutputs(operation->_oper);
  Local<Array> outputsObj = Nan::New<Array>(numOfOutputs);

  for (int i = 0; i < numOfOutputs; i++) {
    TF_Output output = TF_Output{operation->_oper, i};
    int numOfDims = TF_GraphGetTensorNumDims(graph->_graph, output, status);
    if (TF_GetCode(status) != TF_OK) {
      ThrowStatusError();
      return;
    }

    TF_DataType type = TF_OperationOutputType(output);
    Local<Object> item = Nan::New<Object>();
    Local<Array> shape;

    // The `numOfDims` possibly to be -1
    if (numOfDims > 0) {
      int64_t dims[numOfDims];
      TF_GraphGetTensorShape(graph->_graph, output, dims, numOfDims, status);
      if (TF_GetCode(status) != TF_OK) {
        ThrowStatusError();
        return;
      }
      shape = Nan::New<Array>(numOfDims);
      for (int i = 0; i < numOfDims; i++) {
        Nan::Set(shape, i, Nan::New<Number>(dims[i]));
      }
    } else {
      // if the `numOfDims` is -1, shape should be [], as a scalar value.
      shape = Nan::New<Array>(0);
    }
    Nan::Set(item, Nan::New<String>("shape").ToLocalChecked(), shape);
    Nan::Set(item, 
      Nan::New<String>("type").ToLocalChecked(), 
      Nan::New<Number>((int)type));
    Nan::Set(outputsObj, i, item);
  }
  info.GetReturnValue().Set(outputsObj);
}

NAN_METHOD(Operation::SetAttrType) {
  if (info.Length() != 2) {
    Nan::ThrowTypeError("attr name and type are required");
    return;
  }
  TensorflowNode::Operation* operation = ObjectWrap::Unwrap<TensorflowNode::Operation>(info.This());
  V8_STRING_TO_CSTR(name, info[0]);
  TF_DataType type = (TF_DataType)info[1]->Uint32Value();
  TF_SetAttrType(operation->_description, name, type);
  info.GetReturnValue().Set(info.This());
}

NAN_METHOD(Operation::SetAttrBool) {
  if (info.Length() != 2) {
    Nan::ThrowTypeError("attr name and value are required");
    return;
  }
  TensorflowNode::Operation* operation = ObjectWrap::Unwrap<TensorflowNode::Operation>(info.This());
  V8_STRING_TO_CSTR(name, info[0]);
  bool value = info[1]->BooleanValue();
  TF_SetAttrBool(operation->_description, name, value);
  info.GetReturnValue().Set(info.This());
}

NAN_METHOD(Operation::SetAttrInt) {
  if (info.Length() != 2) {
    Nan::ThrowTypeError("attr name and value are required");
    return;
  }
  TensorflowNode::Operation* operation = ObjectWrap::Unwrap<TensorflowNode::Operation>(info.This());
  V8_STRING_TO_CSTR(name, info[0]);
  int64_t value = info[1]->IntegerValue();
  TF_SetAttrInt(operation->_description, name, value);
  info.GetReturnValue().Set(info.This());
}

NAN_METHOD(Operation::SetAttrFloat) {
  if (info.Length() != 2) {
    Nan::ThrowTypeError("attr name and value are required");
    return;
  }
  TensorflowNode::Operation* operation = ObjectWrap::Unwrap<TensorflowNode::Operation>(info.This());
  V8_STRING_TO_CSTR(name, info[0]);
  double value = info[1]->NumberValue();
  TF_SetAttrFloat(operation->_description, name, (float)value);
  info.GetReturnValue().Set(info.This());
}

NAN_METHOD(Operation::SetAttrString) {
  if (info.Length() != 2) {
    Nan::ThrowTypeError("attr name and value are required");
    return;
  }
  TensorflowNode::Operation* operation = ObjectWrap::Unwrap<TensorflowNode::Operation>(info.This());
  V8_STRING_TO_CSTR(name, info[0]);
  V8_STRING_TO_CSTR(value, info[1]);
  size_t len = info[1]->ToString()->Length();
  TF_SetAttrString(operation->_description, name, value, len);
  info.GetReturnValue().Set(info.This());
}

NAN_METHOD(Operation::SetAttrShape) {
  if (info.Length() != 2) {
    Nan::ThrowTypeError("attr name and dims/shape are required");
  }

  TensorflowNode::Operation* operation = ObjectWrap::Unwrap<TensorflowNode::Operation>(info.This());
  V8_STRING_TO_CSTR(name, info[0]);
  Local<Object> maybeDims = info[1]->ToObject();
  if (!maybeDims->IsArray()) {
    Nan::ThrowError("The second parameter `shape` should be an array");
    return;
  }

  Local<String> lenstr = Nan::New("length").ToLocalChecked();
  size_t numOfDims = Nan::Get(maybeDims, lenstr).ToLocalChecked()->Uint32Value();
  int64_t dims[numOfDims];

  for (size_t i = 0; i < numOfDims; i++) {
    dims[i] = Nan::Get(maybeDims, i).ToLocalChecked()->Int32Value();
  }

  TF_SetAttrShape(operation->_description, name, dims, numOfDims);
  info.GetReturnValue().Set(info.This());
}

NAN_METHOD(Operation::SetAttrTensor) {
  if (info.Length() != 2) {
    Nan::ThrowTypeError("attr name and tensor are required");
    return;
  }
  TensorflowNode::Operation* operation = ObjectWrap::Unwrap<TensorflowNode::Operation>(info.This());
  V8_STRING_TO_CSTR(name, info[0]);
  TensorflowNode::Tensor* tensor = ObjectWrap::Unwrap<TensorflowNode::Tensor>(info[1]->ToObject());
  TF_SetAttrTensor(operation->_description, name, tensor->_tensor, status);
  if (TF_GetCode(status) != TF_OK) {
    return ThrowStatusError();
  }
  info.GetReturnValue().Set(info.This());
}

NAN_METHOD(Operation::AddInput) {
  if (info.Length() == 0) {
    Nan::ThrowTypeError("input is required");
    return;
  }

  TF_Output input;
  {
    int index = info[0]->Int32Value();
    TensorflowNode::Operation* op = ObjectWrap::Unwrap<TensorflowNode::Operation>(info[1]->ToObject());
    input = TF_Output{op->_oper, index};
  }

  TensorflowNode::Operation* operation = ObjectWrap::Unwrap<TensorflowNode::Operation>(info.This());
  TF_AddInput(operation->_description, input);
  info.GetReturnValue().Set(info.This());
}

NAN_METHOD(Operation::AddInputList) {
  if (info.Length() == 0 || !info[0]->IsArray()) {
    Nan::ThrowTypeError("inputs is required and be an Array");
    return;
  }

  Local<Array> list = Local<Array>::Cast(info[0]);
  size_t len = list->Length();
  TF_Output inputs[len];

  for (size_t i = 0; i < len; i++) {
    Local<Object> item = Nan::Get(list, i).ToLocalChecked()->ToObject();
    Local<Value> indexKey = Nan::New<String>("index").ToLocalChecked();
    Local<Value> opKey = Nan::New<String>("op").ToLocalChecked();

    int index = Nan::Get(item, indexKey).ToLocalChecked()->Int32Value();
    Local<Object> op = Nan::Get(item, opKey).ToLocalChecked()->ToObject();

    TensorflowNode::Operation* operation = ObjectWrap::Unwrap<TensorflowNode::Operation>(op);
    inputs[i] = TF_Output{operation->_oper, index};
  }

  TensorflowNode::Operation* operation = ObjectWrap::Unwrap<TensorflowNode::Operation>(info.This());
  TF_AddInputList(operation->_description, inputs, len);
  info.GetReturnValue().Set(info.This());
}

NAN_METHOD(Operation::AddControlInput) {
  if (info.Length() == 0) {
    Nan::ThrowTypeError("input is required");
    return;
  }
  TensorflowNode::Operation* input = ObjectWrap::Unwrap<TensorflowNode::Operation>(info[0]->ToObject());
  TensorflowNode::Operation* operation = ObjectWrap::Unwrap<TensorflowNode::Operation>(info.This());
  TF_AddControlInput(operation->_description, input->_oper);
  info.GetReturnValue().Set(info.This());
}

NAN_METHOD(Operation::Finish) {
  TensorflowNode::Operation* operation = ObjectWrap::Unwrap<TensorflowNode::Operation>(info.This());
  if (operation->_oper != NULL) {
    return Nan::ThrowError("finish is can be called once");
  }

  TF_Operation* oper = TF_FinishOperation(operation->_description, status);
  if (TF_GetCode(status) == TF_OK) {
    operation->_oper = oper;
  } else {
    ThrowStatusError();
  }
}

Local<Object>
Operation::NewFromOperation(Local<Object> graph, TF_Operation* oper) {
  Nan::EscapableHandleScope scope;
  Local<Function> tmpl = Nan::New(constructor());
  Local<Value> argv[1] = { graph };
  Local<Object> target = Nan::NewInstance(tmpl, 1, argv).ToLocalChecked();
  TensorflowNode::Operation *operation = ObjectWrap::Unwrap<TensorflowNode::Operation>(target);
  operation->_oper = oper;
  return scope.Escape(target);
}

Operation::Operation() {
  // Placeholder
}

Operation::Operation(TF_Graph* graph, const char* type, const char* name) {
  _description = TF_NewOperation(graph, type, name);
  _oper = NULL;
}

Operation::~Operation() {
  // TODO
}

inline Nan::Persistent<v8::Function>& 
Operation::constructor() {
  static Nan::Persistent<v8::Function> my_constructor;
  return my_constructor;
}

}
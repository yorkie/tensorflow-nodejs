#pragma once

#include <stdio.h>
#include <node.h>
#include <nan.h>
#include "tensorflow/c/c_api.h"

using namespace v8;

namespace TensorflowNode {

/**
 * @class Session
 * @extends Nan::ObjectWrap
 */
class Session : public Nan::ObjectWrap {
public:
  static inline Nan::Persistent<v8::Function>& constructor();
  static NAN_MODULE_INIT(Init);

  /**
   * @constructor
   * @param {Graph} graph - the graph on this session
   * @param {Object} options - the session options
   * @param {String} options.target - the target: "local"/"ip:port"/"host:port"
   * @param {String} options.config - the config
   */
  static NAN_METHOD(New);

  /**
   * @static
   * @method load
   */
  static NAN_METHOD(Load);

  /**
   * @method close
   */
  static NAN_METHOD(Close);

  /**
   * @method destroy
   */
  static NAN_METHOD(Destory);

  /**
   * @method run
   * @param {Graph} fetches
   * @param {Object} feed
   * @param {Object} options
   * @param {Object} metadata
   */
  static NAN_METHOD(Run);

private:
  explicit Session(TF_Graph* graph, const char* target, ArrayBuffer::Contents options);
  explicit Session(TF_Graph* graph, const char* target, ArrayBuffer::Contents options, 
    const char* exportDir, const char* const* tags, int tagsLen);
  ~Session();
  void DeleteInputValues();
  void DeleteOutputValues();
  void SetInputs(std::vector<std::pair<TF_Operation*, TF_Tensor*>> inputs);
  void SetOutputs(std::initializer_list<TF_Operation*> outputs);
  void SetTargets(std::initializer_list<TF_Operation*> targets);

  TF_Session* _session;
  std::vector<TF_Output> inputs_;
  std::vector<TF_Tensor*> input_values_;
  std::vector<TF_Output> outputs_;
  std::vector<TF_Tensor*> output_values_;
  std::vector<TF_Operation*> targets_;
};

}
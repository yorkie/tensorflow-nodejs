module.exports = function(tensorflow) {
  'use strict';

  const Graph = tensorflow.Graph;

  /**
   * @class Graph
   * @module flow
   */
  return {

    /**
     * @class Graph.flow
     * @method switch
     */
    switch: Graph.createOperationBuilder(
      'Switch',
      function(descriptor, data, pred, type) {
        descriptor.addInput(0, data);
        descriptor.addInput(0, pred);
        descriptor.setAttrType('T', type || tensorflow.dtype.int32);
      }
    ),

    /**
     * @class Graph.flow
     * @method merge
     */
    merge: Graph.createOperationBuilder(
      'Merge',
      function(descriptor, inputs, type) {
        descriptor.addInputList(inputs.map((op) => {
          return { op, index: 0 };
        }));
        descriptor.setAttrType('T', type || tensorflow.dtype.int32);
      }
    ),

    /**
     * @class Graph.flow
     * @method enter
     */
    enter: Graph.createOperationBuilder(
      'Enter',
      function(descriptor, data, type, options) {
        descriptor.addInput(0, data);
        descriptor.setAttrType('T', type || tensorflow.dtype.int32);
        if (options && options.frameName)
          descriptor.setAttrString('frame_name', options.frameName);
        if (options && typeof options.isConstant === 'boolean')
          descriptor.setAttrBool('is_constant', options.is_constant);
        if (options && typeof options.parallelIterations === 'number')
          descriptor.setAttrInt('parallel_iterations', options.parallelIterations);
      }
    ),

    /**
     * @class Graph.flow
     * @method exit
     */
    exit: Graph.createOperationBuilder(
      'Exit',
      function(descriptor, data, type) {
        descriptor.addInput(0, input);
        descriptor.setAttrType('T', type || tensorflow.dtype.int32);
      }
    ),

    /**
     * @class Graph.flow
     * @method abort
     * @param {Error} error
     */
    abort: Graph.createOperationBuilder(
      'Abort',
      function(descriptor, error) {
        if (!(error instanceof Error))
          throw new TypeError('the error must be an instance of `Error`.');
        descriptor.setAttrString('error_msg', error.message || 'unknow error.');
        // FIXME(Yorkie): in core, it uses this API to let `Abort` could be done without
        // an abnormal signal, this is not strict, especially in JavaScript. Then that's
        // also the reason why we make this function accept only an `Error` instance.
        // 
        // See this commit from core for more details:
        // https://github.com/tensorflow/tensorflow/commit/70eb79ed25532fd336bbca2ca5c0849a28ec9191
        descriptor.setAttrBool('exit_without_error', false);
      }
    ),

    /**
     * @class Graph.flow
     * @method trigger
     */
    trigger: Graph.createOperationBuilder(
      'ControlTrigger',
      function(descriptor) {
        // Do nothing, Just serves as a control trigger for scheduling.
        // Only useful as a placeholder for control edges.
      }
    ),

  };
};


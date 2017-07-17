module.exports = function(tensorflow) {
  'use strict';

  const Graph = tensorflow.Graph;

  /**
   * @class Graph
   * @module training
   */
  return {

    /**
     * @method applyGradientDescent
     */
    applyGradientDescent: Graph.createOperationBuilder(
      'ApplyGradientDescent',
      function(descriptor, ref, alpha, delta, useLocking) {
        descriptor.addInput(0, ref);
        descriptor.addInput(0, alpha);
        descriptor.addInput(0, delta);
        if (useLocking) {
          descriptor.setAttrBool('use_locking', true);
        }
      }
    ),

    /**
     * @method resourceApplyGradientDescent
     */
    resourceApplyGradientDescent: Graph.createOperationBuilder(
      'ResourceApplyGradientDescent',
      function(descriptor, ref, alpha, delta, useLocking) {
        descriptor.addInput(0, ref);
        descriptor.addInput(0, alpha);
        descriptor.addInput(0, delta);
        if (useLocking) {
          descriptor.setAttrBool('use_locking', true);
        }
      }
    ),

  };
};


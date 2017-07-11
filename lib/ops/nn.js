module.exports = function(tensorflow) {
  'use strict';

  const Graph = tensorflow.Graph;

  /**
   * @class Graph
   * @module nn
   */
  return {
    
    /**
     * @class Graph.nn
     * @method l2loss
     */
    l2loss: Graph.createOperationBuilder(
      'L2Loss',
      function(descriptor, t) {
        descriptor.addInput(0, t);
      }
    ),

    /**
     * @class Graph.nn
     * @method lrn
     */
    lrn: Graph.createOperationBuilder(
      'LRN',
      function(descriptor, input, options) {
        descriptor.addInput(0, input);
        options = options || {};
        if (typeof options.radius === 'number')
          descriptor.setAttrInt('depth_radius', options.radius);
        if (typeof options.bias === 'number')
          descriptor.setAttrFloat('bias', options.bias);
        if (typeof options.alpha === 'number')
          descriptor.setAttrFloat('alpha', options.alpha);
        if (typeof options.beta === 'number')
          descriptor.setAttrFloat('beta', options.beta);
      }
    ),

    /**
     * @class Graph.nn
     * @method relu
     */
    relu: Graph.createOperationBuilder(
      'Relu',
      function(descriptor, features) {
        descriptor.addInput(0, features);
      }
    ),

    /**
     * @class Graph.nn
     * @method relu6
     */
    relu6: Graph.createOperationBuilder(
      'Relu6',
      function(descriptor, features) {
        descriptor.addInput(0, features);
      }
    ),

    /**
     * @class Graph.nn
     * @method elu
     */
    elu: Graph.createOperationBuilder(
      'Elu',
      function(descriptor, features) {
        descriptor.addInput(0, features);
      }
    ),

    /**
     * @class Graph.nn
     * @method softplus
     */
    softplus: Graph.createOperationBuilder(
      'Softplus',
      function(descriptor, features) {
        descriptor.addInput(0, features);
      }
    ),

    /**
     * @class Graph.nn
     * @method softsign
     */
    softsign: Graph.createOperationBuilder(
      'Softsign',
      function(descriptor, features) {
        descriptor.addInput(0, features);
      }
    ),

    /**
     * @class Graph.nn
     * @method softmax
     */
    softmax: Graph.createOperationBuilder(
      'Softmax', 
      function(descriptor, logits) {
        descriptor.addInput(0, logits);
      }
    ),
    
    /**
     * @class Graph.nn
     * @method logSoftmax
     */
    logSoftmax: Graph.createOperationBuilder(
      'LogSoftmax',
      function(descriptor, logits) {
        descriptor.addInput(0, logits);
      }
    ),

    /**
     * @class Graph.nn
     * @method inTopK
     */
    inTopK: Graph.createOperationBuilder(
      'InTopK',
      function(descriptor, predictions, targets, K) {
        descriptor.addInput(0, predictions);
        descriptor.addInput(0, targets);
        descriptor.setAttrInt('k', K);
      }
    ),

  };

};


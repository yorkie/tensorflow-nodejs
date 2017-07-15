module.exports = function(tensorflow) {
  'use strict';

  const Graph = tensorflow.Graph;

  /**
   * @class Graph
   * @module state
   */
  return {

    /**
     * @class Graph.state
     * @method variable
     */
    variable: Graph.createOperationBuilder(
      'VariableV2',
      function(descriptor, type, shape, options) {
        descriptor.setAttrType('dtype', type);
        descriptor.setAttrShape('shape', shape);
        if (options && options.container) {
          descriptor.setAttrString('container', options.container);
        }
        if (options && options.sharedName === 'string') {
          descriptor.setAttrString('shared_name', options.sharedName);
        }
      }
    ),

    /**
     * @class Graph.state
     * @method assign
     */
    assign: Graph.createOperationBuilder(
      'Assign',
      function(descriptor, variable, value, options) {
        const output = tensorflow.protoOfStrictly(value);
        descriptor.addInput(0, variable);
        descriptor.addInput(0, value);
        descriptor.setAttrType('T', output.type);
        if (options && options.validateShape) {
          descriptor.setAttrBool('validate_shape', options.validateShape);
        }
        if (options && options.useLocking) {
          descriptor.setAttrBool('use_locking', options.useLocking);
        }
      }
    )

  };

};


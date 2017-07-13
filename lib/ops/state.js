module.exports = function(tensorflow) {
  'use strict';

  const Graph = tensorflow.Graph;

  function variableOpHandler(descriptor, type, shape, options) {
    descriptor.setAttrType('dtype', type);
    descriptor.setAttrShape('shape', shape);
    if (options) {
      if (typeof options.container === 'string') {
        descriptor.setAttrString('container', options.container);
      }
      if (typeof options.sharedName === 'string') {
        descriptor.setAttrString('shared_name', options.sharedName);
      }
    }
  }

  /**
   * @class Graph
   * @method var
   */
  Graph.prototype.variable = Graph.createOperationBuilder(
    'Variable', variableOpHandler
  );

  /**
   * @class Graph
   * @method var2
   */
  Graph.prototype.variable2 = Graph.createOperationBuilder(
    'Variable2', variableOpHandler
  );

};


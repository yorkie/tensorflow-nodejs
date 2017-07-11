module.exports = function(tensorflow) {
  'use strict';

  const Graph = tensorflow.Graph;

  function noop() {
    // Placeholder
  }

  /**
   * @class Graph
   * @method const
   */
  Graph.prototype.noop = Graph.createOperationBuilder('NoOp', noop);

};


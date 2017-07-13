'use strict';

module.exports = function(tensorflow) {

  let Graph = tensorflow.Graph;

  /**
   * Operation ID is for generate the operation name if not presents
   */
  // TODO(Yorkie): implement the type-based id object to manage all created operations.
  let operationId = 0;

  /**
   * @class Graph
   * @static
   * @method createOperationBuilder
   * @param {String} type
   * @param {String} initializer
   */
  Graph.createOperationBuilder = function createOperationBuilder(type, initializer) {
    return function createOperationFromBuilder() {
      // FIXME(Yorkie): we auto-generate the operation name here, user needs a way
      // to customize its op?
      const args = [].slice.call(arguments);
      const name = `${type}_${operationId++}`;
      const op = new tensorflow.Operation(this, type, name);
      initializer.apply(null, [op].concat(args));
      op.finish();
      return op;
    };
  };

}
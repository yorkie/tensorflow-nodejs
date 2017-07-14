'use strict';

module.exports = function(tensorflow) {

  let Graph = tensorflow.Graph;

  /**
   * Operation ID is for generate the operation name if not presents
   */
  // TODO(Yorkie): implement the type-based id object to manage all created operations.
  let operationId = 0;

  /**
   * @class OperationsList
   */
  class OperationsList {
    /**
     * @method constructor
     */
    constructor() {
      this._list = [];
    }
    /**
     * @method get
     * @param {String} name
     */
    get(name) {
      return this._list.reduce((val, op) => {
        if (val === null && name === op.name) {
          return op;
        } else {
          return val;
        }
      }, null);
    }
    /**
     * @method append
     * @param {Operation} op
     */
    append(op) {
      if (!(op instanceof tensorflow.Operation)) {
        throw new TypeError('the second argument must be an instance of tensorflow.Operation');
      }
      this._list.filter((exist) => {
        return exist.name === op.name;
      }).map((_, index) => {
        // TODO(Yorkie): delete the operation from core
        delete this._list[index];
      });
      this._list.push(op);
    }
    /**
     * @method list
     * @param {String} type
     */
    list(type) {
      return this._list.filter((op) => op.type === type);
    }
  }

  /**
   * @module operations
   */
  Graph.prototype.operations = new OperationsList();

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
      // `this` references to Graph instance.
      const op = new tensorflow.Operation(this, type, name);
      initializer.apply(this, [op].concat(args));
      op.finish();

      this.operations.append(op);
      return op;
    };
  };

}
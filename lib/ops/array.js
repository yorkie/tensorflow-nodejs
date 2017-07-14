module.exports = function(tensorflow) {
  'use strict';

  const Graph = tensorflow.Graph;

  /**
   * @class Graph
   * @method placeholder
   * @param {String} name
   */
  Graph.prototype.placeholder = Graph.createOperationBuilder(
    'Placeholder', 
    function(descriptor, type, shape) {
      descriptor.setAttrType('dtype', type || tensorflow.dtype.int32);
      if (Array.isArray(shape)) {
        descriptor.setAttrShape('shape', shape);
      }
    }
  );

  /**
   * @class Graph
   * @method const
   */
  Graph.prototype.const = Graph.createOperationBuilder(
    'Const',
    function(descriptor, val, type, shape) {
      if (!shape && Array.isArray(type)) {
        shape = type;
        type = undefined;
      }
      const value = tensorflow.Tensor.from(val, type, shape);
      descriptor.setAttrType('dtype', value.type);
      descriptor.setAttrTensor('value', value);
    }
  );

  /**
   * @class Graph
   * @method zerosLike
   * @param {String} type
   */
  Graph.prototype.zerosLike = Graph.createOperationBuilder(
    'ZerosLike',
    function(descriptor, x, type) {
      descriptor.addInput(0, x);
      descriptor.setAttrType('T', type || tensorflow.dtype.int32);
    }
  );

  /**
   * @class Graph
   * @method zeros
   * @param {String} type
   */
  Graph.prototype.zeros = Graph.prototype.zerosLike;

  /**
   * @class Graph
   * @method diag
   * @param {String} diagonal
   */
  Graph.prototype.diag = Graph.createOperationBuilder(
    'Diag',
    function(descriptor, diagonal) {
      descriptor.addInput(0, diagonal);
    }
  );

  /**
   * @class Graph
   * @method reverse
   * @param {Tensor} tensor
   * @param {Array} dims
   */
  Graph.prototype.reverse = Graph.createOperationBuilder(
    'Reverse',
    function(descriptor, tensor, dims) {
      descriptor.addInput(0, tensor);
      descriptor.addInput(0, dims);
    }
  );

  /**
   * @class Graph
   * @method shape
   * @param {Tensor} input
   * @param {Number} type
   */
  Graph.prototype.shape = Graph.createOperationBuilder(
    'Shape',
    function(descriptor, input, type) {
      descriptor.addInput(0, input);
      descriptor.setAttrType('type', type || tensorflow.dtype.int32);
    }
  );

};


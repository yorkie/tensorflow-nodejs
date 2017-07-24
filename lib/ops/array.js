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
   * @method constant
   */
  Graph.prototype.constant = Graph.createOperationBuilder(
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
    function(descriptor, x) {
      descriptor.addInput(0, x);
      descriptor.setAttrType('T', tensorflow.protoOfStrictly(x).type);
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
    function(descriptor, input) {
      descriptor.addInput(0, input);
      descriptor.setAttrType('T', tensorflow.protoOfStrictly(input).type);
    }
  );

  /**
   * @class Graph
   * @method rank
   * @param {Operation} input
   * @param {Type} type
   */
  Graph.prototype.rank = Graph.createOperationBuilder(
    'Rank',
    function(descriptor, input) {
      descriptor.addInput(0, input);
      descriptor.setAttrType('T', tensorflow.protoOfStrictly(input).type);
    }
  );

  /**
   * @class Graph
   * @method size
   * @param {Operation} input
   */
  Graph.prototype.size = Graph.createOperationBuilder(
    'Size',
    function(descriptor, input) {
      descriptor.addInput(0, input);
      descriptor.setAttrType('T', tensorflow.protoOfStrictly(input).type);
    }
  );

  /**
   * @class Graph
   * @method identity
   * @param {Operation} input
   */
  Graph.prototype.identity = Graph.createOperationBuilder(
    'Identity',
    function(descriptor, input) {
      descriptor.addInput(0, input);
      descriptor.setAttrType('T', tensorflow.protoOfStrictly(input).type);
    }
  );

  /**
   * @class Graph
   * @method onehot
   */
  Graph.prototype.onehot = Graph.createOperationBuilder(
    'OneHot',
    function(descriptor, indices, depth, on, off) {
      depth = this.constant(depth, tensorflow.dtype.int32, []);
      // // TODO(Yorkie): currently we let the on/off values to be float32..
      on = this.constant(on || 1.0, tensorflow.dtype.float32, []);
      off = this.constant(off || 0.0, tensorflow.dtype.float32, []);
      descriptor.addInput(0, indices);
      descriptor.addInput(0, depth);
      descriptor.addInput(0, on);
      descriptor.addInput(0, off);
      descriptor.setAttrType('T', tensorflow.dtype.float32);
    }
  );

};


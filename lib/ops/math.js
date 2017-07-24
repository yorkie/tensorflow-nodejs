module.exports = function(tensorflow) {
  'use strict';

  const Graph = tensorflow.Graph;

  function unary(name) {
    return Graph.createOperationBuilder(
      name, 
      function(descriptor, val) {
        descriptor.addInput(0, val);
      }
    );
  }

  function binary(name) {
    return Graph.createOperationBuilder(
      name,
      function(descriptor, left, right) {
        descriptor.addInput(0, left);
        descriptor.addInput(0, right);
      }
    );
  }

  /**
   * @class Graph
   * @method cast
   */
  Graph.prototype.cast = Graph.createOperationBuilder(
    'Cast',
    function(descriptor, input, from, to) {
      descriptor.addInput(0, input);
      descriptor.setAttrType('SrcT', from);
      descriptor.setAttrType('DstT', to);
    }
  );

  /**
   * @class Graph
   * @method abs
   */
  Graph.prototype.abs = unary('Abs');

  /**
   * @class Graph
   * @method neg
   */
  Graph.prototype.neg = unary('Neg');

  /**
   * @class Graph
   * @method inv
   */
  Graph.prototype.inv = unary('Inv');

  /**
   * @class Graph
   * @method square
   */
  Graph.prototype.square = unary('Square');

  /**
   * @class Graph
   * @method round
   */
  Graph.prototype.round = unary('Round');

  /**
   * @class Graph
   * @method tan
   */
  Graph.prototype.tan = unary('Tan');

  /**
   * @class Graph
   * @method asin
   */
  Graph.prototype.asin = unary('Asin');

  /**
   * @class Graph
   * @method acos
   */
  Graph.prototype.acos = unary('Acos');

  /**
   * @class Graph
   * @method atan
   */
  Graph.prototype.atan = unary('Atan');

  /**
   * @class Graph
   * @method sign
   */
  Graph.prototype.sign = unary('Sign');

  /**
   * @class Graph
   * @method floor
   */
  Graph.prototype.floor = unary('Floor');

  /**
   * @class Graph
   * @method ceil
   */
  Graph.prototype.ceil = unary('Ceil');

  /**
   * @class Graph
   * @method rint
   */
  Graph.prototype.rint = unary('Rint');

  /**
   * @class Graph
   * @method add
   */
  Graph.prototype.add = binary('Add');

  /**
   * @class Graph
   * @method sub
   */
  Graph.prototype.sub = binary('Sub');

  /**
   * @class Graph
   * @method mul
   */
  Graph.prototype.mul = binary('Mul');

  /**
   * @class Graph
   * @method div
   */
  Graph.prototype.div = binary('Div');

  /**
   * @class Graph
   * @method mod
   */
  Graph.prototype.mod = binary('Mod');

  /**
   * @class Graph
   * @method pow
   */
  Graph.prototype.pow = binary('Pow');

  /**
   * @class Graph
   * @method max
   */
  Graph.prototype.max = binary('Maximum');

  /**
   * @class Graph
   * @method min
   */
  Graph.prototype.min = binary('Minimum');


  /**
   * @class Graph
   * @method addN
   */
  Graph.prototype.addN = Graph.createOperationBuilder(
    'AddN',
    function(descriptor, nums) {
      descriptor.addInputList(nums.map((op) => {
        return { op, index: 0 };
      }));
    }
  );

  /**
   * @class Graph
   * @method matmul
   */
  Graph.prototype.matmul = Graph.createOperationBuilder(
    'MatMul',
    function(descriptor, a, b, options) {
      descriptor.addInput(0, a);
      descriptor.addInput(0, b);
      // TODO(Yorkie): transpose_a/b
    }
  );

  /**
   * @class Graph
   * @method mean
   */
  Graph.prototype.mean = Graph.createOperationBuilder(
    'Mean',
    function(descriptor, input, reductionOn, keepDims) {
      descriptor.addInput(0, input);
      descriptor.addInput(0, reductionOn);
      descriptor.setAttrBool('keep_dims', keepDims || false);
    }
  );

  /**
   * @class Graph
   * @method reduceMean
   */
  Graph.prototype.reduceMean = Graph.createOperationBuilder(
    'Mean',
    function(descriptor, input, reductionOn, keepDims) {
      descriptor.addInput(0, input);
      descriptor.addInput(0, reductionOn);
      descriptor.setAttrBool('keep_dims', keepDims || false);
    }
  );

  /**
   * @class Graph
   * @method lt
   */
  Graph.prototype.lt = binary('Less');

  /**
   * @class Graph
   * @method lte
   */
  Graph.prototype.lte = binary('LessEqual');

  /**
   * @class Graph
   * @method gt
   */
  Graph.prototype.gt = binary('Greater');

  /**
   * @class Graph
   * @method gte
   */
  Graph.prototype.gte = binary('GreaterEqual');

  /**
   * @class Graph
   * @method equal
   */
  Graph.prototype.equal = binary('Equal');

  /**
   * @class Graph
   * @method not
   */
  Graph.prototype.not = unary('LogicalNot');

  /**
   * @class Graph
   * @method and
   */
  Graph.prototype.and = binary('LogicalAnd');

  /**
   * @class Graph
   * @method or
   */
  Graph.prototype.or = binary('LogicalOr');


};

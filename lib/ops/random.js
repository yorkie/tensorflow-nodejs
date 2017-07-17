module.exports = function(tensorflow) {
  'use strict';

  const Graph = tensorflow.Graph;

  /**
   * @class Graph
   * @module random
   */
  return {

    /**
     * @method randomUniform
     */
    randomUniform: Graph.createOperationBuilder(
      'RandomUniform',
      function(descriptor, shape, seeds) {
        const _shape = this.const(shape, tensorflow.dtype.int32, [shape.length]);
        descriptor.addInput(0, _shape);
        descriptor.setAttrType('dtype', tensorflow.dtype.float32);
        if (Array.isArray(seeds)) {
          descriptor.setAttrInt('seed', seeds[0]);
          descriptor.setAttrInt('seed2', seeds[1]);
        }
      }
    ),

    /**
     * @method randomUniformInt
     */
    randomUniformInt: Graph.createOperationBuilder(
      'RandomUniformInt',
      function(descriptor, shape, range, seeds) {
        if (!Array.isArray(range) || range.length < 2) {
          throw new TypeError('range should be an array with 2 elements');
        }
        const _shape = this.const(shape, tensorflow.dtype.int32, [shape.length]);
        descriptor.addInput(0, _shape);
        descriptor.addInput(0, range[0]);
        descriptor.addInput(0, range[1]);
        descriptor.setAttrType('dtype', tensorflow.dtype.float32);
        if (Array.isArray(seeds)) {
          descriptor.setAttrInt('seed', seeds[0]);
          descriptor.setAttrInt('seed2', seeds[1]);
        }
      }
    ),

    /**
     * @method randomGamma
     */
    randomGamma: Graph.createOperationBuilder(
      'RandomGamma',
      function(descriptor, shape, alpha, seeds) {
        const _shape = this.const(shape, tensorflow.dtype.int32, [shape.length]);
        descriptor.addInput(0, _shape);
        descriptor.addInput(0, alpha);
        descriptor.setAttrType('dtype', tensorflow.dtype.float32);
        if (Array.isArray(seeds)) {
          descriptor.setAttrInt('seed', seeds[0]);
          descriptor.setAttrInt('seed2', seeds[1]);
        }
      }
    ),

    /**
     * @method randomPoisson
     */
    randomPoisson: Graph.createOperationBuilder(
      'RandomPoisson',
      function(descriptor, shape, rate, seeds) {
        const _shape = this.const(shape, tensorflow.dtype.int32, [shape.length]);
        descriptor.addInput(0, _shape);
        descriptor.addInput(0, rate);
        descriptor.setAttrType('dtype', tensorflow.dtype.float32);
        if (Array.isArray(seeds)) {
          descriptor.setAttrInt('seed', seeds[0]);
          descriptor.setAttrInt('seed2', seeds[1]);
        }
      }
    ),

  };
};


module.exports = function(tensorflow) {
  'use strict';

  const Graph = tensorflow.Graph;

  return {

    /**
     * Converts each string in the input Tensor to its hash mod 
     * by a number of buckets.
     * @class Graph.string
     * @method toHashBucketFast
     */
    toHashBucketFast: Graph.createOperationBuilder(
      'StringToHashBucketFast',
      function(descriptor, input, numOfBucket) {
        descriptor.addInput(0, input);
        if (typeof numOfBucket === 'number') {
          if (numOfBucket < 1)
            throw new TypeError('`numOfBucket` should be a number >= 1');
          descriptor.setAttrInt('num_buckets', numOfBucket);
        }
      }
    ),

    /**
     * Converts each string in the input Tensor to its hash mod 
     * by a number of buckets.
     * @class Graph.string
     * @method toHashBucket
     */
    toHashBucket: Graph.createOperationBuilder(
      'StringToHashBucket',
      function(descriptor, input, numOfBucket) {
        descriptor.addInput(0, input);
        if (typeof numOfBucket === 'number') {
          if (numOfBucket < 1)
            throw new TypeError('`numOfBucket` should be a number >= 1');
          descriptor.setAttrInt('num_buckets', numOfBucket);
        }
      }
    ),

    /**
     * Joins a string Tensor across the given dimensions.
     * @class Graph.string
     * @method reduceJoin
     * @param {String} input
     * @param {Number} indices
     * @param {Object} options
     * @param {Boolean} options.keepDims
     * @param {String} options.separator
     */
    reduceJoin: Graph.createOperationBuilder(
      'ReduceJoin',
      function(descriptor, input, indices, options) {
        descriptor.addInput(0, input);
        descriptor.addInput(0, indices);
        options = options || {};
        if (options.keepDims)
          descriptor.setAttrBool('keep_dims', options.keepDims);
        if (options.separator)
          descriptor.setAttrString('separator', options.separator);
      }
    ),

    /**
     * Converts each entry in the given tensor to strings.
     * Supports many numeric types and boolean.
     * @class Graph.string
     * @method asString
     * @param {Any} input
     * @param {Object} options
     * @param {Number} options.precision
     * @param {Boolean} options.scientific
     * @param {Boolean} options.shortest
     * @param {Number} options.width
     * @param {String} options.fill
     */
    asString: Graph.createOperationBuilder(
      'AsString',
      function(descriptor, input, options) {
        descriptor.addInput(0, input);
        options = options || {};
        if (options.precision)
          descriptor.setAttrInt('precision', options.precision);
        if (options.scientific)
          descriptor.setAttrBool('scientific', options.scientific);
        if (options.shortest)
          descriptor.setAttrBool('shortest', options.shortest);
        if (options.width)
          descriptor.setAttrInt('width', options.width);
        if (options.fill)
          descriptor.setAttrString('fill', options.fill);
      }
    ),

    /**
     * @class Graph.string
     * @method join
     */
    join: Graph.createOperationBuilder(
      'StringJoin',
      function(descriptor, inputs, sep) {
        descriptor.addInputList(inputs.map((op) => {
          return { op, index: 0 };
        }));
        descriptor.setAttrInt('N', inputs.length);
        descriptor.setAttrString('separator', sep);
      }
    ),

    /**
     * @class Graph.string
     * @method substr
     */
    substr: Graph.createOperationBuilder(
      'Substr', 
      function(descriptor, input, pos, len) {
        descriptor.addInput(0, input);
        descriptor.addInput(0, pos);
        descriptor.addInput(0, len);
      }
    ),

    /**
     * @class Graph.string
     * @method split
     */
    split: Graph.createOperationBuilder(
      'StringSplit',
      function(descriptor, input, delimiter) {
        descriptor.addInput(0, input);
        descriptor.addInput(0, delimiter);
      }
    ),

  };

};


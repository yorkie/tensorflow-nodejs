module.exports = function(tensorflow) {
  'use strict';

  const Graph = tensorflow.Graph;

  /**
   * @class Graph
   * @module base64
   */
  return {

    /**
     * @class Graph.base64
     * @method encode
     */
    encode: Graph.createOperationBuilder(
      'EncodeBase64',
      function(descriptor, input, pad) {
        descriptor.addInput(0, input);
        if (typeof pad === 'bool') {
          descriptor.setAttrBool('pad', pad);
        }
      }
    ),

    /**
     * @class Graph.base64
     * @method decode
     */
    decode: Graph.createOperationBuilder(
      'DecodeBase64',
      function(descriptor, input) {
        descriptor.addInput(0, input);
      }
    ),
  };
};


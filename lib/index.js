'use strict';

const boa = require('@pipcook/boa');
const pytf = boa.import('tensorflow');

const tf = module.exports = {

  /**
   * returns the tensorflow versions
   */
  get version() {
    const pyver = pytf.version;
    return {
      COMPILER_VERSION: pyver.COMPILER_VERSION,
      GIT_VERSION: pyver.GIT_VERSION,
      GRAPH_DEF_VERSION: pyver.GRAPH_DEF_VERSION,
      GRAPH_DEF_VERSION_MIN_CONSUMER: pyver.GRAPH_DEF_VERSION_MIN_CONSUMER,
      GRAPH_DEF_VERSION_MIN_PRODUCER: pyver.GRAPH_DEF_VERSION_MIN_PRODUCER,
      VERSION: pyver.VERSION
    };
  },

  /**
   * Creates a constant tensor from a tensor-like object.
   * @param {*} val 
   * @param {*} dtype 
   * @param {*} shape 
   * @param {*} name 
   */
  constant(val, dtype, shape, name = 'Const') {
    return pytf.constant(val, dtype, shape, name);
  },

  /**
   * Implementation of the Keras API meant to be a high-level API for TensorFlow.
   */
  keras: require('./keras'),
};

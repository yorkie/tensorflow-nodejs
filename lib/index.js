'use strict';

const protobuf = require('protobufjs');
const tensorflow = module.exports = require('bindings')('tensorflow.node');
const Graph = tensorflow.Graph;
const Tensor = tensorflow.Tensor;
const Session = tensorflow.Session;

// proto
const ConfigProto = protobuf.loadSync('protobuf/config.proto').lookupType('ConfigProto');

// load modules
require('./tensor')(tensorflow);
require('./session')(tensorflow);
require('./graph')(tensorflow);
require('./train')(tensorflow);

// utils for pass a class constructor and arguments, and return the
// instance of the given class.
function passthroughModule(initializer, args) {
  return new (Function.prototype.bind.apply(initializer, [null].concat(args)));
}

/**
 * @method protoOfStrictly
 */
tensorflow.protoOfStrictly = function(operation) {
  if (operation.outputs.length > 1) {
    // TODO(Yorkie): doesnt support a variable to an operation which owns more than 1 outputs
    throw new TypeError(`No support for an operation which owns > 1 outputs`);
  }
  return operation.outputs[0];
};

/**
 * @method createSession
 */
tensorflow.createSession = function(graph, options) {
  options = options || {};
  options.target = options.target || 'local';
  options.config = Object.assign({
    deviceCount: {
      GPU: 0,
      CPU: 8,
    },
    intraOpParallelismThreads: 10,
    interOpParallelismThreads: 10,
    usePerSessionThreads: true,
    placementPeriod: 30
  }, options.config);

  // start verifying the config
  const verificationErr = ConfigProto.verify(options.config);
  if (verificationErr) {
    throw new Error(verificationErr);
  }
  const texture = ConfigProto.create(options.config);
  const config = ConfigProto.encode(texture).finish();

  function getSafeArrayBuffer(arr) {
    const offset = arr.byteOffset;
    const length = arr.length;
    return arr.buffer.slice(offset, offset + length);
  }
  return passthroughModule(tensorflow.Session, [
    graph,
    options.target,
    getSafeArrayBuffer(config)
  ]);
};

/**
 * @method createGraph
 */
tensorflow.createGraph = function() {
  let graph = passthroughModule(tensorflow.Graph, arguments);
  graph.initialize();
  return graph;
};

/**
 * @method createTensor
 */
tensorflow.createTensor = function() {
  return passthroughModule(tensorflow.Tensor, arguments);
};

// default graph and session.
const defaultGraph = tensorflow.createGraph();
const defaultSession = tensorflow.createSession(defaultGraph);

/**
 * @method graph
 */
tensorflow.graph = function() {
  return defaultGraph;
};

/**
 * @method session
 */
tensorflow.session = function() {
  return defaultSession;
};

/**
 * @method tensor
 */
tensorflow.tensor = function() {
  return tensorflow.Tensor.from.apply(tensorflow.Tensor, arguments);
}

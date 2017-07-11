'use strict';

const fs = require('fs');
const path = require('path');
const protobuf = require('protobufjs');
const tensorflow = module.exports = require('bindings')('tensorflow.node');
const Graph = tensorflow.Graph;
const Tensor = tensorflow.Tensor;
const Session = tensorflow.Session;

/**
 * protobuf types
 */
const ConfigProto = protobuf.loadSync('protobuf/config.proto').lookupType('ConfigProto');
// to be more...

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

// load the defined ops
function loadAndInitOps(root) {
  fs.readdirSync(path.join(__dirname, root)).forEach((filename) => {
    const name = filename.replace(/\.js$/, '');
    const obj = require(path.join(__dirname, root, name))(tensorflow);
    if (obj) {
      tensorflow.Graph.prototype[name] = obj;
    }
  });
}

// load the modules under ./ops
loadAndInitOps('./ops');

/**
 * @class Tensor
 * @method getViewData
 */
Tensor.prototype.getViewData = function getViewData() {
  let viewData;

  function read(val, byte, method) {
    let outs = [];
    const len = val.length / byte;
    for (let i = 0; i < len; i++) {
      if (typeof method === 'function') {
        outs.push(method.call(val, i * byte));
      } else if (typeof val[method] === 'function') {
        outs.push(val[method](i * byte));
      } else {
        throw new TypeError(`The method "${method}" should be a function`);
      }
    }
    return outs;
  }

  switch (this.type) {
  case tensorflow.dtype.bool:
    viewData = read(this.data, 1, (offset) => {
      return !!this.readInt8(offset);
    });
    break;
  case tensorflow.dtype.int8:
    viewData = read(this.data, 1, 'readInt8');
    break;
  case tensorflow.dtype.int16:
    viewData = read(this.data, 2, 'readInt16LE');
    break;
  case tensorflow.dtype.int32:
    viewData = read(this.data, 4, 'readInt32LE');
    break;
  case tensorflow.dtype.int64:
    viewData = read(this.data, 8, (offset) => {
      return this.readInt(offset, 8);
    });
    break;
  case tensorflow.dtype.uint8:
    viewData = read(this.data, 1, 'readUInt8');
    break;
  case tensorflow.dtype.uint16:
    viewData = read(this.data, 2, 'readUInt16LE');
    break;
  case tensorflow.dtype.float16:
  case tensorflow.dtype.float32:
    viewData = read(this.data, 4, 'readFloatLE');
    break;
  case tensorflow.dtype.float64:
    viewData = read(this.data, 8, 'readDoubleLE');
    break;
  case tensorflow.dtype.string:
    viewData = this.data.toString();
    break;
  }
  return viewData;
};


/**
 * @class Session
 * @method run
 * @param {Graph} fetches
 * @param {Graph} feeds
 * @param {Object} options
 */
Session.prototype.run = function sessionRun(fetches, feeds, options) {
  return this._run(fetches, feeds, options).reduce((result, tensor) => {
    return result.concat(tensor.getViewData());
  }, []);
};

// utils for pass a class constructor and arguments, and return the
// instance of the given class.
function passthroughModule(initializer, args) {
  return new (Function.prototype.bind.apply(initializer, [null].concat(args)));
}

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
  return passthroughModule(tensorflow.Graph, arguments);
};

/**
 * @method createTensor
 */
tensorflow.createTensor = function() {
  return passthroughModule(tensorflow.Tensor, arguments);
};

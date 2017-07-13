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

  function createTensorData(val, byte, method, size) {
    if (!Array.isArray(val))
      val = [val];

    // if size is less than the val's length, throw an error to report user
    if (size < val.length) {
      throw new Error('The elements number is too short');
    }

    // fill rest with the last val item for.
    if (size > val.length) {
      const base = val.length - 1;
      for (let i = base + 1; i < size; i++) {
        val[i] = val[base];
      }
    }

    const data = Buffer.alloc(val.length * byte);
    val.forEach((v, i) => {
      if (typeof method === 'function') {
        method.call(data, v, i * byte);
      } else if (typeof data[method] === 'function') {
        data[method](v, i * byte);
      } else {
        throw new TypeError(`The method "${method}" is not a function`);
      }
    });
    return data;
  }

  function int8(val, size) {
    return createTensorData(val, 1, 'writeInt8', size);
  }

  function int16(val, size) {
    return createTensorData(val, 2, 'writeInt16LE', size);
  }

  function int32(val, size) {
    return createTensorData(val, 4, 'writeInt32LE', size);
  }

  function int64(val, size) {
    return createTensorData(val, 8, (v, byte) => {
      this.writeIntLE(v, byte, 8);
    }, size);
  }

  function uint8(val, size) {
    return createTensorData(val, 1, 'writeUInt8', size);
  }

  function uint16(val, size) {
    return createTensorData(val, 2, 'writeUInt16LE', size);
  }

  function float(val, size) {
    return createTensorData(val, 4, 'writeFloatLE', size);
  }

  function double(val, size) {
    return createTensorData(val, 8, 'writeDoubleLE', size);
  }

  function string(val) {
    return Buffer.from(val);
  }

  function typeOf(val) {
    const typename = typeof val;
    if (typename === 'string') {
      return tensorflow.dtype.string;
    } else if (typename === 'bool') {
      return tensorflow.dtype.boolean;
    }

    if (typename === 'object' && Array.isArray(val)) {
      const type = val.map(fetchTypename).reduce(checkAndReturn);
      if (type) {
        return type;
      } else {
        throw new TypeError('The array element should be in same type, but: ' + val);
      }
    }
    return tensorflow.dtype.int32;
  }

  function fetchTypename(v) {
    return typeOf(v);
  }

  function checkAndReturn(type, v) {
    if (type === null) {
      return v;
    } else if (type !== false && type === v) {
      return type;
    } else {
      return false;
    }
  }

  /**
   * @class Graph
   * @method const
   */
  Graph.prototype.const = Graph.createOperationBuilder(
    'Const',
    function(descriptor, val, type, shape) {
      let data;

      if (!shape && Array.isArray(type)) {
        shape = type;
        type = undefined;
      }

      if (!shape) {
        throw new TypeError('shape is required');
      }

      // FIXME(Yorkie): should more strict check for type
      if (!type) {
        type = typeOf(val);
      }

      switch (type) {
      case tensorflow.dtype.bool:
      case tensorflow.dtype.int8:
        data = int8(val, shape[0]);
        break;
      case tensorflow.dtype.int32:
        data = int32(val, shape[0]);
        break;
      case tensorflow.dtype.int64:
        data = int64(val, shape[0]);
        break;
      case tensorflow.dtype.uint8:
        data = uint8(val, shape[0]);
        break;
      case tensorflow.dtype.uint16:
        data = uint16(val, shape[0]);
        break;
      // FIXME(Yorkie): currently we don't support float16
      case tensorflow.dtype.float16:
      case tensorflow.dtype.float32:
        data = float(val, shape[0]);
        break;
      case tensorflow.dtype.float64:
        data = double(val, shape[0]);
        break;
      case tensorflow.dtype.string:
        data = string(val);
        shape = [0, 1];
        break;
      }
      const value = new tensorflow.Tensor(type, shape, { data });
      descriptor.setAttrType('dtype', type);
      descriptor.setAttrTensor('value', value);
      descriptor._tensor = value;
    }
  );

  /**
   * @class Graph
   * @method zerosLike
   * @param {String} type
   */
  Graph.prototype.zerosLike = Graph.createOperationBuilder(
    'ZerosLike',
    function(descriptor, type, x) {
      descriptor.setAttrType(type || tensorflow.dtype.int32);
      descriptor.addInput(0, x);
    }
  );

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
   * @method fill
   * @param {Array} dims
   * @param {Array} value
   */
  Graph.prototype.fill = Graph.createOperationBuilder(
    'Fill',
    function(descriptor, dims, value) {
      descriptor.addInput(0, dims);
      descriptor.addInput(0, value);
      // TODO(Yorkie): currently only supports int32
      descriptor.setAttrType('type', tensorflow.dtype.int32);
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


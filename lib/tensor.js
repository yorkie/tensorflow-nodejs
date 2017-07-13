'use strict';

module.exports = function(tensorflow) {

  let Tensor = tensorflow.Tensor;

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

  Tensor.from = function(val, type, shape) {
    let data;

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
    return new tensorflow.Tensor(type, shape, { data });
  };

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

};
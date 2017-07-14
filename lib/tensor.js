'use strict';

module.exports = function(tensorflow) {

  let Tensor = tensorflow.Tensor;

  function length(shape, size) {
    let len = 1;
    for (let i = 0; i < shape.length; i++) {
      len = len * shape[i];
    }
    return len * size;
  }

  function tensor(val, type, shape) {
    const size = tensorflow.dtype.sizeOf(type);
    let data = Buffer.alloc(length(shape, size), 0);
    let offset = 0;

    function _flat(_val, _shape) {
      if (_val === undefined || _val === null)
        return;
      // when val is not an array and shape = [1], we should convert val
      // to array.
      if (_shape[0] === 1 && _shape.length === 1 && !Array.isArray(_val)) {
        _val = [ _val ];
      }
      if (_shape.length === 0) {
        if (type === tensorflow.dtype.bool || type === tensorflow.dtype.int8) {
          data.writeInt8(_val, offset);
        } else if (type === tensorflow.dtype.int32) {
          data.writeInt32LE(_val, offset);
        } else if (type === tensorflow.dtype.int64) {
          data.writeIntLE(_val, offset, size);
        } else if (type === tensorflow.dtype.uint8) {
          data.writeUInt8(_val, offset);
        } else if (type === tensorflow.dtype.uint16) {
          data.writeUInt16LE(_val, offset);
        } else if (type === tensorflow.dtype.float16 ||
          type === tensorflow.dtype.float32) {
          data.writeFloatLE(_val, offset);
        } else if (type === tensorflow.dtype.float64) {
          data.writeDoubleLE(_val, offset);
        } else if (type === tensorflow.dtype.string) {
          // TODO
        }
        offset += size;
      } else {
        for (let i = 0; i < _shape[0]; i++) {
          _flat(_val[i], _shape.slice(1), size);
        }
      }
    }
    _flat(val, shape);
    return new tensorflow.Tensor(type, shape, { data })
  }

  function typeOf(val) {
    const typename = typeof val;
    if (typename === 'string') {
      return tensorflow.dtype.string;
    } else if (typename === 'bool') {
      return tensorflow.dtype.boolean;
    }

    if (typename === 'object' && Array.isArray(val)) {
      const type = val.map(typeOf).reduce(checkAndReturn);
      if (type) {
        return type;
      } else {
        throw new TypeError('The array element should be in same type, but: ' + val);
      }
    }
    return tensorflow.dtype.int32;
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
    if (!shape && !Array.isArray(val)) {
      shape = [1];
    }
    if (!shape) {
      throw new TypeError('shape is required');
    }

    return tensor(val, type, shape);
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

    if (this.shape.length === 1 && this.shape[0] === 1) {
      viewData = viewData[0];
    }
    return viewData;
  };

};
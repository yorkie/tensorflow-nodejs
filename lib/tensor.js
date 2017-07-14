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
    const type = this.type;
    const size = tensorflow.dtype.sizeOf(type);
    let offset = 0;

    function _defalt(_data, _shape) {
      let res = [];
      if (!_shape.length) {
        if (type === tensorflow.dtype.bool) {
          // converts to boolean via `!!`.
          res = !!_data.readInt8(offset);
        } else if (type === tensorflow.dtype.int8) {
          res = _data.readInt8(offset);
        } else if (type === tensorflow.dtype.int16) {
          res = _data.readInt16LE(offset);
        } else if (type === tensorflow.dtype.int32) {
          res = _data.readInt32LE(offset);
        } else if (type === tensorflow.dtype.int64) {
          res = _data.readInt(offset, size);
        } else if (type === tensorflow.dtype.uint8) {
          res = _data.readUInt8(offset);
        } else if (type === tensorflow.dtype.uint16) {
          res = _data.readUInt16LE(offset);
        } else if (type === tensorflow.dtype.float16 ||
          type === tensorflow.dtype.float32) {
          res = _data.readFloatLE(offset);
        } else if (type === tensorflow.dtype.float64) {
          res = _data.readDoubleLE(offset);
        } else if (type === tensorflow.dtype.string) {
          // TODO(Yorkie): support string decoding.
        }
        offset += size;
      } else {
        for (let i = 0; i < _shape[0]; i++) {
          res[i] = _defalt(_data, _shape.slice(1));
        }
      }
      return res;
    }
    return _defalt(this.data, this.shape);
  };

};
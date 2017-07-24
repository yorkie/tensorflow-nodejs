'use strict';

module.exports = function(tensorflow) {
  const BN = require('bn.js').BN;

  let Tensor = tensorflow.Tensor;

  function length(shape, size) {
    let len = 1;
    for (let i = 0; i < shape.length; i++) {
      len = len * shape[i];
    }
    return len * size;
  }

  function tensor(val, type, shape) {
    let data;
    if (Buffer.isBuffer(val)) {
      if (!type || !shape)
        throw new Error('`type` and `shape` are required when val is a buffer.');

      // This builds an ArrayBuffer, and pass the new Buffer from the ArrayBuffer.
      // If we directly make the following assignment:
      //   data = val;
      //
      // The data.buffer is the full buffer from the Node.js's buffer pool, it causes
      // the data to be read is wrong. Now we crop the fully correct buffer would fix
      // the problem.
      const ab = val.buffer.slice(val.offset, val.offset + val.byteLength);
      data = Buffer.from(ab);
    } else if (type !== tensorflow.dtype.string) {
      const size = tensorflow.dtype.sizeOf(type);
      data = Buffer.alloc(length(shape, size), 0);

      let offset = 0;
      _flat(val, shape, (_val, _shape) => {
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
          throw new TypeError('string type should not be reached with shape');
        }
        offset += size;
      });
    } else {
      let header = [];
      let contents = [];
      let read = 0;
      _flat(val, shape, (_val, _shape) => {
        const encoded = Tensor._stringEncode(_val);
        const offset = new BN(read).toBuffer('le', 8);
        header.push(offset);
        contents.push(encoded);
        read += encoded.length;
      });

      const headerSize = contents.length * 8;
      const contentsSize = contents.reduce((size, data) => {
        return size + data.byteLength;
      }, 0);
      data = Buffer.alloc(headerSize + contentsSize, 0);
      header.concat(contents).reduce((offset, buf) => {
        buf.copy(data, offset);
        return offset + buf.byteLength;
      }, 0);
    }

    function _flat(_val, _shape, handler) {
      if (_val === undefined || _val === null)
        return;
      // when val is not an array and shape = [1], we should convert val
      // to array.
      if (_shape[0] === 1 && _shape.length === 1 && !Array.isArray(_val)) {
        _val = [ _val ];
      }
      if (_shape.length === 0) {
        handler(_val, _shape);
      } else {
        for (let i = 0; i < _shape[0]; i++) {
          _flat(_val[i], _shape.slice(1), handler);
        }
      }
    }

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
    if (type !== tensorflow.dtype.string) {
      const size = tensorflow.dtype.sizeOf(type);
      let offset = 0;
      return _defalt(this.data, this.shape, (_data, _shape) => {
        let res;
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
          throw new TypeError('string type should not be reached with shape');
        }
        offset += size;
        return res;
      });
    } else {
      const numOfElements = length(this.shape, 1);
      const header = [];
      const contents = this.data.buffer.slice(numOfElements * 8);
      for (let i = 0; i < numOfElements; i++) {
        const offset = new BN(this.data.slice(i * 8, (i + 1) * 8), 'le');
        header.push(offset.toNumber());
      }

      let cursor = 0;
      // FIXME(Yorkie): contents is an ArrayBuffer instance, not the Buffer.
      // Because the low-level bridge API accepts the ArrayBuffer, this problem
      // is caused by its inconsistance between the Buffer and ArrayBuffer when
      // slice.
      // So the answer is passing the ArrayBuffer directly here.
      return _defalt(contents, this.shape, (_data, _shape) => {
        const start = header[cursor];
        const end = header[cursor + 1] ? (header[cursor + 1]) : undefined;
        if (typeof start !== 'number')
          throw new TypeError('number must not be here');

        cursor += 1;
        return Tensor._stringDecode(_data.slice(start, end));
      });
    }

    function _defalt(_data, _shape, handler) {
      let res = [];
      if (!_shape.length) {
        res = handler(_data, _shape, handler);
      } else {
        for (let i = 0; i < _shape[0]; i++) {
          res[i] = _defalt(_data, _shape.slice(1), handler);
        }
      }
      return res;
    }
  };

};
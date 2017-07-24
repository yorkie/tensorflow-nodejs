'use strict';

const assert = require('assert');
const tf = require('../');

const tensor = tf.tensor(Buffer.from([1, 2]), tf.dtype.int8, [2]);
assert.deepEqual(tensor.getViewData(), [1, 2]);


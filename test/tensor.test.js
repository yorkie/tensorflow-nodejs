'use strict';

const assert = require('assert');
const tf = require('../');

test('tensor', () => {
  const tensor = tf.tensor(Buffer.from([1, 2]), tf.dtype.int8, [2]);
  assert.deepEqual(tensor.getViewData(), [1, 2]);

  const text = 'foobar and yorkie are good friends';
  const tensorTxt = tf.tensor(text);
  assert.equal(tensorTxt.getViewData(), text);
});


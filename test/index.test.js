'use strict';

const assert = require('assert');
const tf = require('../');
test('index', () => {
  assert.equal(tf.version, '1.2.1');
});

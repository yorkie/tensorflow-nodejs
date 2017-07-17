'use strict';

const assert = require('assert');
const tf = require('../');

const text = 'foobar and yorkie are good friends';
const tensor = tf.tensor(text);
assert.equal(tensor.getViewData(), text);
'use strict';

const assert = require('assert');
const tf = require('../');
const graph = tf.graph();
const session = tf.session();

function assertRandomUniform(shape) {
  const test = graph.random.randomUniform(shape);
  assert.equal(session.run(test).length, shape[0]);
}

assertRandomUniform([2]);
assertRandomUniform([2, 3]);
assertRandomUniform([3, 1, 2]);

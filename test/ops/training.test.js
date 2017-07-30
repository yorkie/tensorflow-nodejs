'use strict';

const assert = require('assert');
const tf = require('../../');
const graph = tf.graph();
const session = tf.session();

test('op.training', () => {
  const loss = graph.constant([100, 20], 3, [2]);
  // TODO: fix it
  // new tf.train.GradientDescentOptimizer(0.5).minimize(loss);    
});
'use strict';

const assert = require('assert');
const tf = require('../');
const graph = tf.graph();
const session = tf.session();

const loss = graph.constant([100, 20], 3, [2]);
new tf.train.GradientDescentOptimizer(0.5).minimize(loss)
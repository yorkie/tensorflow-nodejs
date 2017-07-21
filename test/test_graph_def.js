'use strict';

const assert = require('assert');
const tf = require('../');
const graph = tf.graph();
const session = tf.session();

graph.load('test/data/graph_def.pbtxt');

const op = graph.operations.get('my_variable/Assign');
const res = session.run(op);
assert.equal(res, 1000);

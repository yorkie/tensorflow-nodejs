'use strict';

const assert = require('assert');
const tf = require('../');
const graph = tf.graph();
const session = tf.session();

graph.load('test/fixtures/graph_def.pbtxt');

const op = graph.operations.get('my_variable/Assign');
const res = session.run(op);

test('graph', () => {
  assert.equal(res, 1000);      
});

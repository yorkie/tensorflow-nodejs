'use strict';

const assert = require('assert');
const tf = require('../');
const graph = tf.createGraph();
const session = tf.createSession(graph);

const test_const = graph.const([[1, 2], [3, 4]], tf.dtype.float32, [2, 2]);
assert.deepEqual(session.run(test_const), [[1, 2], [3, 4]]);

const test_shape = graph.shape(test_const);
assert.deepEqual(session.run(test_shape), [2, 2]);

const test_rank = graph.rank(test_const);
assert.equal(session.run(test_rank), 2);

const test_size = graph.size(test_const);
assert.equal(session.run(test_size), 4);
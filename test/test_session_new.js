'use strict';

const assert = require('assert');
const tf = require('../');

const graph = tf.graph();
const session = tf.createSession(graph, {
  model: {
    root: 'test/saved_models/1',
    tags: ['serve']
  }
});
assert.equal(session instanceof tf.Session, true);

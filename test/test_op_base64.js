'use strict';

const assert = require('assert');
const tf = require('../');
const graph = tf.graph();
const session = tf.session();

function assertEncodeBase64(actual) {
  const text = graph.constant(actual);
  const result = session.run(graph.base64.encode(text));
  assert.equal(new Buffer(result[0], 'base64').toString(), actual);
}

function assertDecodeBase64(actual) {
  const encoded = new Buffer(actual).toString('base64');
  const text = graph.constant(encoded);
  const result = session.run(graph.base64.decode(text));
  assert.equal(result[0], actual);
}

assertEncodeBase64('foobar');
assertEncodeBase64('yorkie are working hard for tensorflow');
assertDecodeBase64('beep boop');
assertDecodeBase64('TensorFlow Node.js provides idiomatic JavaScript language bindings');

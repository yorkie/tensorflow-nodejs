'use strict';

const tf = require('./');
const graph = tf.createGraph();

const x = graph.placeholder(tf.dtype.int32, [2]);
// const W = graph.variable(graph.zeros([784, 10]));
// const b = graph.variable(graph.zeros([10]));
// const y = graph.nn.softmax(graph.matmul(x, W) + b);

const session = tf.createSession(graph);
const feeds = graph.const([10, 20], [2]);
const res = session.run(x, [feeds, feeds._tensor]);
// const res = session.run(feeds);

console.log('result is:', res);

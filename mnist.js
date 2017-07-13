'use strict';

const tf = require('./');
const graph = tf.createGraph();

const x = graph.placeholder(tf.dtype.int32);
const y = graph.const(30, [1]);
const z = graph.addN([x, y]);

// const W = graph.variable(graph.zeros([784, 10]));
// const b = graph.variable(graph.zeros([10]));
// const y = graph.nn.softmax(graph.matmul(x, W) + b);

const session = tf.createSession(graph);
const feeds = tf.Tensor.from(1000, false, [1]);
const res = session.run(z, [x, feeds]);

console.log('result is:', res);

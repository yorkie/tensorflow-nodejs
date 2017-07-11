'use strict';

const tf = require('./');
const graph = tf.createGraph();

const x = graph.placeholder(tf.dtype.float32, [null, 784]);
const W = graph.variable(graph.zeros([784, 10]));
const b = graph.variable(graph.zeros([10]));
const y = graph.nn.softmax(graph.matmul(x, W) + b);

// const session = tf.createSession(graph);
// const result = session.run(neg);
// console.log(result);

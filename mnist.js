'use strict';

const tf = require('./');
const graph = tf.createGraph();

// const x = graph.placeholder();
// const y = graph.const(30);
// const z = graph.add(x, y);

const x = graph.const([5, 103], 3, [100]);
const z = graph.zeros(x);

// const W = graph.variable(graph.zeros([784, 10]));
// const b = graph.variable(graph.zeros([10]));
// const y = graph.nn.softmax(graph.matmul(x, W) + b);

const session = tf.createSession(graph);
const res = session.run(z);
console.log('result is:', res);

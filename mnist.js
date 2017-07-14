'use strict';

const tf = require('./');
const graph = tf.createGraph();


const x = graph.const([[1, 2], [3, 4]], tf.dtype.float32, [2, 2]);
const w = graph.variable(x);
const y = graph.nn.softmax(graph.matmul(w, w));

const session = tf.createSession(graph);
const res = session.run(y);
console.log('result is:', res);

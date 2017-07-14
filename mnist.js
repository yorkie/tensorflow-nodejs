'use strict';

const tf = require('./');
const graph = tf.createGraph();

// const x = graph.placeholder();
// const y = graph.const(30);
// const z = graph.add(x, y);

const x = graph.const([[5, 6], [10, 5]], 3, [2, 2]);
const w = graph.variable(x);
const y = graph.matmul(w, w);

const session = tf.createSession(graph);
const res = session.run(y);
console.log('result is:', res);

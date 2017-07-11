'use strict';

const tf = require('./');
const graph = tf.createGraph();

// // const output = graph.placeholder();
// const beep = graph.const(200);
// const boop = graph.const(1000);
// const sum = graph.addN([boop, beep, boop]);
// const neg = graph.neg(sum);
// // const less = graph.approximateEqual(boop, sum);
// console.log(graph.nn);

const str = graph.const([10000, 30], [382]);

const session = tf.createSession(graph);
const result = session.run(str);
console.log(result);

# <img alt="TensorFlow" src="https://www.tensorflow.org/images/tf_logo_transp.png" width="170"/> Node.js Binding

TensorFlow Node.js provides idiomatic [Node.js](https://nodejs.org) language
bindings for [TensorFlow](http://tensorflow.org).

**Notice:** This project is still under active development and not guaranteed to have a
stable API. This is especially true because the underlying TensorFlow C API has not yet
been stabilized as well.

## Usage

```js
'use strict';

const tf = require('tensorflow2');
const graph = tf.createGraph();

const beep = graph.const(200);
const boop = graph.const(1000);
const sum = graph.addN([boop, beep, boop]);

const session = tf.createSession(graph);
const result = session.run(sum);
console.log(result);
```

## Installation

```sh
$ npm install tensorflow2 --save
```

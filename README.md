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

const x = graph.placeholder();
const y = graph.const(1000);
const sum = graph.add(x, y);

const session = tf.createSession(graph);
const result = session.run(sum, tf.Tensor.from(1048));
console.log(result); // 2048
```

## Installation

```sh
$ npm install tensorflow2 --save
```

## License

MIT

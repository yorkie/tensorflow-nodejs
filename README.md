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

const x = graph.const([[1, 2], [3, 4]], tf.dtype.float32, [2, 2]);
const w = graph.variable(x);
const y = graph.nn.softmax(graph.matmul(w, w));

const session = tf.createSession(graph);
const res = session.run(y);
```

## Operations

There are the following operations that we supported in this library.

### State

The state is managed by users for saving, restoring machine states.

- [x] `variable` Holds state in the form of a tensor that persists across steps. Outputs a ref to the tensor state so it may be read or modified.

- [x] `assign` Update 'ref' by assigning 'value' to it. This operation outputs "ref" after the assignment is done. This makes it easier to chain operations that need to use the reset value.

### Array

- [x] `placeholder` A placeholder op for a value that will be fed into the computation.
- [x] `const` Returns a constant tensor.
- [x] `reverse` Reverses specific dimensions of a tensor.

### Neural networks

In this module, it implements the following alogrithms for represents neural networks.

- [x] `softmax` Computes softmax activations.
- [x] `l2loss` L2 Loss, Computes half the L2 norm of a tensor without the `sqrt`.
- [x] `lrn` Local Response Normalization. The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last dimension), and each vector is normalized independently.  Within a given vector, each component is divided by the weighted, squared sum of inputs within `depth_radius`. For details, see [Krizhevsky et al., ImageNet classification with deep convolutional neural networks (NIPS 2012)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).
- [x] `relu` Computes rectified linear: `max(features, 0)`.
- [x] `elu` Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise. See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](http://arxiv.org/abs/1511.0728).
- [x] `inTopK` Says whether the targets are in the top `K` predictions.

## Installation

```sh
$ npm install tensorflow2 --save
```

## License

MIT

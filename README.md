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

#### `variable`

Holds state in the form of a tensor that persists across steps. Outputs a ref to the tensor state so it may be read or modified.

#### `assign`

Update 'ref' by assigning 'value' to it. This operation outputs "ref" after the assignment is done. This makes it easier to chain operations that need to use the reset value.

### Array

#### `placeholder`

A placeholder op for a value that will be fed into the computation.

N.B. This operation will fail with an error if it is executed. It is
intended as a way to represent a value that will always be fed, and to
provide attrs that enable the fed value to be checked at runtime.

#### `const`

Returns a constant tensor.

#### `reverse`

Reverses specific dimensions of a tensor.

Given a `tensor`, and a `bool` tensor `dims` representing the dimensions
of `tensor`, this operation reverses each dimension i of `tensor` where
`dims[i]` is `True`.

`tensor` can have up to 8 dimensions. The number of dimensions
of `tensor` must equal the number of elements in `dims`. In other words:

```
rank(tensor) = size(dims)
```

### Neural networks

In this module, it implements the following alogrithms for represents neural networks.

#### `softmax`

Computes softmax activations.

#### `l2loss`

L2 Loss, Computes half the L2 norm of a tensor without the `sqrt`: 

```
output = sum(t ** 2) / 2
```

#### `lrn`

Local Response Normalization.

The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last dimension), and each vector is normalized independently.  Within a given vector, each component is divided by the weighted, squared sum of inputs within `depth_radius`.  In detail,

```
sqr_sum[a, b, c, d] =
    sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
output = input / (bias + alpha * sqr_sum) ** beta
```

For details, see [Krizhevsky et al., ImageNet classification with deep convolutional neural networks (NIPS 2012)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

#### `relu`

Computes rectified linear: `max(features, 0)`.

#### `elu`

Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise. See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](http://arxiv.org/abs/1511.0728)

#### `inTopK`

Says whether the targets are in the top `K` predictions.

This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the prediction for the target class is among the top `k` predictions among all predictions for example `i`. Note that the behavior of `InTopK` differs from the `TopK` op in its handling of ties; if multiple classes have the same prediction value and straddle the top-`k` boundary, all of those classes are considered to be in the top `k`.

## Installation

```sh
$ npm install tensorflow2 --save
```

## License

MIT

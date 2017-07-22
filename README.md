# <img alt="TensorFlow" src="https://www.tensorflow.org/images/tf_logo_transp.png" width="170"/> for Node.js

| NPM | Dependency | Build |
|-----|------------|-------|
|[![NPM version][npm-image]][npm-url]|[![Dependency Status][david-image]][david-url]|[![Build Status][travis-image]][travis-url]|

[npm-image]: https://img.shields.io/npm/v/tensorflow2.svg?style=flat-square
[npm-url]: https://npmjs.org/package/tensorflow2
[travis-image]: https://img.shields.io/travis/yorkie/tensorflow-nodejs.svg?style=flat-square
[travis-url]: https://travis-ci.org/yorkie/tensorflow-nodejs
[david-image]: http://img.shields.io/david/yorkie/tensorflow-nodejs.svg?style=flat-square
[david-url]: https://david-dm.org/yorkie/tensorflow-nodejs

[TensorFlow] Node.js provides idiomatic JavaScript language bindings and a high layer 
API for Node.js users.

**Notice:** This project is still under active development and not guaranteed to have a
stable API. This is especially true because the underlying TensorFlow C API has not yet
been stabilized as well.

## Installation

```sh
$ npm install tensorflow2 --save
```

## Usage

### Run a predefined graph

The ability to run a predefined graph is the most basic function for any [TensorFlow] client library.

> Given a `GraphDef` (or `MetaGraphDef`) protocol message, be able to create a session, run queries, and get tensor results. This is sufficient for a mobile app or server that wants to run inference on a pre-trained model.

Output the `GraphDef` binary format from your Python script:

```python
import tensorflow as tf
import os

def main():
  v = tf.Variable(1000, name='my_variable')
  sess = tf.Session()
  tf.train.write_graph(sess.graph_def, tmpdir, 'graph.pb', as_text=False)
```

And load the `graph.pb` to your JavaScript runtime:

```js
'use strict';

const tf = require('tensorflow2');
const graph = tf.graph();
const session = tf.session();

graph.load('/path/to/graph.pb');

// load the op by name
const op = graph.operations.get('my_variable/Assign');

// the following outputs the 1000
const res = session.run(op);
```

### Graph construction

> At least one function per defined TensorFlow op that adds an operation to the graph. Ideally these functions would be automatically generated so they stay in sync as the op definitions are modified.

```js
'use strict';

const tf = require('tensorflow2');
const graph = tf.graph();

const x = graph.constant([[1, 2], [3, 4]], tf.dtype.float32, [2, 2]);
const w = graph.variable(x);
const y = graph.nn.softmax(graph.matmul(w, w));

const session = tf.session();
const res = session.run(y);
```

## Operations

There are the following operations that we supported in this library.

### State

The state is managed by users for saving, restoring machine states.

- [x] `variable` Holds state in the form of a tensor that persists across steps. Outputs a ref to the tensor state so it may be read or modified.
- [x] `assign` Update 'ref' by assigning 'value' to it. This operation outputs "ref" after the assignment is done. This makes it easier to chain operations that need to use the reset value.

### Random

- [x] `randomUniform` Outputs random values from a uniform distribution.
- [x] `randomUniformInt` Outputs random integers from a uniform distribution.
- [x] `randomGamma` Outputs random values from the Gamma distribution(s) described by alpha.
- [x] `randomPoisson` Outputs random values from the Poisson distribution(s) described by rate.

### Array

- [x] `placeholder` A placeholder op for a value that will be fed into the computation.
- [x] `const` Returns a constant tensor.
- [x] `reverse` Reverses specific dimensions of a tensor.
- [x] `shape` Returns the shape of a tensor.
- [x] `rank` Returns the rank of a tensor.
- [x] `size` Returns the size of a tensor.

### Base64

- [x] `encode` Encode strings into web-safe base64 format.
- [x] `decode` Decode web-safe base64-encoded strings.

### Flow

- [x] `switch` Forwards `data` to the output port determined by `pred`.
- [x] `merge` Forwards the value of an available tensor from `inputs` to `output`.
- [x] `enter` Creates or finds a child frame, and makes `data` available to the child frame.
- [x] `exit` Exits the current frame to its parent frame.
- [x] `abort` Raise an exception to abort the process when called.
- [x] `trigger` Does nothing and serves as a control trigger for scheduling.

### Image

- [x] `decodeJpeg` Decode a JPEG-encoded image to a uint8 tensor.
- [x] `encodeJpeg` JPEG-encode an image.
- [x] `resizeArea` Resize `images` to `size` using area interpolation.
- [x] `resizeBicubic` Resize `images` to `size` using bicubic interpolation.
- [x] `resizeBilinear` Resize `images` to `size` using bilinear interpolation.
- [x] `resizeNearestNeighbor` Resize `images` to `size` using nearest neighbor interpolation.
- [x] `randomCorp` Randomly crop `image`.

### Audio

- [x] `decodeWav` Decode a 16-bit PCM WAV file to a float tensor.
- [x] `encodeWav` Encode audio data using the WAV file format.
- [x] `spectrogram` Produces a visualization of audio data over time.
- [x] `mfcc` Transforms a spectrogram into a form that's useful for speech recognition.

### Neural networks

In this module, it implements the following algorithms for representing neural networks.

- [x] `softmax` Computes softmax activations.
- [x] `l2loss` L2 Loss, Computes half the L2 norm of a tensor without the `sqrt`.
- [x] `lrn` Local Response Normalization. The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last dimension), and each vector is normalized independently.  Within a given vector, each component is divided by the weighted, squared sum of inputs within `depth_radius`. For details, see [Krizhevsky et al., ImageNet classification with deep convolutional neural networks (NIPS 2012)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).
- [x] `relu` Computes rectified linear: `max(features, 0)`.
- [x] `elu` Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise. See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](http://arxiv.org/abs/1511.0728).
- [x] `inTopK` Says whether the targets are in the top `K` predictions.

## Tests

```sh
$ npm test
```

## License

[MIT](./LICENSE) licensed @ 2017

[TensorFlow]: http://tensorflow.org

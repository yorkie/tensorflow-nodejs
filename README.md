# <img alt="TensorFlow" src="https://www.tensorflow.org/images/tf_logo_transp.png" width="170"/> for Node.js

| NPM | Dependency | Build | Coverage |
|-----|------------|-------|----------|
|[![NPM version][npm-image]][npm-url]|[![Dependency Status][david-image]][david-url]|[![Build Status][travis-image]][travis-url]|[![Coverage][coveralls-image]][coveralls-url]

[npm-image]: https://img.shields.io/npm/v/tensorflow2.svg?style=flat-square
[npm-url]: https://npmjs.org/package/tensorflow2
[travis-image]: https://img.shields.io/travis/yorkie/tensorflow-nodejs.svg?style=flat-square
[travis-url]: https://travis-ci.org/yorkie/tensorflow-nodejs
[david-image]: http://img.shields.io/david/yorkie/tensorflow-nodejs.svg?style=flat-square
[david-url]: https://david-dm.org/yorkie/tensorflow-nodejs
[coveralls-image]: https://img.shields.io/codecov/c/github/yorkie/tensorflow-nodejs.svg?style=flat-square
[coveralls-url]: https://codecov.io/github/yorkie/tensorflow-nodejs?branch=master

This library wraps [Tensorflow][] Python for Node.js developers, it's powered by [@pipcook/boa](https://github.com/alibaba/pipcook/blob/master/docs/manual/intro-to-boa.md).

**Notice:** This project is still under active development and not guaranteed to have a
stable API. This is especially true because the underlying TensorFlow C API has not yet
been stabilized as well.

## Installation

```sh
$ npm install tensorflow2 --save
```

## Usage

```js
const tf = require('tensorflow2');

// load mnist dataset.
const dataset = tf.keras.dataset.mnist();
// {
//   train: { x: [Getter], y: [Getter] },
//   test: { x: [Getter], y: [Getter] }
// }

// create model.
const model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten({
    input_shape: [28, 28]
  }),
  tf.keras.layers.Dense(128, {
    activation: 'relu'
  }),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
]);
model.summary();

// compile the model.
const loss_fn = tf.keras.losses.SparseCategoricalCrossentropy({ from_logits: true });
model.compile({
  optimizer: 'adam',
  loss: loss_fn,
  metrics: [ 'accuracy' ],
});

// train the model.
model.fit(dataset.train.x, dataset.train.y, { epochs: 5 });

// save the model
model.save('your-model.h5');
```

See [examples/mnist.js](./examples/mnist.js) for complete example.

## Tests

```sh
$ npm test
```

## License

[MIT](./LICENSE) licensed @ 2020

[TensorFlow]: http://tensorflow.org

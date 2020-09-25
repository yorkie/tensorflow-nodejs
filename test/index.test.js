'use strict';

const assert = require('assert');
const tf = require('../');

test('index', () => {
  console.log(tf.version);
  console.log(tf.constant([100, 200, 300]));
});

test('leras.datasets', () => {
  const dataset = tf.keras.datasets.mnist();
  console.log(dataset);
});

test('keras.layers and model', () => {
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
  console.log(model.summary());
});
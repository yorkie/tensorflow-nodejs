const tf = require('../');

const dataset = tf.keras.datasets.mnist();
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

const loss_fn = tf.keras.losses.SparseCategoricalCrossentropy({ from_logits: true });
model.compile({
  optimizer: 'adam',
  loss: loss_fn,
  metrics: [ 'accuracy' ],
});
console.log('compiled model');

model.fit(dataset.train.x, dataset.train.y, { epochs: 5 });
console.log('train done');

model.evaluate(dataset.test.x, dataset.test.y, { verbose: 2 });
model.save(__dirname + '/mnist.h5');

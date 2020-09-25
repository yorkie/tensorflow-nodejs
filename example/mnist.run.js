const tf = require('../');

const dataset = tf.keras.datasets.mnist();
const model = tf.keras.models.load(__dirname + '/mnist.h5');
model.summary();

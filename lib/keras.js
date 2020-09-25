'use strict';

const boa = require('@pipcook/boa');
const { tuple } = boa.builtins();
const { keras } = boa.import('tensorflow');

module.exports = {

  /**
   * datasets
   */
  datasets: {
    /**
     * Load the mnist dataset.
     * @param {string} path path where to cache the dataset locally (relative to ~/.keras/datasets).
     */
    mnist(path) {
      let dataset;
      if (typeof path === 'string') {
        const opts = boa.kwargs({ path });
        dataset = keras.datasets.mnist.load_data(opts);
      } else {
        dataset = keras.datasets.mnist.load_data();
      }
      const [ train, test ] = dataset;
      return {
        train: {
          get x() { return train[0]; },
          get y() { return train[1]; },
        },
        test: {
          get x() { return test[0]; },
          get y() { return test[1]; },
        },
      };
    },
  },

  /**
   * models
   */
  models: {
    Sequential: keras.models.Sequential
  },

  /**
   * layers
   */
  layers: {
    Flatten(opts) {
      if (!opts) {
        return keras.layers.Flatten();
      }
      if (opts.input_shape) {
        opts.input_shape = tuple(opts.input_shape);
      }
      return keras.layers.Flatten(boa.kwargs(opts));
    },
    Dense(units, opts) {
      if (!opts) {
        return keras.layers.Dense(units);
      }
      return keras.layers.Dense(units, boa.kwargs(opts));
    },
    Dropout: keras.layers.Dropout,
  },

  /**
   * losses
   */
  losses: {
    SparseCategoricalCrossentropy: keras.losses.SparseCategoricalCrossentropy
  }
};

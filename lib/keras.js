'use strict';

const boa = require('@pipcook/boa');
const { tuple } = boa.builtins();
const { keras } = boa.import('tensorflow');

module.exports = {

  /**
   * datasets
   */
  datasets: {
    mnist: keras.datasets.mnist,
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

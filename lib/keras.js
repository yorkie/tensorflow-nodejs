'use strict';

const boa = require('@pipcook/boa');
const { tuple } = boa.builtins();
const { keras } = boa.import('tensorflow');

/**
 * Model groups layers into an object with training and inference features.
 */
class Model {
  /**
   * 
   * @param {*} pymodel the Python model.
   */
  constructor(pymodel) {
    this.pymodel = pymodel;
  }
  /**
   * Compile the model.
   * @param {object} opts 
   * @param {string} opts.optimizer
   * @param {*} opts.loss
   * @param {*} opts.metrics
   */
  compile(opts) {
    return this.pymodel.compile(boa.kwargs(opts));
  }
  /**
   * Train the model with tensor x and y.
   * @param {*} x 
   * @param {*} y 
   * @param {*} opts 
   * @param {*} opts.epochs
   */
  fit(x, y, opts) {
    return this.pymodel.fit(x, y, boa.kwargs(opts));
  }
  /**
   * Evaluate the model.
   * @param {*} x 
   * @param {*} y 
   * @param {*} opts 
   * @param {number} opts.verbose
   */
  evaluate(x, y, opts) {
    return this.pymodel.fit(x, y, boa.kwargs(opts));
  }
  /**
   * Save the model.
   * @param {*} path 
   */
  save(pathname) {
    return this.pymodel.save(pathname);
  }
  /**
   * Output the summary of this model.
   */
  summary() {
    return this.pymodel.summary();
  }
}

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
    /**
     * Loads a model saved via `model.save()`.
     */
    load: keras.models.load_model,
    /**
     * Sequential model.
     */
    Sequential(layers, name) {
      const pymodel = keras.models.Sequential(layers, name);
      return new Model(pymodel);
    }
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
    SparseCategoricalCrossentropy(opts) {
      return keras.losses.SparseCategoricalCrossentropy(boa.kwargs(opts));
    },
  }
};

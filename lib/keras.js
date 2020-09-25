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
   * Generates output predictions for the input samples.
   * @param {*} x 
   * @param {*} opts 
   */
  predict(x, opts) {
    return this.pymodel.predict(x, boa.kwargs(opts));
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
  /**
   * set the current model if trainable.
   */
  setTrainable(val) {
    this.pymodel.trainable = !!val;
  }
  /**
   * Returns a JSON/YAML string containing the network configuration.
   * @param {*} format 
   */
  toString(format = 'json') {
    if (format === 'yaml') {
      return this.pymodel.to_yaml();
    } else {
      return this.pymodel.to_json();
    }
  }
}

class Dataset {
  constructor(data) {
    this._x = data[0];
    this._y = data[1];
  }
  get x() {
    return this._x;
  }
  get y() {
    return this._y;
  }
}

module.exports = {

  /**
   * Keras Applications are canned architectures with pre-trained weights.
   */
  applications: {
    /**
     * Instantiates the MobileNet architecture.
     * @param {*} opts 
     */
    MobileNet(opts) {
      let pymodel;
      if (!opts) {
        pymodel = keras.applications.MobileNet();
      }
      pymodel = keras.applications.MobileNet(boa.kwargs(opts));
      return new Model(pymodel);
    },
    /**
     * Instantiates the MobileNet architecture.
     * @param {*} opts 
     */
    ResNet50(opts) {
      let pymodel;
      if (!opts) {
        pymodel = keras.applications.ResNet50();
      }
      pymodel = keras.applications.ResNet50(boa.kwargs(opts));
      return new Model(pymodel);
    },
    /**
     * Instantiates the Inception-ResNet v2 architecture.
     * @param {*} opts 
     */
    InceptionResNetV2(opts) {
      let pymodel;
      if (!opts) {
        pymodel = keras.applications.InceptionResNetV2();
      }
      pymodel = keras.applications.InceptionResNetV2(boa.kwargs(opts));
      return new Model(pymodel);
    },
    /**
     * Instantiates the Inception v3 architecture.
     * @param {*} opts 
     */
    InceptionV3(opts) {
      let pymodel;
      if (!opts) {
        pymodel = keras.applications.InceptionV3();
      }
      pymodel = keras.applications.InceptionV3(boa.kwargs(opts));
      return new Model(pymodel);
    }
  },

  /**
   * datasets
   */
  datasets: {
    /**
     * Loads the MNIST dataset.
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
        train: new Dataset(train),
        test: new Dataset(test),
      };
    },
    /**
     * Loads the fashion MNIST dataset.
     * @param {*} path 
     */
    fashion_mnist(path) {
      let dataset;
      if (typeof path === 'string') {
        const opts = boa.kwargs({ path });
        dataset = keras.datasets.mnist.load_data(opts);
      } else {
        dataset = keras.datasets.mnist.load_data();
      }
      const [ train, test ] = dataset;
      return {
        train: new Dataset(train),
        test: new Dataset(test),
      };
    },
    /**
     * Loads the IMDB dataset.
     * @param {*} path 
     * @param {*} opts 
     */
    imdb(path, opts) {
      let dataset;
      if (typeof path === 'string') {
        const opts = boa.kwargs({ path });
        dataset = keras.datasets.imdb.load_data(opts);
      } else {
        dataset = keras.datasets.imdb.load_data();
      }
      const [ train, test ] = dataset;
      return {
        train: new Dataset(train),
        test: new Dataset(test),
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
    
    /**
     * Just your regular densely-connected NN layer.
     * @param {*} units 
     * @param {*} opts 
     */
    Dense(units, opts) {
      if (!opts) {
        return keras.layers.Dense(units);
      }
      return keras.layers.Dense(units, boa.kwargs(opts));
    },
    /**
     * Applies Dropout to the input.
     */
    Dropout: keras.layers.Dropout,
    /**
     * Turns positive integers (indexes) into dense vectors of fixed size.
     * @param {*} inputDim 
     * @param {*} outputDim 
     * @param {*} opts 
     */
    Embedding(inputDim, outputDim, opts) {
      if (!opts) {
        return keras.layers.LSTM(inputDim, outputDim);
      }
      return keras.layers.LSTM(inputDim, outputDim, boa.kwargs(opts));
    },
    /**
     * Flattens the input. Does not affect the batch size.
     * @param {*} opts 
     */
    Flatten(opts) {
      if (!opts) {
        return keras.layers.Flatten();
      }
      if (opts.input_shape) {
        opts.input_shape = tuple(opts.input_shape);
      }
      return keras.layers.Flatten(boa.kwargs(opts));
    },
    /**
     * Long Short-Term Memory layer - Hochreiter 1997.
     * @param {*} units 
     * @param {*} opts 
     */
    LSTM(units, opts) {
      if (!opts) {
        return keras.layers.LSTM(units);
      }
      return keras.layers.LSTM(units, boa.kwargs(opts));
    },
    /**
     * Base class for recurrent layers.
     * @param {*} cell 
     * @param {*} opts 
     */
    RNN(cell, opts) {
      if (!opts) {
        return keras.layers.RNN(units);
      }
      return keras.layers.RNN(units, boa.kwargs(opts));
    }
  },

  /**
   * losses
   */
  losses: {
    SparseCategoricalCrossentropy(opts) {
      return keras.losses.SparseCategoricalCrossentropy(boa.kwargs(opts));
    },
  },

  /**
   * utils
   */
  utils: {
    /**
     * Converts a Keras model to dot format and save to a file.
     * @param {Model} model 
     * @param {*} opts
     */
    plot(model, opts = {}) {
      if (!(model instanceof Model)) {
        throw new TypeError('must be a Model');
      }
      return keras.utils.plot_model(model.pymodel, boa.kwargs(opts));
    }
  }
};

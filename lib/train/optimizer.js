'use strict';

/**
 * @class Optimizer
 */
class Optimizer {

  /**
   * @method constructor
   * @param {String} name
   * @param {Boolean} locking
   */
  constructor(name, locking) {
    if (!name)
      throw new TypeError('Must specify the optimizer name.');
    this._name = name;
    this._locking = locking;
    this._slots = {};
  }

  /**
   * @method minimize
   * @param {Operation} loss
   * @param {Object} options
   */
  minimize(loss, options) {
    // TODO
  }

  /**
   * @method computeGradients
   * @param {Operation} loss
   * @param {Object} options
   */
  computeGradients(loss, options) {
    if (!options.gate || 
      ['graph', 'operator'].indexOf(options.gate) === -1) {
      options.gate = null;
    }
    if (!options.variables) {
      options.variables = loss._graph.trainableVariables();
    }
    // TODO
  }

  /**
   * @method computeGradients
   * @param {Operation} loss
   * @param {Object} options
   */
  applyGradients(gradientsAndVars, options) {
    // TODO
  }
}

exports.Optimizer = Optimizer;

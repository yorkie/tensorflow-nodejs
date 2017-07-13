'use strict';

module.exports = function(tensorflow) {

  let Session = tensorflow.Session;

  /**
   * @class Session
   * @method run
   * @param {Graph} fetches
   * @param {Graph} feeds
   * @param {Object} options
   */
  Session.prototype.run = function sessionRun(fetches, feeds, options) {
    return this._run(fetches, feeds, options).reduce((result, tensor) => {
      return result.concat(tensor.getViewData());
    }, []);
  };

};
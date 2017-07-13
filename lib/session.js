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
    let placeholders = this._graph.operations.list('Placeholder');
    if (placeholders.length === 0) {
      // if we don't have placeholder, the feeds are not allowed to pass
      feeds = false;
    } else {
      if (placeholders.length === 1 && feeds instanceof tensorflow.Tensor) {
        feeds = [ placeholders[0], feeds ];
      }
    }
    // console.log(feeds);
    return this._run(fetches, feeds, options).reduce((result, tensor) => {
      return result.concat(tensor.getViewData());
    }, []);
  };

};
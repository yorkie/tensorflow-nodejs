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
      if (placeholders.length === 1 && !Array.isArray(feeds)) {
        let tensor = feeds;
        if (!(tensor instanceof tensorflow.Tensor)) {
          tensor = tensorflow.Tensor.from(tensor);
        }
        feeds = [ [ placeholders[0], tensor ], ];
      } else {
        // multi placeholders
        feeds = placeholders.map((placeholder) => {
          let tensor = feeds[placeholder.name];
          if (!(tensor instanceof tensorflow.Tensor)) {
            tensor = tensorflow.Tensor.from(tensor);
          }
          return [ placeholder, tensor ];
        });
      }
      // TODO
    }

    const tensors = this._run(fetches, feeds, options);
    if (tensors.length === 1) {
      return tensors[0].getViewData();
    } else {
      return tensors
        .map((tensor) => tensor.getViewData());
    }
  };

};
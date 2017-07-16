module.exports = function(tensorflow) {
  'use strict';

  const Graph = tensorflow.Graph;

  function resizeHandler(descriptor, image, size, alignCorners) {
    descriptor.addInput(0, images);
    descriptor.addInput(0, size);
    descriptor.setAttrBool('align_corners', alignCorners || false);
  }

  /**
   * @class Graph
   * @module image
   */
  return {

    /**
     * @class Graph.image
     * @method decodeJpeg
     */
    decodeJpeg: Graph.createOperationBuilder(
      'DecodeJpeg',
      function(descriptor, contents, options) {
        options = Object.assign({
          channels: 3,
          ratio: 1,
          fancyUpscaling: true,
          recoverTruncated: false,
          acceptableFraction: 1.0,
          dctMethod: '',
        }, options);

        descriptor.addInput(0, contents);
        descriptor.setAttrInt('channels', options.channels);
        descriptor.setAttrInt('ratio', options.ratio);
        descriptor.setAttrBool('fancy_upscaling', options.fancyUpscaling);
        descriptor.setAttrBool('try_recover_truncated', options.recoverTruncated);
        descriptor.setAttrFloat('acceptable_fraction', options.acceptableFraction);
        descriptor.setAttrString('dct_method', options.dctMethod);
      }
    ),

    /**
     * @class Graph.image
     * @method encodeJpeg
     */
    encodeJpeg: Graph.createOperationBuilder(
      'EncodeJpeg',
      function(descriptor, image, options) {
        options = Object.assign({
          format: 'rgb',
          quality: 95,
          progressive: false,
          optimizeSize: false,
          chromaDownsampling: true,
          density: {
            x: 300,
            y: 300,
            unit: 'in'
          },
          metadata: ''
        }, options);

        if (options.format !== 'grey' || options.format !== 'rgb') {
          options.format = '';
        }
        descriptor.addInput(0, image);
        descriptor.setAttrString('format', options.format);
        descriptor.setAttrInt('quality', options.quality);
        descriptor.setAttrBool('progressive', options.progressive);
        descriptor.setAttrBool('optimize_size', options.optimizeSize);
        descriptor.setAttrBool('chroma_downsampling', options.chromaDownsampling);
        descriptor.setAttrString('density_unit', options.density.unit);
        descriptor.setAttrInt('x_density', options.density.x);
        descriptor.setAttrInt('y_density', options.density.y);
        descriptor.setAttrString('xmp_metadata', options.metadata);
      }
    ),

    /**
     * @class Graph.image
     * @method resizeArea
     */
    resizeArea: Graph.createOperationBuilder(
      'ResizeArea', resizeHandler),

    /**
     * @class Graph.image
     * @method resizeBicubic
     */
    resizeBicubic: Graph.createOperationBuilder(
      'ResizeBicubic', resizeHandler),

    /**
     * @class Graph.image
     * @method resizeBilinear
     */
    resizeBilinear: Graph.createOperationBuilder(
      'ResizeBilinear', resizeHandler),

    /**
     * @class Graph.image
     * @method resizeNearestNeighbor
     */
    resizeNearestNeighbor: Graph.createOperationBuilder(
      'ResizeNearestNeighbor', resizeHandler),

    /**
     * @class Graph.image
     * @method randomCorp
     */
    randomCorp: Graph.createOperationBuilder(
      'RandomCrop',
      function(descriptor, image, size, seeds) {
        descriptor.addInput(0, image);
        descriptor.addInput(0, size);
        if (seeds && typeof seeds[0] === 'number')
          descriptor.setAttrInt('seed', seeds[0]);
        if (seeds && typeof seeds[1] === 'number')
          descriptor.setAttrInt('seed2', seeds[1]);
      }
    ),

  };

};


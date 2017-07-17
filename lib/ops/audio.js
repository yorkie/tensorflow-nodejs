module.exports = function(tensorflow) {
  'use strict';

  const Graph = tensorflow.Graph;

  /**
   * @class Graph
   * @module audio
   */
  return {

    /**
     * @class Graph.audio
     * @method decodeWav
     */
    decodeWav: Graph.createOperationBuilder(
      'DecodeWav',
      function(descriptor, contents, options) {
        descriptor.addInput(0, contents);
        if (options && typeof options.channels === 'number')
          descriptor.setAttrInt('desired_channels', options.channels);
        if (options && typeof options.samples === 'number')
          descriptor.setAttrInt('desired_samples', options.samples);
      }
    ),

    /**
     * @class Graph.audio
     * @method encodeWav
     */
    encodeWav: Graph.createOperationBuilder(
      'EncodeWav',
      function(descriptor, audio, sampleRate) {
        descriptor.addInput(0, audio);
        descriptor.addInput(0, sampleRate);
      }
    ),

    /**
     * @class Graph.audio
     * @method spectrogram
     */
    spectrogram: Graph.createOperationBuilder(
      'AudioSpectrogram',
      function(descriptor, input, windowSize, stride) {
        descriptor.addInput(0, input);
        descriptor.setAttrInt('window_size', windowSize);
        descriptor.setAttrInt('stride', stride);
      }
    ),

    /**
     * @class Graph.audio
     * @method mfcc
     */
    mfcc: Graph.createOperationBuilder(
      'Mfcc',
      function(descriptor, spectrogram, sampleRate, options) {
        descriptor.addInput(0, spectrogram);
        descriptor.addInput(0, sampleRate);
        if (options && options.frequencyLimit) {
          descriptor.setAttrFloat(
            'lower_frequency_limit', options.frequencyLimit[0]);
          descriptor.setAttrFloat(
            'upper_frequency_limit', options.frequencyLimit[1]);
        }
        if (options && options.filterbankChannelCount)
          descriptor.setAttrInt(
            'filterbank_channel_count', options.filterbankChannelCount);
        if (options && options.DCTCoefficientCount)
          descriptor.setAttrInt(
            'dct_coefficient_count', options.DCTCoefficientCount);
      }
    ),

  };

};


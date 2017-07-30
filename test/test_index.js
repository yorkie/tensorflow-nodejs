'use strict';

const assert = require('assert');
const tf = require('../');
const oplist = tf.Graph.getAllOpList();

assert(oplist.length > 0);
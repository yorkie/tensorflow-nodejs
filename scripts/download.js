'use strict';

const os = require('os');
const fs = require('fs');
const path = require('path');
const https = require('https');
const tar = require('tar-fs');
const unzip = require('unzip');
const gunzip = require('gunzip-maybe');

const version = '1.2.1';
const TF_TYPE = process.env.TF_TYPE || 'cpu';
const DOWNLOAD_URL = 'https://storage.googleapis.com/tensorflow/libtensorflow/' +
  `libtensorflow-${TF_TYPE}-${os.platform()}-x86_64-${version}.tar.gz`;
const PROTOBUF_URL = 'https://storage.googleapis.com/tensorflow/libtensorflow/' +
  `libtensorflow_proto-${version}.zip`;

if (!fs.existsSync('./tensorflow')) {
  https.get(DOWNLOAD_URL, (res) => {
    if (res.statusCode !== 200) {
      throw new Error(DOWNLOAD_URL + ' ' + res.statusMessage);
    } else {
      console.log(DOWNLOAD_URL + ' is finished downloaded.');
    }
    res.pipe(gunzip()).pipe(tar.extract('./tensorflow'));
  });
} else {
  console.log('Skiped, tensorflow library and header are exists');
}

https.get(PROTOBUF_URL, (res) => {
  if (res.statusCode !== 200) {
    throw new Error(PROTOBUF_URL + ' ' + res.statusMessage);
  } else {
    console.log('Done,', PROTOBUF_URL + ' is finished downloaded.');
  }
  res.pipe(unzip.Extract({
    path: './protobuf'
  }));
});

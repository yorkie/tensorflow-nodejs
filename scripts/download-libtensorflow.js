'use strict';

const os = require('os');
const fs = require('fs');
const path = require('path');
const https = require('https');
const tar = require('tar');

const TF_TYPE = process.env.TF_TYPE || 'cpu';
const DOWNLOAD_URL = 'https://storage.googleapis.com/tensorflow/libtensorflow/' +
  `libtensorflow-${TF_TYPE}-${os.platform()}-x86_64-1.2.1.tar.gz`;

https.get(DOWNLOAD_URL, (res) => {
  if (res.statusCode !== 200) {
    throw new Error(DOWNLOAD_URL + ' ' + res.statusMessage);
  } else {
    console.log(DOWNLOAD_URL + ' is finished downloaded.');
  }
  res.pipe(tar.t()).on('entry', (entry) => {
    if (entry.type !== 'File')
      return;
    let headerPath = path.join(__dirname, '../tensorflow/', entry.path);
    entry.pipe(fs.createWriteStream(headerPath));
  }).on('end', () => {
    console.log('download done');
    // const libPath = path.join(__dirname, '../tensorflow/lib/libtensorflow.so');
    // fs.rename(libPath, '/usr/local/lib/libtensorflow.so', (err) => {
    //   if (err)
    //     throw err;
    //   console.log('../tensorflow/lib/libtensorflow.so -> /usr/local/lib/libtensorflow.so');
    // });
  });
});
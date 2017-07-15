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
  res.pipe(tar.t()).on('entry', (entry) => {
    if (entry.type !== 'File')
      return;
    let headerPath = path.join(__dirname, '../tensorflow/', entry.path);
    entry.pipe(fs.createWriteStream(headerPath));
  });
});
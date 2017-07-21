{
  "targets": [{
    "target_name": "tensorflow",
    "include_dirs": [
      "<!(node -e \"require('nan')\")",
      "./",
      "./tensorflow/include",
    ],
    "sources": [
      "src/binding.cc",
      "src/buffer.cc",
      "src/dtype.cc",
      "src/graph.cc",
      "src/internal.h",
      "src/library.cc",
      "src/operation.cc",
      "src/session.cc",
      "src/tensor.cc",
    ],
    "cflags!": [ "-fno-exceptions" ],
    "cflags_cc!": [ "-fno-exceptions" ],
    "xcode_settings": {
      "CLANG_CXX_LIBRARY": "libc++",
      "MACOSX_DEPLOYMENT_TARGET": "10.12",
    },
    "libraries": [
      "-Wl,-rpath,$(PWD)/tensorflow/lib",
      "-L$(PWD)/tensorflow/lib",
      "-ltensorflow",
    ],
  }]
}

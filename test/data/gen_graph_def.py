import tensorflow as tf
import os

tmpdir = os.path.dirname(os.path.realpath(__file__))

def main():
  v = tf.Variable(1000, name='my_variable')
  sess = tf.Session()
  tf.train.write_graph(sess.graph_def, tmpdir, 'graph_def.pbtxt', as_text=False)

if __name__ == "__main__":
    main()

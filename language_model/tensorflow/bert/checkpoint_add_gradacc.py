"""Add GradientAccumulator to checkpoint file."""

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "old",
    "//je-d/home/staging-brain-gpu-dedicated/bert/pretrained_model/converted",
    "old checkpoint file")
flags.DEFINE_string("new",
                    "//je-d/home/tpu-perf-team/shibow/bert_add_gradacc",
                    "new checkpoint file")


def main(unused_argv):
  reader = tf.train.NewCheckpointReader(FLAGS.old)
  shapes = reader.get_variable_to_shape_map()
  dtypes = reader.get_variable_to_dtype_map()

  tf.reset_default_graph()
  with tf.Session() as sess:
    for n in shapes:
      logging.info(n)
      logging.info(shapes[n])
      logging.info(dtypes[n])
      tf.keras.backend.set_value(
          tf.get_variable(n, shapes[n], dtypes[n]),
          np.array(reader.get_tensor(n)))
      tf.keras.backend.set_value(
          tf.get_variable(n + "/GradientAccumulator", shapes[n], dtypes[n]),
          np.zeros(shapes[n]))
    tf.train.Saver().save(sess, FLAGS.new)


if __name__ == "__main__":
  app.run(main)


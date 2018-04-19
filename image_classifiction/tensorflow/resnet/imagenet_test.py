# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.resnet import imagenet_main
from official.utils.testing import integration

tf.logging.set_verbosity(tf.logging.ERROR)

_BATCH_SIZE = 32
_LABEL_CLASSES = 1001


class BaseTest(tf.test.TestCase):

  def tearDown(self):
    super(BaseTest, self).tearDown()
    tf.gfile.DeleteRecursively(self.get_temp_dir())

  def _tensor_shapes_helper(self, resnet_size, version, dtype, with_gpu):
    """Checks the tensor shapes after each phase of the ResNet model."""
    def reshape(shape):
      """Returns the expected dimensions depending on if a GPU is being used."""

      # If a GPU is used for the test, the shape is returned (already in NCHW
      # form). When GPU is not used, the shape is converted to NHWC.
      if with_gpu:
        return shape
      return shape[0], shape[2], shape[3], shape[1]

    graph = tf.Graph()

    with graph.as_default(), self.test_session(
        graph=graph, use_gpu=with_gpu, force_gpu=with_gpu):
      model = imagenet_main.ImagenetModel(
          resnet_size=resnet_size,
          data_format='channels_first' if with_gpu else 'channels_last',
          version=version,
          dtype=dtype
      )
      inputs = tf.random_uniform([1, 224, 224, 3])
      output = model(inputs, training=True)

      initial_conv = graph.get_tensor_by_name('resnet_model/initial_conv:0')
      max_pool = graph.get_tensor_by_name('resnet_model/initial_max_pool:0')
      block_layer1 = graph.get_tensor_by_name('resnet_model/block_layer1:0')
      block_layer2 = graph.get_tensor_by_name('resnet_model/block_layer2:0')
      block_layer3 = graph.get_tensor_by_name('resnet_model/block_layer3:0')
      block_layer4 = graph.get_tensor_by_name('resnet_model/block_layer4:0')
      reduce_mean = graph.get_tensor_by_name('resnet_model/final_reduce_mean:0')
      dense = graph.get_tensor_by_name('resnet_model/final_dense:0')

      self.assertAllEqual(initial_conv.shape, reshape((1, 64, 112, 112)))
      self.assertAllEqual(max_pool.shape, reshape((1, 64, 56, 56)))

      # The number of channels after each block depends on whether we're
      # using the building_block or the bottleneck_block.
      if resnet_size < 50:
        self.assertAllEqual(block_layer1.shape, reshape((1, 64, 56, 56)))
        self.assertAllEqual(block_layer2.shape, reshape((1, 128, 28, 28)))
        self.assertAllEqual(block_layer3.shape, reshape((1, 256, 14, 14)))
        self.assertAllEqual(block_layer4.shape, reshape((1, 512, 7, 7)))
        self.assertAllEqual(reduce_mean.shape, reshape((1, 512, 1, 1)))
      else:
        self.assertAllEqual(block_layer1.shape, reshape((1, 256, 56, 56)))
        self.assertAllEqual(block_layer2.shape, reshape((1, 512, 28, 28)))
        self.assertAllEqual(block_layer3.shape, reshape((1, 1024, 14, 14)))
        self.assertAllEqual(block_layer4.shape, reshape((1, 2048, 7, 7)))
        self.assertAllEqual(reduce_mean.shape, reshape((1, 2048, 1, 1)))

      self.assertAllEqual(dense.shape, (1, _LABEL_CLASSES))
      self.assertAllEqual(output.shape, (1, _LABEL_CLASSES))

  def tensor_shapes_helper(self, resnet_size, version, with_gpu=False):
    self._tensor_shapes_helper(resnet_size=resnet_size, version=version,
                               dtype=tf.float32, with_gpu=with_gpu)
    self._tensor_shapes_helper(resnet_size=resnet_size, version=version,
                               dtype=tf.float16, with_gpu=with_gpu)

  def test_tensor_shapes_resnet_18_v1(self):
    self.tensor_shapes_helper(18, version=1)

  def test_tensor_shapes_resnet_18_v2(self):
    self.tensor_shapes_helper(18, version=2)

  def test_tensor_shapes_resnet_34_v1(self):
    self.tensor_shapes_helper(34, version=1)

  def test_tensor_shapes_resnet_34_v2(self):
    self.tensor_shapes_helper(34, version=2)

  def test_tensor_shapes_resnet_50_v1(self):
    self.tensor_shapes_helper(50, version=1)

  def test_tensor_shapes_resnet_50_v2(self):
    self.tensor_shapes_helper(50, version=2)

  def test_tensor_shapes_resnet_101_v1(self):
    self.tensor_shapes_helper(101, version=1)

  def test_tensor_shapes_resnet_101_v2(self):
    self.tensor_shapes_helper(101, version=2)

  def test_tensor_shapes_resnet_152_v1(self):
    self.tensor_shapes_helper(152, version=1)

  def test_tensor_shapes_resnet_152_v2(self):
    self.tensor_shapes_helper(152, version=2)

  def test_tensor_shapes_resnet_200_v1(self):
    self.tensor_shapes_helper(200, version=1)

  def test_tensor_shapes_resnet_200_v2(self):
    self.tensor_shapes_helper(200, version=2)

  @unittest.skipUnless(tf.test.is_built_with_cuda(), 'requires GPU')
  def test_tensor_shapes_resnet_18_with_gpu_v1(self):
    self.tensor_shapes_helper(18, version=1, with_gpu=True)

  @unittest.skipUnless(tf.test.is_built_with_cuda(), 'requires GPU')
  def test_tensor_shapes_resnet_18_with_gpu_v2(self):
    self.tensor_shapes_helper(18, version=2, with_gpu=True)

  @unittest.skipUnless(tf.test.is_built_with_cuda(), 'requires GPU')
  def test_tensor_shapes_resnet_34_with_gpu_v1(self):
    self.tensor_shapes_helper(34, version=1, with_gpu=True)

  @unittest.skipUnless(tf.test.is_built_with_cuda(), 'requires GPU')
  def test_tensor_shapes_resnet_34_with_gpu_v2(self):
    self.tensor_shapes_helper(34, version=2, with_gpu=True)

  @unittest.skipUnless(tf.test.is_built_with_cuda(), 'requires GPU')
  def test_tensor_shapes_resnet_50_with_gpu_v1(self):
    self.tensor_shapes_helper(50, version=1, with_gpu=True)

  @unittest.skipUnless(tf.test.is_built_with_cuda(), 'requires GPU')
  def test_tensor_shapes_resnet_50_with_gpu_v2(self):
    self.tensor_shapes_helper(50, version=2, with_gpu=True)

  @unittest.skipUnless(tf.test.is_built_with_cuda(), 'requires GPU')
  def test_tensor_shapes_resnet_101_with_gpu_v1(self):
    self.tensor_shapes_helper(101, version=1, with_gpu=True)

  @unittest.skipUnless(tf.test.is_built_with_cuda(), 'requires GPU')
  def test_tensor_shapes_resnet_101_with_gpu_v2(self):
    self.tensor_shapes_helper(101, version=2, with_gpu=True)

  @unittest.skipUnless(tf.test.is_built_with_cuda(), 'requires GPU')
  def test_tensor_shapes_resnet_152_with_gpu_v1(self):
    self.tensor_shapes_helper(152, version=1, with_gpu=True)

  @unittest.skipUnless(tf.test.is_built_with_cuda(), 'requires GPU')
  def test_tensor_shapes_resnet_152_with_gpu_v2(self):
    self.tensor_shapes_helper(152, version=2, with_gpu=True)

  @unittest.skipUnless(tf.test.is_built_with_cuda(), 'requires GPU')
  def test_tensor_shapes_resnet_200_with_gpu_v1(self):
    self.tensor_shapes_helper(200, version=1, with_gpu=True)

  @unittest.skipUnless(tf.test.is_built_with_cuda(), 'requires GPU')
  def test_tensor_shapes_resnet_200_with_gpu_v2(self):
    self.tensor_shapes_helper(200, version=2, with_gpu=True)

  def _resnet_model_fn_helper(self, mode, version, dtype, multi_gpu):
    """Tests that the EstimatorSpec is given the appropriate arguments."""
    with tf.Graph().as_default() as g:
      tf.train.create_global_step()

      input_fn = imagenet_main.get_synth_input_fn()
      dataset = input_fn(True, '', _BATCH_SIZE)
      iterator = dataset.make_one_shot_iterator()
      features, labels = iterator.get_next()
      spec = imagenet_main.imagenet_model_fn(
          features, labels, mode, {
              'dtype': dtype,
              'resnet_size': 50,
              'data_format': 'channels_last',
              'batch_size': _BATCH_SIZE,
              'version': version,
              'loss_scale': 128 if dtype == tf.float16 else 1,
              'multi_gpu': multi_gpu,
          })

      predictions = spec.predictions
      self.assertAllEqual(predictions['probabilities'].shape,
                          (_BATCH_SIZE, _LABEL_CLASSES))
      self.assertEqual(predictions['probabilities'].dtype, tf.float32)
      self.assertAllEqual(predictions['classes'].shape, (_BATCH_SIZE,))
      self.assertEqual(predictions['classes'].dtype, tf.int64)

      if mode != tf.estimator.ModeKeys.PREDICT:
        loss = spec.loss
        self.assertAllEqual(loss.shape, ())
        self.assertEqual(loss.dtype, tf.float32)

      if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = spec.eval_metric_ops
        self.assertAllEqual(eval_metric_ops['accuracy'][0].shape, ())
        self.assertAllEqual(eval_metric_ops['accuracy'][1].shape, ())
        self.assertEqual(eval_metric_ops['accuracy'][0].dtype, tf.float32)
        self.assertEqual(eval_metric_ops['accuracy'][1].dtype, tf.float32)

        tensors_to_check = ('initial_conv:0', 'initial_max_pool:0',
                            'block_layer1:0', 'block_layer2:0',
                            'block_layer3:0', 'block_layer4:0',
                            'final_reduce_mean:0', 'final_dense:0')

        for tensor_name in tensors_to_check:
          tensor = g.get_tensor_by_name('resnet_model/' + tensor_name)
          self.assertEqual(tensor.dtype, dtype,
                           'Tensor {} has dtype {}, while dtype {} was '
                           'expected'.format(tensor, tensor.dtype,
                                             dtype))

  def resnet_model_fn_helper(self, mode, version, multi_gpu=False):
    self._resnet_model_fn_helper(mode=mode, version=version, dtype=tf.float32,
                                 multi_gpu=multi_gpu)
    self._resnet_model_fn_helper(mode=mode, version=version, dtype=tf.float16,
                                 multi_gpu=multi_gpu)

  def test_resnet_model_fn_train_mode_v1(self):
    self.resnet_model_fn_helper(tf.estimator.ModeKeys.TRAIN, version=1)

  def test_resnet_model_fn_train_mode_v2(self):
    self.resnet_model_fn_helper(tf.estimator.ModeKeys.TRAIN, version=2)

  def test_resnet_model_fn_train_mode_multi_gpu_v1(self):
    self.resnet_model_fn_helper(tf.estimator.ModeKeys.TRAIN, version=1,
                                multi_gpu=True)

  def test_resnet_model_fn_train_mode_multi_gpu_v2(self):
    self.resnet_model_fn_helper(tf.estimator.ModeKeys.TRAIN, version=2,
                                multi_gpu=True)

  def test_resnet_model_fn_eval_mode_v1(self):
    self.resnet_model_fn_helper(tf.estimator.ModeKeys.EVAL, version=1)

  def test_resnet_model_fn_eval_mode_v2(self):
    self.resnet_model_fn_helper(tf.estimator.ModeKeys.EVAL, version=2)

  def test_resnet_model_fn_predict_mode_v1(self):
    self.resnet_model_fn_helper(tf.estimator.ModeKeys.PREDICT, version=1)

  def test_resnet_model_fn_predict_mode_v2(self):
    self.resnet_model_fn_helper(tf.estimator.ModeKeys.PREDICT, version=2)

  def _test_imagenetmodel_shape(self, version):
    batch_size = 135
    num_classes = 246

    model = imagenet_main.ImagenetModel(
        50, data_format='channels_last', num_classes=num_classes,
        version=version)

    fake_input = tf.random_uniform([batch_size, 224, 224, 3])
    output = model(fake_input, training=True)

    self.assertAllEqual(output.shape, (batch_size, num_classes))

  def test_imagenetmodel_shape_v1(self):
    self._test_imagenetmodel_shape(version=1)

  def test_imagenetmodel_shape_v2(self):
    self._test_imagenetmodel_shape(version=2)

  def test_imagenet_end_to_end_synthetic_v1(self):
    integration.run_synthetic(
        main=imagenet_main.main, tmp_root=self.get_temp_dir(),
        extra_flags=['-v', '1']
    )

  def test_imagenet_end_to_end_synthetic_v2(self):
    integration.run_synthetic(
        main=imagenet_main.main, tmp_root=self.get_temp_dir(),
        extra_flags=['-v', '2']
    )

  def test_imagenet_end_to_end_synthetic_v1_tiny(self):
    integration.run_synthetic(
        main=imagenet_main.main, tmp_root=self.get_temp_dir(),
        extra_flags=['-v', '1', '-rs', '18']
    )

  def test_imagenet_end_to_end_synthetic_v2_tiny(self):
    integration.run_synthetic(
        main=imagenet_main.main, tmp_root=self.get_temp_dir(),
        extra_flags=['-v', '2', '-rs', '18']
    )

  def test_imagenet_end_to_end_synthetic_v1_huge(self):
    integration.run_synthetic(
        main=imagenet_main.main, tmp_root=self.get_temp_dir(),
        extra_flags=['-v', '1', '-rs', '200']
    )

  def test_imagenet_end_to_end_synthetic_v2_huge(self):
    integration.run_synthetic(
        main=imagenet_main.main, tmp_root=self.get_temp_dir(),
        extra_flags=['-v', '2', '-rs', '200']
    )


if __name__ == '__main__':
  tf.test.main()

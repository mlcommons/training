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
"""Tests for object_detection.tflearn.inputs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

import numpy as np
import tensorflow as tf

from object_detection import inputs
from object_detection.core import preprocessor
from object_detection.core import standard_fields as fields
from object_detection.utils import config_util

FLAGS = tf.flags.FLAGS


def _get_configs_for_model(model_name):
  """Returns configurations for model."""
  fname = os.path.join(tf.resource_loader.get_data_files_path(),
                       'samples/configs/' + model_name + '.config')
  label_map_path = os.path.join(tf.resource_loader.get_data_files_path(),
                                'data/pet_label_map.pbtxt')
  data_path = os.path.join(tf.resource_loader.get_data_files_path(),
                           'test_data/pets_examples.record')
  configs = config_util.get_configs_from_pipeline_file(fname)
  return config_util.merge_external_params_with_configs(
      configs,
      train_input_path=data_path,
      eval_input_path=data_path,
      label_map_path=label_map_path)


class InputsTest(tf.test.TestCase):

  def test_faster_rcnn_resnet50_train_input(self):
    """Tests the training input function for FasterRcnnResnet50."""
    configs = _get_configs_for_model('faster_rcnn_resnet50_pets')
    configs['train_config'].unpad_groundtruth_tensors = True
    model_config = configs['model']
    model_config.faster_rcnn.num_classes = 37
    train_input_fn = inputs.create_train_input_fn(
        configs['train_config'], configs['train_input_config'], model_config)
    features, labels = train_input_fn()

    self.assertAllEqual([1, None, None, 3],
                        features[fields.InputDataFields.image].shape.as_list())
    self.assertEqual(tf.float32, features[fields.InputDataFields.image].dtype)
    self.assertAllEqual([1],
                        features[inputs.HASH_KEY].shape.as_list())
    self.assertEqual(tf.int32, features[inputs.HASH_KEY].dtype)
    self.assertAllEqual(
        [1, 50, 4],
        labels[fields.InputDataFields.groundtruth_boxes].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_boxes].dtype)
    self.assertAllEqual(
        [1, 50, model_config.faster_rcnn.num_classes],
        labels[fields.InputDataFields.groundtruth_classes].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_classes].dtype)
    self.assertAllEqual(
        [1, 50],
        labels[fields.InputDataFields.groundtruth_weights].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_weights].dtype)

  def test_faster_rcnn_resnet50_eval_input(self):
    """Tests the eval input function for FasterRcnnResnet50."""
    configs = _get_configs_for_model('faster_rcnn_resnet50_pets')
    model_config = configs['model']
    model_config.faster_rcnn.num_classes = 37
    eval_input_fn = inputs.create_eval_input_fn(
        configs['eval_config'], configs['eval_input_config'], model_config)
    features, labels = eval_input_fn()

    self.assertAllEqual([1, None, None, 3],
                        features[fields.InputDataFields.image].shape.as_list())
    self.assertEqual(tf.float32, features[fields.InputDataFields.image].dtype)
    self.assertAllEqual(
        [1, None, None, 3],
        features[fields.InputDataFields.original_image].shape.as_list())
    self.assertEqual(tf.uint8,
                     features[fields.InputDataFields.original_image].dtype)
    self.assertAllEqual([1], features[inputs.HASH_KEY].shape.as_list())
    self.assertEqual(tf.int32, features[inputs.HASH_KEY].dtype)
    self.assertAllEqual(
        [1, None, 4],
        labels[fields.InputDataFields.groundtruth_boxes].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_boxes].dtype)
    self.assertAllEqual(
        [1, None, model_config.faster_rcnn.num_classes],
        labels[fields.InputDataFields.groundtruth_classes].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_classes].dtype)
    self.assertAllEqual(
        [1, None],
        labels[fields.InputDataFields.groundtruth_area].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_area].dtype)
    self.assertAllEqual(
        [1, None],
        labels[fields.InputDataFields.groundtruth_is_crowd].shape.as_list())
    self.assertEqual(
        tf.bool, labels[fields.InputDataFields.groundtruth_is_crowd].dtype)
    self.assertAllEqual(
        [1, None],
        labels[fields.InputDataFields.groundtruth_difficult].shape.as_list())
    self.assertEqual(
        tf.int32, labels[fields.InputDataFields.groundtruth_difficult].dtype)

  def test_ssd_inceptionV2_train_input(self):
    """Tests the training input function for SSDInceptionV2."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    model_config = configs['model']
    model_config.ssd.num_classes = 37
    batch_size = configs['train_config'].batch_size
    train_input_fn = inputs.create_train_input_fn(
        configs['train_config'], configs['train_input_config'], model_config)
    features, labels = train_input_fn()

    self.assertAllEqual([batch_size, 300, 300, 3],
                        features[fields.InputDataFields.image].shape.as_list())
    self.assertEqual(tf.float32, features[fields.InputDataFields.image].dtype)
    self.assertAllEqual([batch_size],
                        features[inputs.HASH_KEY].shape.as_list())
    self.assertEqual(tf.int32, features[inputs.HASH_KEY].dtype)
    self.assertAllEqual(
        [batch_size],
        labels[fields.InputDataFields.num_groundtruth_boxes].shape.as_list())
    self.assertEqual(tf.int32,
                     labels[fields.InputDataFields.num_groundtruth_boxes].dtype)
    self.assertAllEqual(
        [batch_size, 50, 4],
        labels[fields.InputDataFields.groundtruth_boxes].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_boxes].dtype)
    self.assertAllEqual(
        [batch_size, 50, model_config.ssd.num_classes],
        labels[fields.InputDataFields.groundtruth_classes].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_classes].dtype)
    self.assertAllEqual(
        [batch_size, 50],
        labels[fields.InputDataFields.groundtruth_weights].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_weights].dtype)

  def test_ssd_inceptionV2_eval_input(self):
    """Tests the eval input function for SSDInceptionV2."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    model_config = configs['model']
    model_config.ssd.num_classes = 37
    eval_input_fn = inputs.create_eval_input_fn(
        configs['eval_config'], configs['eval_input_config'], model_config)
    features, labels = eval_input_fn()

    self.assertAllEqual([1, 300, 300, 3],
                        features[fields.InputDataFields.image].shape.as_list())
    self.assertEqual(tf.float32, features[fields.InputDataFields.image].dtype)
    self.assertAllEqual(
        [1, None, None, 3],
        features[fields.InputDataFields.original_image].shape.as_list())
    self.assertEqual(tf.uint8,
                     features[fields.InputDataFields.original_image].dtype)
    self.assertAllEqual([1], features[inputs.HASH_KEY].shape.as_list())
    self.assertEqual(tf.int32, features[inputs.HASH_KEY].dtype)
    self.assertAllEqual(
        [1, None, 4],
        labels[fields.InputDataFields.groundtruth_boxes].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_boxes].dtype)
    self.assertAllEqual(
        [1, None, model_config.ssd.num_classes],
        labels[fields.InputDataFields.groundtruth_classes].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_classes].dtype)
    self.assertAllEqual(
        [1, None],
        labels[fields.InputDataFields.groundtruth_area].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_area].dtype)
    self.assertAllEqual(
        [1, None],
        labels[fields.InputDataFields.groundtruth_is_crowd].shape.as_list())
    self.assertEqual(
        tf.bool, labels[fields.InputDataFields.groundtruth_is_crowd].dtype)
    self.assertAllEqual(
        [1, None],
        labels[fields.InputDataFields.groundtruth_difficult].shape.as_list())
    self.assertEqual(
        tf.int32, labels[fields.InputDataFields.groundtruth_difficult].dtype)

  def test_predict_input(self):
    """Tests the predict input function."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    predict_input_fn = inputs.create_predict_input_fn(
        model_config=configs['model'])
    serving_input_receiver = predict_input_fn()

    image = serving_input_receiver.features[fields.InputDataFields.image]
    receiver_tensors = serving_input_receiver.receiver_tensors[
        inputs.SERVING_FED_EXAMPLE_KEY]
    self.assertEqual([1, 300, 300, 3], image.shape.as_list())
    self.assertEqual(tf.float32, image.dtype)
    self.assertEqual(tf.string, receiver_tensors.dtype)

  def test_error_with_bad_train_config(self):
    """Tests that a TypeError is raised with improper train config."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    configs['model'].ssd.num_classes = 37
    train_input_fn = inputs.create_train_input_fn(
        train_config=configs['eval_config'],  # Expecting `TrainConfig`.
        train_input_config=configs['train_input_config'],
        model_config=configs['model'])
    with self.assertRaises(TypeError):
      train_input_fn()

  def test_error_with_bad_train_input_config(self):
    """Tests that a TypeError is raised with improper train input config."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    configs['model'].ssd.num_classes = 37
    train_input_fn = inputs.create_train_input_fn(
        train_config=configs['train_config'],
        train_input_config=configs['model'],  # Expecting `InputReader`.
        model_config=configs['model'])
    with self.assertRaises(TypeError):
      train_input_fn()

  def test_error_with_bad_train_model_config(self):
    """Tests that a TypeError is raised with improper train model config."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    configs['model'].ssd.num_classes = 37
    train_input_fn = inputs.create_train_input_fn(
        train_config=configs['train_config'],
        train_input_config=configs['train_input_config'],
        model_config=configs['train_config'])  # Expecting `DetectionModel`.
    with self.assertRaises(TypeError):
      train_input_fn()

  def test_error_with_bad_eval_config(self):
    """Tests that a TypeError is raised with improper eval config."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    configs['model'].ssd.num_classes = 37
    eval_input_fn = inputs.create_eval_input_fn(
        eval_config=configs['train_config'],  # Expecting `EvalConfig`.
        eval_input_config=configs['eval_input_config'],
        model_config=configs['model'])
    with self.assertRaises(TypeError):
      eval_input_fn()

  def test_error_with_bad_eval_input_config(self):
    """Tests that a TypeError is raised with improper eval input config."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    configs['model'].ssd.num_classes = 37
    eval_input_fn = inputs.create_eval_input_fn(
        eval_config=configs['eval_config'],
        eval_input_config=configs['model'],  # Expecting `InputReader`.
        model_config=configs['model'])
    with self.assertRaises(TypeError):
      eval_input_fn()

  def test_error_with_bad_eval_model_config(self):
    """Tests that a TypeError is raised with improper eval model config."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    configs['model'].ssd.num_classes = 37
    eval_input_fn = inputs.create_eval_input_fn(
        eval_config=configs['eval_config'],
        eval_input_config=configs['eval_input_config'],
        model_config=configs['eval_config'])  # Expecting `DetectionModel`.
    with self.assertRaises(TypeError):
      eval_input_fn()


class DataAugmentationFnTest(tf.test.TestCase):

  def test_apply_image_and_box_augmentation(self):
    data_augmentation_options = [
        (preprocessor.resize_image, {
            'new_height': 20,
            'new_width': 20,
            'method': tf.image.ResizeMethod.NEAREST_NEIGHBOR
        }),
        (preprocessor.scale_boxes_to_pixel_coordinates, {}),
    ]
    data_augmentation_fn = functools.partial(
        inputs.augment_input_data,
        data_augmentation_options=data_augmentation_options)
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np.random.rand(10, 10, 3).astype(np.float32)),
        fields.InputDataFields.groundtruth_boxes:
            tf.constant(np.array([[.5, .5, 1., 1.]], np.float32))
    }
    augmented_tensor_dict = data_augmentation_fn(tensor_dict=tensor_dict)
    with self.test_session() as sess:
      augmented_tensor_dict_out = sess.run(augmented_tensor_dict)

    self.assertAllEqual(
        augmented_tensor_dict_out[fields.InputDataFields.image].shape,
        [20, 20, 3]
    )
    self.assertAllClose(
        augmented_tensor_dict_out[fields.InputDataFields.groundtruth_boxes],
        [[10, 10, 20, 20]]
    )

  def test_include_masks_in_data_augmentation(self):
    data_augmentation_options = [
        (preprocessor.resize_image, {
            'new_height': 20,
            'new_width': 20,
            'method': tf.image.ResizeMethod.NEAREST_NEIGHBOR
        })
    ]
    data_augmentation_fn = functools.partial(
        inputs.augment_input_data,
        data_augmentation_options=data_augmentation_options)
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np.random.rand(10, 10, 3).astype(np.float32)),
        fields.InputDataFields.groundtruth_instance_masks:
            tf.constant(np.zeros([2, 10, 10], np.uint8))
    }
    augmented_tensor_dict = data_augmentation_fn(tensor_dict=tensor_dict)
    with self.test_session() as sess:
      augmented_tensor_dict_out = sess.run(augmented_tensor_dict)

    self.assertAllEqual(
        augmented_tensor_dict_out[fields.InputDataFields.image].shape,
        [20, 20, 3])
    self.assertAllEqual(augmented_tensor_dict_out[
        fields.InputDataFields.groundtruth_instance_masks].shape, [2, 20, 20])

  def test_include_keypoints_in_data_augmentation(self):
    data_augmentation_options = [
        (preprocessor.resize_image, {
            'new_height': 20,
            'new_width': 20,
            'method': tf.image.ResizeMethod.NEAREST_NEIGHBOR
        }),
        (preprocessor.scale_boxes_to_pixel_coordinates, {}),
    ]
    data_augmentation_fn = functools.partial(
        inputs.augment_input_data,
        data_augmentation_options=data_augmentation_options)
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np.random.rand(10, 10, 3).astype(np.float32)),
        fields.InputDataFields.groundtruth_boxes:
            tf.constant(np.array([[.5, .5, 1., 1.]], np.float32)),
        fields.InputDataFields.groundtruth_keypoints:
            tf.constant(np.array([[[0.5, 1.0], [0.5, 0.5]]], np.float32))
    }
    augmented_tensor_dict = data_augmentation_fn(tensor_dict=tensor_dict)
    with self.test_session() as sess:
      augmented_tensor_dict_out = sess.run(augmented_tensor_dict)

    self.assertAllEqual(
        augmented_tensor_dict_out[fields.InputDataFields.image].shape,
        [20, 20, 3]
    )
    self.assertAllClose(
        augmented_tensor_dict_out[fields.InputDataFields.groundtruth_boxes],
        [[10, 10, 20, 20]]
    )
    self.assertAllClose(
        augmented_tensor_dict_out[fields.InputDataFields.groundtruth_keypoints],
        [[[10, 20], [10, 10]]]
    )


def _fake_model_preprocessor_fn(image):
  return (image, tf.expand_dims(tf.shape(image)[1:], axis=0))


def _fake_image_resizer_fn(image, mask):
  return (image, mask, tf.shape(image))


class DataTransformationFnTest(tf.test.TestCase):

  def test_combine_additional_channels_if_present(self):
    image = np.random.rand(4, 4, 3).astype(np.float32)
    additional_channels = np.random.rand(4, 4, 2).astype(np.float32)
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(image),
        fields.InputDataFields.image_additional_channels:
            tf.constant(additional_channels),
        fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([1, 1], np.int32))
    }

    input_transformation_fn = functools.partial(
        inputs.transform_input_data,
        model_preprocess_fn=_fake_model_preprocessor_fn,
        image_resizer_fn=_fake_image_resizer_fn,
        num_classes=1)
    with self.test_session() as sess:
      transformed_inputs = sess.run(
          input_transformation_fn(tensor_dict=tensor_dict))
    self.assertAllEqual(transformed_inputs[fields.InputDataFields.image].dtype,
                        tf.float32)
    self.assertAllEqual(transformed_inputs[fields.InputDataFields.image].shape,
                        [4, 4, 5])
    self.assertAllClose(transformed_inputs[fields.InputDataFields.image],
                        np.concatenate((image, additional_channels), axis=2))

  def test_returns_correct_class_label_encodings(self):
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np.random.rand(4, 4, 3).astype(np.float32)),
        fields.InputDataFields.groundtruth_boxes:
            tf.constant(np.array([[0, 0, 1, 1], [.5, .5, 1, 1]], np.float32)),
        fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([3, 1], np.int32))
    }
    num_classes = 3
    input_transformation_fn = functools.partial(
        inputs.transform_input_data,
        model_preprocess_fn=_fake_model_preprocessor_fn,
        image_resizer_fn=_fake_image_resizer_fn,
        num_classes=num_classes)
    with self.test_session() as sess:
      transformed_inputs = sess.run(
          input_transformation_fn(tensor_dict=tensor_dict))

    self.assertAllClose(
        transformed_inputs[fields.InputDataFields.groundtruth_classes],
        [[0, 0, 1], [1, 0, 0]])

  def test_returns_correct_merged_boxes(self):
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np.random.rand(4, 4, 3).astype(np.float32)),
        fields.InputDataFields.groundtruth_boxes:
            tf.constant(np.array([[.5, .5, 1, 1], [.5, .5, 1, 1]], np.float32)),
        fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([3, 1], np.int32))
    }

    num_classes = 3
    input_transformation_fn = functools.partial(
        inputs.transform_input_data,
        model_preprocess_fn=_fake_model_preprocessor_fn,
        image_resizer_fn=_fake_image_resizer_fn,
        num_classes=num_classes,
        merge_multiple_boxes=True)

    with self.test_session() as sess:
      transformed_inputs = sess.run(
          input_transformation_fn(tensor_dict=tensor_dict))
    self.assertAllClose(
        transformed_inputs[fields.InputDataFields.groundtruth_boxes],
        [[.5, .5, 1., 1.]])
    self.assertAllClose(
        transformed_inputs[fields.InputDataFields.groundtruth_classes],
        [[1, 0, 1]])

  def test_returns_resized_masks(self):
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np.random.rand(4, 4, 3).astype(np.float32)),
        fields.InputDataFields.groundtruth_instance_masks:
            tf.constant(np.random.rand(2, 4, 4).astype(np.float32)),
        fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([3, 1], np.int32))
    }
    def fake_image_resizer_fn(image, masks=None):
      resized_image = tf.image.resize_images(image, [8, 8])
      results = [resized_image]
      if masks is not None:
        resized_masks = tf.transpose(
            tf.image.resize_images(tf.transpose(masks, [1, 2, 0]), [8, 8]),
            [2, 0, 1])
        results.append(resized_masks)
      results.append(tf.shape(resized_image))
      return results

    num_classes = 3
    input_transformation_fn = functools.partial(
        inputs.transform_input_data,
        model_preprocess_fn=_fake_model_preprocessor_fn,
        image_resizer_fn=fake_image_resizer_fn,
        num_classes=num_classes,
        retain_original_image=True)
    with self.test_session() as sess:
      transformed_inputs = sess.run(
          input_transformation_fn(tensor_dict=tensor_dict))
    self.assertAllEqual(transformed_inputs[
        fields.InputDataFields.original_image].dtype, tf.uint8)
    self.assertAllEqual(transformed_inputs[
        fields.InputDataFields.original_image].shape, [4, 4, 3])
    self.assertAllEqual(transformed_inputs[
        fields.InputDataFields.groundtruth_instance_masks].shape, [2, 8, 8])

  def test_applies_model_preprocess_fn_to_image_tensor(self):
    np_image = np.random.randint(256, size=(4, 4, 3))
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np_image),
        fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([3, 1], np.int32))
    }
    def fake_model_preprocessor_fn(image):
      return (image / 255., tf.expand_dims(tf.shape(image)[1:], axis=0))

    num_classes = 3
    input_transformation_fn = functools.partial(
        inputs.transform_input_data,
        model_preprocess_fn=fake_model_preprocessor_fn,
        image_resizer_fn=_fake_image_resizer_fn,
        num_classes=num_classes)

    with self.test_session() as sess:
      transformed_inputs = sess.run(
          input_transformation_fn(tensor_dict=tensor_dict))
    self.assertAllClose(transformed_inputs[fields.InputDataFields.image],
                        np_image / 255.)
    self.assertAllClose(transformed_inputs[fields.InputDataFields.
                                           true_image_shape],
                        [4, 4, 3])

  def test_applies_data_augmentation_fn_to_tensor_dict(self):
    np_image = np.random.randint(256, size=(4, 4, 3))
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np_image),
        fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([3, 1], np.int32))
    }
    def add_one_data_augmentation_fn(tensor_dict):
      return {key: value + 1 for key, value in tensor_dict.items()}

    num_classes = 4
    input_transformation_fn = functools.partial(
        inputs.transform_input_data,
        model_preprocess_fn=_fake_model_preprocessor_fn,
        image_resizer_fn=_fake_image_resizer_fn,
        num_classes=num_classes,
        data_augmentation_fn=add_one_data_augmentation_fn)
    with self.test_session() as sess:
      augmented_tensor_dict = sess.run(
          input_transformation_fn(tensor_dict=tensor_dict))

    self.assertAllEqual(augmented_tensor_dict[fields.InputDataFields.image],
                        np_image + 1)
    self.assertAllEqual(
        augmented_tensor_dict[fields.InputDataFields.groundtruth_classes],
        [[0, 0, 0, 1], [0, 1, 0, 0]])

  def test_applies_data_augmentation_fn_before_model_preprocess_fn(self):
    np_image = np.random.randint(256, size=(4, 4, 3))
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np_image),
        fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([3, 1], np.int32))
    }
    def mul_two_model_preprocessor_fn(image):
      return (image * 2, tf.expand_dims(tf.shape(image)[1:], axis=0))
    def add_five_to_image_data_augmentation_fn(tensor_dict):
      tensor_dict[fields.InputDataFields.image] += 5
      return tensor_dict

    num_classes = 4
    input_transformation_fn = functools.partial(
        inputs.transform_input_data,
        model_preprocess_fn=mul_two_model_preprocessor_fn,
        image_resizer_fn=_fake_image_resizer_fn,
        num_classes=num_classes,
        data_augmentation_fn=add_five_to_image_data_augmentation_fn)
    with self.test_session() as sess:
      augmented_tensor_dict = sess.run(
          input_transformation_fn(tensor_dict=tensor_dict))

    self.assertAllEqual(augmented_tensor_dict[fields.InputDataFields.image],
                        (np_image + 5) * 2)


if __name__ == '__main__':
  tf.test.main()

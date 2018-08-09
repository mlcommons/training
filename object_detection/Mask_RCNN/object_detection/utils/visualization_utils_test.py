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

"""Tests for object_detection.utils.visualization_utils."""
import logging
import os

import numpy as np
import PIL.Image as Image
import tensorflow as tf

from object_detection.utils import visualization_utils

_TESTDATA_PATH = 'object_detection/test_images'


class VisualizationUtilsTest(tf.test.TestCase):

  def create_colorful_test_image(self):
    """This function creates an image that can be used to test vis functions.

    It makes an image composed of four colored rectangles.

    Returns:
      colorful test numpy array image.
    """
    ch255 = np.full([100, 200, 1], 255, dtype=np.uint8)
    ch128 = np.full([100, 200, 1], 128, dtype=np.uint8)
    ch0 = np.full([100, 200, 1], 0, dtype=np.uint8)
    imr = np.concatenate((ch255, ch128, ch128), axis=2)
    img = np.concatenate((ch255, ch255, ch0), axis=2)
    imb = np.concatenate((ch255, ch0, ch255), axis=2)
    imw = np.concatenate((ch128, ch128, ch128), axis=2)
    imu = np.concatenate((imr, img), axis=1)
    imd = np.concatenate((imb, imw), axis=1)
    image = np.concatenate((imu, imd), axis=0)
    return image

  def create_test_image_with_five_channels(self):
    return np.full([100, 200, 5], 255, dtype=np.uint8)

  def test_draw_bounding_box_on_image(self):
    test_image = self.create_colorful_test_image()
    test_image = Image.fromarray(test_image)
    width_original, height_original = test_image.size
    ymin = 0.25
    ymax = 0.75
    xmin = 0.4
    xmax = 0.6

    visualization_utils.draw_bounding_box_on_image(test_image, ymin, xmin, ymax,
                                                   xmax)
    width_final, height_final = test_image.size

    self.assertEqual(width_original, width_final)
    self.assertEqual(height_original, height_final)

  def test_draw_bounding_box_on_image_array(self):
    test_image = self.create_colorful_test_image()
    width_original = test_image.shape[0]
    height_original = test_image.shape[1]
    ymin = 0.25
    ymax = 0.75
    xmin = 0.4
    xmax = 0.6

    visualization_utils.draw_bounding_box_on_image_array(
        test_image, ymin, xmin, ymax, xmax)
    width_final = test_image.shape[0]
    height_final = test_image.shape[1]

    self.assertEqual(width_original, width_final)
    self.assertEqual(height_original, height_final)

  def test_draw_bounding_boxes_on_image(self):
    test_image = self.create_colorful_test_image()
    test_image = Image.fromarray(test_image)
    width_original, height_original = test_image.size
    boxes = np.array([[0.25, 0.75, 0.4, 0.6],
                      [0.1, 0.1, 0.9, 0.9]])

    visualization_utils.draw_bounding_boxes_on_image(test_image, boxes)
    width_final, height_final = test_image.size

    self.assertEqual(width_original, width_final)
    self.assertEqual(height_original, height_final)

  def test_draw_bounding_boxes_on_image_array(self):
    test_image = self.create_colorful_test_image()
    width_original = test_image.shape[0]
    height_original = test_image.shape[1]
    boxes = np.array([[0.25, 0.75, 0.4, 0.6],
                      [0.1, 0.1, 0.9, 0.9]])

    visualization_utils.draw_bounding_boxes_on_image_array(test_image, boxes)
    width_final = test_image.shape[0]
    height_final = test_image.shape[1]

    self.assertEqual(width_original, width_final)
    self.assertEqual(height_original, height_final)

  def test_draw_bounding_boxes_on_image_tensors(self):
    """Tests that bounding box utility produces reasonable results."""
    category_index = {1: {'id': 1, 'name': 'dog'}, 2: {'id': 2, 'name': 'cat'}}

    fname = os.path.join(_TESTDATA_PATH, 'image1.jpg')
    image_np = np.array(Image.open(fname))
    images_np = np.stack((image_np, image_np), axis=0)

    with tf.Graph().as_default():
      images_tensor = tf.constant(value=images_np, dtype=tf.uint8)
      boxes = tf.constant([[[0.4, 0.25, 0.75, 0.75], [0.5, 0.3, 0.6, 0.9]],
                           [[0.25, 0.25, 0.75, 0.75], [0.1, 0.3, 0.6, 1.0]]])
      classes = tf.constant([[1, 1], [1, 2]], dtype=tf.int64)
      scores = tf.constant([[0.8, 0.1], [0.6, 0.5]])
      images_with_boxes = (
          visualization_utils.draw_bounding_boxes_on_image_tensors(
              images_tensor,
              boxes,
              classes,
              scores,
              category_index,
              min_score_thresh=0.2))

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())

        # Write output images for visualization.
        images_with_boxes_np = sess.run(images_with_boxes)
        self.assertEqual(images_np.shape, images_with_boxes_np.shape)
        for i in range(images_with_boxes_np.shape[0]):
          img_name = 'image_' + str(i) + '.png'
          output_file = os.path.join(self.get_temp_dir(), img_name)
          logging.info('Writing output image %d to %s', i, output_file)
          image_pil = Image.fromarray(images_with_boxes_np[i, ...])
          image_pil.save(output_file)

  def test_draw_bounding_boxes_on_image_tensors_with_additional_channels(self):
    """Tests the case where input image tensor has more than 3 channels."""
    category_index = {1: {'id': 1, 'name': 'dog'}}
    image_np = self.create_test_image_with_five_channels()
    images_np = np.stack((image_np, image_np), axis=0)

    with tf.Graph().as_default():
      images_tensor = tf.constant(value=images_np, dtype=tf.uint8)
      boxes = tf.constant(0, dtype=tf.float32, shape=[2, 0, 4])
      classes = tf.constant(0, dtype=tf.int64, shape=[2, 0])
      scores = tf.constant(0, dtype=tf.float32, shape=[2, 0])
      images_with_boxes = (
          visualization_utils.draw_bounding_boxes_on_image_tensors(
              images_tensor,
              boxes,
              classes,
              scores,
              category_index,
              min_score_thresh=0.2))

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())

        final_images_np = sess.run(images_with_boxes)
        self.assertEqual((2, 100, 200, 3), final_images_np.shape)

  def test_draw_keypoints_on_image(self):
    test_image = self.create_colorful_test_image()
    test_image = Image.fromarray(test_image)
    width_original, height_original = test_image.size
    keypoints = [[0.25, 0.75], [0.4, 0.6], [0.1, 0.1], [0.9, 0.9]]

    visualization_utils.draw_keypoints_on_image(test_image, keypoints)
    width_final, height_final = test_image.size

    self.assertEqual(width_original, width_final)
    self.assertEqual(height_original, height_final)

  def test_draw_keypoints_on_image_array(self):
    test_image = self.create_colorful_test_image()
    width_original = test_image.shape[0]
    height_original = test_image.shape[1]
    keypoints = [[0.25, 0.75], [0.4, 0.6], [0.1, 0.1], [0.9, 0.9]]

    visualization_utils.draw_keypoints_on_image_array(test_image, keypoints)
    width_final = test_image.shape[0]
    height_final = test_image.shape[1]

    self.assertEqual(width_original, width_final)
    self.assertEqual(height_original, height_final)

  def test_draw_mask_on_image_array(self):
    test_image = np.asarray([[[0, 0, 0], [0, 0, 0]],
                             [[0, 0, 0], [0, 0, 0]]], dtype=np.uint8)
    mask = np.asarray([[0, 1],
                       [1, 1]], dtype=np.uint8)
    expected_result = np.asarray([[[0, 0, 0], [0, 0, 127]],
                                  [[0, 0, 127], [0, 0, 127]]], dtype=np.uint8)
    visualization_utils.draw_mask_on_image_array(test_image, mask,
                                                 color='Blue', alpha=.5)
    self.assertAllEqual(test_image, expected_result)

  def test_add_cdf_image_summary(self):
    values = [0.1, 0.2, 0.3, 0.4, 0.42, 0.44, 0.46, 0.48, 0.50]
    visualization_utils.add_cdf_image_summary(values, 'PositiveAnchorLoss')
    cdf_image_summary = tf.get_collection(key=tf.GraphKeys.SUMMARIES)[0]
    with self.test_session():
      cdf_image_summary.eval()

  def test_add_hist_image_summary(self):
    values = [0.1, 0.2, 0.3, 0.4, 0.42, 0.44, 0.46, 0.48, 0.50]
    bins = [0.01 * i for i in range(101)]
    visualization_utils.add_hist_image_summary(values, bins,
                                               'ScoresDistribution')
    hist_image_summary = tf.get_collection(key=tf.GraphKeys.SUMMARIES)[0]
    with self.test_session():
      hist_image_summary.eval()


if __name__ == '__main__':
  tf.test.main()

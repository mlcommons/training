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

"""Faster R-CNN meta-architecture definition.

General tensorflow implementation of Faster R-CNN detection models.

See Faster R-CNN: Ren, Shaoqing, et al.
"Faster R-CNN: Towards real-time object detection with region proposal
networks." Advances in neural information processing systems. 2015.

We allow for three modes: number_of_stages={1, 2, 3}. In case of 1 stage,
all of the user facing methods (e.g., predict, postprocess, loss) can be used as
if the model consisted only of the RPN, returning class agnostic proposals
(these can be thought of as approximate detections with no associated class
information).  In case of 2 stages, proposals are computed, then passed
through a second stage "box classifier" to yield (multi-class) detections.
Finally, in case of 3 stages which is only used during eval, proposals are
computed, then passed through a second stage "box classifier" that will compute
refined boxes and classes, and then features are pooled from the refined and
non-maximum suppressed boxes and are passed through the box classifier again. If
number of stages is 3 during training it will be reduced to two automatically.

Implementations of Faster R-CNN models must define a new
FasterRCNNFeatureExtractor and override three methods: `preprocess`,
`_extract_proposal_features` (the first stage of the model), and
`_extract_box_classifier_features` (the second stage of the model). Optionally,
the `restore_fn` method can be overridden.  See tests for an example.

A few important notes:
+ Batching conventions:  We support batched inference and training where
all images within a batch have the same resolution.  Batch sizes are determined
dynamically via the shape of the input tensors (rather than being specified
directly as, e.g., a model constructor).

A complication is that due to non-max suppression, we are not guaranteed to get
the same number of proposals from the first stage RPN (region proposal network)
for each image (though in practice, we should often get the same number of
proposals).  For this reason we pad to a max number of proposals per image
within a batch. This `self.max_num_proposals` property is set to the
`first_stage_max_proposals` parameter at inference time and the
`second_stage_batch_size` at training time since we subsample the batch to
be sent through the box classifier during training.

For the second stage of the pipeline, we arrange the proposals for all images
within the batch along a single batch dimension.  For example, the input to
_extract_box_classifier_features is a tensor of shape
`[total_num_proposals, crop_height, crop_width, depth]` where
total_num_proposals is batch_size * self.max_num_proposals.  (And note that per
the above comment, a subset of these entries correspond to zero paddings.)

+ Coordinate representations:
Following the API (see model.DetectionModel definition), our outputs after
postprocessing operations are always normalized boxes however, internally, we
sometimes convert to absolute --- e.g. for loss computation.  In particular,
anchors and proposal_boxes are both represented as absolute coordinates.

Images are resized in the `preprocess` method.

The Faster R-CNN meta architecture has two post-processing methods
`_postprocess_rpn` which is applied after first stage and
`_postprocess_box_classifier` which is applied after second stage. There are
three different ways post-processing can happen depending on number_of_stages
configured in the meta architecture:

1. When number_of_stages is 1:
  `_postprocess_rpn` is run as part of the `postprocess` method where
  true_image_shapes is used to clip proposals, perform non-max suppression and
  normalize them.
2. When number of stages is 2:
  `_postprocess_rpn` is run as part of the `_predict_second_stage` method where
  `resized_image_shapes` is used to clip proposals, perform non-max suppression
  and normalize them. In this case `postprocess` method skips `_postprocess_rpn`
  and only runs `_postprocess_box_classifier` using `true_image_shapes` to clip
  detections, perform non-max suppression and normalize them.
3. When number of stages is 3:
  `_postprocess_rpn` is run as part of the `_predict_second_stage` using
  `resized_image_shapes` to clip proposals, perform non-max suppression and
  normalize them. Subsequently, `_postprocess_box_classifier` is run as part of
  `_predict_third_stage` using `true_image_shapes` to clip detections, peform
  non-max suppression and normalize them. In this case, the `postprocess` method
  skips both `_postprocess_rpn` and `_postprocess_box_classifier`.
"""
from abc import abstractmethod
from functools import partial
import tensorflow as tf

from object_detection.anchor_generators import grid_anchor_generator
from object_detection.core import balanced_positive_negative_sampler as sampler
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import box_predictor
from object_detection.core import losses
from object_detection.core import model
from object_detection.core import post_processing
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner
from object_detection.utils import ops
from object_detection.utils import shape_utils

slim = tf.contrib.slim


class FasterRCNNFeatureExtractor(object):
  """Faster R-CNN Feature Extractor definition."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               batch_norm_trainable=False,
               reuse_weights=None,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: A boolean indicating whether the training version of the
        computation graph should be constructed.
      first_stage_features_stride: Output stride of extracted RPN feature map.
      batch_norm_trainable: Whether to update batch norm parameters during
        training or not. When training with a relative large batch size
        (e.g. 8), it could be desirable to enable batch norm update.
      reuse_weights: Whether to reuse variables. Default is None.
      weight_decay: float weight decay for feature extractor (default: 0.0).
    """
    self._is_training = is_training
    self._first_stage_features_stride = first_stage_features_stride
    self._train_batch_norm = (batch_norm_trainable and is_training)
    self._reuse_weights = reuse_weights
    self._weight_decay = weight_decay

  @abstractmethod
  def preprocess(self, resized_inputs):
    """Feature-extractor specific preprocessing (minus image resizing)."""
    pass

  def extract_proposal_features(self, preprocessed_inputs, scope):
    """Extracts first stage RPN features.

    This function is responsible for extracting feature maps from preprocessed
    images.  These features are used by the region proposal network (RPN) to
    predict proposals.

    Args:
      preprocessed_inputs: A [batch, height, width, channels] float tensor
        representing a batch of images.
      scope: A scope name.

    Returns:
      rpn_feature_map: A tensor with shape [batch, height, width, depth]
      activations: A dictionary mapping activation tensor names to tensors.
    """
    with tf.variable_scope(scope, values=[preprocessed_inputs]):
      return self._extract_proposal_features(preprocessed_inputs, scope)

  @abstractmethod
  def _extract_proposal_features(self, preprocessed_inputs, scope):
    """Extracts first stage RPN features, to be overridden."""
    pass

  def extract_box_classifier_features(self, proposal_feature_maps, scope):
    """Extracts second stage box classifier features.

    Args:
      proposal_feature_maps: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
        representing the feature map cropped to each proposal.
      scope: A scope name.

    Returns:
      proposal_classifier_features: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, height, width, depth]
        representing box classifier features for each proposal.
    """
    with tf.variable_scope(
        scope, values=[proposal_feature_maps], reuse=tf.AUTO_REUSE):
      return self._extract_box_classifier_features(proposal_feature_maps, scope)

  @abstractmethod
  def _extract_box_classifier_features(self, proposal_feature_maps, scope):
    """Extracts second stage box classifier features, to be overridden."""
    pass

  def restore_from_classification_checkpoint_fn(
      self,
      first_stage_feature_extractor_scope,
      second_stage_feature_extractor_scope):
    """Returns a map of variables to load from a foreign checkpoint.

    Args:
      first_stage_feature_extractor_scope: A scope name for the first stage
        feature extractor.
      second_stage_feature_extractor_scope: A scope name for the second stage
        feature extractor.

    Returns:
      A dict mapping variable names (to load from a checkpoint) to variables in
      the model graph.
    """
    variables_to_restore = {}
    for variable in tf.global_variables():
      for scope_name in [first_stage_feature_extractor_scope,
                         second_stage_feature_extractor_scope]:
        if variable.op.name.startswith(scope_name):
          var_name = variable.op.name.replace(scope_name + '/', '')
          variables_to_restore[var_name] = variable
    return variables_to_restore


class FasterRCNNMetaArch(model.DetectionModel):
  """Faster R-CNN Meta-architecture definition."""

  def __init__(self,
               is_training,
               num_classes,
               image_resizer_fn,
               feature_extractor,
               number_of_stages,
               first_stage_anchor_generator,
               first_stage_atrous_rate,
               first_stage_box_predictor_arg_scope_fn,
               first_stage_box_predictor_kernel_size,
               first_stage_box_predictor_depth,
               first_stage_minibatch_size,
               first_stage_positive_balance_fraction,
               first_stage_nms_score_threshold,
               first_stage_nms_iou_threshold,
               first_stage_max_proposals,
               first_stage_localization_loss_weight,
               first_stage_objectness_loss_weight,
               initial_crop_size,
               maxpool_kernel_size,
               maxpool_stride,
               second_stage_mask_rcnn_box_predictor,
               second_stage_batch_size,
               second_stage_balance_fraction,
               second_stage_non_max_suppression_fn,
               second_stage_score_conversion_fn,
               second_stage_localization_loss_weight,
               second_stage_classification_loss_weight,
               second_stage_classification_loss,
               second_stage_mask_prediction_loss_weight=1.0,
               hard_example_miner=None,
               parallel_iterations=16,
               add_summaries=True):
    """FasterRCNNMetaArch Constructor.

    Args:
      is_training: A boolean indicating whether the training version of the
        computation graph should be constructed.
      num_classes: Number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      image_resizer_fn: A callable for image resizing.  This callable
        takes a rank-3 image tensor of shape [height, width, channels]
        (corresponding to a single image), an optional rank-3 instance mask
        tensor of shape [num_masks, height, width] and returns a resized rank-3
        image tensor, a resized mask tensor if one was provided in the input. In
        addition this callable must also return a 1-D tensor of the form
        [height, width, channels] containing the size of the true image, as the
        image resizer can perform zero padding. See protos/image_resizer.proto.
      feature_extractor: A FasterRCNNFeatureExtractor object.
      number_of_stages:  An integer values taking values in {1, 2, 3}. If
        1, the function will construct only the Region Proposal Network (RPN)
        part of the model. If 2, the function will perform box refinement and
        other auxiliary predictions all in the second stage. If 3, it will
        extract features from refined boxes and perform the auxiliary
        predictions on the non-maximum suppressed refined boxes.
        If is_training is true and the value of number_of_stages is 3, it is
        reduced to 2 since all the model heads are trained in parallel in second
        stage during training.
      first_stage_anchor_generator: An anchor_generator.AnchorGenerator object
        (note that currently we only support
        grid_anchor_generator.GridAnchorGenerator objects)
      first_stage_atrous_rate: A single integer indicating the atrous rate for
        the single convolution op which is applied to the `rpn_features_to_crop`
        tensor to obtain a tensor to be used for box prediction. Some feature
        extractors optionally allow for producing feature maps computed at
        denser resolutions.  The atrous rate is used to compensate for the
        denser feature maps by using an effectively larger receptive field.
        (This should typically be set to 1).
      first_stage_box_predictor_arg_scope_fn: A function to construct tf-slim
        arg_scope for conv2d, separable_conv2d and fully_connected ops for the
        RPN box predictor.
      first_stage_box_predictor_kernel_size: Kernel size to use for the
        convolution op just prior to RPN box predictions.
      first_stage_box_predictor_depth: Output depth for the convolution op
        just prior to RPN box predictions.
      first_stage_minibatch_size: The "batch size" to use for computing the
        objectness and location loss of the region proposal network. This
        "batch size" refers to the number of anchors selected as contributing
        to the loss function for any given image within the image batch and is
        only called "batch_size" due to terminology from the Faster R-CNN paper.
      first_stage_positive_balance_fraction: Fraction of positive examples
        per image for the RPN. The recommended value for Faster RCNN is 0.5.
      first_stage_nms_score_threshold: Score threshold for non max suppression
        for the Region Proposal Network (RPN).  This value is expected to be in
        [0, 1] as it is applied directly after a softmax transformation.  The
        recommended value for Faster R-CNN is 0.
      first_stage_nms_iou_threshold: The Intersection Over Union (IOU) threshold
        for performing Non-Max Suppression (NMS) on the boxes predicted by the
        Region Proposal Network (RPN).
      first_stage_max_proposals: Maximum number of boxes to retain after
        performing Non-Max Suppression (NMS) on the boxes predicted by the
        Region Proposal Network (RPN).
      first_stage_localization_loss_weight: A float
      first_stage_objectness_loss_weight: A float
      initial_crop_size: A single integer indicating the output size
        (width and height are set to be the same) of the initial bilinear
        interpolation based cropping during ROI pooling.
      maxpool_kernel_size: A single integer indicating the kernel size of the
        max pool op on the cropped feature map during ROI pooling.
      maxpool_stride: A single integer indicating the stride of the max pool
        op on the cropped feature map during ROI pooling.
      second_stage_mask_rcnn_box_predictor: Mask R-CNN box predictor to use for
        the second stage.
      second_stage_batch_size: The batch size used for computing the
        classification and refined location loss of the box classifier.  This
        "batch size" refers to the number of proposals selected as contributing
        to the loss function for any given image within the image batch and is
        only called "batch_size" due to terminology from the Faster R-CNN paper.
      second_stage_balance_fraction: Fraction of positive examples to use
        per image for the box classifier. The recommended value for Faster RCNN
        is 0.25.
      second_stage_non_max_suppression_fn: batch_multiclass_non_max_suppression
        callable that takes `boxes`, `scores`, optional `clip_window` and
        optional (kwarg) `mask` inputs (with all other inputs already set)
        and returns a dictionary containing tensors with keys:
        `detection_boxes`, `detection_scores`, `detection_classes`,
        `num_detections`, and (optionally) `detection_masks`. See
        `post_processing.batch_multiclass_non_max_suppression` for the type and
        shape of these tensors.
      second_stage_score_conversion_fn: Callable elementwise nonlinearity
        (that takes tensors as inputs and returns tensors).  This is usually
        used to convert logits to probabilities.
      second_stage_localization_loss_weight: A float indicating the scale factor
        for second stage localization loss.
      second_stage_classification_loss_weight: A float indicating the scale
        factor for second stage classification loss.
      second_stage_classification_loss: Classification loss used by the second
        stage classifier. Either losses.WeightedSigmoidClassificationLoss or
        losses.WeightedSoftmaxClassificationLoss.
      second_stage_mask_prediction_loss_weight: A float indicating the scale
        factor for second stage mask prediction loss. This is applicable only if
        second stage box predictor is configured to predict masks.
      hard_example_miner:  A losses.HardExampleMiner object (can be None).
      parallel_iterations: (Optional) The number of iterations allowed to run
        in parallel for calls to tf.map_fn.
      add_summaries: boolean (default: True) controlling whether summary ops
        should be added to tensorflow graph.

    Raises:
      ValueError: If `second_stage_batch_size` > `first_stage_max_proposals` at
        training time.
      ValueError: If first_stage_anchor_generator is not of type
        grid_anchor_generator.GridAnchorGenerator.
    """
    # TODO(rathodv): add_summaries is currently unused. Respect that directive
    # in the future.
    super(FasterRCNNMetaArch, self).__init__(num_classes=num_classes)

    if is_training and second_stage_batch_size > first_stage_max_proposals:
      raise ValueError('second_stage_batch_size should be no greater than '
                       'first_stage_max_proposals.')
    if not isinstance(first_stage_anchor_generator,
                      grid_anchor_generator.GridAnchorGenerator):
      raise ValueError('first_stage_anchor_generator must be of type '
                       'grid_anchor_generator.GridAnchorGenerator.')

    self._is_training = is_training
    self._image_resizer_fn = image_resizer_fn
    self._feature_extractor = feature_extractor
    self._number_of_stages = number_of_stages

    # The first class is reserved as background.
    unmatched_cls_target = tf.constant(
        [1] + self._num_classes * [0], dtype=tf.float32)
    self._proposal_target_assigner = target_assigner.create_target_assigner(
        'FasterRCNN', 'proposal')
    self._detector_target_assigner = target_assigner.create_target_assigner(
        'FasterRCNN', 'detection', unmatched_cls_target=unmatched_cls_target)
    # Both proposal and detector target assigners use the same box coder
    self._box_coder = self._proposal_target_assigner.box_coder

    # (First stage) Region proposal network parameters
    self._first_stage_anchor_generator = first_stage_anchor_generator
    self._first_stage_atrous_rate = first_stage_atrous_rate
    self._first_stage_box_predictor_arg_scope_fn = (
        first_stage_box_predictor_arg_scope_fn)
    self._first_stage_box_predictor_kernel_size = (
        first_stage_box_predictor_kernel_size)
    self._first_stage_box_predictor_depth = first_stage_box_predictor_depth
    self._first_stage_minibatch_size = first_stage_minibatch_size
    self._first_stage_sampler = sampler.BalancedPositiveNegativeSampler(
        positive_fraction=first_stage_positive_balance_fraction)
    self._first_stage_box_predictor = box_predictor.ConvolutionalBoxPredictor(
        self._is_training, num_classes=1,
        conv_hyperparams_fn=self._first_stage_box_predictor_arg_scope_fn,
        min_depth=0, max_depth=0, num_layers_before_predictor=0,
        use_dropout=False, dropout_keep_prob=1.0, kernel_size=1,
        box_code_size=self._box_coder.code_size)

    self._first_stage_nms_score_threshold = first_stage_nms_score_threshold
    self._first_stage_nms_iou_threshold = first_stage_nms_iou_threshold
    self._first_stage_max_proposals = first_stage_max_proposals

    self._first_stage_localization_loss = (
        losses.WeightedSmoothL1LocalizationLoss())
    self._first_stage_objectness_loss = (
        losses.WeightedSoftmaxClassificationLoss())
    self._first_stage_loc_loss_weight = first_stage_localization_loss_weight
    self._first_stage_obj_loss_weight = first_stage_objectness_loss_weight

    # Per-region cropping parameters
    self._initial_crop_size = initial_crop_size
    self._maxpool_kernel_size = maxpool_kernel_size
    self._maxpool_stride = maxpool_stride

    self._mask_rcnn_box_predictor = second_stage_mask_rcnn_box_predictor

    self._second_stage_batch_size = second_stage_batch_size
    self._second_stage_sampler = sampler.BalancedPositiveNegativeSampler(
        positive_fraction=second_stage_balance_fraction)

    self._second_stage_nms_fn = second_stage_non_max_suppression_fn
    self._second_stage_score_conversion_fn = second_stage_score_conversion_fn

    self._second_stage_localization_loss = (
        losses.WeightedSmoothL1LocalizationLoss())
    self._second_stage_classification_loss = second_stage_classification_loss
    self._second_stage_mask_loss = (
        losses.WeightedSigmoidClassificationLoss())
    self._second_stage_loc_loss_weight = second_stage_localization_loss_weight
    self._second_stage_cls_loss_weight = second_stage_classification_loss_weight
    self._second_stage_mask_loss_weight = (
        second_stage_mask_prediction_loss_weight)
    self._hard_example_miner = hard_example_miner
    self._parallel_iterations = parallel_iterations

    if self._number_of_stages <= 0 or self._number_of_stages > 3:
      raise ValueError('Number of stages should be a value in {1, 2, 3}.')

  @property
  def first_stage_feature_extractor_scope(self):
    return 'FirstStageFeatureExtractor'

  @property
  def second_stage_feature_extractor_scope(self):
    return 'SecondStageFeatureExtractor'

  @property
  def first_stage_box_predictor_scope(self):
    return 'FirstStageBoxPredictor'

  @property
  def second_stage_box_predictor_scope(self):
    return 'SecondStageBoxPredictor'

  @property
  def max_num_proposals(self):
    """Max number of proposals (to pad to) for each image in the input batch.

    At training time, this is set to be the `second_stage_batch_size` if hard
    example miner is not configured, else it is set to
    `first_stage_max_proposals`. At inference time, this is always set to
    `first_stage_max_proposals`.

    Returns:
      A positive integer.
    """
    if self._is_training and not self._hard_example_miner:
      return self._second_stage_batch_size
    return self._first_stage_max_proposals

  @property
  def anchors(self):
    if not self._anchors:
      raise RuntimeError('anchors have not been constructed yet!')
    if not isinstance(self._anchors, box_list.BoxList):
      raise RuntimeError('anchors should be a BoxList object, but is not.')
    return self._anchors

  def preprocess(self, inputs):
    """Feature-extractor specific preprocessing.

    See base class.

    For Faster R-CNN, we perform image resizing in the base class --- each
    class subclassing FasterRCNNMetaArch is responsible for any additional
    preprocessing (e.g., scaling pixel values to be in [-1, 1]).

    Args:
      inputs: a [batch, height_in, width_in, channels] float tensor representing
        a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: a [batch, height_out, width_out, channels] float
        tensor representing a batch of images.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.
    Raises:
      ValueError: if inputs tensor does not have type tf.float32
    """
    if inputs.dtype is not tf.float32:
      raise ValueError('`preprocess` expects a tf.float32 tensor')
    with tf.name_scope('Preprocessor'):
      outputs = shape_utils.static_or_dynamic_map_fn(
          self._image_resizer_fn,
          elems=inputs,
          dtype=[tf.float32, tf.int32],
          parallel_iterations=self._parallel_iterations)
      resized_inputs = outputs[0]
      true_image_shapes = outputs[1]
      return (self._feature_extractor.preprocess(resized_inputs),
              true_image_shapes)

  def _compute_clip_window(self, image_shapes):
    """Computes clip window for non max suppression based on image shapes.

    This function assumes that the clip window's left top corner is at (0, 0).

    Args:
      image_shapes: A 2-D int32 tensor of shape [batch_size, 3] containing
      shapes of images in the batch. Each row represents [height, width,
      channels] of an image.

    Returns:
      A 2-D float32 tensor of shape [batch_size, 4] containing the clip window
      for each image in the form [ymin, xmin, ymax, xmax].
    """
    clip_heights = image_shapes[:, 0]
    clip_widths = image_shapes[:, 1]
    clip_window = tf.to_float(tf.stack([tf.zeros_like(clip_heights),
                                        tf.zeros_like(clip_heights),
                                        clip_heights, clip_widths], axis=1))
    return clip_window

  def predict(self, preprocessed_inputs, true_image_shapes):
    """Predicts unpostprocessed tensors from input tensor.

    This function takes an input batch of images and runs it through the
    forward pass of the network to yield "raw" un-postprocessed predictions.
    If `number_of_stages` is 1, this function only returns first stage
    RPN predictions (un-postprocessed).  Otherwise it returns both
    first stage RPN predictions as well as second stage box classifier
    predictions.

    Other remarks:
    + Anchor pruning vs. clipping: following the recommendation of the Faster
    R-CNN paper, we prune anchors that venture outside the image window at
    training time and clip anchors to the image window at inference time.
    + Proposal padding: as described at the top of the file, proposals are
    padded to self._max_num_proposals and flattened so that proposals from all
    images within the input batch are arranged along the same batch dimension.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Returns:
      prediction_dict: a dictionary holding "raw" prediction tensors:
        1) rpn_box_predictor_features: A 4-D float32 tensor with shape
          [batch_size, height, width, depth] to be used for predicting proposal
          boxes and corresponding objectness scores.
        2) rpn_features_to_crop: A 4-D float32 tensor with shape
          [batch_size, height, width, depth] representing image features to crop
          using the proposal boxes predicted by the RPN.
        3) image_shape: a 1-D tensor of shape [4] representing the input
          image shape.
        4) rpn_box_encodings:  3-D float tensor of shape
          [batch_size, num_anchors, self._box_coder.code_size] containing
          predicted boxes.
        5) rpn_objectness_predictions_with_background: 3-D float tensor of shape
          [batch_size, num_anchors, 2] containing class
          predictions (logits) for each of the anchors.  Note that this
          tensor *includes* background class predictions (at class index 0).
        6) anchors: A 2-D tensor of shape [num_anchors, 4] representing anchors
          for the first stage RPN (in absolute coordinates).  Note that
          `num_anchors` can differ depending on whether the model is created in
          training or inference mode.

        (and if number_of_stages > 1):
        7) refined_box_encodings: a 3-D tensor with shape
          [total_num_proposals, num_classes, self._box_coder.code_size]
          representing predicted (final) refined box encodings, where
          total_num_proposals=batch_size*self._max_num_proposals. If using
          a shared box across classes the shape will instead be
          [total_num_proposals, 1, self._box_coder.code_size].
        8) class_predictions_with_background: a 3-D tensor with shape
          [total_num_proposals, num_classes + 1] containing class
          predictions (logits) for each of the anchors, where
          total_num_proposals=batch_size*self._max_num_proposals.
          Note that this tensor *includes* background class predictions
          (at class index 0).
        9) num_proposals: An int32 tensor of shape [batch_size] representing the
          number of proposals generated by the RPN.  `num_proposals` allows us
          to keep track of which entries are to be treated as zero paddings and
          which are not since we always pad the number of proposals to be
          `self.max_num_proposals` for each image.
        10) proposal_boxes: A float32 tensor of shape
          [batch_size, self.max_num_proposals, 4] representing
          decoded proposal bounding boxes in absolute coordinates.
        11) mask_predictions: (optional) a 4-D tensor with shape
          [total_num_padded_proposals, num_classes, mask_height, mask_width]
          containing instance mask predictions.

    Raises:
      ValueError: If `predict` is called before `preprocess`.
    """
    (rpn_box_predictor_features, rpn_features_to_crop, anchors_boxlist,
     image_shape) = self._extract_rpn_feature_maps(preprocessed_inputs)
    (rpn_box_encodings, rpn_objectness_predictions_with_background
    ) = self._predict_rpn_proposals(rpn_box_predictor_features)

    # The Faster R-CNN paper recommends pruning anchors that venture outside
    # the image window at training time and clipping at inference time.
    clip_window = tf.to_float(tf.stack([0, 0, image_shape[1], image_shape[2]]))
    if self._is_training:
      (rpn_box_encodings, rpn_objectness_predictions_with_background,
       anchors_boxlist) = self._remove_invalid_anchors_and_predictions(
           rpn_box_encodings, rpn_objectness_predictions_with_background,
           anchors_boxlist, clip_window)
    else:
      anchors_boxlist = box_list_ops.clip_to_window(
          anchors_boxlist, clip_window)

    self._anchors = anchors_boxlist
    prediction_dict = {
        'rpn_box_predictor_features': rpn_box_predictor_features,
        'rpn_features_to_crop': rpn_features_to_crop,
        'image_shape': image_shape,
        'rpn_box_encodings': rpn_box_encodings,
        'rpn_objectness_predictions_with_background':
        rpn_objectness_predictions_with_background,
        'anchors': self._anchors.get()
    }

    if self._number_of_stages >= 2:
      prediction_dict.update(self._predict_second_stage(
          rpn_box_encodings,
          rpn_objectness_predictions_with_background,
          rpn_features_to_crop,
          self._anchors.get(), image_shape, true_image_shapes))

    if self._number_of_stages == 3:
      prediction_dict = self._predict_third_stage(
          prediction_dict, true_image_shapes)

    return prediction_dict

  def _image_batch_shape_2d(self, image_batch_shape_1d):
    """Takes a 1-D image batch shape tensor and converts it to a 2-D tensor.

    Example:
    If 1-D image batch shape tensor is [2, 300, 300, 3]. The corresponding 2-D
    image batch tensor would be [[300, 300, 3], [300, 300, 3]]

    Args:
      image_batch_shape_1d: 1-D tensor of the form [batch_size, height,
        width, channels].

    Returns:
      image_batch_shape_2d: 2-D tensor of shape [batch_size, 3] were each row is
        of the form [height, width, channels].
    """
    return tf.tile(tf.expand_dims(image_batch_shape_1d[1:], 0),
                   [image_batch_shape_1d[0], 1])

  def _predict_second_stage(self, rpn_box_encodings,
                            rpn_objectness_predictions_with_background,
                            rpn_features_to_crop,
                            anchors,
                            image_shape,
                            true_image_shapes):
    """Predicts the output tensors from second stage of Faster R-CNN.

    Args:
      rpn_box_encodings: 4-D float tensor of shape
        [batch_size, num_valid_anchors, self._box_coder.code_size] containing
        predicted boxes.
      rpn_objectness_predictions_with_background: 2-D float tensor of shape
        [batch_size, num_valid_anchors, 2] containing class
        predictions (logits) for each of the anchors.  Note that this
        tensor *includes* background class predictions (at class index 0).
      rpn_features_to_crop: A 4-D float32 tensor with shape
        [batch_size, height, width, depth] representing image features to crop
        using the proposal boxes predicted by the RPN.
      anchors: 2-D float tensor of shape
        [num_anchors, self._box_coder.code_size].
      image_shape: A 1D int32 tensors of size [4] containing the image shape.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Returns:
      prediction_dict: a dictionary holding "raw" prediction tensors:
        1) refined_box_encodings: a 3-D tensor with shape
          [total_num_proposals, num_classes, self._box_coder.code_size]
          representing predicted (final) refined box encodings, where
          total_num_proposals=batch_size*self._max_num_proposals. If using a
          shared box across classes the shape will instead be
          [total_num_proposals, 1, self._box_coder.code_size].
        2) class_predictions_with_background: a 3-D tensor with shape
          [total_num_proposals, num_classes + 1] containing class
          predictions (logits) for each of the anchors, where
          total_num_proposals=batch_size*self._max_num_proposals.
          Note that this tensor *includes* background class predictions
          (at class index 0).
        3) num_proposals: An int32 tensor of shape [batch_size] representing the
          number of proposals generated by the RPN.  `num_proposals` allows us
          to keep track of which entries are to be treated as zero paddings and
          which are not since we always pad the number of proposals to be
          `self.max_num_proposals` for each image.
        4) proposal_boxes: A float32 tensor of shape
          [batch_size, self.max_num_proposals, 4] representing
          decoded proposal bounding boxes in absolute coordinates.
        5) proposal_boxes_normalized: A float32 tensor of shape
          [batch_size, self.max_num_proposals, 4] representing decoded proposal
          bounding boxes in normalized coordinates. Can be used to override the
          boxes proposed by the RPN, thus enabling one to extract features and
          get box classification and prediction for externally selected areas
          of the image.
        6) box_classifier_features: a 4-D float32 tensor representing the
          features for each proposal.
    """
    image_shape_2d = self._image_batch_shape_2d(image_shape)
    proposal_boxes_normalized, _, num_proposals = self._postprocess_rpn(
        rpn_box_encodings, rpn_objectness_predictions_with_background,
        anchors, image_shape_2d, true_image_shapes)

    flattened_proposal_feature_maps = (
        self._compute_second_stage_input_feature_maps(
            rpn_features_to_crop, proposal_boxes_normalized))

    box_classifier_features = (
        self._feature_extractor.extract_box_classifier_features(
            flattened_proposal_feature_maps,
            scope=self.second_stage_feature_extractor_scope))

    box_predictions = self._mask_rcnn_box_predictor.predict(
        [box_classifier_features],
        num_predictions_per_location=[1],
        scope=self.second_stage_box_predictor_scope,
        predict_boxes_and_classes=True)

    refined_box_encodings = tf.squeeze(
        box_predictions[box_predictor.BOX_ENCODINGS],
        axis=1, name='all_refined_box_encodings')
    class_predictions_with_background = tf.squeeze(
        box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
        axis=1, name='all_class_predictions_with_background')

    absolute_proposal_boxes = ops.normalized_to_image_coordinates(
        proposal_boxes_normalized, image_shape, self._parallel_iterations)

    prediction_dict = {
        'refined_box_encodings': refined_box_encodings,
        'class_predictions_with_background':
        class_predictions_with_background,
        'num_proposals': num_proposals,
        'proposal_boxes': absolute_proposal_boxes,
        'box_classifier_features': box_classifier_features,
        'proposal_boxes_normalized': proposal_boxes_normalized,
    }

    return prediction_dict

  def _predict_third_stage(self, prediction_dict, image_shapes):
    """Predicts non-box, non-class outputs using refined detections.

    For training, masks as predicted directly on the box_classifier_features,
    which are region-features from the initial anchor boxes.
    For inference, this happens after calling the post-processing stage, such
    that masks are only calculated for the top scored boxes.

    Args:
     prediction_dict: a dictionary holding "raw" prediction tensors:
        1) refined_box_encodings: a 3-D tensor with shape
          [total_num_proposals, num_classes, self._box_coder.code_size]
          representing predicted (final) refined box encodings, where
          total_num_proposals=batch_size*self._max_num_proposals. If using a
          shared box across classes the shape will instead be
          [total_num_proposals, 1, self._box_coder.code_size].
        2) class_predictions_with_background: a 3-D tensor with shape
          [total_num_proposals, num_classes + 1] containing class
          predictions (logits) for each of the anchors, where
          total_num_proposals=batch_size*self._max_num_proposals.
          Note that this tensor *includes* background class predictions
          (at class index 0).
        3) num_proposals: An int32 tensor of shape [batch_size] representing the
          number of proposals generated by the RPN.  `num_proposals` allows us
          to keep track of which entries are to be treated as zero paddings and
          which are not since we always pad the number of proposals to be
          `self.max_num_proposals` for each image.
        4) proposal_boxes: A float32 tensor of shape
          [batch_size, self.max_num_proposals, 4] representing
          decoded proposal bounding boxes in absolute coordinates.
        5) box_classifier_features: a 4-D float32 tensor representing the
          features for each proposal.
      image_shapes: A 2-D int32 tensors of shape [batch_size, 3] containing
        shapes of images in the batch.

    Returns:
      prediction_dict: a dictionary that in addition to the input predictions
      does hold the following predictions as well:
        1) mask_predictions: a 4-D tensor with shape
          [batch_size, max_detection, mask_height, mask_width] containing
          instance mask predictions.
    """
    if self._is_training:
      curr_box_classifier_features = prediction_dict['box_classifier_features']
      detection_classes = prediction_dict['class_predictions_with_background']
      mask_predictions = self._mask_rcnn_box_predictor.predict(
          [curr_box_classifier_features],
          num_predictions_per_location=[1],
          scope=self.second_stage_box_predictor_scope,
          predict_boxes_and_classes=False,
          predict_auxiliary_outputs=True)
      prediction_dict['mask_predictions'] = tf.squeeze(mask_predictions[
          box_predictor.MASK_PREDICTIONS], axis=1)
    else:
      detections_dict = self._postprocess_box_classifier(
          prediction_dict['refined_box_encodings'],
          prediction_dict['class_predictions_with_background'],
          prediction_dict['proposal_boxes'],
          prediction_dict['num_proposals'],
          image_shapes)
      prediction_dict.update(detections_dict)
      detection_boxes = detections_dict[
          fields.DetectionResultFields.detection_boxes]
      detection_classes = detections_dict[
          fields.DetectionResultFields.detection_classes]
      rpn_features_to_crop = prediction_dict['rpn_features_to_crop']
      batch_size = tf.shape(detection_boxes)[0]
      max_detection = tf.shape(detection_boxes)[1]
      flattened_detected_feature_maps = (
          self._compute_second_stage_input_feature_maps(
              rpn_features_to_crop, detection_boxes))
      curr_box_classifier_features = (
          self._feature_extractor.extract_box_classifier_features(
              flattened_detected_feature_maps,
              scope=self.second_stage_feature_extractor_scope))

      mask_predictions = self._mask_rcnn_box_predictor.predict(
          [curr_box_classifier_features],
          num_predictions_per_location=[1],
          scope=self.second_stage_box_predictor_scope,
          predict_boxes_and_classes=False,
          predict_auxiliary_outputs=True)

      detection_masks = tf.squeeze(mask_predictions[
          box_predictor.MASK_PREDICTIONS], axis=1)

      _, num_classes, mask_height, mask_width = (
          detection_masks.get_shape().as_list())
      _, max_detection = detection_classes.get_shape().as_list()
      if num_classes > 1:
        detection_masks = self._gather_instance_masks(
            detection_masks, detection_classes)

      prediction_dict[fields.DetectionResultFields.detection_masks] = (
          tf.reshape(detection_masks,
                     [batch_size, max_detection, mask_height, mask_width]))

    return prediction_dict

  def _gather_instance_masks(self, instance_masks, classes):
    """Gathers the masks that correspond to classes.

    Args:
      instance_masks: A 4-D float32 tensor with shape
        [K, num_classes, mask_height, mask_width].
      classes: A 2-D int32 tensor with shape [batch_size, max_detection].

    Returns:
      masks: a 3-D float32 tensor with shape [K, mask_height, mask_width].
    """
    _, num_classes, height, width = instance_masks.get_shape().as_list()
    k = tf.shape(instance_masks)[0]
    instance_masks = tf.reshape(instance_masks, [-1, height, width])
    classes = tf.to_int32(tf.reshape(classes, [-1]))
    gather_idx = tf.range(k) * num_classes + classes
    return tf.gather(instance_masks, gather_idx)

  def _extract_rpn_feature_maps(self, preprocessed_inputs):
    """Extracts RPN features.

    This function extracts two feature maps: a feature map to be directly
    fed to a box predictor (to predict location and objectness scores for
    proposals) and a feature map from which to crop regions which will then
    be sent to the second stage box classifier.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] image tensor.

    Returns:
      rpn_box_predictor_features: A 4-D float32 tensor with shape
        [batch, height, width, depth] to be used for predicting proposal boxes
        and corresponding objectness scores.
      rpn_features_to_crop: A 4-D float32 tensor with shape
        [batch, height, width, depth] representing image features to crop using
        the proposals boxes.
      anchors: A BoxList representing anchors (for the RPN) in
        absolute coordinates.
      image_shape: A 1-D tensor representing the input image shape.
    """
    image_shape = tf.shape(preprocessed_inputs)
    rpn_features_to_crop, _ = self._feature_extractor.extract_proposal_features(
        preprocessed_inputs, scope=self.first_stage_feature_extractor_scope)

    feature_map_shape = tf.shape(rpn_features_to_crop)
    anchors = box_list_ops.concatenate(
        self._first_stage_anchor_generator.generate([(feature_map_shape[1],
                                                      feature_map_shape[2])]))
    with slim.arg_scope(self._first_stage_box_predictor_arg_scope_fn()):
      kernel_size = self._first_stage_box_predictor_kernel_size
      rpn_box_predictor_features = slim.conv2d(
          rpn_features_to_crop,
          self._first_stage_box_predictor_depth,
          kernel_size=[kernel_size, kernel_size],
          rate=self._first_stage_atrous_rate,
          activation_fn=tf.nn.relu6)
    return (rpn_box_predictor_features, rpn_features_to_crop,
            anchors, image_shape)

  def _predict_rpn_proposals(self, rpn_box_predictor_features):
    """Adds box predictors to RPN feature map to predict proposals.

    Note resulting tensors will not have been postprocessed.

    Args:
      rpn_box_predictor_features: A 4-D float32 tensor with shape
        [batch, height, width, depth] to be used for predicting proposal boxes
        and corresponding objectness scores.

    Returns:
      box_encodings: 3-D float tensor of shape
        [batch_size, num_anchors, self._box_coder.code_size] containing
        predicted boxes.
      objectness_predictions_with_background: 3-D float tensor of shape
        [batch_size, num_anchors, 2] containing class
        predictions (logits) for each of the anchors.  Note that this
        tensor *includes* background class predictions (at class index 0).

    Raises:
      RuntimeError: if the anchor generator generates anchors corresponding to
        multiple feature maps.  We currently assume that a single feature map
        is generated for the RPN.
    """
    num_anchors_per_location = (
        self._first_stage_anchor_generator.num_anchors_per_location())
    if len(num_anchors_per_location) != 1:
      raise RuntimeError('anchor_generator is expected to generate anchors '
                         'corresponding to a single feature map.')
    box_predictions = self._first_stage_box_predictor.predict(
        [rpn_box_predictor_features],
        num_anchors_per_location,
        scope=self.first_stage_box_predictor_scope)

    box_encodings = tf.concat(
        box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
    objectness_predictions_with_background = tf.concat(
        box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
        axis=1)
    return (tf.squeeze(box_encodings, axis=2),
            objectness_predictions_with_background)

  def _remove_invalid_anchors_and_predictions(
      self,
      box_encodings,
      objectness_predictions_with_background,
      anchors_boxlist,
      clip_window):
    """Removes anchors that (partially) fall outside an image.

    Also removes associated box encodings and objectness predictions.

    Args:
      box_encodings: 3-D float tensor of shape
        [batch_size, num_anchors, self._box_coder.code_size] containing
        predicted boxes.
      objectness_predictions_with_background: 3-D float tensor of shape
        [batch_size, num_anchors, 2] containing class
        predictions (logits) for each of the anchors.  Note that this
        tensor *includes* background class predictions (at class index 0).
      anchors_boxlist: A BoxList representing num_anchors anchors (for the RPN)
        in absolute coordinates.
      clip_window: a 1-D tensor representing the [ymin, xmin, ymax, xmax]
        extent of the window to clip/prune to.

    Returns:
      box_encodings: 4-D float tensor of shape
        [batch_size, num_valid_anchors, self._box_coder.code_size] containing
        predicted boxes, where num_valid_anchors <= num_anchors
      objectness_predictions_with_background: 2-D float tensor of shape
        [batch_size, num_valid_anchors, 2] containing class
        predictions (logits) for each of the anchors, where
        num_valid_anchors <= num_anchors.  Note that this
        tensor *includes* background class predictions (at class index 0).
      anchors: A BoxList representing num_valid_anchors anchors (for the RPN) in
        absolute coordinates.
    """
    pruned_anchors_boxlist, keep_indices = box_list_ops.prune_outside_window(
        anchors_boxlist, clip_window)
    def _batch_gather_kept_indices(predictions_tensor):
      return shape_utils.static_or_dynamic_map_fn(
          partial(tf.gather, indices=keep_indices),
          elems=predictions_tensor,
          dtype=tf.float32,
          parallel_iterations=self._parallel_iterations,
          back_prop=True)
    return (_batch_gather_kept_indices(box_encodings),
            _batch_gather_kept_indices(objectness_predictions_with_background),
            pruned_anchors_boxlist)

  def _flatten_first_two_dimensions(self, inputs):
    """Flattens `K-d` tensor along batch dimension to be a `(K-1)-d` tensor.

    Converts `inputs` with shape [A, B, ..., depth] into a tensor of shape
    [A * B, ..., depth].

    Args:
      inputs: A float tensor with shape [A, B, ..., depth].  Note that the first
        two and last dimensions must be statically defined.
    Returns:
      A float tensor with shape [A * B, ..., depth] (where the first and last
        dimension are statically defined.
    """
    combined_shape = shape_utils.combined_static_and_dynamic_shape(inputs)
    flattened_shape = tf.stack([combined_shape[0] * combined_shape[1]] +
                               combined_shape[2:])
    return tf.reshape(inputs, flattened_shape)

  def postprocess(self, prediction_dict, true_image_shapes):
    """Convert prediction tensors to final detections.

    This function converts raw predictions tensors to final detection results.
    See base class for output format conventions.  Note also that by default,
    scores are to be interpreted as logits, but if a score_converter is used,
    then scores are remapped (and may thus have a different interpretation).

    If number_of_stages=1, the returned results represent proposals from the
    first stage RPN and are padded to have self.max_num_proposals for each
    image; otherwise, the results can be interpreted as multiclass detections
    from the full two-stage model and are padded to self._max_detections.

    Args:
      prediction_dict: a dictionary holding prediction tensors (see the
        documentation for the predict method.  If number_of_stages=1, we
        expect prediction_dict to contain `rpn_box_encodings`,
        `rpn_objectness_predictions_with_background`, `rpn_features_to_crop`,
        and `anchors` fields.  Otherwise we expect prediction_dict to
        additionally contain `refined_box_encodings`,
        `class_predictions_with_background`, `num_proposals`,
        `proposal_boxes` and, optionally, `mask_predictions` fields.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Returns:
      detections: a dictionary containing the following fields
        detection_boxes: [batch, max_detection, 4]
        detection_scores: [batch, max_detections]
        detection_classes: [batch, max_detections]
          (this entry is only created if rpn_mode=False)
        num_detections: [batch]

    Raises:
      ValueError: If `predict` is called before `preprocess`.
    """

    with tf.name_scope('FirstStagePostprocessor'):
      if self._number_of_stages == 1:
        proposal_boxes, proposal_scores, num_proposals = self._postprocess_rpn(
            prediction_dict['rpn_box_encodings'],
            prediction_dict['rpn_objectness_predictions_with_background'],
            prediction_dict['anchors'],
            true_image_shapes,
            true_image_shapes)
        return {
            fields.DetectionResultFields.detection_boxes: proposal_boxes,
            fields.DetectionResultFields.detection_scores: proposal_scores,
            fields.DetectionResultFields.num_detections:
                tf.to_float(num_proposals),
        }

    # TODO(jrru): Remove mask_predictions from _post_process_box_classifier.
    with tf.name_scope('SecondStagePostprocessor'):
      if (self._number_of_stages == 2 or
          (self._number_of_stages == 3 and self._is_training)):
        mask_predictions = prediction_dict.get(box_predictor.MASK_PREDICTIONS)
        detections_dict = self._postprocess_box_classifier(
            prediction_dict['refined_box_encodings'],
            prediction_dict['class_predictions_with_background'],
            prediction_dict['proposal_boxes'],
            prediction_dict['num_proposals'],
            true_image_shapes,
            mask_predictions=mask_predictions)
        return detections_dict

    if self._number_of_stages == 3:
      # Post processing is already performed in 3rd stage. We need to transfer
      # postprocessed tensors from `prediction_dict` to `detections_dict`.
      detections_dict = {}
      for key in prediction_dict:
        if key == fields.DetectionResultFields.detection_masks:
          detections_dict[key] = tf.sigmoid(prediction_dict[key])
        elif 'detection' in key:
          detections_dict[key] = prediction_dict[key]
      return detections_dict

  def _postprocess_rpn(self,
                       rpn_box_encodings_batch,
                       rpn_objectness_predictions_with_background_batch,
                       anchors,
                       image_shapes,
                       true_image_shapes):
    """Converts first stage prediction tensors from the RPN to proposals.

    This function decodes the raw RPN predictions, runs non-max suppression
    on the result.

    Note that the behavior of this function is slightly modified during
    training --- specifically, we stop the gradient from passing through the
    proposal boxes and we only return a balanced sampled subset of proposals
    with size `second_stage_batch_size`.

    Args:
      rpn_box_encodings_batch: A 3-D float32 tensor of shape
        [batch_size, num_anchors, self._box_coder.code_size] containing
        predicted proposal box encodings.
      rpn_objectness_predictions_with_background_batch: A 3-D float tensor of
        shape [batch_size, num_anchors, 2] containing objectness predictions
        (logits) for each of the anchors with 0 corresponding to background
        and 1 corresponding to object.
      anchors: A 2-D tensor of shape [num_anchors, 4] representing anchors
        for the first stage RPN.  Note that `num_anchors` can differ depending
        on whether the model is created in training or inference mode.
      image_shapes: A 2-D tensor of shape [batch, 3] containing the shapes of
        images in the batch.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Returns:
      proposal_boxes: A float tensor with shape
        [batch_size, max_num_proposals, 4] representing the (potentially zero
        padded) proposal boxes for all images in the batch.  These boxes are
        represented as normalized coordinates.
      proposal_scores:  A float tensor with shape
        [batch_size, max_num_proposals] representing the (potentially zero
        padded) proposal objectness scores for all images in the batch.
      num_proposals: A Tensor of type `int32`. A 1-D tensor of shape [batch]
        representing the number of proposals predicted for each image in
        the batch.
    """
    rpn_box_encodings_batch = tf.expand_dims(rpn_box_encodings_batch, axis=2)
    rpn_encodings_shape = shape_utils.combined_static_and_dynamic_shape(
        rpn_box_encodings_batch)
    tiled_anchor_boxes = tf.tile(
        tf.expand_dims(anchors, 0), [rpn_encodings_shape[0], 1, 1])
    proposal_boxes = self._batch_decode_boxes(rpn_box_encodings_batch,
                                              tiled_anchor_boxes)
    proposal_boxes = tf.squeeze(proposal_boxes, axis=2)
    rpn_objectness_softmax_without_background = tf.nn.softmax(
        rpn_objectness_predictions_with_background_batch)[:, :, 1]
    clip_window = self._compute_clip_window(image_shapes)
    (proposal_boxes, proposal_scores, _, _, _,
     num_proposals) = post_processing.batch_multiclass_non_max_suppression(
         tf.expand_dims(proposal_boxes, axis=2),
         tf.expand_dims(rpn_objectness_softmax_without_background,
                        axis=2),
         self._first_stage_nms_score_threshold,
         self._first_stage_nms_iou_threshold,
         self._first_stage_max_proposals,
         self._first_stage_max_proposals,
         clip_window=clip_window)
    if self._is_training:
      proposal_boxes = tf.stop_gradient(proposal_boxes)
      if not self._hard_example_miner:
        (groundtruth_boxlists, groundtruth_classes_with_background_list, _,
         _) = self._format_groundtruth_data(true_image_shapes)
        (proposal_boxes, proposal_scores,
         num_proposals) = self._unpad_proposals_and_sample_box_classifier_batch(
             proposal_boxes, proposal_scores, num_proposals,
             groundtruth_boxlists, groundtruth_classes_with_background_list)
    # normalize proposal boxes
    def normalize_boxes(args):
      proposal_boxes_per_image = args[0]
      image_shape = args[1]
      normalized_boxes_per_image = box_list_ops.to_normalized_coordinates(
          box_list.BoxList(proposal_boxes_per_image), image_shape[0],
          image_shape[1], check_range=False).get()
      return normalized_boxes_per_image
    normalized_proposal_boxes = shape_utils.static_or_dynamic_map_fn(
        normalize_boxes, elems=[proposal_boxes, image_shapes], dtype=tf.float32)
    return normalized_proposal_boxes, proposal_scores, num_proposals

  def _unpad_proposals_and_sample_box_classifier_batch(
      self,
      proposal_boxes,
      proposal_scores,
      num_proposals,
      groundtruth_boxlists,
      groundtruth_classes_with_background_list):
    """Unpads proposals and samples a minibatch for second stage.

    Args:
      proposal_boxes: A float tensor with shape
        [batch_size, num_proposals, 4] representing the (potentially zero
        padded) proposal boxes for all images in the batch.  These boxes are
        represented in absolute coordinates.
      proposal_scores:  A float tensor with shape
        [batch_size, num_proposals] representing the (potentially zero
        padded) proposal objectness scores for all images in the batch.
      num_proposals: A Tensor of type `int32`. A 1-D tensor of shape [batch]
        representing the number of proposals predicted for each image in
        the batch.
      groundtruth_boxlists: A list of BoxLists containing (absolute) coordinates
        of the groundtruth boxes.
      groundtruth_classes_with_background_list: A list of 2-D one-hot
        (or k-hot) tensors of shape [num_boxes, num_classes+1] containing the
        class targets with the 0th index assumed to map to the background class.

    Returns:
      proposal_boxes: A float tensor with shape
        [batch_size, second_stage_batch_size, 4] representing the (potentially
        zero padded) proposal boxes for all images in the batch.  These boxes
        are represented in absolute coordinates.
      proposal_scores:  A float tensor with shape
        [batch_size, second_stage_batch_size] representing the (potentially zero
        padded) proposal objectness scores for all images in the batch.
      num_proposals: A Tensor of type `int32`. A 1-D tensor of shape [batch]
        representing the number of proposals predicted for each image in
        the batch.
    """
    single_image_proposal_box_sample = []
    single_image_proposal_score_sample = []
    single_image_num_proposals_sample = []
    for (single_image_proposal_boxes,
         single_image_proposal_scores,
         single_image_num_proposals,
         single_image_groundtruth_boxlist,
         single_image_groundtruth_classes_with_background) in zip(
             tf.unstack(proposal_boxes),
             tf.unstack(proposal_scores),
             tf.unstack(num_proposals),
             groundtruth_boxlists,
             groundtruth_classes_with_background_list):
      static_shape = single_image_proposal_boxes.get_shape()
      sliced_static_shape = tf.TensorShape([tf.Dimension(None),
                                            static_shape.dims[-1]])
      single_image_proposal_boxes = tf.slice(
          single_image_proposal_boxes,
          [0, 0],
          [single_image_num_proposals, -1])
      single_image_proposal_boxes.set_shape(sliced_static_shape)

      single_image_proposal_scores = tf.slice(single_image_proposal_scores,
                                              [0],
                                              [single_image_num_proposals])
      single_image_boxlist = box_list.BoxList(single_image_proposal_boxes)
      single_image_boxlist.add_field(fields.BoxListFields.scores,
                                     single_image_proposal_scores)
      sampled_boxlist = self._sample_box_classifier_minibatch(
          single_image_boxlist,
          single_image_groundtruth_boxlist,
          single_image_groundtruth_classes_with_background)
      sampled_padded_boxlist = box_list_ops.pad_or_clip_box_list(
          sampled_boxlist,
          num_boxes=self._second_stage_batch_size)
      single_image_num_proposals_sample.append(tf.minimum(
          sampled_boxlist.num_boxes(),
          self._second_stage_batch_size))
      bb = sampled_padded_boxlist.get()
      single_image_proposal_box_sample.append(bb)
      single_image_proposal_score_sample.append(
          sampled_padded_boxlist.get_field(fields.BoxListFields.scores))
    return (tf.stack(single_image_proposal_box_sample),
            tf.stack(single_image_proposal_score_sample),
            tf.stack(single_image_num_proposals_sample))

  def _format_groundtruth_data(self, true_image_shapes):
    """Helper function for preparing groundtruth data for target assignment.

    In order to be consistent with the model.DetectionModel interface,
    groundtruth boxes are specified in normalized coordinates and classes are
    specified as label indices with no assumed background category.  To prepare
    for target assignment, we:
    1) convert boxes to absolute coordinates,
    2) add a background class at class index 0
    3) groundtruth instance masks, if available, are resized to match
       image_shape.

    Args:
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Returns:
      groundtruth_boxlists: A list of BoxLists containing (absolute) coordinates
        of the groundtruth boxes.
      groundtruth_classes_with_background_list: A list of 2-D one-hot
        (or k-hot) tensors of shape [num_boxes, num_classes+1] containing the
        class targets with the 0th index assumed to map to the background class.
      groundtruth_masks_list: If present, a list of 3-D tf.float32 tensors of
        shape [num_boxes, image_height, image_width] containing instance masks.
        This is set to None if no masks exist in the provided groundtruth.
    """
    groundtruth_boxlists = [
        box_list_ops.to_absolute_coordinates(
            box_list.BoxList(boxes), true_image_shapes[i, 0],
            true_image_shapes[i, 1])
        for i, boxes in enumerate(
            self.groundtruth_lists(fields.BoxListFields.boxes))
    ]
    groundtruth_classes_with_background_list = [
        tf.to_float(
            tf.pad(one_hot_encoding, [[0, 0], [1, 0]], mode='CONSTANT'))
        for one_hot_encoding in self.groundtruth_lists(
            fields.BoxListFields.classes)]

    groundtruth_masks_list = self._groundtruth_lists.get(
        fields.BoxListFields.masks)
    if groundtruth_masks_list is not None:
      resized_masks_list = []
      for mask in groundtruth_masks_list:
        _, resized_mask, _ = self._image_resizer_fn(
            # Reuse the given `image_resizer_fn` to resize groundtruth masks.
            # `mask` tensor for an image is of the shape [num_masks,
            # image_height, image_width]. Below we create a dummy image of the
            # the shape [image_height, image_width, 1] to use with
            # `image_resizer_fn`.
            image=tf.zeros(tf.stack([tf.shape(mask)[1], tf.shape(mask)[2], 1])),
            masks=mask)
        resized_masks_list.append(resized_mask)

      groundtruth_masks_list = resized_masks_list
    groundtruth_weights_list = None
    if self.groundtruth_has_field(fields.BoxListFields.weights):
      groundtruth_weights_list = self.groundtruth_lists(
          fields.BoxListFields.weights)

    return (groundtruth_boxlists, groundtruth_classes_with_background_list,
            groundtruth_masks_list, groundtruth_weights_list)

  def _sample_box_classifier_minibatch(self,
                                       proposal_boxlist,
                                       groundtruth_boxlist,
                                       groundtruth_classes_with_background):
    """Samples a mini-batch of proposals to be sent to the box classifier.

    Helper function for self._postprocess_rpn.

    Args:
      proposal_boxlist: A BoxList containing K proposal boxes in absolute
        coordinates.
      groundtruth_boxlist: A Boxlist containing N groundtruth object boxes in
        absolute coordinates.
      groundtruth_classes_with_background: A tensor with shape
        `[N, self.num_classes + 1]` representing groundtruth classes. The
        classes are assumed to be k-hot encoded, and include background as the
        zero-th class.

    Returns:
      a BoxList contained sampled proposals.
    """
    (cls_targets, cls_weights, _, _, _) = self._detector_target_assigner.assign(
        proposal_boxlist, groundtruth_boxlist,
        groundtruth_classes_with_background)
    # Selects all boxes as candidates if none of them is selected according
    # to cls_weights. This could happen as boxes within certain IOU ranges
    # are ignored. If triggered, the selected boxes will still be ignored
    # during loss computation.
    cls_weights += tf.to_float(tf.equal(tf.reduce_sum(cls_weights), 0))
    positive_indicator = tf.greater(tf.argmax(cls_targets, axis=1), 0)
    sampled_indices = self._second_stage_sampler.subsample(
        tf.cast(cls_weights, tf.bool),
        self._second_stage_batch_size,
        positive_indicator)
    return box_list_ops.boolean_mask(proposal_boxlist, sampled_indices)

  def _compute_second_stage_input_feature_maps(self, features_to_crop,
                                               proposal_boxes_normalized):
    """Crops to a set of proposals from the feature map for a batch of images.

    Helper function for self._postprocess_rpn. This function calls
    `tf.image.crop_and_resize` to create the feature map to be passed to the
    second stage box classifier for each proposal.

    Args:
      features_to_crop: A float32 tensor with shape
        [batch_size, height, width, depth]
      proposal_boxes_normalized: A float32 tensor with shape [batch_size,
        num_proposals, box_code_size] containing proposal boxes in
        normalized coordinates.

    Returns:
      A float32 tensor with shape [K, new_height, new_width, depth].
    """
    def get_box_inds(proposals):
      proposals_shape = proposals.get_shape().as_list()
      if any(dim is None for dim in proposals_shape):
        proposals_shape = tf.shape(proposals)
      ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
      multiplier = tf.expand_dims(
          tf.range(start=0, limit=proposals_shape[0]), 1)
      return tf.reshape(ones_mat * multiplier, [-1])

    cropped_regions = tf.image.crop_and_resize(
        features_to_crop,
        self._flatten_first_two_dimensions(proposal_boxes_normalized),
        get_box_inds(proposal_boxes_normalized),
        (self._initial_crop_size, self._initial_crop_size))
    return slim.max_pool2d(
        cropped_regions,
        [self._maxpool_kernel_size, self._maxpool_kernel_size],
        stride=self._maxpool_stride)

  def _postprocess_box_classifier(self,
                                  refined_box_encodings,
                                  class_predictions_with_background,
                                  proposal_boxes,
                                  num_proposals,
                                  image_shapes,
                                  mask_predictions=None):
    """Converts predictions from the second stage box classifier to detections.

    Args:
      refined_box_encodings: a 3-D float tensor with shape
        [total_num_padded_proposals, num_classes, self._box_coder.code_size]
        representing predicted (final) refined box encodings. If using a shared
        box across classes the shape will instead be
        [total_num_padded_proposals, 1, 4]
      class_predictions_with_background: a 3-D tensor float with shape
        [total_num_padded_proposals, num_classes + 1] containing class
        predictions (logits) for each of the proposals.  Note that this tensor
        *includes* background class predictions (at class index 0).
      proposal_boxes: a 3-D float tensor with shape
        [batch_size, self.max_num_proposals, 4] representing decoded proposal
        bounding boxes in absolute coordinates.
      num_proposals: a 1-D int32 tensor of shape [batch] representing the number
        of proposals predicted for each image in the batch.
      image_shapes: a 2-D int32 tensor containing shapes of input image in the
        batch.
      mask_predictions: (optional) a 4-D float tensor with shape
        [total_num_padded_proposals, num_classes, mask_height, mask_width]
        containing instance mask prediction logits.

    Returns:
      A dictionary containing:
        `detection_boxes`: [batch, max_detection, 4]
        `detection_scores`: [batch, max_detections]
        `detection_classes`: [batch, max_detections]
        `num_detections`: [batch]
        `detection_masks`:
          (optional) [batch, max_detections, mask_height, mask_width]. Note
          that a pixel-wise sigmoid score converter is applied to the detection
          masks.
    """
    refined_box_encodings_batch = tf.reshape(
        refined_box_encodings,
        [-1,
         self.max_num_proposals,
         refined_box_encodings.shape[1],
         self._box_coder.code_size])
    class_predictions_with_background_batch = tf.reshape(
        class_predictions_with_background,
        [-1, self.max_num_proposals, self.num_classes + 1]
    )
    refined_decoded_boxes_batch = self._batch_decode_boxes(
        refined_box_encodings_batch, proposal_boxes)
    class_predictions_with_background_batch = (
        self._second_stage_score_conversion_fn(
            class_predictions_with_background_batch))
    class_predictions_batch = tf.reshape(
        tf.slice(class_predictions_with_background_batch,
                 [0, 0, 1], [-1, -1, -1]),
        [-1, self.max_num_proposals, self.num_classes])
    clip_window = self._compute_clip_window(image_shapes)
    mask_predictions_batch = None
    if mask_predictions is not None:
      mask_height = mask_predictions.shape[2].value
      mask_width = mask_predictions.shape[3].value
      mask_predictions = tf.sigmoid(mask_predictions)
      mask_predictions_batch = tf.reshape(
          mask_predictions, [-1, self.max_num_proposals,
                             self.num_classes, mask_height, mask_width])
    (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks, _,
     num_detections) = self._second_stage_nms_fn(
         refined_decoded_boxes_batch,
         class_predictions_batch,
         clip_window=clip_window,
         change_coordinate_frame=True,
         num_valid_boxes=num_proposals,
         masks=mask_predictions_batch)
    detections = {
        fields.DetectionResultFields.detection_boxes: nmsed_boxes,
        fields.DetectionResultFields.detection_scores: nmsed_scores,
        fields.DetectionResultFields.detection_classes: nmsed_classes,
        fields.DetectionResultFields.num_detections: tf.to_float(num_detections)
    }
    if nmsed_masks is not None:
      detections[fields.DetectionResultFields.detection_masks] = nmsed_masks
    return detections

  def _batch_decode_boxes(self, box_encodings, anchor_boxes):
    """Decodes box encodings with respect to the anchor boxes.

    Args:
      box_encodings: a 4-D tensor with shape
        [batch_size, num_anchors, num_classes, self._box_coder.code_size]
        representing box encodings.
      anchor_boxes: [batch_size, num_anchors, self._box_coder.code_size]
        representing decoded bounding boxes. If using a shared box across
        classes the shape will instead be
        [total_num_proposals, 1, self._box_coder.code_size].

    Returns:
      decoded_boxes: a
        [batch_size, num_anchors, num_classes, self._box_coder.code_size]
        float tensor representing bounding box predictions (for each image in
        batch, proposal and class). If using a shared box across classes the
        shape will instead be
        [batch_size, num_anchors, 1, self._box_coder.code_size].
    """
    combined_shape = shape_utils.combined_static_and_dynamic_shape(
        box_encodings)
    num_classes = combined_shape[2]
    tiled_anchor_boxes = tf.tile(
        tf.expand_dims(anchor_boxes, 2), [1, 1, num_classes, 1])
    tiled_anchors_boxlist = box_list.BoxList(
        tf.reshape(tiled_anchor_boxes, [-1, 4]))
    decoded_boxes = self._box_coder.decode(
        tf.reshape(box_encodings, [-1, self._box_coder.code_size]),
        tiled_anchors_boxlist)
    return tf.reshape(decoded_boxes.get(),
                      tf.stack([combined_shape[0], combined_shape[1],
                                num_classes, 4]))

  def loss(self, prediction_dict, true_image_shapes, scope=None):
    """Compute scalar loss tensors given prediction tensors.

    If number_of_stages=1, only RPN related losses are computed (i.e.,
    `rpn_localization_loss` and `rpn_objectness_loss`).  Otherwise all
    losses are computed.

    Args:
      prediction_dict: a dictionary holding prediction tensors (see the
        documentation for the predict method.  If number_of_stages=1, we
        expect prediction_dict to contain `rpn_box_encodings`,
        `rpn_objectness_predictions_with_background`, `rpn_features_to_crop`,
        `image_shape`, and `anchors` fields.  Otherwise we expect
        prediction_dict to additionally contain `refined_box_encodings`,
        `class_predictions_with_background`, `num_proposals`, and
        `proposal_boxes` fields.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.
      scope: Optional scope name.

    Returns:
      a dictionary mapping loss keys (`first_stage_localization_loss`,
        `first_stage_objectness_loss`, 'second_stage_localization_loss',
        'second_stage_classification_loss') to scalar tensors representing
        corresponding loss values.
    """
    with tf.name_scope(scope, 'Loss', prediction_dict.values()):
      (groundtruth_boxlists, groundtruth_classes_with_background_list,
       groundtruth_masks_list, groundtruth_weights_list
      ) = self._format_groundtruth_data(true_image_shapes)
      loss_dict = self._loss_rpn(
          prediction_dict['rpn_box_encodings'],
          prediction_dict['rpn_objectness_predictions_with_background'],
          prediction_dict['anchors'], groundtruth_boxlists,
          groundtruth_classes_with_background_list, groundtruth_weights_list)
      if self._number_of_stages > 1:
        loss_dict.update(
            self._loss_box_classifier(
                prediction_dict['refined_box_encodings'],
                prediction_dict['class_predictions_with_background'],
                prediction_dict['proposal_boxes'],
                prediction_dict['num_proposals'],
                groundtruth_boxlists,
                groundtruth_classes_with_background_list,
                groundtruth_weights_list,
                prediction_dict['image_shape'],
                prediction_dict.get('mask_predictions'),
                groundtruth_masks_list,
            ))
    return loss_dict

  def _loss_rpn(self, rpn_box_encodings,
                rpn_objectness_predictions_with_background, anchors,
                groundtruth_boxlists, groundtruth_classes_with_background_list,
                groundtruth_weights_list):
    """Computes scalar RPN loss tensors.

    Uses self._proposal_target_assigner to obtain regression and classification
    targets for the first stage RPN, samples a "minibatch" of anchors to
    participate in the loss computation, and returns the RPN losses.

    Args:
      rpn_box_encodings: A 4-D float tensor of shape
        [batch_size, num_anchors, self._box_coder.code_size] containing
        predicted proposal box encodings.
      rpn_objectness_predictions_with_background: A 2-D float tensor of shape
        [batch_size, num_anchors, 2] containing objectness predictions
        (logits) for each of the anchors with 0 corresponding to background
        and 1 corresponding to object.
      anchors: A 2-D tensor of shape [num_anchors, 4] representing anchors
        for the first stage RPN.  Note that `num_anchors` can differ depending
        on whether the model is created in training or inference mode.
      groundtruth_boxlists: A list of BoxLists containing coordinates of the
        groundtruth boxes.
      groundtruth_classes_with_background_list: A list of 2-D one-hot
        (or k-hot) tensors of shape [num_boxes, num_classes+1] containing the
        class targets with the 0th index assumed to map to the background class.
      groundtruth_weights_list: A list of 1-D tf.float32 tensors of shape
        [num_boxes] containing weights for groundtruth boxes.

    Returns:
      a dictionary mapping loss keys (`first_stage_localization_loss`,
        `first_stage_objectness_loss`) to scalar tensors representing
        corresponding loss values.
    """
    with tf.name_scope('RPNLoss'):
      (batch_cls_targets, batch_cls_weights, batch_reg_targets,
       batch_reg_weights, _) = target_assigner.batch_assign_targets(
           self._proposal_target_assigner, box_list.BoxList(anchors),
           groundtruth_boxlists,
           len(groundtruth_boxlists) * [None], groundtruth_weights_list)
      batch_cls_targets = tf.squeeze(batch_cls_targets, axis=2)

      def _minibatch_subsample_fn(inputs):
        cls_targets, cls_weights = inputs
        return self._first_stage_sampler.subsample(
            tf.cast(cls_weights, tf.bool),
            self._first_stage_minibatch_size, tf.cast(cls_targets, tf.bool))
      batch_sampled_indices = tf.to_float(shape_utils.static_or_dynamic_map_fn(
          _minibatch_subsample_fn,
          [batch_cls_targets, batch_cls_weights],
          dtype=tf.bool,
          parallel_iterations=self._parallel_iterations,
          back_prop=True))

      # Normalize by number of examples in sampled minibatch
      normalizer = tf.reduce_sum(batch_sampled_indices, axis=1)
      batch_one_hot_targets = tf.one_hot(
          tf.to_int32(batch_cls_targets), depth=2)
      sampled_reg_indices = tf.multiply(batch_sampled_indices,
                                        batch_reg_weights)

      localization_losses = self._first_stage_localization_loss(
          rpn_box_encodings, batch_reg_targets, weights=sampled_reg_indices)
      objectness_losses = self._first_stage_objectness_loss(
          rpn_objectness_predictions_with_background,
          batch_one_hot_targets, weights=batch_sampled_indices)
      localization_loss = tf.reduce_mean(
          tf.reduce_sum(localization_losses, axis=1) / normalizer)
      objectness_loss = tf.reduce_mean(
          tf.reduce_sum(objectness_losses, axis=1) / normalizer)

      localization_loss = tf.multiply(self._first_stage_loc_loss_weight,
                                      localization_loss,
                                      name='localization_loss')
      objectness_loss = tf.multiply(self._first_stage_obj_loss_weight,
                                    objectness_loss, name='objectness_loss')
      loss_dict = {localization_loss.op.name: localization_loss,
                   objectness_loss.op.name: objectness_loss}
    return loss_dict

  def _loss_box_classifier(self,
                           refined_box_encodings,
                           class_predictions_with_background,
                           proposal_boxes,
                           num_proposals,
                           groundtruth_boxlists,
                           groundtruth_classes_with_background_list,
                           groundtruth_weights_list,
                           image_shape,
                           prediction_masks=None,
                           groundtruth_masks_list=None):
    """Computes scalar box classifier loss tensors.

    Uses self._detector_target_assigner to obtain regression and classification
    targets for the second stage box classifier, optionally performs
    hard mining, and returns losses.  All losses are computed independently
    for each image and then averaged across the batch.
    Please note that for boxes and masks with multiple labels, the box
    regression and mask prediction losses are only computed for one label.

    This function assumes that the proposal boxes in the "padded" regions are
    actually zero (and thus should not be matched to).


    Args:
      refined_box_encodings: a 3-D tensor with shape
        [total_num_proposals, num_classes, box_coder.code_size] representing
        predicted (final) refined box encodings. If using a shared box across
        classes this will instead have shape
        [total_num_proposals, 1, box_coder.code_size].
      class_predictions_with_background: a 2-D tensor with shape
        [total_num_proposals, num_classes + 1] containing class
        predictions (logits) for each of the anchors.  Note that this tensor
        *includes* background class predictions (at class index 0).
      proposal_boxes: [batch_size, self.max_num_proposals, 4] representing
        decoded proposal bounding boxes.
      num_proposals: A Tensor of type `int32`. A 1-D tensor of shape [batch]
        representing the number of proposals predicted for each image in
        the batch.
      groundtruth_boxlists: a list of BoxLists containing coordinates of the
        groundtruth boxes.
      groundtruth_classes_with_background_list: a list of 2-D one-hot
        (or k-hot) tensors of shape [num_boxes, num_classes + 1] containing the
        class targets with the 0th index assumed to map to the background class.
      groundtruth_weights_list: A list of 1-D tf.float32 tensors of shape
        [num_boxes] containing weights for groundtruth boxes.
      image_shape: a 1-D tensor of shape [4] representing the image shape.
      prediction_masks: an optional 4-D tensor with shape [total_num_proposals,
        num_classes, mask_height, mask_width] containing the instance masks for
        each box.
      groundtruth_masks_list: an optional list of 3-D tensors of shape
        [num_boxes, image_height, image_width] containing the instance masks for
        each of the boxes.

    Returns:
      a dictionary mapping loss keys ('second_stage_localization_loss',
        'second_stage_classification_loss') to scalar tensors representing
        corresponding loss values.

    Raises:
      ValueError: if `predict_instance_masks` in
        second_stage_mask_rcnn_box_predictor is True and
        `groundtruth_masks_list` is not provided.
    """
    with tf.name_scope('BoxClassifierLoss'):
      paddings_indicator = self._padded_batched_proposals_indicator(
          num_proposals, self.max_num_proposals)
      proposal_boxlists = [
          box_list.BoxList(proposal_boxes_single_image)
          for proposal_boxes_single_image in tf.unstack(proposal_boxes)]
      batch_size = len(proposal_boxlists)

      num_proposals_or_one = tf.to_float(tf.expand_dims(
          tf.maximum(num_proposals, tf.ones_like(num_proposals)), 1))
      normalizer = tf.tile(num_proposals_or_one,
                           [1, self.max_num_proposals]) * batch_size

      (batch_cls_targets_with_background, batch_cls_weights, batch_reg_targets,
       batch_reg_weights, _) = target_assigner.batch_assign_targets(
           self._detector_target_assigner, proposal_boxlists,
           groundtruth_boxlists, groundtruth_classes_with_background_list,
           groundtruth_weights_list)

      class_predictions_with_background = tf.reshape(
          class_predictions_with_background,
          [batch_size, self.max_num_proposals, -1])

      flat_cls_targets_with_background = tf.reshape(
          batch_cls_targets_with_background,
          [batch_size * self.max_num_proposals, -1])
      one_hot_flat_cls_targets_with_background = tf.argmax(
          flat_cls_targets_with_background, axis=1)
      one_hot_flat_cls_targets_with_background = tf.one_hot(
          one_hot_flat_cls_targets_with_background,
          flat_cls_targets_with_background.get_shape()[1])

      # If using a shared box across classes use directly
      if refined_box_encodings.shape[1] == 1:
        reshaped_refined_box_encodings = tf.reshape(
            refined_box_encodings,
            [batch_size, self.max_num_proposals, self._box_coder.code_size])
      # For anchors with multiple labels, picks refined_location_encodings
      # for just one class to avoid over-counting for regression loss and
      # (optionally) mask loss.
      else:
        # We only predict refined location encodings for the non background
        # classes, but we now pad it to make it compatible with the class
        # predictions
        refined_box_encodings_with_background = tf.pad(
            refined_box_encodings, [[0, 0], [1, 0], [0, 0]])
        refined_box_encodings_masked_by_class_targets = tf.boolean_mask(
            refined_box_encodings_with_background,
            tf.greater(one_hot_flat_cls_targets_with_background, 0))
        reshaped_refined_box_encodings = tf.reshape(
            refined_box_encodings_masked_by_class_targets,
            [batch_size, self.max_num_proposals, self._box_coder.code_size])

      second_stage_loc_losses = self._second_stage_localization_loss(
          reshaped_refined_box_encodings,
          batch_reg_targets, weights=batch_reg_weights) / normalizer
      second_stage_cls_losses = ops.reduce_sum_trailing_dimensions(
          self._second_stage_classification_loss(
              class_predictions_with_background,
              batch_cls_targets_with_background,
              weights=batch_cls_weights),
          ndims=2) / normalizer

      second_stage_loc_loss = tf.reduce_sum(
          tf.boolean_mask(second_stage_loc_losses, paddings_indicator))
      second_stage_cls_loss = tf.reduce_sum(
          tf.boolean_mask(second_stage_cls_losses, paddings_indicator))

      if self._hard_example_miner:
        (second_stage_loc_loss, second_stage_cls_loss
        ) = self._unpad_proposals_and_apply_hard_mining(
            proposal_boxlists, second_stage_loc_losses,
            second_stage_cls_losses, num_proposals)
      localization_loss = tf.multiply(self._second_stage_loc_loss_weight,
                                      second_stage_loc_loss,
                                      name='localization_loss')

      classification_loss = tf.multiply(self._second_stage_cls_loss_weight,
                                        second_stage_cls_loss,
                                        name='classification_loss')

      loss_dict = {localization_loss.op.name: localization_loss,
                   classification_loss.op.name: classification_loss}
      second_stage_mask_loss = None
      if prediction_masks is not None:
        if groundtruth_masks_list is None:
          raise ValueError('Groundtruth instance masks not provided. '
                           'Please configure input reader.')

        # Create a new target assigner that matches the proposals to groundtruth
        # and returns the mask targets.
        # TODO(rathodv): Move `unmatched_cls_target` from constructor to assign
        # function. This will enable reuse of a single target assigner for both
        # class targets and mask targets.
        mask_target_assigner = target_assigner.create_target_assigner(
            'FasterRCNN', 'detection',
            unmatched_cls_target=tf.zeros(image_shape[1:3], dtype=tf.float32))
        (batch_mask_targets, _, _,
         batch_mask_target_weights, _) = target_assigner.batch_assign_targets(
             mask_target_assigner, proposal_boxlists, groundtruth_boxlists,
             groundtruth_masks_list, groundtruth_weights_list)

        # Pad the prediction_masks with to add zeros for background class to be
        # consistent with class predictions.
        if prediction_masks.get_shape().as_list()[1] == 1:
          # Class agnostic masks or masks for one-class prediction. Logic for
          # both cases is the same since background predictions are ignored
          # through the batch_mask_target_weights.
          prediction_masks_masked_by_class_targets = prediction_masks
        else:
          prediction_masks_with_background = tf.pad(
              prediction_masks, [[0, 0], [1, 0], [0, 0], [0, 0]])
          prediction_masks_masked_by_class_targets = tf.boolean_mask(
              prediction_masks_with_background,
              tf.greater(one_hot_flat_cls_targets_with_background, 0))

        mask_height = prediction_masks.shape[2].value
        mask_width = prediction_masks.shape[3].value
        reshaped_prediction_masks = tf.reshape(
            prediction_masks_masked_by_class_targets,
            [batch_size, -1, mask_height * mask_width])

        batch_mask_targets_shape = tf.shape(batch_mask_targets)
        flat_gt_masks = tf.reshape(batch_mask_targets,
                                   [-1, batch_mask_targets_shape[2],
                                    batch_mask_targets_shape[3]])

        # Use normalized proposals to crop mask targets from image masks.
        flat_normalized_proposals = box_list_ops.to_normalized_coordinates(
            box_list.BoxList(tf.reshape(proposal_boxes, [-1, 4])),
            image_shape[1], image_shape[2]).get()

        flat_cropped_gt_mask = tf.image.crop_and_resize(
            tf.expand_dims(flat_gt_masks, -1),
            flat_normalized_proposals,
            tf.range(flat_normalized_proposals.shape[0].value),
            [mask_height, mask_width])

        batch_cropped_gt_mask = tf.reshape(
            flat_cropped_gt_mask,
            [batch_size, -1, mask_height * mask_width])

        second_stage_mask_losses = ops.reduce_sum_trailing_dimensions(
            self._second_stage_mask_loss(
                reshaped_prediction_masks,
                batch_cropped_gt_mask,
                weights=batch_mask_target_weights),
            ndims=2) / (
                mask_height * mask_width * tf.maximum(
                    tf.reduce_sum(
                        batch_mask_target_weights, axis=1, keep_dims=True
                    ), tf.ones((batch_size, 1))))
        second_stage_mask_loss = tf.reduce_sum(
            tf.boolean_mask(second_stage_mask_losses, paddings_indicator))

      if second_stage_mask_loss is not None:
        mask_loss = tf.multiply(self._second_stage_mask_loss_weight,
                                second_stage_mask_loss, name='mask_loss')
        loss_dict[mask_loss.op.name] = mask_loss
    return loss_dict

  def _padded_batched_proposals_indicator(self,
                                          num_proposals,
                                          max_num_proposals):
    """Creates indicator matrix of non-pad elements of padded batch proposals.

    Args:
      num_proposals: Tensor of type tf.int32 with shape [batch_size].
      max_num_proposals: Maximum number of proposals per image (integer).

    Returns:
      A Tensor of type tf.bool with shape [batch_size, max_num_proposals].
    """
    batch_size = tf.size(num_proposals)
    tiled_num_proposals = tf.tile(
        tf.expand_dims(num_proposals, 1), [1, max_num_proposals])
    tiled_proposal_index = tf.tile(
        tf.expand_dims(tf.range(max_num_proposals), 0), [batch_size, 1])
    return tf.greater(tiled_num_proposals, tiled_proposal_index)

  def _unpad_proposals_and_apply_hard_mining(self,
                                             proposal_boxlists,
                                             second_stage_loc_losses,
                                             second_stage_cls_losses,
                                             num_proposals):
    """Unpads proposals and applies hard mining.

    Args:
      proposal_boxlists: A list of `batch_size` BoxLists each representing
        `self.max_num_proposals` representing decoded proposal bounding boxes
        for each image.
      second_stage_loc_losses: A Tensor of type `float32`. A tensor of shape
        `[batch_size, self.max_num_proposals]` representing per-anchor
        second stage localization loss values.
      second_stage_cls_losses: A Tensor of type `float32`. A tensor of shape
        `[batch_size, self.max_num_proposals]` representing per-anchor
        second stage classification loss values.
      num_proposals: A Tensor of type `int32`. A 1-D tensor of shape [batch]
        representing the number of proposals predicted for each image in
        the batch.

    Returns:
      second_stage_loc_loss: A scalar float32 tensor representing the second
        stage localization loss.
      second_stage_cls_loss: A scalar float32 tensor representing the second
        stage classification loss.
    """
    for (proposal_boxlist, single_image_loc_loss, single_image_cls_loss,
         single_image_num_proposals) in zip(
             proposal_boxlists,
             tf.unstack(second_stage_loc_losses),
             tf.unstack(second_stage_cls_losses),
             tf.unstack(num_proposals)):
      proposal_boxlist = box_list.BoxList(
          tf.slice(proposal_boxlist.get(),
                   [0, 0], [single_image_num_proposals, -1]))
      single_image_loc_loss = tf.slice(single_image_loc_loss,
                                       [0], [single_image_num_proposals])
      single_image_cls_loss = tf.slice(single_image_cls_loss,
                                       [0], [single_image_num_proposals])
      return self._hard_example_miner(
          location_losses=tf.expand_dims(single_image_loc_loss, 0),
          cls_losses=tf.expand_dims(single_image_cls_loss, 0),
          decoded_boxlist_list=[proposal_boxlist])

  def restore_map(self,
                  fine_tune_checkpoint_type='detection',
                  load_all_detection_checkpoint_vars=False):
    """Returns a map of variables to load from a foreign checkpoint.

    See parent class for details.

    Args:
      fine_tune_checkpoint_type: whether to restore from a full detection
        checkpoint (with compatible variable names) or to restore from a
        classification checkpoint for initialization prior to training.
        Valid values: `detection`, `classification`. Default 'detection'.
       load_all_detection_checkpoint_vars: whether to load all variables (when
         `fine_tune_checkpoint_type` is `detection`). If False, only variables
         within the feature extractor scopes are included. Default False.

    Returns:
      A dict mapping variable names (to load from a checkpoint) to variables in
      the model graph.
    Raises:
      ValueError: if fine_tune_checkpoint_type is neither `classification`
        nor `detection`.
    """
    if fine_tune_checkpoint_type not in ['detection', 'classification']:
      raise ValueError('Not supported fine_tune_checkpoint_type: {}'.format(
          fine_tune_checkpoint_type))
    if fine_tune_checkpoint_type == 'classification':
      return self._feature_extractor.restore_from_classification_checkpoint_fn(
          self.first_stage_feature_extractor_scope,
          self.second_stage_feature_extractor_scope)

    variables_to_restore = tf.global_variables()
    variables_to_restore.append(slim.get_or_create_global_step())
    # Only load feature extractor variables to be consistent with loading from
    # a classification checkpoint.
    include_patterns = None
    if not load_all_detection_checkpoint_vars:
      include_patterns = [
          self.first_stage_feature_extractor_scope,
          self.second_stage_feature_extractor_scope
      ]
    feature_extractor_variables = tf.contrib.framework.filter_variables(
        variables_to_restore, include_patterns=include_patterns)
    return {var.op.name: var for var in feature_extractor_variables}

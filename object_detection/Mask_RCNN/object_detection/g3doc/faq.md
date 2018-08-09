# Frequently Asked Questions

## Q: AttributeError: 'module' object has no attribute 'BackupHandler'
A: This BackupHandler (tf.contrib.slim.tfexample_decoder.BackupHandler) was
introduced in tensorflow 1.5.0 so runing with earlier versions may cause this
issue. It now has been replaced by
object_detection.data_decoders.tf_example_decoder.BackupHandler. Whoever sees
this issue should be able to resolve it by syncing your fork to HEAD.
Same for LookupTensor.

## Q: AttributeError: 'module' object has no attribute 'LookupTensor'
A: Similar to BackupHandler, syncing your fork to HEAD should make it work.

## Q: Why can't I get the inference time as reported in model zoo?
A: The inference time reported in model zoo is mean time of testing hundreds of
images with an internal machine. As mentioned in
[Tensorflow detection model zoo](detection_model_zoo.md), this speed depends
highly on one's specific hardware configuration and should be treated more as
relative timing.

# MLPerf™ Training Reference Implementations

This is a repository of reference implementations for the MLPerf training benchmarks. These implementations are valid as starting points for benchmark implementations but are not fully optimized and are not intended to be used for "real" performance measurements of software frameworks or hardware. 

Please see the [MLPerf Training Benchmark](https://arxiv.org/abs/1910.01500) paper for a detailed description of the motivation and guiding principles behind the benchmark suite. If you use any part of this benchmark (e.g., reference implementations, submissions, etc.) in academic work, please cite the following:

```
@misc{mattson2019mlperf,
    title={MLPerf Training Benchmark},
    author={Peter Mattson and Christine Cheng and Cody Coleman and Greg Diamos and Paulius Micikevicius and David Patterson and Hanlin Tang and Gu-Yeon Wei and Peter Bailis and Victor Bittorf and David Brooks and Dehao Chen and Debojyoti Dutta and Udit Gupta and Kim Hazelwood and Andrew Hock and Xinyuan Huang and Atsushi Ike and Bill Jia and Daniel Kang and David Kanter and Naveen Kumar and Jeffery Liao and Guokai Ma and Deepak Narayanan and Tayo Oguntebi and Gennady Pekhimenko and Lillian Pentecost and Vijay Janapa Reddi and Taylor Robie and Tom St. John and Tsuguchika Tabaru and Carole-Jean Wu and Lingjie Xu and Masafumi Yamazaki and Cliff Young and Matei Zaharia},
    year={2019},
    eprint={1910.01500},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

These reference implementations are still very much "alpha" or "beta" quality. They could be improved in many ways. Please file issues or pull requests to help us improve quality.

# Contents

We provide reference implementations for benchmarks in the MLPerf suite, as well as several benchmarks under development. 

Each reference implementation provides the following:
 
* Code that implements the model in at least one framework.
* A Dockerfile which can be used to run the benchmark in a container.
* A script which downloads the appropriate dataset.
* A script which runs and times training the model.
* Documentation on the dataset, model, and machine setup.

# Running Benchmarks

Follow instructions on the Readme of each benchmark. Generally, a benchmark can be run with the following steps:

1. Setup docker & dependencies. There is a shared script (install_cuda_docker.sh) to do this. Some benchmarks will have additional setup, mentioned in their READMEs.
2. Download the dataset using `./download_dataset.sh`. This should be run outside of docker, on your host machine. This should be run from the directory it is in (it may make assumptions about CWD).
3. Optionally, run `verify_dataset.sh` to ensure the was successfully downloaded.
4. Build and run the docker image, the command to do this is included with each Benchmark. 

Each benchmark will run until the target quality is reached and then stop, printing timing results.

Some these benchmarks are rather slow or take a long time to run on the reference hardware. We expect to see significant performance improvements with more hardware and optimized implementations.

# MLPerf Training v5.1 (Submission Deadline Oct 10, 2025)

| Model | reference implementation | framework* | dataset | model parameter count**
| ---- | ---- | ---- | ---- | ----
| retinanet | [single_stage_detector](https://github.com/mlcommons/training/tree/master/single_stage_detector) | pytorch | OpenImages | 37M
| flux.1 | text_to_image | torchtitan | CC12M, COCO-2014 subset for eval | 12B
| llama3.1_8b | small_llm_pretraining | NeMo | C4 | 8b
| llama2_70b_lora | [llama2_70b_lora](https://github.com/mlcommons/training/tree/master/llama2_70b_lora) | pytorch | SCROLLS GovReport | 70B
| llama3.1_405b | [large_language_model_pretraining](https://github.com/mlcommons/training/tree/master/large_language_model_pretraining) | NeMo | C4 | 405B
| dlrm_dcnv2 | [recommendation_v2](https://github.com/mlcommons/training/tree/master/recommendation_v2/torchrec_dlrm) | torchrec | Criteo 3.5TB multi-hot | 167M
| rgat | [graph_neural_network](https://github.com/mlcommons/training/tree/master/graph_neural_network) | GLT | IGBFull | 25M

*Framework here is given for the reference implementation. Submitters are free to use their own frameworks to run the benchmark.
 
**Model parameter count is not the same as active parameter that are being trained in the benchmark. 

# MLPerf Training v5.0 (Submission Deadline May 2, 2025)

* Framework here is given for the reference implementation. Submitters are free to use their own frameworks to run the benchmark. 
* Model parameter count is not the same as active parameter that are being trained in the benchmark. 

| Model | reference implementation | framework | dataset | model parameter count
| ---- | ---- | ---- | ---- | ----
| retinanet | [single_stage_detector](https://github.com/mlcommons/training/tree/master/single_stage_detector) | pytorch | OpenImages | 37M
| flux.1-schnell| [text_to_image](https://github.com/mlcommons/training/tree/master/text_to_image) | pytorch | CC12M subset | 11.9B
| bert | [language_model](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert) | tensorflow | Wikipedia 2020/01/01 | 340M
| llama3.1_405b | [large_language_model_pretraining](https://github.com/mlcommons/training/tree/master/large_language_model_pretraining) | NeMo | C4 | 405B
| llama2_70b_lora | [llama2_70b_lora](https://github.com/mlcommons/training/tree/master/llama2_70b_lora) | pytorch | SCROLLS GovReport | 70B
| dlrm_dcnv2 | [recommendation_v2](https://github.com/mlcommons/training/tree/master/recommendation_v2/torchrec_dlrm) | torchrec | Criteo 3.5TB multi-hot | 167M
| rgat | [graph_neural_network](https://github.com/mlcommons/training/tree/master/graph_neural_network) | GLT | IGBFull | 25M

# MLPerf Training v4.1 (Submission Deadline Oct 11, 2024)
*Framework here is given for the reference implementation. Submitters are free to use their own frameworks to run the benchmark.

| model | reference implementation | framework | dataset | model parameter count
| ---- | ---- | ---- | ---- | ----
| RetinaNet | [vision/object detection](https://github.com/mlcommons/training/tree/master/single_stage_detector) | pytorch | OpenImages | 37M
| Stable Diffusionv2 | [image generation](https://github.com/mlcommons/training/tree/master/stable_diffusion) | pytorch | LAION-400M-filtered | 865M
| BERT-large | [language/nlp](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert) | tensorflow | Wikipedia 2020/01/01 | 340M
| GPT3 | [language/llm](https://github.com/mlcommons/training/tree/master/large_language_model) | paxml,megatron-lm | C4 | 175B
| LLama2 70B-LoRA | [language/LLM fine-tuning](https://github.com/mlcommons/training/tree/master/llama2_70b_lora) | pytorch | SCROLLS GovReport | 70B
| DLRMv2 | [recommendation](https://github.com/mlcommons/training/tree/master/recommendation_v2/torchrec_dlrm) | torchrec | Criteo 3.5TB multi-hot | 167M
| RGAT | [GNN](https://github.com/mlcommons/training/tree/master/graph_neural_network) | pytorch | IGBFull | 25M 

*Note model parameter count is not the same as active parameter that are being trained in the benchmark. 

# MLPerf Training v4.0 (Submission Deadline May 10, 2024)
*Framework here is given for the reference implementation. Submitters are free to use their own frameworks to run the benchmark.

| model | reference implementation | framework | dataset
| ---- | ---- | ---- | ---- |
| resnet50v1.5 | [vision/classification_and_detection](https://github.com/mlcommons/training/tree/master/image_classification) | tensorflow2 | Imagenet
| RetinaNet | [vision/object detection](https://github.com/mlcommons/training/tree/master/single_stage_detector) | pytorch | OpenImages
| 3DUnet | [vision/image segmentation](https://github.com/mlcommons/training/tree/master/image_segmentation/pytorch) | pytorch | KiTS19
| Stable Diffusionv2 | [image generation](https://github.com/mlcommons/training/tree/master/stable_diffusion) | pytorch | LAION-400M-filtered
| BERT-large | [language/nlp](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert) | tensorflow | Wikipedia 2020/01/01
| GPT3 | [language/llm](https://github.com/mlcommons/training/tree/master/large_language_model) | paxml,megatron-lm | C4
| LLama2 70B-LoRA | [language/LLM fine-tuning](https://github.com/mlcommons/training/tree/master/llama2_70b_lora) | pytorch | SCROLLS GovReport
| DLRMv2 | [recommendation](https://github.com/mlcommons/training/tree/master/recommendation_v2/torchrec_dlrm) | torchrec | Criteo 3.5TB multi-hot
| RGAT | [GNN](https://github.com/mlcommons/training/tree/master/graph_neural_network) | pytorch | IGBFull



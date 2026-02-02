### Release Notes for sd_sandbox version 1.7.0

This release of the sd_sandbox project correspond to the RyzenAI version 1.7.0 release. There are some important updates:

* In this release, only text to image generation is supported 
* All models use bfp16 precision for improved performance on supported hardware
* There is no support for SD3,SD 3.5 models, or models that use ControlNet (This will be available with the release of RAI 1.7.1)
    * All pipelines and pipeline groups related to SD3, SD3.5, and ControlNet have been removed from the `pipeline_configs.yaml` file
    * References to SD3, SD3.5, and ControlNet have been removed from the README.md file
* There is no dynamic shape support as this is a feature of the SD3 and SD 3.5 models
* Support has been added for Segmind-Vega model

* A path to the `onnx_custom_ops.dll` used by the models must be provided in the  `pipeline_configs.yaml` file

* A `traceback` flag has been added, that when set to true, will provide detailed error messages in case of failures during pipeline execution.

* An `image-only` flag has been added, that when set to true, will generate images only without profiling (batch mode). This disables warmup and profiling rounds for fastest execution.






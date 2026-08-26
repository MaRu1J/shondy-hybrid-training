# shondy-hybrid-training

Standalone Python training and model-bundle export for the ShonDy dense-grid
hybrid surrogate.

The project consumes Teacher HDF5 files and publishes schema-v2 model bundles.
It has no dependency on the ShonDy C++ solver, CMake, or a solver checkout.
The shared interface is the released
`shondy-hybrid-contract==1.0.0` Python wheel.

## Install

Install the contract wheel from the approved package index, then install this
project with its locked third-party dependencies in the release environment:

```bash
python -m pip install shondy-hybrid-contract==1.0.0
python -m pip install .
```

The CUDA 12.8 release image uses `Dockerfile.cuda12.8` and installs the same
contract version from the configured package index. It does not copy contract
or solver source code into the image.

## CLI

```bash
shondy-train-hybrid --help
```

Training accepts published schema-v2 Teacher trajectories and emits the model
bundle files required by the Runtime contract, including metadata, ONNX
artifacts, the native particle MLP description, and exact SHA256 validation.

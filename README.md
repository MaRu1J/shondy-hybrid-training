# shondy-hybrid-training

Standalone Python training and model-bundle export for the ShonDy dense-grid
hybrid surrogate.

The project consumes Teacher Schema 3 HDF5 files and publishes Model Bundle
Schema 3 artifacts.
It has no dependency on the ShonDy C++ solver, CMake, or a solver checkout.
The shared interface is the released
`shondy-hybrid-contract==2.0.0` Python wheel.

## Install

Install the contract wheel from the approved package index, then install this
project with its locked third-party dependencies in the release environment:

```bash
python -m pip install shondy-hybrid-contract==2.0.0
python -m pip install .
```

The CUDA 12.8 release image uses `Dockerfile.cuda12.8` and installs the same
contract version from the configured package index. It does not copy contract
or solver source code into the image.

## CLI

```bash
shondy-train-hybrid --help
```

Training accepts published Schema 3 Teacher trajectories and emits the model
bundle files required by the Runtime contract, including metadata, ONNX
artifacts, the native particle MLP description, and exact SHA256 validation.
The artifact metadata owns a finite positive `aiDeltaTime`; every input
trajectory in one training run must use the same value.

## Training throughput

The CLI fully validates every selected Teacher frame once, then trains through a
lighter pipeline that keeps HDF5 files open, prefetches upcoming frames, and
reuses deterministic P2G grids. These execution optimizations do not change the
model architecture, channel contract, loss, dtype, or exported artifacts.

The default host-memory limits are 8 GiB for P2G grids and 16 GiB for minimal
training-frame tensors. Set either limit to zero to disable that cache, or lower
the values on memory-constrained hosts:

```bash
shondy-train-hybrid teacher.h5 \
  --output model-bundle \
  --device cuda \
  --preprocessing-device cuda \
  --prefetch-frames 2 \
  --dynamic-grid-cache-gib 8 \
  --training-frame-cache-gib 16
```

Cache occupancy is printed before optimization starts and is recorded in
`training-metrics.json`. Caches are released before model export.

## Loss curve

Generate a self-contained SVG loss curve from an exported artifact bundle. The
plotter uses only the Python standard library:

```bash
python3 plot_loss_curve.py /path/to/model-bundle
```

The default output is `/path/to/model-bundle/training-loss.svg`. Use `--output`
to choose another SVG path.

By default, the one-time P2G statistics pass uses `--device`. Set
`--preprocessing-device cpu` to preserve the legacy CPU reduction numerics while
keeping model training on CUDA; CUDA P2G is faster but floating-point atomic
reduction order can produce small, contract-valid differences.

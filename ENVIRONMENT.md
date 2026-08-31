# Reproducible training environment

The release-qualified baseline is `Dockerfile.cuda12.8`. Its CUDA/cuDNN base
image is pinned by digest and Python dependencies are pinned in
`requirements-cuda12.8.lock`, including the transitive dependency closure.
The image installs that closure with `--no-deps` and runs `pip check`, so pip
cannot silently resolve newer transitive packages. The contract is installed
as the released `shondy-hybrid-contract==2.0.0` wheel from the configured
package index; the solver repository is not part of the build context:

```bash
docker build \
  --build-arg SHONDY_HYBRID_CONTRACT_INDEX_URL=https://<internal-python-index>/simple \
  -f Dockerfile.cuda12.8 \
  -t shondy-hybrid-training:0.1.0 .
```

Run with the NVIDIA container runtime and mount Teacher input read-only plus a
separate output directory. Export validation is successful only when ONNX
Runtime reports `CUDAExecutionProvider` as the active first provider.
Provider availability checked during image build is not sufficient because a
Docker build has no GPU. Release qualification must run the golden grid and
particle ONNX sessions with `session.disable_cpu_ep_fallback=1` under
`docker run --gpus` and confirm CUDA is the first provider.

The `pyproject.toml` entry point is `shondy-train-hybrid`. A developer install
outside the container must install `shondy-hybrid-contract==2.0.0` first. Such
an install is convenient for development but is not the reproducibility
baseline.

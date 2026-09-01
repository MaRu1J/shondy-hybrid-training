# ShonDy Hybrid 端到端操作手册（CLion + `shondy-ai`）

这份手册按实际产物链路操作：

```text
Contract wheel/CMake package
        -> Teacher Writer 写出 Teacher HDF5
        -> Training 读取该 HDF5 并导出 Model Bundle
        -> Runtime 读取该 Bundle 做 hybrid rollout
```

手册针对当前已经贯通的 Schema 3 实现。不要把旧的 Schema 2 示例、旧的
`aiDeltaTime=1e-4` 或旧模型目录混入这条链路。

## 0. 当前版本和目录

四个项目的当前接口矩阵如下。`registry SHA256` 是 Contract 2.0.0 的冻结
registry 内容哈希，四个项目必须使用同一个值。

| 项目/字段 | 当前值 |
| --- | --- |
| Contract package | `2.0.0` |
| Contract registry SHA256 | `7867be1813a2ab05708598811fd24748fc71715ecaa584c30dbb38aef468c12b` |
| Teacher HDF5 schema | `3` |
| Model Bundle schema | `3` |
| Runtime certification profile | `fixed-wall-v2` |
| `aiDeltaTime` | 由 Teacher 配置写入 HDF5 根属性和 frame 属性；Training 复制到 model metadata；Runtime 默认读取 model metadata，若 controlData 也填写则必须相同 |
| Wall storage | Schema 3 compact：参考网格/拓扑只在 `/wallGeometry` 保存一次；运动壁面逐帧保存 `wallState` 位姿和速度，不逐帧复制完整 wall grid |
| Wall pose | `translationXYZ-rotationWXYZ`，右手系主动单位四元数，顺序 `WXYZ` |
| Grid interpolation | `cellCenteredTrilinear8Cell`，3 个 padding cells，不允许越界 clamp |
| Grid Encoder compact-v1 | encoder `[32,48,64,96]`，decoder `[64,48,32]`，latent `16`，参数 `2,072,464` |
| Grid Encoder large-v1 | encoder `[64,96,128,192]`，decoder `[128,96,64]`，latent `16`，参数 `8,250,640` |
| Particle MLP（native ABI） | `[34,128,128,64,3]`，参数 `29,443` |

固定路径：

```bash
export CONTRACT_REPO=/home/ruijin/project/shondy-hybrid-contract
export TEACHER_REPO=/home/ruijin/project/shondy-teacher-writer
export TRAINING_REPO=/home/ruijin/project/shondy-hybrid-training
export RUNTIME_REPO=/home/ruijin/project/shondy-hybrid-runtime

# 本次已验证的 Contract 本地构建产物；重新构建后换成新的目录。
export CONTRACT_CMAKE=/tmp/shondy-contract-rheMv1/cmake
export CONTRACT_WHEEL=/tmp/shondy-contract-rheMv1/dist/shondy_hybrid_contract-2.0.0-py3-none-any.whl
```

`/tmp` 可能被系统清理。路径不存在时，回到第 1 节重新构建 Contract，不要
从其他 solver source tree 复制一份 contract。

## 1. 准备 Conda 环境和安全工作目录

每个终端窗口都先执行：

```bash
source /home/ruijin/.local/miniforge3/etc/profile.d/conda.sh
conda activate shondy-ai

which python
python -V
python -c 'import torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available())'
nvidia-smi
```

Python 应该来自 `.../envs/shondy-ai/bin/python`，CUDA 机器上
`torch.cuda.is_available()` 应为 `True`。没有 GPU 时可以运行 Teacher 和部分
CPU 检查，但不能完成 Runtime 的 CUDA ONNX 验证。

不要直接改仓库里已有的案例，因为其中可能有用户留下的 HDF5、JSON 或日志。
先复制到一个从未使用过的 `/tmp` 目录；目标已存在就换一个名字：

```bash
export TEACHER_CASE=/tmp/shondy-teacher-case-schema3-ai1e3
test ! -e "$TEACHER_CASE" || { echo "目标已存在，请换目录" >&2; exit 1; }
cp -a "$TEACHER_REPO/applicationTests/fundamentalCases/03_damBreak3D" "$TEACHER_CASE"

export RUNTIME_CASE=/tmp/shondy-runtime-case-schema3-ai1e3
test ! -e "$RUNTIME_CASE" || { echo "目标已存在，请换目录" >&2; exit 1; }
cp -a "$RUNTIME_REPO/applicationTests/fundamentalCases/03_damBreak3D" "$RUNTIME_CASE"
```

Teacher HDF5 和 Runtime 的 `controlData.h5` 不是同一个东西：前者是训练
trajectory，后者是 solver 的初始/控制数据。不要互相覆盖。

## 2. 准备 Contract wheel 和 CMake package

### 2.0 本地重新构建 Contract（可选）

如果第 0 节的 `/tmp/shondy-contract-rheMv1` 已不存在，先在
`shondy-ai` 中从 Contract 仓库重新构建。构建目录使用新的临时目录，不要
删除旧的构建目录：

```bash
cd "$CONTRACT_REPO"
source /home/ruijin/.local/miniforge3/etc/profile.d/conda.sh
conda activate shondy-ai

export CONTRACT_BUILD=/tmp/shondy-contract-local-$(date +%Y%m%d-%H%M%S)
mkdir -p "$CONTRACT_BUILD/dist"
python -m pip install build pytest
python -m pytest -q
python -m build --outdir "$CONTRACT_BUILD/dist"

cmake -S . -B "$CONTRACT_BUILD/cmake-build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$CONTRACT_BUILD/cmake-build"
cmake --install "$CONTRACT_BUILD/cmake-build" --prefix "$CONTRACT_BUILD/cmake"

export CONTRACT_WHEEL="$CONTRACT_BUILD/dist/shondy_hybrid_contract-2.0.0-py3-none-any.whl"
export CONTRACT_CMAKE="$CONTRACT_BUILD/cmake"
sha256sum "$CONTRACT_WHEEL" "$CONTRACT_BUILD/dist/shondy_hybrid_contract-2.0.0.tar.gz"
```

如果测试或构建报告 `VERSION`、registry 或 schema 不是第 0 节的值，停止
后续流程，先处理 Contract 版本不一致。

### 2.1 使用已经构建的本地 wheel

在 `shondy-ai` 中安装同一份 wheel，并立即打印版本和 registry hash：

```bash
python -m pip install --force-reinstall "$CONTRACT_WHEEL"

python - <<'PY'
from shondy_hybrid_contract import CONTRACT_V2, REGISTRY_V2_SHA256
print("contractVersion:", CONTRACT_V2["contractPackage"]["version"])
print("registrySha256:", REGISTRY_V2_SHA256)
PY
```

本次已核验 wheel 的 SHA256 是：

```text
87119b4ec86f72a3e01864d8880b2f587cd3dd38ae138c03cc5d2fcd258e3b2c
```

重新构建时以 `sha256sum "$CONTRACT_WHEEL"` 的实际输出为准，并把这个值
记录到实验日志。

### 2.2 在 CLion 中让 C++ 项目使用它

Teacher Writer 和 Runtime 都要做一次下面的设置：

1. 用 CLion 打开对应仓库根目录，不要打开 Contract 的 `src` 子目录。
2. 打开 `Settings | Build, Execution, Deployment | CMake`。
3. 选择 `releaseCUDA` profile（没有 NVIDIA/CUDA 时可先选 `releaseCPU` 做编译检查）。
4. 在 **CMake options** 加入一行：

   ```text
   -DSHONDY_HYBRID_CONTRACT_DIR=/tmp/shondy-contract-rheMv1/cmake
   ```

   这里的目录必须包含
   `lib/cmake/shondy-hybrid-contract/shondy-hybrid-contract-config.cmake`。
5. 确认 `VCPKG_ROOT` 已在 CLion 的 Toolchains 或 Environment 中设置；点击
   **Reload CMake Project**。
6. 在 CLion 的 CMake 输出中确认找到的是 `shondy-hybrid-contract 2.0.0`。

Teacher Writer 的命令行等价构建（仅用于确认 CLion 配置没有遗漏）：

```bash
cd "$TEACHER_REPO"
cmake --preset releaseCUDA -DSHONDY_HYBRID_CONTRACT_DIR="$CONTRACT_CMAKE"
cmake --build --preset releaseCUDA -j8
```

命令行等价检查（不替代 CLion 设置）：

```bash
test -f "$CONTRACT_CMAKE/lib/cmake/shondy-hybrid-contract/shondy-hybrid-contract-config.cmake"
grep -n 'PACKAGE_VERSION "2.0.0"' \
  "$CONTRACT_CMAKE/lib/cmake/shondy-hybrid-contract/shondy-hybrid-contract-config-version.cmake"
```

## 3. Teacher Writer：在 CLion 选择案例并写出 1e-3 数据

### 3.1 选择正确的 CMake target

在 Teacher Writer 的 CLion 窗口中：

1. CMake profile 选择 `releaseCUDA`。
2. 点击右上角运行配置下拉框，选择 CMake target **`shonDy-solver`**，不要
   选择 `shonDy-solver-test`；后者只运行测试，不写 Teacher 数据。
3. 如果列表中没有目标，先执行 `Build | Rebuild Project`，并检查第 2.2 节的
   Contract 路径。
4. 打开 **Run | Edit Configurations...**，新增或复制一个 `shonDy-solver`
   配置：
   - Executable：`build/releaseCUDA/shonDy-solver`
   - Working directory：`$TEACHER_CASE`（也可以保持仓库根目录，因为下面使用绝对路径）
   - Program arguments：

     ```text
     -c /tmp/shondy-teacher-case-schema3-ai1e3 -t 1
     ```

   CLion 直接启动的进程就是单 MPI rank。Teacher Writer 当前要求单 rank，
   不要在这个步骤改成 `mpirun -np 4`。

### 3.2 修改 Teacher 的 `controlData.json`

用 CLion 打开复制后的：

```text
/tmp/shondy-teacher-case-schema3-ai1e3/controlData.json
```

只修改 `solverControls.neuralNetworksSetting` 和
`solverControls.outputSetting.endTime`，不要替换整个 JSON。最小可用值如下：

```json
"neuralNetworksSetting": {
    "writeTeacherMacroFrames": true,
    "aiDeltaTime": 0.001,
    "mlGridCellSize": 0.08,
    "teacherFrameStride": 1,
    "teacherCompression": "none",
    "teacherCompressionLevel": 1,
    "teacherTrajectoryId": "schema3-ai1e3-0400-0900",
    "teacherDebugDenseWallCache": false,
    "conditions": {},
    "trainStart": 0.4,
    "trainEnd": 0.9
}
```

同时将：

```json
"solverControls": {
    "outputSetting": {
        "endTime": 0.9
    }
}
```

改成与实际案例的其余 output 字段合并后的结果。`endTime` 必须是
`aiDeltaTime` 的整数倍；`trainStart=0.4` 和 `trainEnd=0.9` 也必须落在
整数 AI tick 上。

字段含义：

- `writeTeacherMacroFrames=true`：开启 Teacher 写出；为 `false` 不会生成 HDF5。
- `aiDeltaTime=0.001`：物理 AI macro tick，不能用 `teacherFrameStride` 代替。
- `teacherFrameStride=1`：每个 macro tick 都写一帧；它只控制写出选择，不改变
  物理时间步。
- `teacherCompression`：只能是 `none` 或 `deflate`；需要压缩时可用
  `deflate` 加 1 到 9 的 `teacherCompressionLevel`。
- `teacherTrajectoryId`：安全 ASCII 标识符，会出现在输出文件名中，不能包含
  `/`、空格或 `..`。
- `teacherDebugDenseWallCache=false`：正式 compact 产物保持关闭。只在做旧
  dense rasterizer parity 调试时临时打开，不能用它掩盖 compact 数据缺失。
- `trainStart/trainEnd`：写出的时间窗口是 `[trainStart, trainEnd)`；必须非空。

当前 `fixed-wall-v2` 认证路径只覆盖单 rank、fixed wall。Teacher Writer 的
Schema 3 可以表达 prescribed/sampled wall，但当前 Runtime 尚未认证 moving
wall、inlet/outlet、split/merge、porous media 和 pre-water 组合；第一次贯通
请使用现有 `03_damBreak3D` fixed-wall 案例。

### 3.3 运行和确认输出

点击 CLion 右上角绿色运行按钮，或者在终端执行同一个 binary：

```bash
cd "$TEACHER_REPO"
build/releaseCUDA/shonDy-solver \
  -c "$TEACHER_CASE" \
  -t 1
```

`teacherTrajectoryId` 为上例时，输出应在 case 目录中：

```text
/tmp/shondy-teacher-case-schema3-ai1e3/ai-teacher-frames-schema3-ai1e3-0400-0900.h5
```

如果文件名不同，以 solver 日志中打印的实际路径为准；不要移动或覆盖已有
HDF5，后续训练直接使用这一个实际文件。

## 4. 验证 Teacher HDF5

先用 Contract CLI 做 Schema 3 校验：

```bash
export TEACHER_H5="$TEACHER_CASE/ai-teacher-frames-schema3-ai1e3-0400-0900.h5"
shondy-contract validate-teacher-v3 "$TEACHER_H5"
```

应输出 `valid: ...h5`。再查看关键时间和 compact wall 内容：

```bash
python - "$TEACHER_H5" <<'PY'
import sys
import h5py

path = sys.argv[1]
with h5py.File(path, "r") as f:
    frames = sorted(f["frames"])
    first, last = f["frames"][frames[0]], f["frames"][frames[-1]]
    print("contractVersion:", f.attrs["contractVersion"])
    print("schemaVersion:", f.attrs["schemaVersion"])
    print("registry:", f.attrs["contractRegistrySha256"])
    print("aiDeltaTime:", f.attrs["aiDeltaTime"])
    print("frameCount:", len(frames))
    print("first:", first.name, first.attrs["timeStart"], first.attrs["timeEnd"])
    print("last:", last.name, last.attrs["timeStart"], last.attrs["timeEnd"])
    print("wallGeometry datasets:", sorted(f["wallGeometry"].keys()))
    print("first frame datasets:", sorted(first.keys()))
PY

sha256sum "$TEACHER_H5"
```

检查点：

- 根属性和每个 frame 的 `aiDeltaTime` 都是 `0.001`。
- frame 名称是 12 位 macro tick；`timeEnd-timeStart=0.001`。
- `substepCount` 为正，particle 的 `staticId`、`valid`、target shape 合法。
- compact wall 的参考顶点和三角拓扑只在 `/wallGeometry`；fixed wall 不需要
  每帧 `wallState`，moving wall 才需要逐帧位姿/速度。
- 不应出现把 body UUID 或完整 wall grid 重复写入每个 frame 的旧格式。

本次已验证过的参考文件是 500 帧、macro tick `400..899`、时间 `[0.4,0.9]`，
SHA256：

```text
cde6e802f48982d3609d7a64019c085fd0d1087ade5cd03a52f0099d216ea32b
```

## 5. Training：使用刚才的 HDF5 训练和导出 Bundle

### 5.1 训练前检查

```bash
cd "$TRAINING_REPO"
source /home/ruijin/.local/miniforge3/etc/profile.d/conda.sh
conda activate shondy-ai

python -c 'import shondy_hybrid_contract as c; print(c.CONTRACT_V2["contractPackage"]["version"])'
python train_hybrid_model.py --help | sed -n '1,100p'
test -r "$TEACHER_H5"
```

所有输入 trajectory 必须使用同一个 Contract registry、grid、particle
diameter、channel 顺序和 `aiDeltaTime`。单 trajectory 时要显式使用
`--split-fractions 1 0 0`，否则默认的 validation/test split 无法提供独立
trajectory。

### 5.2 先做 smoke（验证接线，不代表完整训练）

输出目录必须不存在或为空；不要复用旧 bundle：

```bash
export COMPACT_SMOKE=/tmp/shondy-training-compact-smoke
test ! -e "$COMPACT_SMOKE" || { echo "目标已存在，请换目录" >&2; exit 1; }

python train_hybrid_model.py "$TEACHER_H5" \
  --output "$COMPACT_SMOKE" \
  --model-profile compact-v1 \
  --epochs 1 \
  --frame-subset-count 16 \
  --device cuda \
  --preprocessing-device cuda \
  --seed 1729 \
  --prefetch-frames 2 \
  --progress-interval-seconds 1 \
  --split-fractions 1 0 0 \
  --collision-post-process full
```

`--frame-subset-count` 只用于快速 smoke，正式训练删除它。不要使用
`--skip-cuda-export-validation`，因为 Runtime 需要 CUDA ONNX provider。

large profile 的接线 smoke 使用新的目录：

```bash
export LARGE_SMOKE=/tmp/shondy-training-large-smoke
test ! -e "$LARGE_SMOKE" || { echo "目标已存在，请换目录" >&2; exit 1; }

python train_hybrid_model.py "$TEACHER_H5" \
  --output "$LARGE_SMOKE" \
  --model-profile large-v1 \
  --epochs 1 \
  --frame-subset-count 16 \
  --device cuda \
  --preprocessing-device cuda \
  --seed 1729 \
  --split-fractions 1 0 0 \
  --collision-post-process full
```

### 5.3 导出一份可用于 Runtime 的模型

下面是 500 帧、20 epoch 的 compact 示例。显存不足时可以先使用上一节的
smoke bundle 验证程序接线，但报告中必须标明它是 smoke，不要称为完整训练。

```bash
export BUNDLE=/tmp/shondy-training-my-run
test ! -e "$BUNDLE" || { echo "目标已存在，请换目录" >&2; exit 1; }

python train_hybrid_model.py "$TEACHER_H5" \
  --output "$BUNDLE" \
  --model-profile compact-v1 \
  --epochs 20 \
  --learning-rate 1e-4 \
  --weight-decay 0 \
  --device cuda \
  --preprocessing-device cuda \
  --seed 1729 \
  --prefetch-frames 2 \
  --progress-interval-seconds 1 \
  --dynamic-grid-cache-gib 8 \
  --training-frame-cache-gib 16 \
  --split-fractions 1 0 0 \
  --checkpoint-epochs 5 10 15 20 \
  --collision-post-process full
```

程序成功结束后，`$BUNDLE` 至少应有：

```text
model-metadata.json
grid-encoder.onnx
particle-mlp.onnx
particle-mlp-native.json
export-validation.json
training-metrics.json
```

查看元数据和每个 artifact 的 SHA256：

```bash
jq '{contractVersion, schemaVersion, contractRegistrySha256, aiDeltaTime,
    certificationProfile, architecture}' "$BUNDLE/model-metadata.json"
sha256sum "$BUNDLE/model-metadata.json" \
  "$BUNDLE/grid-encoder.onnx" \
  "$BUNDLE/particle-mlp.onnx" \
  "$BUNDLE/particle-mlp-native.json" \
  "$BUNDLE/export-validation.json" \
  "$BUNDLE/training-metrics.json"
```

确认 `export-validation.json` 显示 PyTorch/ONNX 数值校验通过，并且 provider
包含 `CUDAExecutionProvider`。`model-metadata.json` 应显示：

- Contract `2.0.0`、schema `3`、registry SHA 与第 0 节完全一致；
- `aiDeltaTime=0.001`、profile 与 Teacher 一致；
- Grid Encoder profile、widths、参数量与 `compact-v1` 或 `large-v1` 一致；
- Particle MLP 固定为 `[34,128,128,64,3]`、`29,443` 参数。

如果打算直接使用上一节的 smoke 结果做 Runtime 接线测试，把 Runtime JSON 中
的 `hybridModelDirectory` 改成 `$COMPACT_SMOKE` 或 `$LARGE_SMOKE` 对应的
绝对路径；JSON 中不能写 shell 的 `$BUNDLE` 变量。smoke 只证明数据链路和
CUDA 导出可用，不代表模型质量。

最后做跨产物校验，而不是只校验 JSON：

```bash
shondy-contract validate-v3 "$TEACHER_H5" "$BUNDLE"
```

它会同时检查 Teacher HDF5、模型 metadata、artifact 文件 SHA256、grid
geometry、channel 顺序、wall transform 和 `aiDeltaTime`。

## 6. Runtime：在 CLion 设置模型并运行多个 AI interval

### 6.1 构建 Runtime

Runtime 必须使用第 2 节同一份 CMake package：

1. 用 CLion 打开 `/home/ruijin/project/shondy-hybrid-runtime`。
2. CMake profile 选择 `releaseCUDA`。
3. CMake options 设置：

   ```text
   -DSHONDY_HYBRID_CONTRACT_DIR=/tmp/shondy-contract-rheMv1/cmake
   ```

4. Reload CMake Project，选择 target `shonDy-solver`，执行 Build。

命令行等价操作：

```bash
cd "$RUNTIME_REPO"
cmake --preset releaseCUDA -DSHONDY_HYBRID_CONTRACT_DIR="$CONTRACT_CMAKE"
cmake --build --preset releaseCUDA -j8
```

不要删除整个 `build` 目录来解决配置问题；如果切换了依赖前缀，使用一个新的
ignored CMake build 目录或在 CLion 中重新配置，并保留原有目录。

### 6.2 设置 Runtime 案例

用 CLion 打开复制后的：

```text
/tmp/shondy-runtime-case-schema3-ai1e3/controlData.json
```

在现有 `solverControls.neuralNetworksSetting` 中加入或修改这些字段：

```json
"neuralNetworksSetting": {
    "runtimeMode": "hybrid-with-fallback",
    "hybridModelDirectory": "/tmp/shondy-training-my-run",
    "aiDeltaTime": 0.001,
    "recoveryPhysicalSteps": 3,
    "maxConsecutiveAiFailures": 3,
    "aiOnlyStopOnFailure": true
}
```

`aiDeltaTime` 可以省略，此时以 `model-metadata.json` 为准；第一次手动贯通
建议显式填 `0.001`，这样 JSON 本身也能被检查。填了就必须与模型 metadata
完全相同，不能写回旧值 `1e-4`。

将此案例的 `solverControls.outputSetting.endTime` 设置为 `0.006`，或者在
Run 配置用 `-e 0.006` 覆盖，以便先跨越 6 个 AI interval 完成短跑。模型目录
使用绝对路径，且不要把 bundle 文件复制到仓库并提交。

### 6.3 配置 CLion Run/Debug

在 **Run | Edit Configurations...** 中新增/复制 `shonDy-solver`：

- Executable：`build/releaseCUDA/shonDy-solver`
- Working directory：`/tmp/shondy-runtime-case-schema3-ai1e3`
- Program arguments：

  ```text
  -c /tmp/shondy-runtime-case-schema3-ai1e3 -t 1 -e 0.006
  ```

- Environment：激活 `shondy-ai` 后，将 ONNX Runtime 和 CUDA 动态库放进
  `LD_LIBRARY_PATH`。当前 Linux 环境可在终端生成一行值后粘贴到 CLion：

  ```bash
  NVIDIA_LIBS=$(find "$CONDA_PREFIX/lib/python3.12/site-packages/nvidia" \
    -mindepth 2 -maxdepth 2 -type d -name lib -printf '%p:')
  echo "${RUNTIME_REPO}/build/releaseCUDA/onnxruntime-runtime:${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  ```

  若 `onnxruntime-runtime` 目录不存在，说明 Runtime 尚未完成 CUDA/ONNX
  构建；先回到 6.1 检查 CMake 输出。

也可以在已激活 `shondy-ai` 的终端直接运行，便于先看完整日志：

```bash
cd "$RUNTIME_REPO"
NVIDIA_LIBS=$(find "$CONDA_PREFIX/lib/python3.12/site-packages/nvidia" \
  -mindepth 2 -maxdepth 2 -type d -name lib -printf '%p:')
export LD_LIBRARY_PATH="$RUNTIME_REPO/build/releaseCUDA/onnxruntime-runtime:${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
build/releaseCUDA/shonDy-solver \
  -c "$RUNTIME_CASE" -t 1 -e 0.006
```

### 6.4 判断推理是否真的成功

日志中应能看到类似这些信息：

```text
Hybrid Runtime mode      : hybrid-with-fallback
Hybrid model directory   : /tmp/shondy-training-my-run
Hybrid AI delta time     : 1.00e-03 s
Hybrid AI step           : accepted
```

短跑结束时检查：

```text
Hybrid AI step count     : 6
Hybrid fallback count    : 0
Hybrid final termination time: 6.00e-03 s
```

至少需要多个 AI interval。确认没有时间漂移、没有越过 `endTime`、没有因为
metadata/registry/artifact SHA/architecture/`aiDeltaTime` 不一致而 fallback。
如果出现 fallback，保留完整日志并记录原因；`hybrid-with-fallback` 允许预期的
物理恢复，但首次认证不应出现未解释的 fallback。

## 7. Contract 发布到 GitHub（管理员步骤）

代码 push 和 GitHub Release 不是一回事。Contract 的 release workflow 在
`.github/workflows/release.yml` 中只对 `v*` tag 触发，并且要求 tag 必须等于
`v$(VERSION)`。当前版本是 `2.0.0`，所以 tag 必须是 `v2.0.0`；不能覆盖已有
tag/release。

在有 GitHub 写权限且安装了 `gh` 或可使用 GitHub UI 的环境执行：

```bash
cd /home/ruijin/project/shondy-hybrid-contract
git fetch origin
git checkout main
git pull --ff-only origin main
git show --stat --oneline c793869aded2762c4bc4f0fd2bf0105c60923e2d
cat VERSION
git tag -l v2.0.0
```

确认 `VERSION` 是 `2.0.0`、目标 commit 已在远端、`v2.0.0` 尚不存在后：

```bash
git tag -a v2.0.0 c793869aded2762c4bc4f0fd2bf0105c60923e2d \
  -m "shondy-hybrid-contract 2.0.0"
git push origin v2.0.0
```

tag push 会触发 GitHub Actions：重新运行 Python 测试，构建 wheel、sdist、
CMake archive 和 `SHA256SUMS`，再创建 GitHub Release。检查 workflow 成功并
下载 release assets 后，消费者才可以改用类似下面的公开 URL：

```bash
python -m pip install \
  https://github.com/MaRu1J/shondy-hybrid-contract/releases/download/v2.0.0/shondy_hybrid_contract-2.0.0-py3-none-any.whl
```

若 tag 已存在、workflow 失败或没有 GitHub 权限，不要 force push、删除 tag 或
手工替换 release；继续使用已核验的本地 wheel/CMake package，并把失败链接和
日志交给仓库管理员。当前本机之前检查过没有 `gh` 命令，因此发布步骤需要在
有权限的 GitHub 环境完成。

## 8. 常见错误排查

| 现象 | 原因和处理 |
| --- | --- |
| `ModuleNotFoundError: shondy_hybrid_contract` | 没激活 `shondy-ai` 或未安装本地 wheel；重新执行第 1、2.1 节。 |
| CMake `Could not find shondy-hybrid-contract` | `SHONDY_HYBRID_CONTRACT_DIR` 必须是包含 `lib/cmake/.../config.cmake` 的 prefix，不是 Contract 源码的 `src`。Reload CMake。 |
| Teacher `aiDeltaTime` 非法或窗口为空 | `writeTeacherMacroFrames=true`，`aiDeltaTime>0`，`endTime/trainStart/trainEnd` 都必须是 `0.001` 的整数倍，且 `trainEnd>trainStart`。 |
| HDF5 schema/registry mismatch | Teacher Writer、Training wheel、Runtime CMake 使用了不同 Contract；全部切回同一份 2.0.0 package，并重新生成产物。 |
| Training 拒绝非空 output | 训练程序保护已有模型；给 `--output` 换一个全新的 `/tmp` 目录，不要删除旧 bundle。 |
| `aiDeltaTime` mismatch | 比较 Teacher 根属性、`model-metadata.json` 和 Runtime controlData；三者应为同一个有限正值。不要把它固定回旧的 `1e-4`。 |
| bundle artifact SHA mismatch | 文件被改写、拷贝不完整或 metadata 不是这组 ONNX；重新导出到新目录并再次运行 `shondy-contract validate-v3`。 |
| `CUDAExecutionProvider` 不可用 | 检查 `shondy-ai` 的 CUDA/ONNX Runtime 包、`LD_LIBRARY_PATH` 和 NVIDIA 驱动；不要用 CPU provider 冒充 release validation。 |
| Runtime 报 particle diameter/grid/channel 不一致 | Runtime case 必须与 Teacher 和 bundle 的物理案例一致；不要将另一个案例的 bundle 直接套用。 |
| Runtime 一启动就拒绝 moving wall 或 inlet/outlet | 当前 Runtime 首个认证 profile 是 `fixed-wall-v2`；先用 `03_damBreak3D` fixed-wall 完成贯通。 |
| 只看到一个 AI step 或最终时间越过 `endTime` | `-e`、`outputSetting.endTime`、model `aiDeltaTime` 没对齐；使用 `-e 0.006` 做 6 interval 短跑并检查 scheduler 日志。 |
| CLion 找不到 `shonDy-solver` | 选错 target 或没有 `BUILD_PF2=ON`；重新 Reload CMake，确认选择 `releaseCUDA` 和 `shonDy-solver`。 |

## 9. 一次完整操作后的留档

建议把下面信息保存到实验记录，而不是提交到任何仓库：

```bash
git -C "$CONTRACT_REPO" rev-parse HEAD
git -C "$TEACHER_REPO" rev-parse HEAD
git -C "$TRAINING_REPO" rev-parse HEAD
git -C "$RUNTIME_REPO" rev-parse HEAD
sha256sum "$CONTRACT_WHEEL" "$TEACHER_H5"
sha256sum "$BUNDLE"/model-metadata.json "$BUNDLE"/grid-encoder.onnx \
  "$BUNDLE"/particle-mlp.onnx "$BUNDLE"/particle-mlp-native.json
jq '{contractVersion, schemaVersion, contractRegistrySha256, aiDeltaTime,
    certificationProfile, architecture}' "$BUNDLE/model-metadata.json"
```

本次贯通验证使用的参考 commit 是：Contract
`c793869aded2762c4bc4f0fd2bf0105c60923e2d`、Teacher Writer
`aebbf41a1a73b0e616678a65c05f5a86f48fc198`、Training
`3a5fcfeff8c16599074473353ca6a95291166f94`、Runtime
`12bb7ecc0d9038c341396e0d0b0f27fba036c94f`。实际操作时以四个仓库当前
`git rev-parse HEAD` 为准，并记录后续项目实际使用的上游 commit/产物。

# ShonDy Hybrid 模型训练教程

本文介绍如何准备训练环境和 Teacher Schema 3 数据、启动训练、观察训练
进度，以及在哪里查找最终导出的模型。本文所有命令均针对当前 training
项目：

```text
/home/ruijin/project/shondy-hybrid-training
```

## 1. 训练前需要准备什么

开始训练前，需要准备以下内容：

1. 一块能被 PyTorch 识别的 NVIDIA GPU。本文示例使用 CUDA 完成数据预处理、
   模型训练和 ONNX 导出一致性验证。
2. 安装了本项目依赖的 Python 3.12 环境。已经核验过的本机环境是：

   ```text
   /home/ruijin/.local/miniforge3/envs/shondy-ai
   ```

3. `shondy-hybrid-contract==2.0.0`、Teacher Schema 3 和 Model Bundle
   Schema 3。本文命令通过 `PYTHONPATH` 使用已经核验的 contract 源码：

   ```text
   /home/ruijin/project/shondy-hybrid-contract/src
   ```

4. 至少一个可读的 Teacher Schema 3 HDF5 trajectory。训练程序以只读方式
   打开 H5。一次训练所使用的所有 trajectory 必须具有相同的 grid、condition
   schema、particle diameter 和 `aiDeltaTime`。
5. 足够的主机内存和输出磁盘空间。对本文使用的 500-frame 数据，实测动态
   grid cache 约为 0.76 GiB，training frame cache 约为 4.90 GiB。
6. 一个尚不存在的新输出目录，或者一个已经存在但内容为空的目录。训练程序
   会拒绝覆盖非空目录，以保护已有模型、metrics 和 checkpoint。

可复现的容器环境另见 `ENVIRONMENT.md`。本文使用的是已经核验过的本机
Python 环境，而不是 Docker。

## 2. 检查 GPU、Python 和输入数据

开始长时间训练前，建议先执行以下只读检查：

```bash
cd /home/ruijin/project/shondy-hybrid-training

nvidia-smi

PYTHONPATH=/home/ruijin/project/shondy-hybrid-contract/src \
  /home/ruijin/.local/miniforge3/envs/shondy-ai/bin/python \
  -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"

test -r /home/ruijin/project/shondy-teacher-writer/applicationTests/generated/03_damBreak3D-schema3-ai1e3-0400-0900/ai-teacher-frames-schema3-ai1e3-0400-0900.h5
```

`torch.cuda.is_available()` 应输出 `True`。最后一个 `test -r` 命令成功时
不会输出内容。

本文已经核验过的训练数据位于：

```text
/home/ruijin/project/shondy-teacher-writer/applicationTests/generated/03_damBreak3D-schema3-ai1e3-0400-0900/ai-teacher-frames-schema3-ai1e3-0400-0900.h5
```

该文件包含一个 trajectory、500 个真实宏步、tick `400..899`，并且
`aiDeltaTime=0.001`。它不包含 production dense wall grid；训练程序会根据
Schema 3 reference wall geometry 确定性重建 fixed wall grid。

## 3. 使用已核验 H5 开始训练

下面这条命令使用 `compact-v1` profile 训练 20 个 epoch，可以直接在终端中
运行：

```bash
cd /home/ruijin/project/shondy-hybrid-training

PYTHONPATH=/home/ruijin/project/shondy-hybrid-contract/src \
  /home/ruijin/.local/miniforge3/envs/shondy-ai/bin/python \
  train_hybrid_model.py \
  /home/ruijin/project/shondy-teacher-writer/applicationTests/generated/03_damBreak3D-schema3-ai1e3-0400-0900/ai-teacher-frames-schema3-ai1e3-0400-0900.h5 \
  --output /home/ruijin/project/shondy-hybrid-training/training-runs/dambreak3d-schema3-compact-v1-20e \
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

命令中的输出目录当前尚不存在。运行成功后该目录会包含模型文件；再次训练时
不要复用这个非空目录，而应为新实验换一个新的目录名称。

这份 H5 只有一个 trajectory，因此命令必须显式使用 `--split-fractions
1 0 0`。该 trajectory 全部用于训练，validation 和 test 为空，对应指标将为
`null`。要获得有效的 validation/test 指标，必须准备多个相互独立的 Teacher
trajectory。不要把同一 trajectory 的不同 frame 分到不同 split，否则会造成
trajectory 信息泄漏。

## 4. 如何观察训练进度

命令直接连接终端时，每一个 epoch 都会显示动态更新的进度条，例如：

```text
Epoch 3/20 [###############...............] 250/500  50.00% elapsed    8.1s ETA     8.1s
```

其中包含当前 epoch、已处理 frame 数量、百分比、已用时间和预计剩余时间。
每个 epoch 完成后，程序还会输出 training standardized MSE 和仅包含优化循环的
epoch 耗时。

如果输出被重定向到日志或由任务调度系统收集，动态进度条会自动替换为定期
JSON 记录。`--progress-interval-seconds 1` 表示非交互模式下大约每秒输出一条
进度记录。

第一个 epoch 前还需要完成 H5 索引、wall reconstruction、P2G statistics 和
cache 预处理。程序会先输出 selected frames 和 cache population JSON，因此
可以区分“仍在正常预处理”和“训练进程卡住”。

## 5. 输出模型在哪里

上面的示例命令会把完整模型写到：

```text
/home/ruijin/project/shondy-hybrid-training/training-runs/dambreak3d-schema3-compact-v1-20e
```

目录中的主要文件如下：

| 文件 | 用途 |
|---|---|
| `model-metadata.json` | Contract 2.0.0 Model Bundle Schema 3 metadata |
| `training-metrics.json` | loss、epoch 时间、峰值显存、profile、参数量、split 和导出结果 |
| `export-validation.json` | profile 校验和 PyTorch/ONNX 数值一致性结果 |
| `model-state.pt` | 完整 PyTorch model state 和 architecture metadata |
| `grid-encoder.onnx` | 推理用 Grid Encoder |
| `particle-mlp.onnx` | 推理用 Particle MLP |
| `particle-mlp-native.json` | row-major native Particle MLP 参数 |
| `grid-encoder.pt` | TorchScript Grid Encoder 兼容 artifact |
| `particle-mlp.pt` | TorchScript Particle MLP 兼容 artifact |
| `checkpoint-epoch-XXXX.pt` | 由 `--checkpoint-epochs` 请求的恢复 checkpoint |

程序最后一行还会输出 artifact 目录的绝对路径。只有当推理导出、数值一致性
验证全部结束，并且 `training-metrics.json` 已写入后，才算完整训练成功。

## 6. 如何选择模型 profile

`compact-v1` 是默认 profile；空 condition 时有 2,101,907 个可训练参数。
`large-v1` 有 8,280,083 个参数，需要更多训练时间和 GPU 显存。

训练 large profile 时，应换用新的输出目录，并将：

```text
--model-profile compact-v1
```

替换为：

```text
--model-profile large-v1
```

checkpoint 与 profile 绑定。compact checkpoint 不能加载到 large 模型中；
profile 或实际层宽不一致时，加载程序会明确拒绝。

## 7. 从 checkpoint 恢复训练

恢复训练时也必须使用新的输出目录。例如从 epoch 10 的 checkpoint 继续：

```bash
cd /home/ruijin/project/shondy-hybrid-training

PYTHONPATH=/home/ruijin/project/shondy-hybrid-contract/src \
  /home/ruijin/.local/miniforge3/envs/shondy-ai/bin/python \
  train_hybrid_model.py \
  /home/ruijin/project/shondy-teacher-writer/applicationTests/generated/03_damBreak3D-schema3-ai1e3-0400-0900/ai-teacher-frames-schema3-ai1e3-0400-0900.h5 \
  --output /home/ruijin/project/shondy-hybrid-training/training-runs/dambreak3d-schema3-compact-v1-resumed \
  --model-profile compact-v1 \
  --resume-checkpoint /home/ruijin/project/shondy-hybrid-training/training-runs/dambreak3d-schema3-compact-v1-20e/checkpoint-epoch-0010.pt \
  --epochs 10 \
  --device cuda \
  --preprocessing-device cuda \
  --split-fractions 1 0 0 \
  --progress-interval-seconds 1
```

这里的 `--epochs` 表示本次命令实际执行多少个优化 epoch，不会自动减去
checkpoint 中保存的 epoch 编号。

## 8. 多 trajectory 和 validation/test

可以在所有 options 之前依次传入多个 Teacher H5 路径。所有文件必须具有完全
相同的 `aiDeltaTime` 和 model contract。具备足够多相互独立的 trajectory 后，
可以使用：

```text
--split-fractions 0.8 0.1 0.1
```

seeded split 是稳定的，并保证每个 trajectory 只属于 training、validation 或
test 中的一个。如果非零 split 无法分配到 trajectory，训练会明确报错，不会
静默地把同一 trajectory 的 frame 泄漏到不同 split。

## 9. 训练安全规则

- Teacher H5 只能作为只读输入使用。
- 每次实验都使用新的输出目录。
- 对比不同 profile 时，保持 seed 和其他所有 flags 完全一致。
- 不要使用 `teacherFrameStride` 改变物理时间尺度。
- 最终全数据训练不要使用 `--frame-subset-count`。
- release bundle 不要使用 `--skip-cuda-export-validation`。
- 接受模型前检查 `training-metrics.json` 和 `export-validation.json`。

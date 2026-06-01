# 实验技术升级清单

本项目作为新技术实验平台，记录可接入/替换/增强的模块。按「收益 / 风险 / 接入成本」排序。
场景默认：单说话人**唱歌** SVC，7h 数据，16G 显存(4080S)。

> 约定：标 ✅ 的为已接入，🟢 优先试，🟡 值得实验，🔴 大改/谨慎。

---

## 已接入

| 模块 | 实现位置 | 说明 |
|------|----------|------|
| ✅ Lion 优化器 | `modules/optimizers.py`、`train.py` | config `train.optimizer: "lion"` 切换。**必须调小 lr 至 AdamW 的 1/3~1/10**(如 2e-5)，配 `weight_decay: 0.1`。GUI 已可切 |
| ✅ Transformer Flow | `models.py` | config `model.use_transformer_flow: true`，配 `n_layers_trans_flow`。吃显存/吃数据，7h 可试 |
| ✅ 音量增强 vol_aug | `data_utils.py`、`preprocess_*` | 预处理加 `--vol_aug`，连带开 `vol_embedding`。推理输入音量不定时强烈建议开 |
| ✅ bf16 混合精度 | `train.py` | `fp16_run: true` + `half_type: "bf16"` 才生效。4080S 推荐 |

---

## 内容编码器选型（唱歌场景结论）

| 编码器 | 唱歌适配 | 结论 |
|--------|----------|------|
| **vec768l12** (ContentVec) | ⭐ 最佳 | **保持默认**。社区唱歌模型主流选择，发音动态保留好 |
| **wavlmbase+** (WavLM) | 可对照 | 🟡 唯一值得 A/B 的对象，音色解耦更强，咬字可能更清，但提升不保证 |
| whisper-ppg / -large | ❌ 不推荐 | ASR 模型抽 PPG 特征，刻意丢音高；高音/长音/颤音处咬字易糊。large-v2 还特别重。**留给纯语音转换** |
| hubertsoft / cnhubertlarge | 一般 | 无明显唱歌优势 |

---

## 待实验清单（按优先级）

### 🟢 优先（收益明确、风险低）

**1. 声码器 Vocoder：nsf-hifigan → BigVGAN v2 / Vocos**
- 当前 `vocoder_name: "nsf-hifigan"`，音质天花板所在。
- BigVGAN v2：高频/泛化更强，唱歌齿音、气声还原好。Vocos：ConvNeXt+iSTFT，推理更快。
- 接入成本：中。需新增 vocoder 封装、改 `models.py` 生成端与推理 `infer_tool.py`。

**2. F0 提取：确认用 rmvpe/fcpe**
- 唱歌对 f0 精度极敏感。`F0_METHODS` 已含 rmvpe/fcpe。
- **唱歌首选 rmvpe**(抗噪、八度错误少)，**fcpe** 速度快可做对照。避免 crepe/harvest/dio 用于唱歌。
- 接入成本：零，配置即用。

**3. LR 调度：ExponentialLR → CosineAnnealing + warmup**
- 当前 `train.py:111` 指数衰减偏粗。余弦退火对长训练收敛更平滑。
- 接入成本：低，改 `train.py` 调度器构造。

### 🟡 值得实验

**4. 优化器对照：+ Schedule-Free AdamW**
- 已有 AdamW/Lion，可再加 Meta 的 schedule-free 做三方对照，免调度器。

**5. 内容编码器 A/B：vec768l12 vs wavlmbase+**
- 见上方编码器表。唱歌坚持 vec768l12 为主，wavlmbase+ 做对照。

### 🔴 大改 / 谨慎

**6. 扩散头：DDPM → Flow Matching / Rectified Flow**
- `diffusion/` 现为 DDPM 类，FM 可把采样步数砍到几步、质量相当。
- 接入成本：高。需重写 `diffusion/solver.py` 与采样逻辑。

**7. 判别器：MultiPeriod → + MS-STFT / CQT discriminator**
- EnCodec/DAC 同款，对高频更敏感。需改 `models.py` 判别器与 loss，GAN 调参麻烦。

---

## 建议实验路线

Lion(✅) → 声码器 BigVGAN/Vocos(音质杠杆最大) → 确认 rmvpe → 余弦调度 → 编码器 A/B。
优化器与调度器是低垂果实；**声码器才是唱歌音质的真正杠杆**。

> 注意：结构类参数(transformer_flow / vol_embedding / speech_encoder / vocoder)一旦训练即固化，
> 切换 = 重训新模型。做对照请开独立工程，勿在同一 config 反复改。


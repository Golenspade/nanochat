# nanochat

![nanochat logo](dev/nanochat.png)

> 100 美元能买到的最好的 ChatGPT。

这个仓库是一个完整的类 ChatGPT 大语言模型实现，代码库简洁、最小化、易于修改、依赖少。nanochat 设计为在单个 8XH100 节点上运行，通过 [runs/speedrun.sh](runs/speedrun.sh) 这样的脚本端到端运行整个流程。这包括分词、预训练、微调、评估、推理，以及通过简单的 UI 提供 Web 服务，让你可以像使用 ChatGPT 一样与自己的大语言模型对话。nanochat 将成为 Eureka Labs 正在开发的 LLM101n 课程的毕业项目。

## 快速开始

体验魔力的最快方式是运行 [runs/speedrun.sh](runs/speedrun.sh) 脚本，它会训练并推理 100 美元级别的 nanochat。在 8XH100 节点上（24 美元/小时），总运行时间约 4 小时。从你喜欢的云服务商那里启动一个新的 8XH100 GPU 机器（例如我使用并喜欢 [Lambda](https://lambda.ai/service/gpu-cloud)），然后启动训练脚本：

```bash
bash runs/speedrun.sh
```

或者，由于脚本要运行 4 小时，我喜欢在新的 screen 会话 `speedrun` 中这样启动（同时将输出记录到 `speedrun.log`）：

```bash
screen -L -Logfile speedrun.log -S speedrun bash runs/speedrun.sh
```

如果你不太熟悉 screen，可以参考 [screen 速查表](https://gist.github.com/jctosta/af918e1618682638aa82)。你可以在 screen 会话中观看进度，或者用 `Ctrl-a d` 分离会话，然后用 `tail speedrun.log` 查看进度。现在等待 4 小时。完成后，你可以通过类似 ChatGPT 的 Web UI 与你的大语言模型对话。确保你的本地 uv 虚拟环境已激活（运行 `source .venv/bin/activate`），然后启动服务：

```bash
python -m scripts.chat_web
```

然后访问显示的 URL。确保正确访问，例如在 Lambda 上使用你所在节点的公共 IP，后跟端口，例如 [http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/)。然后就可以像平常使用 ChatGPT 一样与你的大语言模型对话了！让它写故事或诗歌。问它你是谁，看看它的幻觉。问它为什么天空是蓝色的。或者为什么是绿色的。speedrun 是一个 4e19 FLOPs 能力的模型，所以有点像在和幼儿园小朋友聊天 :)

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

你还可以 `cat report.md` 查看项目目录中生成的文件，其中包含运行的"成绩单"，即一系列评估和指标。在最后，你会看到一个汇总表，例如：

---

- 字符数：333,989
- 行数：8,304
- 文件数：44
- Token 数（约）：83,497
- 依赖项（uv.lock 行数）：2,004

| 指标            | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2219   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2875   | 0.2807   | -        |
| ARC-Easy        | -        | 0.3561   | 0.3876   | -        |
| GSM8K           | -        | 0.0250   | 0.0455   | 0.0758   |
| HumanEval       | -        | 0.0671   | 0.0854   | -        |
| MMLU            | -        | 0.3111   | 0.3151   | -        |
| ChatCORE        | -        | 0.0730   | 0.0884   | -        |

总运行时间：3小时51分钟

---

（默认情况下你的表格可能缺少 RL 数字）。关于 speedrun 脚本的更多信息以及需要注意和期待的内容，请参考我在仓库讨论区发布的演练：["介绍 nanochat：100 美元能买到的最好的 ChatGPT"](https://github.com/karpathy/nanochat/discussions/1)。

## 更大的模型

不出所料，100 美元不足以训练一个高性能的 ChatGPT 克隆。事实上，大语言模型以其数百万美元的资本支出而闻名。就我们的目的而言，我认为还有两个更有趣的规模。首先是约 300 美元级别的 d26 模型（即 depth=26），训练时间约 12 小时，性能略优于 GPT-2 的 CORE 分数。其次是 1000 美元级别（约 41.6 小时），因为这是一个不错的整数。但这两者尚未完全支持，因此还没有附加到主分支中。

话虽如此，为了让你有个概念，[runs/speedrun.sh](runs/speedrun.sh) 文件训练 GPT-2 级别的 d26 模型所需的示例更改只涉及三处修改：

```bash
...
# 你需要下载更多的数据分片用于预训练
# 获取参数数量，乘以 20 得到 token 数，乘以 4.8 得到字符数，
# 除以 2.5 亿得到分片数量。待办：需要改进这个...
python -m nanochat.dataset -n 450 &
...
# 使用 --depth 增加模型大小。为了不 OOM，将设备批次大小减半 32 -> 16：
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16
...
# 确保在中期训练时使用相同的设置：
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
```

就是这样！最需要注意的是确保有足够的数据分片进行训练（否则代码会循环并在同一训练集上进行更多轮次，稍微降低学习速度），以及管理你的内存/显存，主要是通过减少 `device_batch_size` 直到能够容纳（脚本会自动通过增加梯度累积循环次数来补偿，简单地将并行计算转换为顺序计算）。

关于运行 nanochat 的计算环境的更多信息：

- 代码在 Ampere 8XA100 GPU 节点上也能正常运行，但会稍慢一些。
- 所有代码在单个 GPU 上也能正常运行，只需省略 `torchrun`，并会产生几乎相同的结果（代码会自动切换到梯度累积），但你需要等待 8 倍的时间。
- 如果你的 GPU 显存少于 80GB，你需要调整一些超参数，否则会 OOM / 显存不足。在脚本中查找 `--device_batch_size` 并减少它直到能够容纳。例如从 32（默认）减少到 16、8、4、2，甚至 1。再少的话你需要更了解自己在做什么，并更有创意。
- 大部分代码是相当标准的 PyTorch，所以应该能在任何支持它的平台上运行 - xpu、mps 等，但我还没有开箱即用地实现这些，所以可能需要一些调整。

## 提问

nanochat 设计得简短而精炼。这样做的一个大优势是我们可以将所有文件打包在一起，复制粘贴到你喜欢的大语言模型中提出任意问题。例如，我喜欢使用 [files-to-prompt](https://github.com/simonw/files-to-prompt) 工具这样打包仓库：

```bash
files-to-prompt . -e py -e md -e rs -e html -e toml -e sh --ignore "*target*" --cxml > packaged.txt
```

这包括所有 py、rs、html、toml、sh 文件，排除 `rustbpe/target` 文件夹，并选择 cxml 输出格式。所有内容都写入 `packaged.txt` 文件，目前约 330KB（即远低于最先进大语言模型的约 10 万 token），约 45 个文件中的 8K 行代码。

或者，我推荐使用 Devin/Cognition 的 [DeepWiki](https://deepwiki.com/) 来询问这个仓库的问题。在这个仓库的 URL 中，只需将 github.com 改为 deepwiki.com，就可以开始了。

## 测试

我在这方面投入不多，但存在一些测试，特别是针对分词器的。例如运行：

```bash
python -m pytest tests/ -v
```

## 贡献

nanochat 远未完成。目标是改进微型模型的最先进水平，使其能够在低于 1000 美元的预算下端到端地工作。可访问性不仅关乎总体成本，还关乎认知复杂性 - nanochat 不是一个可详尽配置的大语言模型"框架"；代码库中不会有巨大的配置对象、模型工厂或 if-then-else 怪物。它是一个单一的、连贯的、最小化的、可读的、可修改的、最大程度可分叉的"强基线"代码库，设计为端到端运行并产生一个具体的 ChatGPT 克隆及其成绩单。

## 致谢

- 名称（nanochat）源自我早期的项目 [nanoGPT](https://github.com/karpathy/nanoGPT)，该项目仅涵盖预训练。
- nanochat 还受到 [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) 的启发，该项目通过清晰的指标和排行榜将 nanoGPT 仓库游戏化，并借鉴了其许多想法和一些预训练实现。
- 感谢 [HuggingFace](https://huggingface.co/) 提供的 fineweb 和 smoltalk。
- 感谢 [Lambda](https://lambda.ai/service/gpu-cloud) 为开发此项目提供的算力。
- 感谢首席大语言模型调教师 🧙‍♂️ Alec Radford 的建议/指导。

## 引用

如果你发现 nanochat 对你的研究有帮助，请简单引用为：

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## 许可证

MIT

# 视频搬运算法运行器

两个分支独立执行，不相互替代，不读取历史视频、预制轨迹、蒙版或标定模型。原实验目录保持不变。

- `fixed_r16/`：来自 `r16_speed_v1/tool/算法`。固定亮色中性水印自动标定、逐帧存在性验证、R16 四轮稳健连续性求解、边缘连续性及仿射后处理。保留 NumPy 数学实现和 C++ 内核源码。
- `moving_v6/`：`adaptive_temporal.py`、`coherent_repair.py` 来自移动水印增强 V6；`temporal_repair.py` 复用 V4 的纯几何函数；`spatial_fallback.py` 保留 V4 空间降级；`affine_overlay.py` 来自 V5；`track_logo.py` 来自原通用模板跟踪器。`engine.py` 从本次输入重新执行双向跟踪、动态蒙版放置、场景切分、自适应 donor 选择、连续来源标签及留出集仿射标定。`spatial_continuity.py` 仅复用 R16 的空间求解核心，不调用固定水印检测或固定轨迹。移动分支另用双极性、颜色无关的窄证据隔离和结构置信门控，证据不足时保留旧空间降级。

V6 空间修复区分可写蒙版与取样排除蒙版，只从当前帧中原始可见的背景拟合，保留已通过时序验证的恢复像素。具有足够内部面积的大孔洞使用 `exemplar_texture.py`，从完全避开排除蒙版的原始 11×11 图块逐步合成，减少大孔洞的扩散模糊。孔洞几何以完整取样排除区域计算；部分像素已获时序恢复不会使原始不可见区域变成有两侧观测的细缝。内部合成可以经过完整孔洞，最终只写调用者指定的剩余蒙版。可见背景通过四象限留出平面检验的区域继续插值；细缝维持原空间求解。局部方向置信不能证明大孔洞内部始终沿同一方向，不能独自阻止图块候选。合成像素只以低权重约束接缝，不能进入干净图块库或反算标定。方向延续、图块合成、边缘和仿射拟合均不能证明被完全遮挡的纹理或正文真值。未可靠定位的帧仍保留原图。

`background_registration.py` 为时序供体增加 SIFT 特征和遮罩平移搜索，用于原 LK 未能对齐的大位移。描述子完整取样范围及平移匹配均排除目标和供体自身水印；搜索最优值触边、平坦方向和浮点取消误差范围内的伪极小值不能提供可辨识的平移。有效提议仍需通过原局部可见背景误差、两个独立来源的一致性及场景切点检查。增加候选不降低接收阈值，也不允许跨真实切镜借用背景。

移动跟踪后运行两种独立补检。`tracking_bridge.py` 只在场景内、可信原始锚点间验证有限短缺口，要求正反向图像观测、原始锚点轮廓一致和端点闭合，运动预测不能直接成为接受位置。`support_recovery.py` 使用不可变参考模板的支持区域相关性、分区相关性和有方向轮廓验证独立候选，附近原始锚点只限定搜索区域；补回帧不能成为其他候选的锚点。普通跟踪、桥接和参考补检分别计数，不能把覆盖率当作画质真值。

合法参考的支持区域不足 24 像素时，两种新增补检因证据不足跳过，保留原主追踪的接受和拒绝结果；不会因此让既有可用任务失败。空白或其他无效参考仍由主入口拒绝。

V6 donor 缓存保留原像素：轨迹范围较小时缓存背景上下文并集，范围较大时使用固定网格块，均限制在 96 MiB；动态蒙版缓存限制在 8 MiB / 256 项；标定已计算的时序局部结果限制在 24 MiB / 24 项，并在正式处理时消费一次。缓存规则从输入尺寸、帧率和轨迹推导，不包含素材名称、固定坐标或预制轨迹。

R16 保留原始计算核心、预编译存在性几何、12 像素上下文局部块以及有界多进程传输。相同输入像素、蒙版和计算实现使用同一核心；通用入口将原 FFmpeg 每秒三帧标定改为区间均匀、有上限的标定采样，因此不承诺整个 CLI 与原实验逐像素等价。移动分支是对 V6 修复核心的通用入口重构，保留原始四秒搜索、约每秒十帧候选及最多十二个 donor，未移植旧 V5 横向变形标定链；当前仅支持平移，也不承诺与旧入口逐像素等价。

## 依赖与启动

Python 3.10+、NumPy、OpenCV、带 `libx264`/AAC 的 FFmpeg 和 FFprobe。建议在独立虚拟环境安装：

```bash
python3 -m venv .local/watermark-venv
.local/watermark-venv/bin/pip install -r backend/tools/watermark/requirements.txt
.local/watermark-venv/bin/python backend/tools/watermark/runner.py --check
.local/watermark-venv/bin/python backend/tools/watermark/runner.py --request request.json --output output.mp4
```

`--check` 只输出 JSON：`{"branches":{"fixed-r16":{"available":true,"reason":""},"moving-v6":{"available":true,"reason":""}}}`。缺少 Python 包或 FFmpeg 时也可运行该检查，分支返回不可用。它检查依赖能否导入，不承诺任意素材均符合算法假设。

FFmpeg/FFprobe 默认从 `PATH` 查找，可分别配置 `CANVAS_FFMPEG_PATH`、`CANVAS_FFPROBE_PATH`。固定分支优先使用 C++17 编译器 `CXX` 或 `c++`，把本机原生内核编译到系统临时目录下的独立源码哈希缓存；`CANVAS_WATERMARK_NATIVE_CACHE` 可指定缓存根目录。构建不写入仓库，也不提交二进制。macOS 和 Linux 使用各自动态库格式；Linux 的可选 Accelerate 背景拟合优化使用原 NumPy 实现，其他便携内核可编译。没有编译器或编译失败时使用原数学实现，报告列出构建成功的内核，性能会降低；`solverUsage` 另记录 R16 实际走 native、NumPy 和数值失败降级的帧数。跨浮点实现不承诺逐字节相同。`CANVAS_WATERMARK_DISABLE_NATIVE=1` 强制使用 NumPy。历史模块内的独立实验构建入口不作为产品命令，统一使用 `native_build.py` 或 `runner.py`。

## 请求合同

```json
{
  "input": "/absolute/authorized-local-input.mp4",
  "branch": "moving-v6",
  "startSeconds": 0,
  "durationSeconds": 10,
  "workers": 2,
  "target": "top-left",
  "referenceSeconds": 1,
  "roi": {"x": 40, "y": 30, "width": 160, "height": 60}
}
```

`input` 只接受本地已有文件；业务调用方必须先检查资源归属。`branch` 为 `fixed-r16` 或 `moving-v6`。`durationSeconds=0` 处理到片尾，不截成固定十秒。时间使用非负半向上四舍五入到帧边界，片段帧数为绝对结束帧减绝对开始帧，避免 29.97fps 等相邻十秒段重复或漏帧。报告记录实际起点和时长。`referenceSeconds` 是原视频绝对秒，可在处理片段之外。ROI 是原视频像素整数，右下边界由 `x+width`、`y+height` 决定。

每段只跟踪处理范围前后各四秒，参考图独立读取；参考时刻在范围外时，采用原全局门槛重新定位，不解码参考帧与目标段之间的整片视频。启动时只读取媒体元数据，不再整片扫描 CFR 时间戳。FFmpeg 按时间跳转并解码当前片段、参考和所需上下文，所有读取共用基于平均帧率的输出时间格。后端批任务可传内部字段 `validationCacheDir`，用于共享手动区域的参考标定；不再创建或读取整片时间轴校验缓存。

移动分支必须提供 ROI。固定分支优先使用 ROI；省略时按 `target` 从 `top-left`、`top-right`、`bottom-left`、`bottom-right` 选取原画面角落。`workers` 为 1..16；固定分支使用有界多进程队列且只传局部修复块。移动分支的图像修复线程最多两个，编码线程遵循请求值。

标准输出每行一条 JSON：`{"progress":0,"stage":"probing"}`，直至 `{"progress":100,"stage":"completed","report":{...}}`。错误输出写到 stderr，最后一行为 `ERROR: <不含路径的简短原因>`，并返回非零退出码。最终 MP4 仅在所有帧处理和编码成功后原子落盘，同时写入同名 `.report.json`，包括使用的分支、动态跟踪接受帧数、时序恢复像素、反算像素、空间降级像素和局限。输出路径及报告路径必须尚不存在。

报告共有字段为 `branch`、`sourceDimensions`、`startSeconds`、`durationSeconds`、`frames`、`fps`、`roi`、`algorithm`、`changedPixels`、`native`、`limitations: string[]`。移动分支另有 `trackedFrames`、`untrackedFrames`、`trackingCoverage`（0..1）、`temporalRecoveredPixels`、`inversePixels`、`spatialFallbackPixels`、`calibratedInversePixels`、`spatialContinuity`。后者记录平坦填补、边缘、接缝、取样隔离和结构不确定降级的累计像素使用量，不能当作质量分数。三种修复像素计数为过程使用量，反算可能覆盖先前空间估计，不能直接相加当作独立修改像素数。业务界面应显示跟踪覆盖率及未覆盖帧保留原图的说明。固定分支另有 `targetPixels`、`calibration` 和 `solverUsage`。

内部时间合同另包含 `timelineMode="timestamp-resampled-cfr-v1"`、`outputFps` 与 `sourceTimestampOrigin`。后端用这些字段、报告起止时间与输出媒体元数据核对实际成片，不再预先解码整片来证明输入为 CFR。`sourceTimestampOrigin` 是原视频流的起始 PTS，用户输入秒数仍从画面起点计时。

移动报告另列 `originalTrackedFrames`、`bridgedFrames`、`supportRecoveredFrames`、`contourRecoveredFrames`、`identityRejectedFrames`、`trackingContext` 和 `passthrough`。前四者只统计身份验证后仍接受的轨迹，相加为 `trackedFrames`；`identityRejectedFrames` 记录被撤销的旧定位，不计入覆盖率。`passthrough=true` 表示编码前没有像素修改，必须解释为保留输入，不能声称已去除水印。

`untrackedFrameDetails` 只列本次输出片段中的未处理帧，不包含跟踪上下文。`frames` 每项记录输出时间轴上从零开始的绝对 `frameIndex`、`timeSeconds`、最终接受状态及实际记录的轮廓、时序重试、桥接和参考补检拒绝原因，也保留原检测置信度和竞争峰差距。可变帧率输入的 `frameIndex` 不等于原文件中的压缩帧序号。最多列 100 项且限制在 32 KiB JSON 预算内，`total` 始终为完整未处理帧数，`truncated` 明确表示是否截断；未运行的检查不编造拒绝原因。

移动定位先以参考图相关性提出候选，再用有方向的轮廓距离和独立字形身份检查逐帧复核。轮廓匹配按带符号的梯度方向搜索有限个分离峰，缺笔或多笔拒绝也会触发全局重捕获；相同身份的竞争峰仍须满足唯一性差距。失败时可提取较弱候选边缘，并按模 π 的法线方向重试，支持同一灰色字形跨明暗背景后的局部反极性；这些提案方式不改变像素身份、轮廓覆盖和唯一性接受门槛。参考帧均匀边框给出未知透明度下的像素可行域，并包含加性明暗浮雕边缘的保守范围，连续违反可行域的笔画不能因整体相关性较高而接受。安静背景上另查与字形连接的多余轮廓；复杂背景上的相同颜色、相同形状场景内容仍可能不可辨识，不能保证任意素材零误检。

稀疏图案与背景线相交时，边缘检测可能在水印笔画处断开场景边缘。多余轮廓检查先沿该短边自身的拟合方向查找延续；短边的像素台阶可能使拟合方向偏斜，因此失败后还从整个字形范围外的实际边缘提出直线方向。外部直线必须在既有 1.5 像素容差内穿过该额外边缘的所有像素，再通过相同的逐点延续检查：已知水印支持外连续缺失超过两个采样点即中止，至少观察到六个字形范围外的同向边缘点才豁免。诊断记录方向来源、实际外部见证点及外部候选线。附近不相连的平行线不能替代连续性证据，多余笔画面积门槛不变。

背景切换不再阻止参考水印重定位，但修复供体仍受独立场景边界约束。不超过一秒的缺口可用两侧独立验证锚点限定当前帧搜索区域，仍须找到当前图像中的唯一轮廓峰；新增恢复帧不会继续充当锚点扩张缺口。轮廓证据不替代像素蒙版或背景恢复，也不会把插值位置直接标成成功。高置信且通过身份验证的旧亚像素位置会保留，避免轮廓量化改变已经可靠的配准。逐帧诊断包含轮廓覆盖、竞争峰、标准 alpha 与扩展范围违例、缺笔、多笔、原坐标是否保留和双向搜索锚点；这些是定位证据，不是自然纹理恢复质量分数。

输出保持原像素尺寸，采用视频元数据平均帧率对应的 CFR 时间格，编码为 H.264 CRF16。支持 VFR 和非零视频起始时间戳；VFR 在解码时按输出时间格复制或丢弃帧，所有参考和活动时间均以视频画面起点为零，不保留原始不规则帧间隔。保留全部输入音轨并按同一时间范围重新编码为 AAC 192kbps，以支持精确片段裁切及 MP4 容器；不是音频逐字节复制。旋转显示元数据、奇数尺寸仍会明确拒绝。调用方取消时应终止整个进程组，以同时结束 FFmpeg 和算法 worker。

## 能力边界

- 固定 R16 识别稳定亮色中性标记。持续明亮的场景细节可能混淆，彩色/深色/移动水印不是该检测器的能力。候选不存在或最终没有修改像素会失败，不把未处理视频标为成功。
- 移动 V6 的通用跟踪入口支持形状稳定、平移的标记，所选 ROI 需包含清晰标记及近均匀的边缘背景。过度纹理化的参考边缘、空白参考、过大范围仍返回明确错误。旋转、缩放、换字、严重遮挡尚不支持。
- 动态蒙版每帧由本次参考图和跟踪位置生成，保留亚像素蒙版边缘；未接受跟踪的帧保持原图，也不作为干净 donor。合法参考下整段没有可信水印时也保留输入并输出，报告 `trackedFrames=0`、`passthrough=true`，不让长片中的无水印区间阻断批次。报告的 `trackedFrames` 表明实际覆盖，成功不代表每帧都能去除。
- V6 只有至少两个足够分离、场景一致、互相证实的来源帧才能供给时序像素。其余像素使用证据约束的空间连续性、原空间降级或大孔洞图块合成，`spatialContinuity.texture_pixels` 单列图块合成的过程使用量。仿射反算只用真实时序证据训练，要求训练与留出帧分别满足背景变化和误差阈值；无法辨识时该像素禁用反算。这些分支都如实计数。纹理重复、接缝或被遮挡边缘的错误延伸仍可能出现，不能仅凭跟踪接受帧数判定画质。
- 不透明水印后完全不可见的内容无法保证恢复真实细节。时序一致性和模型置信度都不是隐藏内容真值。编码前从原片构造输出，只写回允许蒙版内的像素；H.264 有损编码仍可能影响周围像素。
- 原帧不降采样，4K 可输入，但吞吐受素材/ROI/帧数影响，本次移植不承诺既有素材以外的实时速度。移动 ROI 最大 196608 像素、单边最大 1024；跟踪上下文最多 100000 帧，超限直接失败。跟踪逆序解码块约 64MiB，donor LRU 约 96MiB；空间修复/配准另有中间数组。固定分支最多取 48 帧、约 2400 万累计像素校准；不写全视频原始帧缓存。

## 自动发现研究入口

`auto_overlay/discover.py` 是独立的纯算法候选发现原型，只接受原视频和采样预算，不读取用户 ROI、参考秒数、活动区间或既有轨迹。它尚未接入 `runner.py` 或生产擦除流程。

```bash
python3 -B backend/tools/watermark/auto_overlay/discover.py \
  --input /absolute/authorized-local-input.mp4 \
  --output .local/cache/overlay-discovery --samples 96
```

原型在整段视频均匀采样，以最长边 1280 像素的灰度图生成候选。不同长度的时间窗提取反复出现的正、负局部对比结构，再以带符号高通形状的一致性降低变化字幕的排名。该步骤仅做位置固定的重复结构发现；彩色区域转换为灰度后若没有亮度对比，当前实现不能识别。候选的参考图和支持像素重新从原分辨率读取，未改变输入视频或输出处理视频。

输出 `discovery.json`、`samples.npz` 和各候选的 `reference.png`、`support.npy`、`support.png`。JSON 记录自动生成的原片坐标、参考帧、采样锚点、每次采样的覆盖率与形状相关性。所有候选的状态均为 `candidate_requires_identity`，`repeatedContrastObserved` 只说明本次观察与重复形状相符，不表示已证明水印存在。支持图不是精细 alpha 或可写修复蒙版，可能包含背景结构，也可能遗漏粗笔画内部。

此入口没有完成移动水印发现、形变动画身份、活动边界逐帧定位或叠加层与场景物体的分离。短事件若没有覆盖至少三个采样时刻会遗漏；长期固定的场景文字、边框或重复纹理仍可能形成候选。同一像素序列既可能来自画面内标牌，也可能来自叠加文字，单靠重复不能消除这种歧义。因此发现框、候选排名和出现次数均不能授权自动擦除，也不能作为整片去水印完成的证明。

`auto_overlay/activity.py` 可继续读取上述自动产物，在候选的参考位置观察组件形状，并对预算内的变化区间进行原分辨率逐帧细化：

```bash
python3 -B backend/tools/watermark/auto_overlay/activity.py \
  --discovery .local/cache/overlay-discovery/discovery.json \
  --output .local/cache/overlay-activity --candidate-limit 8
```

输出为 `activity.json` 和 `activity-refined.json`。前者记录稀疏观察和相符采样段，后者记录实际细化范围与逐帧观察证据。它只在固定参考位置判断 `present` 或 `unknown`，没有全局不存在结论、已接受擦除状态或修复蒙版。目标发生移动、遮挡或形变时，不相符仍可能代表未知，不能直接当作消失。二分提案不是整个区间已经逐帧验证，需按输出中的实际密集覆盖范围理解结论。该入口同样没有接入生产 runner。

## 验证

```bash
python3 -B -m unittest discover -s backend/tools/watermark/tests -v
```

测试在临时目录生成三秒小视频并自动清理。覆盖两个真实 CLI、音轨和裁切时长、区间外参考帧、`durationSeconds=0`、缺依赖检查、R16 四轮 NumPy/native 样例一致性、局部传输与全帧调用的编码前一致性、V6 真正的时序恢复、来源独立性/缺蒙版/切镜拒绝及 V5 留出校准。它们是实现和合同回归，不是复杂真实素材的视觉质量验收。Linux native 构建尚需在对应 Linux 环境验证。

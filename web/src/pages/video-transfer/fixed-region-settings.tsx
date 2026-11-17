import { useState } from "react";
import { Button, InputNumber, Select } from "antd";
import { resourceFileUrl } from "@/services/api/resources";
import type { VideoTransferFixedRegion, VideoTransferRecord, VideoTransferROI, VideoTransferSettings } from "@/services/api/video-transfer";
import { ROISelector } from "./roi-selector";

export function FixedRegionSettings({ settings, source, disabled, onChange }: { settings: VideoTransferSettings; source: VideoTransferRecord; disabled: boolean; onChange: (patch: Partial<VideoTransferSettings>) => void }) {
    const [selectedIndex, setSelectedIndex] = useState(0);
    const regions = settings.fixedRegions || [];
    const index = Math.min(selectedIndex, Math.max(0, regions.length - 1));
    const region = regions[index];

    function addRegion() {
        if (disabled || regions.length >= 4) return;
        setSelectedIndex(regions.length);
        onChange({ fixedRegions: [...regions, { roi: { x: 0, y: 0, width: 0, height: 0 }, referenceSeconds: settings.startSeconds, startSeconds: settings.startSeconds, durationSeconds: 0 }] });
    }

    function updateRegion(patch: Partial<VideoTransferFixedRegion>) {
        onChange({ fixedRegions: regions.map((item, itemIndex) => (itemIndex === index ? { ...item, ...patch } : item)) });
    }

    return (
        <>
            <div className="video-transfer-mode-switch" aria-label="固定水印定位方式">
                <button type="button" aria-pressed={!region} disabled={disabled} onClick={() => onChange({ fixedRegions: [] })}>
                    自动角落
                </button>
                <button
                    type="button"
                    aria-pressed={Boolean(region)}
                    disabled={disabled}
                    onClick={() => {
                        if (!region) addRegion();
                    }}
                >
                    精确区域
                </button>
            </div>
            {region ? (
                <>
                    <div className="video-transfer-region-list" aria-label="固定水印区域">
                        {regions.map((_, itemIndex) => (
                            <button key={itemIndex} type="button" aria-pressed={index === itemIndex} disabled={disabled} onClick={() => setSelectedIndex(itemIndex)}>
                                区域 {itemIndex + 1}
                            </button>
                        ))}
                        <Button size="small" disabled={disabled || regions.length >= 4} onClick={addRegion}>
                            添加区域
                        </Button>
                    </div>
                    <div className="video-transfer-region-heading">
                        <strong>编辑区域 {index + 1}</strong>
                        <Button
                            size="small"
                            type="text"
                            danger
                            disabled={disabled}
                            aria-label={`删除区域 ${index + 1}`}
                            onClick={() => {
                                onChange({ fixedRegions: regions.filter((_, itemIndex) => itemIndex !== index) });
                                setSelectedIndex(Math.max(0, index - 1));
                            }}
                        >
                            删除区域
                        </Button>
                    </div>
                    <p className="video-transfer-hint">最多添加 4 处互不重叠的区域。每处仅在有效时段内处理；文字消失后请结束该区域的有效时段，避免误擦背景。</p>
                    <label>
                        参考时刻（原视频秒数）
                        <InputNumber aria-label="区域参考时刻" disabled={disabled} min={0} max={source.sourceDuration} step={0.1} precision={3} value={region.referenceSeconds} onChange={(value) => updateRegion({ referenceSeconds: value ?? 0 })} />
                    </label>
                    {source.sourceResourceId && source.sourceWidth && source.sourceHeight ? (
                        <ROISelector
                            key={`${source.sourceResourceId}:${index}`}
                            src={resourceFileUrl(source.sourceResourceId)}
                            width={source.sourceWidth}
                            height={source.sourceHeight}
                            seconds={region.referenceSeconds}
                            roi={region.roi}
                            disabled={disabled}
                            ariaLabel={`固定水印区域 ${index + 1}，拖动框选，也可在下方输入坐标`}
                            onChange={(roi) => updateRegion({ roi })}
                        />
                    ) : (
                        <p className="video-transfer-hint">上传视频后，可以在参考画面中拖框选择水印。</p>
                    )}
                    <div className="video-transfer-roi-inputs">
                        {(
                            [
                                ["x", "左侧 X"],
                                ["y", "顶部 Y"],
                                ["width", "宽度"],
                                ["height", "高度"],
                            ] as Array<[keyof VideoTransferROI, string]>
                        ).map(([key, label]) => (
                            <label key={key}>
                                {label}（像素）
                                <InputNumber
                                    aria-label={`区域${label}像素`}
                                    disabled={disabled}
                                    min={key === "width" || key === "height" ? 4 : 0}
                                    precision={0}
                                    value={region.roi[key]}
                                    onChange={(value) => updateRegion({ roi: { ...region.roi, [key]: value ?? 0 } })}
                                />
                            </label>
                        ))}
                    </div>
                    <div className="video-transfer-time-inputs">
                        <label>
                            有效开始（原视频秒数）
                            <InputNumber aria-label="区域开始时间" disabled={disabled} min={0} max={source.sourceDuration} step={0.1} precision={3} value={region.startSeconds} onChange={(value) => updateRegion({ startSeconds: value ?? 0 })} />
                        </label>
                        <label>
                            有效时长（秒）
                            <InputNumber aria-label="区域有效时长" disabled={disabled} min={0} max={source.sourceDuration} step={0.1} precision={3} value={region.durationSeconds} onChange={(value) => updateRegion({ durationSeconds: value ?? 0 })} />
                        </label>
                    </div>
                    <p className="video-transfer-hint">有效时长为 0 时持续至片尾。参考时刻须在有效时段内，并能清楚看到该处水印。实际只处理下方“处理范围”内的画面。</p>
                </>
            ) : (
                <label>
                    固定角落
                    <Select
                        aria-label="固定水印位置"
                        disabled={disabled}
                        value={settings.target}
                        onChange={(target) => onChange({ target })}
                        options={[
                            { value: "top-left", label: "左上角" },
                            { value: "top-right", label: "右上角" },
                            { value: "bottom-left", label: "左下角" },
                            { value: "bottom-right", label: "右下角" },
                        ]}
                    />
                </label>
            )}
            <p className="video-transfer-hint">适合位置固定的亮色中性水印。水印会移动时请选择 V6；低对比度灰色水印可能无法可靠识别。</p>
        </>
    );
}

package app

import (
	"errors"
	"math"

	"yingce/backend/internal/videotransfer"
)

const videoTransferTimelineMode = "timestamp-resampled-cfr-v1"

// These runner fields attest to the time grid used to decode this output. They
// remain internal: source nb_frames is diagnostic, never a normalized frame cap.
type videoTransferRunnerReport struct {
	videotransfer.Summary
	TimelineMode          string   `json:"timelineMode"`
	OutputFPS             *float64 `json:"outputFps"`
	SourceTimestampOrigin *float64 `json:"sourceTimestampOrigin"`
	StartSeconds          *float64 `json:"startSeconds"`
	DurationSeconds       *float64 `json:"durationSeconds"`
}

func validateVideoTransferTimeline(report videoTransferRunnerReport, settings videotransfer.Settings, source, output mediaProbeInfo) error {
	finite := func(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }
	if !finite(source.FrameRate) || source.FrameRate <= 0 || !finite(source.VideoStartSeconds) {
		return errors.New("无法验证视频输出时间轴")
	}
	for _, value := range []*float64{report.OutputFPS, report.SourceTimestampOrigin, report.StartSeconds, report.DurationSeconds} {
		if value == nil || !finite(*value) {
			return errors.New("处理报告缺少有效的输出时间轴")
		}
	}
	if report.TimelineMode != videoTransferTimelineMode || math.Abs(*report.OutputFPS-source.FrameRate) > 1e-6 || math.Abs(*report.SourceTimestampOrigin-source.VideoStartSeconds) > 1e-6 {
		return errors.New("处理报告的输出时间轴与原视频不一致")
	}
	duration := source.VideoDurationSeconds
	if duration <= 0 {
		duration = float64(source.DurationMs) / 1000
	}
	end := duration
	if settings.DurationSeconds > 0 {
		end = math.Min(end, settings.StartSeconds+settings.DurationSeconds)
	}
	first, last := int(math.Round(settings.StartSeconds*source.FrameRate)), int(math.Round(end*source.FrameRate))
	frames := last - first
	actualStart, actualDuration := float64(first)/source.FrameRate, float64(frames)/source.FrameRate
	if frames <= 0 || report.Frames != frames || (output.Frames > 0 && output.Frames != frames) {
		return errors.New("处理结果或报告帧数与请求区间不一致")
	}
	if math.Abs(*report.StartSeconds-actualStart) > 1e-6 || math.Abs(*report.DurationSeconds-actualDuration) > 1e-6 || math.Abs(output.FrameRate-source.FrameRate) > .001 || math.Abs(output.VideoDurationSeconds-actualDuration) > .005 {
		return errors.New("处理结果或报告时长与输出时间轴不一致")
	}
	return nil
}

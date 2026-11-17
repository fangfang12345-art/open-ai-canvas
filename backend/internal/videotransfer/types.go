package videotransfer

import (
	"errors"
	"fmt"
	"math"
	"time"
)

const Fixed = "fixed-r16"
const Moving = "moving-v6"
const TaskType = "video_transfer"
const SegmentDurationSeconds = 10

type Limits struct {
	MaxSourceBytes           int64   `json:"maxSourceBytes"`
	MaxSourceDurationSeconds float64 `json:"maxSourceDurationSeconds"`
	MaxDurationSeconds       float64 `json:"maxDurationSeconds"`
	MaxPixels                int     `json:"maxPixels"`
	MaxWorkers               int     `json:"maxWorkers"`
	MaxMovingROIPixels       int     `json:"maxMovingRoiPixels"`
	MaxMovingROIDimension    int     `json:"maxMovingRoiDimension"`
	SegmentDurationSeconds   float64 `json:"segmentDurationSeconds"`
	MaxConcurrentSegments    int     `json:"maxConcurrentSegments"`
}

func DefaultLimits() Limits {
	return Limits{MaxPixels: 3840 * 2160, MaxWorkers: 16, MaxMovingROIPixels: 196608, MaxMovingROIDimension: 1024, SegmentDurationSeconds: SegmentDurationSeconds, MaxConcurrentSegments: 2}
}

type ROI struct {
	X      float64 `json:"x"`
	Y      float64 `json:"y"`
	Width  float64 `json:"width"`
	Height float64 `json:"height"`
}

type Settings struct {
	Target           string        `json:"target,omitempty"`
	ReferenceSeconds float64       `json:"referenceSeconds"`
	ROI              *ROI          `json:"roi,omitempty"`
	FixedRegions     []FixedRegion `json:"fixedRegions,omitempty"`
	StartSeconds     float64       `json:"startSeconds"`
	DurationSeconds  float64       `json:"durationSeconds"`
	Workers          int           `json:"workers"`
}

type SettingsPatch struct {
	Target           *string        `json:"target,omitempty"`
	ReferenceSeconds *float64       `json:"referenceSeconds,omitempty"`
	ROI              *ROI           `json:"roi,omitempty"`
	FixedRegions     *[]FixedRegion `json:"fixedRegions,omitempty"`
	StartSeconds     *float64       `json:"startSeconds,omitempty"`
	DurationSeconds  *float64       `json:"durationSeconds,omitempty"`
	Workers          *int           `json:"workers,omitempty"`
}

func (s Settings) Merge(p SettingsPatch) Settings {
	if s.ROI != nil {
		roi := *s.ROI
		s.ROI = &roi
	}
	s.FixedRegions = append([]FixedRegion(nil), s.FixedRegions...)
	if p.Target != nil {
		s.Target = *p.Target
	}
	if p.ReferenceSeconds != nil {
		s.ReferenceSeconds = *p.ReferenceSeconds
	}
	if p.ROI != nil {
		roi := *p.ROI
		s.ROI = &roi
	}
	if p.FixedRegions != nil {
		s.FixedRegions = append([]FixedRegion(nil), (*p.FixedRegions)...)
	}
	if p.StartSeconds != nil {
		s.StartSeconds = *p.StartSeconds
	}
	if p.DurationSeconds != nil {
		s.DurationSeconds = *p.DurationSeconds
	}
	if p.Workers != nil {
		s.Workers = *p.Workers
	}
	return s
}

type BranchState struct {
	Settings         Settings  `json:"settings"`
	Status           string    `json:"status"`
	Progress         int       `json:"progress"`
	Stage            string    `json:"stage,omitempty"`
	Error            string    `json:"error,omitempty"`
	OutputResourceID string    `json:"outputResourceId,omitempty"`
	ElapsedSeconds   float64   `json:"elapsedSeconds,omitempty"`
	TaskID           string    `json:"taskId,omitempty"`
	Summary          *Summary  `json:"summary,omitempty"`
	Segments         []Segment `json:"segments"`
}

type Segment struct {
	ID               string   `json:"id"`
	Index            int      `json:"index"`
	StartSeconds     float64  `json:"startSeconds"`
	DurationSeconds  float64  `json:"durationSeconds"`
	Status           string   `json:"status"`
	Progress         int      `json:"progress"`
	Stage            string   `json:"stage,omitempty"`
	Error            string   `json:"error,omitempty"`
	OutputResourceID string   `json:"outputResourceId,omitempty"`
	ElapsedSeconds   float64  `json:"elapsedSeconds,omitempty"`
	Summary          *Summary `json:"summary,omitempty"`
}

// 时间窗口共用原片，在统一输出时间轴上取整；源文件帧数不限制重采样后的帧数。
func Segments(settings Settings, duration float64) []Segment {
	return SegmentsForTimeline(settings, duration, 0)
}

func SegmentsForTimeline(settings Settings, duration, fps float64) []Segment {
	result := make([]Segment, 0)
	if !finite(duration) || duration <= 0 || !finite(settings.StartSeconds) || settings.StartSeconds < 0 || !finite(settings.DurationSeconds) || settings.DurationSeconds < 0 {
		return result
	}
	end := duration
	if settings.DurationSeconds > 0 {
		end = math.Min(end, settings.StartSeconds+settings.DurationSeconds)
	}
	for index, start := 0, settings.StartSeconds; start < end-1e-9; index, start = index+1, settings.StartSeconds+float64(index+1)*SegmentDurationSeconds {
		partEnd := math.Min(start+SegmentDurationSeconds, end)
		if finite(fps) && fps > 0 {
			first, last := math.Round(start*fps), math.Round(partEnd*fps)
			if last <= first {
				continue
			}
		}
		result = append(result, Segment{ID: fmt.Sprintf("segment-%d", index+1), Index: index, StartSeconds: start, DurationSeconds: partEnd - start, Status: "idle"})
	}
	return result
}

type NativeSummary struct {
	Enabled                  bool     `json:"enabled"`
	Kernels                  []string `json:"kernels,omitempty"`
	UnavailableKernels       []string `json:"unavailableKernels,omitempty"`
	BackgroundImplementation string   `json:"backgroundImplementation,omitempty"`
}

type SolverUsage struct {
	RobustNativeFrames   int `json:"robustNativeFrames"`
	RobustNumpyFrames    int `json:"robustNumpyFrames"`
	RobustFallbackFrames int `json:"robustFallbackFrames"`
	AffineFallbackFrames int `json:"affineFallbackFrames"`
}

type Summary struct {
	Frames                  int            `json:"frames"`
	TrackedFrames           int            `json:"trackedFrames"`
	UntrackedFrames         int            `json:"untrackedFrames"`
	TrackingCoverage        float64        `json:"trackingCoverage"`
	ChangedPixels           int64          `json:"changedPixels"`
	TemporalRecoveredPixels int64          `json:"temporalRecoveredPixels"`
	InversePixels           int64          `json:"inversePixels"`
	SpatialFallbackPixels   int64          `json:"spatialFallbackPixels"`
	CalibratedInversePixels int64          `json:"calibratedInversePixels"`
	Limitations             []string       `json:"limitations"`
	Native                  *NativeSummary `json:"native,omitempty"`
	SolverUsage             *SolverUsage   `json:"solverUsage,omitempty"`
}

type Record struct {
	ID               string                 `json:"id"`
	Title            string                 `json:"title"`
	SourceResourceID string                 `json:"sourceResourceId,omitempty"`
	SourceName       string                 `json:"sourceName,omitempty"`
	SourceWidth      int                    `json:"sourceWidth,omitempty"`
	SourceHeight     int                    `json:"sourceHeight,omitempty"`
	SourceDuration   float64                `json:"sourceDuration,omitempty"`
	SourceFrameRate  float64                `json:"sourceFrameRate,omitempty"`
	SourceFrames     int                    `json:"sourceFrames,omitempty"`
	SelectedBranch   string                 `json:"selectedBranch"`
	Branches         map[string]BranchState `json:"branches"`
	Revision         int64                  `json:"revision"`
	CreatedAt        time.Time              `json:"createdAt"`
	UpdatedAt        time.Time              `json:"updatedAt"`
}

type BranchPatch struct {
	Settings SettingsPatch `json:"settings"`
}
type Patch struct {
	ExpectedRevision int64                  `json:"expectedRevision"`
	Title            *string                `json:"title,omitempty"`
	SourceResourceID *string                `json:"sourceResourceId,omitempty"`
	SourceName       *string                `json:"sourceName,omitempty"`
	SelectedBranch   *string                `json:"selectedBranch,omitempty"`
	Branches         map[string]BranchPatch `json:"branches,omitempty"`
}

type CreateRequest struct {
	Title            string `json:"title"`
	SourceResourceID string `json:"sourceResourceId,omitempty"`
	SourceName       string `json:"sourceName,omitempty"`
	SelectedBranch   string `json:"selectedBranch,omitempty"`
}

type RunRequest struct {
	Branch           string `json:"branch"`
	ExpectedRevision int64  `json:"expectedRevision"`
}

type Capability struct {
	ID        string `json:"id"`
	Available bool   `json:"available"`
	Reason    string `json:"reason,omitempty"`
}
type Capabilities struct {
	Branches []Capability `json:"branches"`
	Limits   Limits       `json:"limits"`
}

func ValidBranch(branch string) bool { return branch == Fixed || branch == Moving }
func Active(status string) bool      { return status == "queued" || status == "running" }
func DefaultBranches() map[string]BranchState {
	return map[string]BranchState{
		Fixed:  {Settings: Settings{Target: "top-right", Workers: 10}, Status: "idle"},
		Moving: {Settings: Settings{Workers: 10}, Status: "idle"},
	}
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

func ValidateSettings(branch string, s Settings) error {
	if !ValidBranch(branch) {
		return errors.New("请选择有效的处理分支")
	}
	if !finite(s.StartSeconds) || !finite(s.DurationSeconds) || s.StartSeconds < 0 || s.DurationSeconds < 0 {
		return errors.New("时间范围无效")
	}
	if s.Workers < 1 || s.Workers > 16 {
		return errors.New("并行进程数必须为 1 到 16")
	}
	if len(s.FixedRegions) > 0 {
		if branch != Fixed {
			return errors.New("固定水印区域仅适用于固定 R16")
		}
		return validateFixedRegions(s.FixedRegions)
	}
	if !finite(s.ReferenceSeconds) || s.ReferenceSeconds < 0 {
		return errors.New("参考帧时间无效")
	}
	if branch == Fixed && s.Target != "top-left" && s.Target != "top-right" && s.Target != "bottom-left" && s.Target != "bottom-right" {
		return errors.New("请选择固定水印所在角落")
	}
	if s.ROI != nil {
		r := s.ROI
		if !validROI(*r) {
			return errors.New("水印区域必须使用原片整数像素，宽高至少 4 像素")
		}
	}
	return nil
}

func ValidateRun(branch string, s Settings, width, height int, duration float64, limits Limits) error {
	if err := ValidateSettings(branch, s); err != nil {
		return err
	}
	if s.Workers > limits.MaxWorkers {
		return errors.New("并行进程数超过服务器上限")
	}
	if width <= 0 || height <= 0 || width > 16384 || height > 16384 || width*height > limits.MaxPixels || !finite(duration) || duration <= 0 || (limits.MaxSourceDurationSeconds > 0 && duration > limits.MaxSourceDurationSeconds) {
		return errors.New("视频尺寸或时长超过服务器处理上限，请查看处理能力限制")
	}
	if s.StartSeconds >= duration {
		return errors.New("开始时间必须早于视频结尾")
	}
	length := s.DurationSeconds
	if length == 0 {
		length = duration - s.StartSeconds
	}
	if length <= 0 || (limits.MaxDurationSeconds > 0 && length > limits.MaxDurationSeconds) || s.StartSeconds+length > duration+0.05 {
		return errors.New("处理区间超出视频范围或服务器时长上限")
	}
	if len(s.FixedRegions) > 0 {
		return validateFixedRegionsForMedia(s.FixedRegions, width, height, duration, s.StartSeconds, s.StartSeconds+length)
	}
	if branch == Moving {
		if s.ROI == nil {
			return errors.New("请在参考帧中选择移动水印区域")
		}
		if s.ROI.Width*s.ROI.Height > float64(limits.MaxMovingROIPixels) || s.ROI.Width > float64(limits.MaxMovingROIDimension) || s.ROI.Height > float64(limits.MaxMovingROIDimension) {
			return errors.New("移动水印区域过大，请缩小到水印本体")
		}
		if s.ROI.X+s.ROI.Width > float64(width) || s.ROI.Y+s.ROI.Height > float64(height) {
			return errors.New("水印区域超出原视频尺寸")
		}
		if s.ReferenceSeconds >= duration {
			return errors.New("参考帧必须位于原视频范围内")
		}
	}
	return nil
}

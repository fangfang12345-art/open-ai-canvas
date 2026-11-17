package videotransfer

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
)

type FixedRegion struct {
	ROI              ROI     `json:"roi"`
	ReferenceSeconds float64 `json:"referenceSeconds"`
	StartSeconds     float64 `json:"startSeconds"`
	DurationSeconds  float64 `json:"durationSeconds"`
}

func (r *FixedRegion) UnmarshalJSON(data []byte) error {
	var value struct {
		ROI *struct {
			X      *float64 `json:"x"`
			Y      *float64 `json:"y"`
			Width  *float64 `json:"width"`
			Height *float64 `json:"height"`
		} `json:"roi"`
		ReferenceSeconds *float64 `json:"referenceSeconds"`
		StartSeconds     *float64 `json:"startSeconds"`
		DurationSeconds  *float64 `json:"durationSeconds"`
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&value); err != nil {
		return err
	}
	if value.ROI == nil || value.ReferenceSeconds == nil || value.StartSeconds == nil || value.DurationSeconds == nil {
		return errors.New("固定水印区域必须指定区域、参考帧、开始时间和有效秒数")
	}
	if value.ROI.X == nil || value.ROI.Y == nil || value.ROI.Width == nil || value.ROI.Height == nil {
		return errors.New("固定水印区域必须指定完整的原片像素矩形")
	}
	*r = FixedRegion{ROI: ROI{X: *value.ROI.X, Y: *value.ROI.Y, Width: *value.ROI.Width, Height: *value.ROI.Height}, ReferenceSeconds: *value.ReferenceSeconds, StartSeconds: *value.StartSeconds, DurationSeconds: *value.DurationSeconds}
	return nil
}

func validROI(r ROI) bool {
	return finite(r.X) && finite(r.Y) && finite(r.Width) && finite(r.Height) &&
		math.Trunc(r.X) == r.X && math.Trunc(r.Y) == r.Y && math.Trunc(r.Width) == r.Width && math.Trunc(r.Height) == r.Height &&
		r.X >= 0 && r.Y >= 0 && r.Width >= 4 && r.Height >= 4 && r.X+r.Width <= 16384 && r.Y+r.Height <= 16384
}

func validateFixedRegions(regions []FixedRegion) error {
	if len(regions) > 4 {
		return errors.New("固定水印区域最多 4 个")
	}
	for i, region := range regions {
		if !validROI(region.ROI) {
			return fmt.Errorf("固定水印区域 %d 必须使用原片整数像素，宽高至少 4 像素", i+1)
		}
		if !finite(region.StartSeconds) || !finite(region.DurationSeconds) || !finite(region.ReferenceSeconds) || region.StartSeconds < 0 || region.DurationSeconds < 0 || region.ReferenceSeconds < region.StartSeconds || !finite(region.StartSeconds+region.DurationSeconds) || (region.DurationSeconds > 0 && region.ReferenceSeconds >= region.StartSeconds+region.DurationSeconds) {
			return fmt.Errorf("固定水印区域 %d 的参考帧必须在有效时间范围内，结束时刻不包含在内", i+1)
		}
		for _, previous := range regions[:i] {
			a, b := region.ROI, previous.ROI
			if a.X < b.X+b.Width && b.X < a.X+a.Width && a.Y < b.Y+b.Height && b.Y < a.Y+a.Height {
				return errors.New("固定水印区域不能重叠")
			}
		}
	}
	return nil
}

func validateFixedRegionsForMedia(regions []FixedRegion, width, height int, duration, start, end float64) error {
	intersects := false
	for i, region := range regions {
		if region.ROI.X+region.ROI.Width > float64(width) || region.ROI.Y+region.ROI.Height > float64(height) {
			return fmt.Errorf("固定水印区域 %d 超出原视频尺寸", i+1)
		}
		regionEnd := duration
		if region.DurationSeconds > 0 {
			regionEnd = region.StartSeconds + region.DurationSeconds
		}
		if region.StartSeconds >= duration || regionEnd > duration+1e-6 || region.ReferenceSeconds >= duration {
			return fmt.Errorf("固定水印区域 %d 的有效时间或参考帧超出原视频范围", i+1)
		}
		if start < regionEnd && region.StartSeconds < end {
			intersects = true
		}
	}
	if !intersects {
		return errors.New("处理范围与所有固定水印区域的有效时间均不相交")
	}
	return nil
}

// Match the runner's half-up absolute frame boundaries before dispatch. A
// numerically valid reference just before an end can round onto the excluded frame.
func ValidateFixedRegionFrames(regions []FixedRegion, duration, fps float64) error {
	if len(regions) == 0 {
		return nil
	}
	if err := validateFixedRegions(regions); err != nil {
		return err
	}
	if !finite(fps) || fps <= 0 || !finite(duration) || duration <= 0 {
		return errors.New("无法验证固定水印区域的输出帧率")
	}
	frames := int(math.Round(duration * fps))
	for i, region := range regions {
		first := int(math.Round(region.StartSeconds * fps))
		last := frames
		if region.DurationSeconds > 0 {
			last = min(last, int(math.Round((region.StartSeconds+region.DurationSeconds)*fps)))
		}
		reference := int(math.Round(region.ReferenceSeconds * fps))
		if last-first < 3 {
			return fmt.Errorf("固定水印区域 %d 的有效时间必须包含至少 3 帧", i+1)
		}
		if reference < first || reference >= last {
			return fmt.Errorf("固定水印区域 %d 的参考帧取整后不在有效时间内，请远离结束时刻选择", i+1)
		}
	}
	return nil
}

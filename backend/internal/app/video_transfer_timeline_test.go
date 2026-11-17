package app

import (
	"encoding/json"
	"math"
	"testing"

	"yingce/backend/internal/videotransfer"
)

func TestVideoTransferNormalizedTimelineReport(t *testing.T) {
	const fps = 30000.0 / 1001
	start := 2.5 / fps
	first, last := math.Round(start*fps), math.Round((start+10)*fps)
	duration := (last - first) / fps
	source := mediaProbeInfo{FrameRate: fps, Frames: 12, VideoDurationSeconds: 21.3, VideoStartSeconds: -2.5}
	output := mediaProbeInfo{FrameRate: fps, Frames: int(last - first), VideoDurationSeconds: duration}
	settings := videotransfer.Settings{StartSeconds: start, DurationSeconds: 10}
	fields := map[string]any{"timelineMode": videoTransferTimelineMode, "frames": output.Frames, "outputFps": fps, "sourceTimestampOrigin": source.VideoStartSeconds, "startSeconds": first / fps, "durationSeconds": duration}
	for _, field := range []string{"", "timelineMode", "frames", "outputFps", "sourceTimestampOrigin", "startSeconds", "durationSeconds"} {
		t.Run("missing-"+field, func(t *testing.T) {
			copy := make(map[string]any, len(fields))
			for k, v := range fields {
				copy[k] = v
			}
			delete(copy, field)
			data, _ := json.Marshal(copy)
			var report videoTransferRunnerReport
			if err := json.Unmarshal(data, &report); err != nil {
				t.Fatal(err)
			}
			err := validateVideoTransferTimeline(report, settings, source, output)
			if (err == nil) != (field == "") {
				t.Fatalf("field=%s: %v", field, err)
			}
			if field == "" {
				// Source compressed-frame count must never truncate a VFR output grid.
				badOutput := output
				badOutput.Frames--
				if validateVideoTransferTimeline(report, settings, source, badOutput) == nil {
					t.Fatal("missing decoded output frame accepted")
				}
				wrongOrigin := *report.SourceTimestampOrigin + 1
				report.SourceTimestampOrigin = &wrongOrigin
				if validateVideoTransferTimeline(report, settings, source, output) == nil {
					t.Fatal("wrong source timestamp origin accepted")
				}
			}
		})
	}
}

func TestVideoTransferProbeTimestampDuration(t *testing.T) {
	for _, tc := range []struct {
		name, raw        string
		origin, duration float64
		wantError        bool
	}{
		{"stream", `{"streams":[{"start_time":"5","duration":"19.916667"}],"format":{"duration":"24.916667","format_name":"mov,mp4"}}`, 5, 19.916667, false},
		{"time-base", `{"streams":[{"start_time":"-2","duration_ts":600,"time_base":"1/30"}],"format":{"duration":"18","format_name":"mpegts"}}`, -2, 20, false},
		{"mkv-video-tag", `{"streams":[{"start_time":"5","tags":{"DURATION":"00:00:24.916667"}}],"format":{"duration":"30","format_name":"matroska,webm"}}`, 5, 19.916667, false},
		{"mkv-fallback", `{"streams":[{"start_time":"5"}],"format":{"duration":"24.916667","format_name":"matroska,webm"}}`, 5, 19.916667, false},
		{"zero-origin-fallback", `{"streams":[{"start_time":"0"}],"format":{"duration":"19.916667"}}`, 0, 19.916667, false},
		{"ambiguous-origin", `{"streams":[{"start_time":"5"}],"format":{"duration":"24.916667","format_name":"unknown"}}`, 0, 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			info, err := parseMediaProbeOutputDuration([]byte(tc.raw), true)
			if (err != nil) != tc.wantError {
				t.Fatalf("error=%v wanted=%v", err, tc.wantError)
			}
			if !tc.wantError && (info.VideoStartSeconds != tc.origin || math.Abs(info.VideoDurationSeconds-tc.duration) > 1e-9 || info.DurationMs != int64(math.Ceil(tc.duration*1000))) {
				t.Fatalf("incorrect timestamp duration: %+v", info)
			}
		})
	}
}

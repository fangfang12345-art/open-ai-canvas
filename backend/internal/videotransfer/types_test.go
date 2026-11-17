package videotransfer

import (
	"math"
	"testing"
)

func TestVideoTransferSegments(t *testing.T) {
	for _, tc := range []struct {
		duration float64
		count    int
		tail     float64
	}{{21.3, 3, 1.3}, {20, 2, 10}, {10, 1, 10}, {0, 0, 0}} {
		parts := Segments(Settings{}, tc.duration)
		if len(parts) != tc.count {
			t.Fatalf("duration %v: %+v", tc.duration, parts)
		}
		if tc.count > 0 && math.Abs(parts[tc.count-1].DurationSeconds-tc.tail) > 1e-8 {
			t.Fatalf("tail %+v", parts)
		}
	}
	parts := Segments(Settings{StartSeconds: 7, DurationSeconds: 12}, 30)
	if len(parts) != 2 || parts[1].StartSeconds != 17 || parts[1].DurationSeconds != 2 {
		t.Fatalf("range %+v", parts)
	}
	if parts := SegmentsForTimeline(Settings{}, 20.001, 30); len(parts) != 2 {
		t.Fatalf("empty tail %+v", parts)
	}
	fps := 30000.0 / 1001
	parts = SegmentsForTimeline(Settings{}, 30.03, fps)
	frames, previous := 0, 0
	for _, part := range parts {
		first, last := int(math.Round(part.StartSeconds*fps)), int(math.Round((part.StartSeconds+part.DurationSeconds)*fps))
		if first != previous {
			t.Fatal("non-contiguous frames")
		}
		previous = last
		frames += last - first
	}
	if frames != 900 {
		t.Fatalf("frames=%d", frames)
	}
	if err := ValidateRun(Fixed, DefaultBranches()[Fixed].Settings, 160, 90, 100000, DefaultLimits()); err != nil {
		t.Fatalf("default duration cap: %v", err)
	}
}

func TestVideoTransferRunBounds(t *testing.T) {
	settings := DefaultBranches()[Fixed].Settings
	if err := ValidateRun(Fixed, settings, 3840, 2160, 10, DefaultLimits()); err != nil {
		t.Fatalf("4K original resolution rejected: %v", err)
	}
	moving := DefaultBranches()[Moving].Settings
	moving.StartSeconds = 5
	moving.DurationSeconds = 2
	moving.ReferenceSeconds = 1
	moving.ROI = &ROI{X: 1, Y: 2, Width: 80, Height: 40}
	if err := ValidateRun(Moving, moving, 3840, 2160, 10, DefaultLimits()); err != nil {
		t.Fatalf("reference outside clip but inside source rejected: %v", err)
	}
	for _, tc := range []struct {
		name   string
		mutate func(*Settings)
	}{
		{"missing-mask", func(s *Settings) { s.ROI = nil }},
		{"fractional-pixels", func(s *Settings) { s.ROI = &ROI{X: 0.5, Y: 0, Width: 80, Height: 40} }},
		{"outside-source", func(s *Settings) { s.ReferenceSeconds = 10 }},
		{"nan", func(s *Settings) { s.StartSeconds = math.NaN() }},
		{"clip-too-long", func(s *Settings) { s.DurationSeconds = 8 }},
		{"outside-frame", func(s *Settings) { s.ROI = &ROI{X: 3800, Y: 0, Width: 80, Height: 40} }},
		{"oversized-mask", func(s *Settings) { s.ROI = &ROI{Width: 1024, Height: 1024} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			copy := moving
			tc.mutate(&copy)
			if err := ValidateRun(Moving, copy, 3840, 2160, 10, DefaultLimits()); err == nil {
				t.Fatal("invalid request accepted")
			}
		})
	}
}

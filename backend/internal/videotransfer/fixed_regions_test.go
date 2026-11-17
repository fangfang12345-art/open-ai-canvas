package videotransfer

import (
	"encoding/json"
	"math"
	"testing"
)

func fixedRegionSettings() Settings {
	return Settings{Workers: 4, StartSeconds: 5, DurationSeconds: 10, FixedRegions: []FixedRegion{
		{ROI: ROI{X: 10, Y: 10, Width: 100, Height: 40}, ReferenceSeconds: 1, StartSeconds: 0, DurationSeconds: 10},
		{ROI: ROI{X: 110, Y: 10, Width: 100, Height: 40}, ReferenceSeconds: 18, StartSeconds: 10, DurationSeconds: 0},
	}}
}

func TestFixedRegionValidation(t *testing.T) {
	s := fixedRegionSettings()
	// Distinct areas may touch; each reference is absolute and may lie outside
	// the requested output interval. Legacy coordinates do not select these areas.
	s.Target, s.ReferenceSeconds, s.ROI = "unused", -1, &ROI{Width: 1}
	if err := ValidateRun(Fixed, s, 1920, 1080, 20, DefaultLimits()); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name   string
		branch string
		mutate func(*Settings)
	}{
		{"wrong-branch", Moving, func(s *Settings) {}},
		{"too-many", Fixed, func(s *Settings) { s.FixedRegions = make([]FixedRegion, 5) }},
		{"overlap-even-with-different-times", Fixed, func(s *Settings) { s.FixedRegions[1].ROI.X = 109 }},
		{"fractional-pixel", Fixed, func(s *Settings) { s.FixedRegions[0].ROI.X = .1 }},
		{"missing-roi", Fixed, func(s *Settings) { s.FixedRegions[0].ROI = ROI{} }},
		{"invalid-pixel", Fixed, func(s *Settings) { s.FixedRegions[0].ROI.Width = math.Inf(1) }},
		{"reference-before-region", Fixed, func(s *Settings) { s.FixedRegions[1].ReferenceSeconds = 9 }},
		{"reference-at-exclusive-end", Fixed, func(s *Settings) { s.FixedRegions[0].ReferenceSeconds = 10 }},
		{"invalid-region-time", Fixed, func(s *Settings) { s.FixedRegions[0].StartSeconds = math.NaN() }},
		{"outside-frame", Fixed, func(s *Settings) { s.FixedRegions[1].ROI.X = 1900 }},
		{"reference-at-source-end", Fixed, func(s *Settings) { s.FixedRegions[1].ReferenceSeconds = 20 }},
		{"active-range-outside-source", Fixed, func(s *Settings) { s.FixedRegions[1].DurationSeconds = 11 }},
		{"range-after-source", Fixed, func(s *Settings) { s.FixedRegions[1].StartSeconds = 20; s.FixedRegions[1].ReferenceSeconds = 20 }},
		{"no-overlap-at-exclusive-end", Fixed, func(s *Settings) { s.FixedRegions = s.FixedRegions[:1]; s.StartSeconds = 10; s.DurationSeconds = 10 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			settings := fixedRegionSettings()
			tc.mutate(&settings)
			if err := ValidateRun(tc.branch, settings, 1920, 1080, 20, DefaultLimits()); err == nil {
				t.Fatal("invalid fixed region accepted")
			}
		})
	}
	s = fixedRegionSettings()
	s.FixedRegions = s.FixedRegions[:1]
	s.FixedRegions[0].ROI = ROI{Width: 1920, Height: 1080}
	if err := ValidateRun(Fixed, s, 1920, 1080, 20, DefaultLimits()); err != nil {
		t.Fatalf("fixed ROI incorrectly used moving ROI size limit: %v", err)
	}
}

func TestFixedRegionMergeCopiesAndClears(t *testing.T) {
	original := fixedRegionSettings()
	original.Target = "top-right"
	original.ROI = &ROI{Width: 20, Height: 20}
	copy := original.Merge(SettingsPatch{})
	copy.FixedRegions[0].ROI.Width = 30
	copy.ROI.Width = 40
	if original.FixedRegions[0].ROI.Width != 100 || original.ROI.Width != 20 {
		t.Fatal("merge changed the source settings")
	}
	patchRegions := fixedRegionSettings().FixedRegions
	copy = original.Merge(SettingsPatch{FixedRegions: &patchRegions})
	copy.FixedRegions[0].ROI.Width = 50
	if patchRegions[0].ROI.Width != 100 {
		t.Fatal("merge retained a caller-owned slice")
	}
	for _, payload := range []string{`{}`, `{"fixedRegions":null}`, `{"fixedRegions":[]}`} {
		var patch SettingsPatch
		if err := json.Unmarshal([]byte(payload), &patch); err != nil {
			t.Fatal(err)
		}
		copy = original.Merge(patch)
		if (len(copy.FixedRegions) == 0) != (payload == `{"fixedRegions":[]}`) {
			t.Fatalf("clear semantics incorrect: %s", payload)
		}
		if err := ValidateSettings(Fixed, copy); err != nil {
			t.Fatal(err)
		}
	}
}

func TestFixedRegionRequiredFieldsAndFrameBoundaries(t *testing.T) {
	for _, payload := range []string{
		`null`, `{}`, `{"roi":{"x":0,"y":0,"width":10,"height":10}}`,
		`{"roi":{"y":0,"width":10,"height":10},"referenceSeconds":0,"startSeconds":0,"durationSeconds":0}`,
		`{"roi":{"x":0,"y":0,"width":10,"height":10},"referenceSeconds":null,"startSeconds":0,"durationSeconds":0}`,
		`{"roi":{"x":0,"y":0,"width":10,"height":10},"referenceSeconds":0,"durationSeconds":0}`,
		`{"roi":{"x":0,"y":0,"width":10,"height":10},"referenceSeconds":0,"startSeconds":0}`,
		`{"roi":{"x":0,"y":0,"width":10,"height":10,"input":"private"},"referenceSeconds":0,"startSeconds":0,"durationSeconds":0}`,
	} {
		var region FixedRegion
		if err := json.Unmarshal([]byte(payload), &region); err == nil {
			t.Fatalf("incomplete or unknown region fields accepted: %s", payload)
		}
	}
	for _, fps := range []float64{30, 30000.0 / 1001} {
		regions := []FixedRegion{{ROI: ROI{Width: 10, Height: 10}, StartSeconds: 10 / fps, DurationSeconds: 3 / fps, ReferenceSeconds: 12 / fps}}
		if err := ValidateFixedRegionFrames(regions, 20, fps); err != nil {
			t.Fatalf("three-frame region rejected at %v: %v", fps, err)
		}
		regions[0].ReferenceSeconds = 12.9 / fps
		if err := ValidateFixedRegionFrames(regions, 20, fps); err == nil {
			t.Fatal("reference rounded onto excluded end accepted")
		}
		regions[0].ReferenceSeconds, regions[0].DurationSeconds = 10/fps, 2/fps
		if err := ValidateFixedRegionFrames(regions, 20, fps); err == nil {
			t.Fatal("two-frame region accepted")
		}
		regions[0].DurationSeconds = 0
		if err := ValidateFixedRegionFrames(regions, 12/fps, fps); err == nil {
			t.Fatal("normalized output duration not used for open-ended region")
		}
	}
}

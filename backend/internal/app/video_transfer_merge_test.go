package app

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"yingce/backend/internal/model"
	"yingce/backend/internal/repository"
	"yingce/backend/internal/videotransfer"
)

func assertVideoTransferCompleteMedia(t *testing.T, svc *Service, source string, state videotransfer.BranchState, frames int, duration, start float64) {
	t.Helper()
	resource, err := svc.repo.ResourceForUser("transfer-owner", state.OutputResourceID)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(svc.dataDir, "resources", filepath.FromSlash(resource.ObjectKey))
	info, err := probeMediaFileDuration(context.Background(), path, true)
	if err != nil || info.Frames != frames || math.Abs(info.VideoDurationSeconds-duration) > .005 || resource.Width != info.Width || resource.Height != info.Height || resource.DurationMs != info.DurationMs {
		t.Fatalf("complete metadata=%+v resource=%+v err=%v", info, resource, err)
	}
	frameHashes := func(path string) []string {
		out, err := exec.Command("ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0", "-f", "framemd5", "-").Output()
		if err != nil {
			t.Fatal(err)
		}
		var hashes []string
		for _, line := range strings.Split(string(out), "\n") {
			if strings.HasPrefix(line, "#") || strings.TrimSpace(line) == "" {
				continue
			}
			fields := strings.Split(line, ",")
			hashes = append(hashes, strings.TrimSpace(fields[len(fields)-1]))
		}
		return hashes
	}
	var expected []string
	ordered, _, _, err := orderVideoTransferSegments(state.Segments, mediaProbeInfo{FrameRate: info.FrameRate})
	if err != nil {
		t.Fatal(err)
	}
	for _, part := range ordered {
		r, err := svc.repo.ResourceForUser("transfer-owner", part.OutputResourceID)
		if err != nil {
			t.Fatal(err)
		}
		expected = append(expected, frameHashes(filepath.Join(svc.dataDir, "resources", filepath.FromSlash(r.ObjectKey)))...)
	}
	actual := frameHashes(path)
	if len(actual) != frames || strings.Join(actual, "\n") != strings.Join(expected, "\n") {
		t.Fatal("assembled video differs from the ordered decoded processing results")
	}
	pcm := func(path string, offset float64) []byte {
		out, err := exec.Command("ffmpeg", "-v", "error", "-seek_timestamp", "1", "-ss", fmt.Sprint(offset), "-i", path, "-t", fmt.Sprint(duration), "-map", "0:a:0", "-ac", "1", "-ar", "44100", "-f", "f32le", "-").Output()
		if err != nil {
			t.Fatalf("decode continuous audio: %v", err)
		}
		return out
	}
	a, b := pcm(source, start), pcm(path, 0)
	if math.Abs(float64(len(b)/4)/44100-duration) > .025 {
		t.Fatalf("audio duration=%f expected=%f", float64(len(b)/4)/44100, duration)
	}
	for _, seconds := range []float64{0, 9.8, 19.8} {
		from, to := int(seconds*44100), min(int((seconds+.4)*44100), len(a)/4, len(b)/4)
		if from >= to {
			continue
		}
		var ab, aa, bb float64
		for i := from; i < to; i++ {
			x := float64(math.Float32frombits(binary.LittleEndian.Uint32(a[i*4:])))
			y := float64(math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:])))
			ab, aa, bb = ab+x*y, aa+x*x, bb+y*y
		}
		if correlation := ab / math.Sqrt(aa*bb); math.IsNaN(correlation) || correlation < .98 {
			t.Fatalf("audio alignment near %.1fs: %.6f", seconds, correlation)
		}
	}
}

func TestVideoTransferMergeFractionalRateAndOrder(t *testing.T) {
	testVideoTransferMergeTimestampOrigin(t, 0)
}

func TestVideoTransferMergeNonzeroTimestampOrigin(t *testing.T) {
	testVideoTransferMergeTimestampOrigin(t, 5)
}

func testVideoTransferMergeTimestampOrigin(t *testing.T, origin float64) {
	requireFFmpegTools(t)
	svc, db := videoTransferTestService(t)
	sourceID := videoTransferTestSource(t, svc, db)
	source := os.Getenv("VIDEO_TRANSFER_TEST_SOURCE")
	const fps = 30000.0 / 1001
	// An unaligned half-frame start must use the same half-up boundaries as runner.
	start := 2.5 / fps
	args := []string{"-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=160x90:r=30000/1001:d=21.5", "-f", "lavfi", "-i", "aevalsrc=sin(2*PI*(200*t+20*t*t)):s=44100:d=25"}
	if origin != 0 {
		// Earlier audio makes format.start_time differ from the video origin.
		args = append(args, "-vf", fmt.Sprintf("setpts=PTS+%v/TB", origin), "-af", fmt.Sprintf("asetpts=PTS+%v/TB", origin-2), "-fps_mode", "passthrough")
	}
	args = append(args, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", source)
	if output, err := exec.Command("ffmpeg", args...).CombinedOutput(); err != nil {
		t.Fatalf("%v %s", err, output)
	}
	info, err := probeMediaFileDuration(context.Background(), source, true)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(info.VideoStartSeconds-origin) > 1/fps {
		t.Fatalf("unexpected video origin: %+v", info)
	}
	settings := videotransfer.Settings{StartSeconds: start, DurationSeconds: 21.3, Workers: 2}
	segments := videotransfer.SegmentsForTimeline(settings, info.VideoDurationSeconds, fps)
	for i := range segments {
		part := &segments[i]
		first := int(math.Round(part.StartSeconds * fps))
		last := int(math.Round((part.StartSeconds + part.DurationSeconds) * fps))
		path := filepath.Join(t.TempDir(), "output.mp4")
		filter := fmt.Sprintf("trim=start_frame=%d:end_frame=%d,setpts=PTS-STARTPTS", first, last)
		if output, err := exec.Command("ffmpeg", "-v", "error", "-y", "-i", source, "-vf", filter, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", path).CombinedOutput(); err != nil {
			t.Fatalf("%v %s", err, output)
		}
		partInfo, _ := probeMediaFileDuration(context.Background(), path, true)
		file, _ := os.Open(path)
		stat, _ := file.Stat()
		resource, storeErr := svc.storeVideoTransferOutput("transfer-owner", "part.mp4", stat.Size(), partInfo, file, fmt.Sprint("fractional-", i))
		_ = file.Close()
		if storeErr != nil {
			t.Fatal(storeErr)
		}
		part.Status, part.Progress, part.OutputResourceID = "succeeded", 100, resource.ID
		part.Summary = &videotransfer.Summary{Frames: last - first}
	}
	segments[0], segments[2] = segments[2], segments[0]
	task := &model.Task{ID: "fractional", UserID: "transfer-owner"}
	result, err := svc.mergeVideoTransferOutputs(context.Background(), task, videoTransferTaskInput{SourceResourceID: sourceID, Branch: videotransfer.Fixed}, source, t.TempDir(), info, segments, func(string, int) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	_, first, frames, _ := orderVideoTransferSegments(segments, info)
	assertVideoTransferCompleteMedia(t, svc, source, videotransfer.BranchState{OutputResourceID: result.ID, Segments: segments}, frames, float64(frames)/fps, info.VideoStartSeconds+float64(first)/fps)
	if first != 3 || frames != 638 {
		t.Fatalf("frame boundaries first=%d count=%d", first, frames)
	}
}

func TestVideoTransferMergeValidationAndProgress(t *testing.T) {
	segments := videotransfer.Segments(videotransfer.Settings{}, 21.3)
	for i := range segments {
		segments[i].Status, segments[i].OutputResourceID = "succeeded", strconv.Itoa(i)
		segments[i].Summary = &videotransfer.Summary{Frames: int(math.Round(segments[i].DurationSeconds * 30))}
	}
	segments[0].Progress, segments[1].Progress = 100, 100
	if progress := videoTransferProcessingProgress(segments); progress != 84 {
		t.Fatalf("short final interval must not weigh one third: %d", progress)
	}
	segments[2].Progress = 100
	if progress := videoTransferProcessingProgress(segments); progress != 90 {
		t.Fatalf("assembly budget lost: %d", progress)
	}
	if _, _, frames, err := orderVideoTransferSegments(segments, mediaProbeInfo{FrameRate: 30, Frames: 12}); err != nil || frames != 639 {
		t.Fatalf("source frame count truncated normalized output: frames=%d err=%v", frames, err)
	}
	segments[1].StartSeconds = 10.1
	if _, _, _, err := orderVideoTransferSegments(segments, mediaProbeInfo{FrameRate: 30}); err == nil {
		t.Fatal("gapped/overlapping output accepted")
	}
}

func TestVideoTransferCompleteQuotaKeepsAccountLimits(t *testing.T) {
	svc, _ := videoTransferTestService(t)
	policy, _ := svc.RuntimePolicy()
	size := megabytes(policy.Resource.GeneratedFileMB) + 1
	if _, err := svc.reserveGeneratedResourceQuota("transfer-owner", size); err == nil {
		t.Fatal("generic generated file cap unexpectedly relaxed")
	}
	day, err := svc.reserveVideoTransferCompleteQuota("transfer-owner", size)
	if err != nil {
		t.Fatal(err)
	}
	svc.releaseUserUploadQuota("transfer-owner", day, size)
	if _, err = svc.reserveVideoTransferCompleteQuota("transfer-owner", megabytes(policy.Resource.DailyUploadMB)); err == nil {
		t.Fatal("complete output bypassed account/daily quota")
	}
	limit, _ := svc.storedFileLimit("transfer-owner", policy)
	if _, err = svc.reserveVideoTransferCompleteQuota("transfer-owner", limit); err == nil {
		t.Fatal("complete output bypassed stored capacity")
	}
}

func TestVideoTransferRetryChecksOriginalSnapshot(t *testing.T) {
	for _, changed := range []string{"same", "settings", "source", "foreign-output", "strategy-version", "fixed-region"} {
		t.Run(changed, func(t *testing.T) {
			svc, db := videoTransferTestService(t)
			source := videoTransferTestSource(t, svc, db)
			videoTransferTestRunner(t, "exit 1")
			record, _ := svc.CreateVideoTransfer("transfer-owner", VideoTransferCreateRequest{SourceResourceID: source})
			record, err := svc.RunVideoTransfer(context.Background(), "transfer-owner", record.ID, VideoTransferRunRequest{Branch: videotransfer.Fixed, ExpectedRevision: record.Revision})
			if err != nil {
				t.Fatal(err)
			}
			state := record.Branches[videotransfer.Fixed]
			task, _ := svc.repo.TaskForUser("transfer-owner", state.TaskID)
			var input videoTransferTaskInput
			_ = json.Unmarshal([]byte(task.InputJSON), &input)
			if changed == "settings" {
				input.Settings.Target = "bottom-left"
			}
			if changed == "source" {
				input.SourceResourceID = "different-source"
			}
			if changed == "strategy-version" {
				input.StrategyVersion = "fixed-r16-regions-v1"
			}
			if changed == "fixed-region" {
				input.Settings.FixedRegions = []videotransfer.FixedRegion{{ROI: videotransfer.ROI{Width: 20, Height: 10}, ReferenceSeconds: .5}}
			}
			encoded, _ := json.Marshal(input)
			state.Segments[0].Status, state.Segments[0].Progress, state.Segments[0].OutputResourceID = "succeeded", 100, source
			state.Segments[0].Summary = &videotransfer.Summary{Frames: 30}
			if changed == "foreign-output" {
				foreign := &model.Resource{ID: "foreign-part", UserID: "transfer-other", Status: model.ResourceStatusReady, MimeType: "video/mp4", Size: 1}
				if err = db.Create(foreign).Error; err != nil {
					t.Fatal(err)
				}
				state.Segments[0].OutputResourceID = foreign.ID
			}
			result, _ := json.Marshal(map[string]any{"segments": state.Segments})
			if err = db.Model(&model.Task{}).Where("id = ?", task.ID).Updates(map[string]any{"status": model.TaskStatusFailed, "input_json": string(encoded), "result_json": string(result)}).Error; err != nil {
				t.Fatal(err)
			}
			retried, err := svc.RunVideoTransfer(context.Background(), "transfer-owner", record.ID, VideoTransferRunRequest{Branch: videotransfer.Fixed, ExpectedRevision: record.Revision})
			if err != nil {
				t.Fatal(err)
			}
			resumed := retried.Branches[videotransfer.Fixed].Segments[0].Status == "succeeded"
			if resumed != (changed == "same") {
				t.Fatalf("resumed incompatible snapshot=%v", resumed)
			}
		})
	}
}

func TestVideoTransferConcatenationCancellationAndLeaseConflict(t *testing.T) {
	if os.PathSeparator == '\\' {
		t.Skip("POSIX subprocess fixture")
	}
	for _, mode := range []string{"cancel", "lease"} {
		t.Run(mode, func(t *testing.T) {
			bin := filepath.Join(t.TempDir(), "ffmpeg")
			if err := os.WriteFile(bin, []byte("#!/bin/sh\nprintf 'out_time_us=5000000\\n'\nsleep 30 & wait\n"), 0700); err != nil {
				t.Fatal(err)
			}
			t.Setenv("CANVAS_FFMPEG_PATH", bin)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			started := time.Now()
			err := concatenateVideoTransfer(ctx, "list", "source", "out", 0, 10, func(string, int) error {
				if mode == "lease" {
					return repository.ErrTaskStateConflict
				}
				cancel()
				return nil
			})
			if err == nil || time.Since(started) > 5*time.Second {
				t.Fatalf("did not promptly stop: %v", err)
			}
			if mode == "lease" && !errors.Is(err, repository.ErrTaskStateConflict) {
				t.Fatal(err)
			}
		})
	}
}

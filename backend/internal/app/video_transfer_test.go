package app

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"gorm.io/gorm"
	"yingce/backend/internal/database"
	"yingce/backend/internal/model"
	"yingce/backend/internal/repository"
	"yingce/backend/internal/videotransfer"
)

func TestVideoTransferParallelSegmentsKeepPartialOutputs(t *testing.T) {
	requireFFmpegTools(t)
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 fixture")
	}
	for _, name := range []string{"complete", "partial", "publication-conflict"} {
		t.Run(name, func(t *testing.T) {
			partial, publicationConflict := name == "partial", name == "publication-conflict"
			svc, db := videoTransferTestService(t)
			source := videoTransferTestSource(t, svc, db)
			path := os.Getenv("VIDEO_TRANSFER_TEST_SOURCE")
			if output, err := exec.Command("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=160x90:r=30:d=21.3", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=21.3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", path).CombinedOutput(); err != nil {
				t.Fatalf("%v %s", err, output)
			}
			stat, _ := os.Stat(path)
			if err := db.Model(&model.Resource{}).Where("id = ?", source).Updates(map[string]any{"size": stat.Size(), "duration_ms": 21300}).Error; err != nil {
				t.Fatal(err)
			}
			markers := t.TempDir()
			t.Setenv("VIDEO_TRANSFER_MARKERS", markers)
			t.Setenv("VIDEO_TRANSFER_PARTIAL", "")
			if partial {
				t.Setenv("VIDEO_TRANSFER_PARTIAL", "1")
			}
			videoTransferTestRunner(t, `python3 - "$2" "$4" <<'PY'
import json, os, pathlib, subprocess, sys, time
r=json.load(open(sys.argv[1])); out=sys.argv[2]; index=int(r['startSeconds']//10)
root=pathlib.Path(os.environ['VIDEO_TRANSFER_MARKERS'])
assert pathlib.Path(r['validationCacheDir']) == pathlib.Path(r['input']).parent
assert r['fixedRegions'][0]['referenceSeconds'] == 1
assert r['fixedRegions'][0]['startSeconds'] == 0 and r['fixedRegions'][0]['durationSeconds'] == 12
assert r['fixedRegions'][1]['referenceSeconds'] == 20.5
assert r['fixedRegions'][1]['startSeconds'] == 12 and r['fixedRegions'][1]['durationSeconds'] == 0
(root/('start'+str(index))).write_text(r['input'])
if index < 2:
    until=time.monotonic()+5
    while not all((root/('start'+str(i))).exists() for i in (0,1)):
        assert time.monotonic()<until, 'segments not concurrent'
        time.sleep(.01)
if os.environ.get('VIDEO_TRANSFER_PARTIAL') and index==1:
    raise RuntimeError('intentional segment failure')
subprocess.run(['ffmpeg','-v','error','-y','-ss',str(r['startSeconds']),'-i',r['input'],'-t',str(r['durationSeconds']),'-c:v','libx264','-pix_fmt','yuv420p',out],check=True)
json.dump({'frames':round(r['durationSeconds']*30),'changedPixels':0,'trackingCoverage':0,'timelineMode':'timestamp-resampled-cfr-v1','outputFps':30,'sourceTimestampOrigin':0,'startSeconds':r['startSeconds'],'durationSeconds':r['durationSeconds']},open(out.replace('.mp4','.report.json'),'w'))
PY`)
			record, err := svc.CreateVideoTransfer("transfer-owner", VideoTransferCreateRequest{SourceResourceID: source})
			if err != nil {
				t.Fatal(err)
			}
			if len(record.Branches[videotransfer.Fixed].Segments) != 3 {
				t.Fatalf("upload segments %+v", record)
			}
			regions := []videotransfer.FixedRegion{
				{ROI: videotransfer.ROI{Width: 20, Height: 10}, ReferenceSeconds: 1, DurationSeconds: 12},
				{ROI: videotransfer.ROI{X: 100, Width: 20, Height: 10}, ReferenceSeconds: 20.5, StartSeconds: 12},
			}
			record, err = svc.UpdateVideoTransfer("transfer-owner", record.ID, VideoTransferPatch{ExpectedRevision: record.Revision, Branches: map[string]videotransfer.BranchPatch{videotransfer.Fixed: {Settings: videotransfer.SettingsPatch{FixedRegions: &regions}}}})
			if err != nil {
				t.Fatal(err)
			}
			record, err = svc.RunVideoTransfer(context.Background(), "transfer-owner", record.ID, VideoTransferRunRequest{Branch: videotransfer.Fixed, ExpectedRevision: record.Revision})
			if err != nil {
				t.Fatal(err)
			}
			var unpublished *model.Resource
			if publicationConflict {
				err = db.Callback().Update().Before("gorm:update").Register("test:reject-video-completion", func(tx *gorm.DB) {
					if task, ok := tx.Statement.Dest.(*model.Task); ok && task.Type == videotransfer.TaskType && task.Status == model.TaskStatusSucceeded {
						var result struct {
							ResourceID string `json:"resourceId"`
						}
						_ = json.Unmarshal([]byte(task.ResultJSON), &result)
						var resource model.Resource
						if tx.Session(&gorm.Session{NewDB: true}).Where("user_id = ? AND id = ?", task.UserID, result.ResourceID).First(&resource).Error == nil {
							unpublished = &resource
						}
						tx.AddError(repository.ErrTaskStateConflict)
					}
				})
				if err != nil {
					t.Fatal(err)
				}
			}
			if err = svc.taskWorker().processNextTask(); err != nil && !publicationConflict {
				t.Fatal(err)
			}
			if publicationConflict {
				_ = db.Callback().Update().Remove("test:reject-video-completion")
				if unpublished == nil || unpublished.ID == "" {
					t.Fatal("did not reach final resource publication")
				}
				if _, readErr := svc.repo.ResourceForUser("transfer-owner", unpublished.ID); !errors.Is(readErr, gorm.ErrRecordNotFound) {
					t.Fatalf("unpublished merged resource remains: %v", readErr)
				}
				if _, statErr := os.Stat(filepath.Join(svc.dataDir, "resources", filepath.FromSlash(unpublished.ObjectKey))); !os.IsNotExist(statErr) {
					t.Fatal("unpublished merged file remains")
				}
			}
			finished, err := svc.VideoTransfer("transfer-owner", record.ID)
			if err != nil {
				t.Fatal(err)
			}
			state := finished.Branches[videotransfer.Fixed]
			expected := "succeeded"
			if partial || publicationConflict {
				expected = "failed"
			}
			if state.Status != expected || len(state.Segments) != 3 || (state.OutputResourceID == "") != (partial || publicationConflict) {
				t.Fatalf("state %+v", state)
			}
			shared := ""
			for i, part := range state.Segments {
				if partial && i == 1 {
					if part.Status != "failed" {
						t.Fatal(part)
					}
					continue
				}
				if part.Status != "succeeded" || part.OutputResourceID == "" || part.Summary == nil {
					t.Fatal(part)
				}
				marker, _ := os.ReadFile(filepath.Join(markers, "start"+string(rune('0'+i))))
				if shared != "" && shared != string(marker) {
					t.Fatal("source copied per segment")
				}
				shared = string(marker)
				asset := model.Asset{ID: "segment-asset-" + part.ID, UserID: "transfer-owner", PayloadJSON: `{"data":{"storageKey":"resource:` + part.OutputResourceID + `"}}`}
				if err = db.Create(&asset).Error; err != nil {
					t.Fatal(err)
				}
				if err = svc.DeleteUserAsset("transfer-owner", asset.ID); err == nil {
					t.Fatal("segment output not reference-protected")
				}
			}
			if !partial && !publicationConflict {
				assertVideoTransferCompleteMedia(t, svc, path, state, 639, 21.3, 0)
				// Simulate a worker crash after the final segment commit but before parent completion.
				if err = db.Model(&model.Task{}).Where("id = ?", state.TaskID).Updates(map[string]any{"status": model.TaskStatusQueued, "completed_at": nil, "lease_owner": "", "lease_expires_at": nil}).Error; err != nil {
					t.Fatal(err)
				}
				markerBefore, _ := os.Stat(filepath.Join(markers, "start0"))
				if err = svc.taskWorker().processNextTask(); err != nil {
					t.Fatal(err)
				}
				recovered, err := svc.VideoTransfer("transfer-owner", record.ID)
				if err != nil {
					t.Fatal(err)
				}
				markerAfter, _ := os.Stat(filepath.Join(markers, "start0"))
				if !markerAfter.ModTime().Equal(markerBefore.ModTime()) {
					t.Fatal("recovery reran a successful segment")
				}
				for i, part := range recovered.Branches[videotransfer.Fixed].Segments {
					if part.OutputResourceID != state.Segments[i].OutputResourceID {
						t.Fatal("recovery lost successful output")
					}
				}
			} else {
				// A user retry must preserve successful internal work as well as lease recovery.
				before0, _ := os.Stat(filepath.Join(markers, "start0"))
				before2, _ := os.Stat(filepath.Join(markers, "start2"))
				t.Setenv("VIDEO_TRANSFER_PARTIAL", "")
				retried, retryErr := svc.RunVideoTransfer(context.Background(), "transfer-owner", record.ID, VideoTransferRunRequest{Branch: videotransfer.Fixed, ExpectedRevision: finished.Revision})
				if retryErr != nil {
					t.Fatal(retryErr)
				}
				if retryErr = svc.taskWorker().processNextTask(); retryErr != nil {
					t.Fatal(retryErr)
				}
				retried, retryErr = svc.VideoTransfer("transfer-owner", retried.ID)
				if retryErr != nil || retried.Branches[videotransfer.Fixed].Status != "succeeded" {
					t.Fatalf("retry=%+v %v", retried, retryErr)
				}
				after0, _ := os.Stat(filepath.Join(markers, "start0"))
				after2, _ := os.Stat(filepath.Join(markers, "start2"))
				if !before0.ModTime().Equal(after0.ModTime()) || !before2.ModTime().Equal(after2.ModTime()) {
					t.Fatal("user retry reran completed work")
				}
				assertVideoTransferCompleteMedia(t, svc, path, retried.Branches[videotransfer.Fixed], 639, 21.3, 0)
			}
		})
	}
}

func TestVideoTransferTimeoutScalesWithSegments(t *testing.T) {
	payload, _ := json.Marshal(videoTransferTaskInput{SegmentCount: 100})
	if got := taskExecutionTimeout(&model.Task{Type: videotransfer.TaskType, InputJSON: string(payload)}, RuntimeTaskPolicy{}); got < 200*time.Hour {
		t.Fatalf("timeout=%v", got)
	}
}

func TestVideoTransferLargeReservedFailedUploadRetry(t *testing.T) {
	svc, db := videoTransferTestService(t)
	videoTransferTestSource(t, svc, db)
	input := os.Getenv("VIDEO_TRANSFER_TEST_SOURCE")
	const size = int64(60 << 20)
	if err := os.Truncate(input, size); err != nil {
		t.Fatal(err)
	}
	key := normalizedResourceUploadKey([]string{"large-video-retry"})
	failed := &model.Resource{ID: "failed-video", UserID: "transfer-owner", Kind: "video", Status: model.ResourceStatusFailed, Provider: "local", ObjectKey: "retry/output.mp4", MimeType: "video/mp4", Size: size, UploadKey: key, Width: 160, Height: 90, DurationMs: 1000}
	if err := svc.repo.CreateResource(failed); err != nil {
		t.Fatal(err)
	}
	if err := svc.ReserveVideoTransferUploadSession("transfer-owner", "retry-session", size, time.Now().Add(time.Hour), 32); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(input)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	output, err := svc.UploadVideoTransferSourceFile("transfer-owner", "retry-session", "input.mp4", size, "video", 160, 90, 1000, f, "large-video-retry")
	if err != nil {
		t.Fatal(err)
	}
	if output.ID != failed.ID || output.Status != model.ResourceStatusReady {
		t.Fatalf("retry=%+v", output)
	}
	used, err := svc.repo.DailyUploadBytes("transfer-owner", time.Now().UTC().Format("2006-01-02"))
	if err != nil || used != size {
		t.Fatalf("quota=%d %v", used, err)
	}
}

func TestVideoTransferReservedRetryPublicationFailureRemovesObject(t *testing.T) {
	svc, _ := videoTransferTestService(t)
	failed := &model.Resource{ID: "failed-reserved", UserID: "transfer-owner", Kind: "video", Status: model.ResourceStatusFailed, Provider: "local", ObjectKey: "retry/failed.mp4", MimeType: "video/mp4", Size: 7}
	if err := svc.repo.CreateResource(failed); err != nil {
		t.Fatal(err)
	}
	if _, err := svc.retryStoredResource("transfer-owner", failed, "video", "video/mp4", 7, strings.NewReader("payload"), "expired-reservation"); err == nil {
		t.Fatal("expired reservation published")
	}
	if _, err := os.Stat(filepath.Join(svc.dataDir, "resources", "retry", "failed.mp4")); !os.IsNotExist(err) {
		t.Fatalf("unpublished object remains: %v", err)
	}
	if _, err := svc.repo.ResourceForUser("transfer-owner", failed.ID); !errors.Is(err, gorm.ErrRecordNotFound) {
		t.Fatalf("unpublished resource remains: %v", err)
	}
}

func videoTransferTestService(t *testing.T) (*Service, *gorm.DB) {
	t.Helper()
	db := newSQLiteTestDB(t)
	if err := database.MigrateSchema(db); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"transfer-owner", "transfer-other"} {
		if err := db.Create(&model.User{ID: id, Username: id, Status: model.UserStatusActive}).Error; err != nil {
			t.Fatal(err)
		}
	}
	svc := New(repository.New(db), t.TempDir())
	t.Cleanup(func() { _ = svc.Close() })
	return svc, db
}

func videoTransferTestSource(t *testing.T, svc *Service, db *gorm.DB) string {
	t.Helper()
	requireFFmpegTools(t)
	dir := filepath.Join(svc.dataDir, "resources")
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "source.mp4")
	writeTestVideo(t, path, 160, 90, nil)
	stat, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if err = db.Create(&model.Resource{ID: "source-video", UserID: "transfer-owner", Kind: "media", Status: model.ResourceStatusReady, Provider: "local", ObjectKey: "source.mp4", MimeType: "video/mp4", Size: stat.Size(), Width: 160, Height: 90, DurationMs: 1000}).Error; err != nil {
		t.Fatal(err)
	}
	t.Setenv("VIDEO_TRANSFER_TEST_SOURCE", path)
	return "source-video"
}

func videoTransferTestRunner(t *testing.T, body string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("POSIX subprocess fixture")
	}
	dir := t.TempDir()
	runner := filepath.Join(dir, "runner.py")
	if err := os.WriteFile(runner, []byte("# fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	python := filepath.Join(dir, "python-fixture")
	script := "#!/bin/sh\nset -eu\nshift\nif [ \"$1\" = \"--check\" ]; then\n  printf '%s\\n' '{\"branches\":{\"fixed-r16\":{\"available\":true},\"moving-v6\":{\"available\":true}}}'\n  exit 0\nfi\n" + body + "\n"
	if err := os.WriteFile(python, []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("CANVAS_WATERMARK_PYTHON", python)
	t.Setenv("CANVAS_WATERMARK_RUNNER", runner)
}

func TestVideoTransferIsolationRevisionAndSettings(t *testing.T) {
	svc, db := videoTransferTestService(t)
	if _, err := svc.CreateTask("transfer-owner", CreateTaskRequest{Type: videotransfer.TaskType}); err == nil {
		t.Fatal("generic task endpoint bypassed workflow validation")
	}
	record, err := svc.CreateVideoTransfer("transfer-owner", VideoTransferCreateRequest{Title: "clip alpha"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = svc.VideoTransfer("transfer-other", record.ID); err == nil {
		t.Fatal("cross-user record read accepted")
	}
	if _, err = svc.UpdateVideoTransfer("transfer-other", record.ID, VideoTransferPatch{ExpectedRevision: 1}); err == nil {
		t.Fatal("cross-user write accepted")
	}
	if err = db.Create(&model.Resource{ID: "foreign", UserID: "transfer-other", Status: model.ResourceStatusReady, MimeType: "video/mp4", Size: 10}).Error; err != nil {
		t.Fatal(err)
	}
	foreign := "foreign"
	if _, err = svc.UpdateVideoTransfer("transfer-owner", record.ID, VideoTransferPatch{ExpectedRevision: 1, SourceResourceID: &foreign}); err == nil {
		t.Fatal("cross-user source accepted")
	}
	title := "clip beta"
	updated, err := svc.UpdateVideoTransfer("transfer-owner", record.ID, VideoTransferPatch{ExpectedRevision: 1, Title: &title})
	if err != nil {
		t.Fatal(err)
	}
	if updated.Revision != 2 {
		t.Fatalf("revision=%d", updated.Revision)
	}
	if _, err = svc.UpdateVideoTransfer("transfer-owner", record.ID, VideoTransferPatch{ExpectedRevision: 1, Title: &title}); err == nil {
		t.Fatal("stale revision accepted")
	}
	selected := videotransfer.Moving
	if _, err = svc.UpdateVideoTransfer("transfer-owner", record.ID, VideoTransferPatch{ExpectedRevision: 2, SelectedBranch: &selected, Branches: map[string]videotransfer.BranchPatch{videotransfer.Moving: {}}}); err != nil {
		t.Fatalf("draft without ROI rejected: %v", err)
	}
	page, err := svc.VideoTransfers("transfer-owner", 1, 10, "beta")
	if err != nil || page.Total != 1 {
		t.Fatalf("search=%+v %v", page, err)
	}
	page, err = svc.VideoTransfers("transfer-other", 1, 10, "")
	if err != nil || page.Total != 0 {
		t.Fatalf("user-isolated list=%+v %v", page, err)
	}
}

func TestVideoTransferTaskRunPersistenceAndReferences(t *testing.T) {
	svc, db := videoTransferTestService(t)
	source := videoTransferTestSource(t, svc, db)
	videoTransferTestRunner(t, `cp "$VIDEO_TRANSFER_TEST_SOURCE" "$4"
printf '%s\n' '{"progress":70,"stage":"处理测试视频"}'
printf '%s\n' '{"frames":30,"changedPixels":100,"trackingCoverage":1,"trackedFrames":30,"limitations":["测试处理摘要"],"timelineMode":"timestamp-resampled-cfr-v1","outputFps":30,"sourceTimestampOrigin":0,"startSeconds":0,"durationSeconds":1}' > "${4%.mp4}.report.json"`)
	record, err := svc.CreateVideoTransfer("transfer-owner", VideoTransferCreateRequest{SourceResourceID: source})
	if err != nil {
		t.Fatal(err)
	}
	record, err = svc.RunVideoTransfer(context.Background(), "transfer-owner", record.ID, VideoTransferRunRequest{Branch: videotransfer.Fixed, ExpectedRevision: record.Revision})
	if err != nil {
		t.Fatal(err)
	}
	taskID := record.Branches[videotransfer.Fixed].TaskID
	if taskID == "" || record.Branches[videotransfer.Fixed].Status != "queued" {
		t.Fatalf("not queued: %+v", record)
	}
	if _, err = svc.RunVideoTransfer(context.Background(), "transfer-owner", record.ID, VideoTransferRunRequest{Branch: videotransfer.Fixed, ExpectedRevision: record.Revision}); err == nil {
		t.Fatal("duplicate active run accepted")
	}
	if err = svc.taskWorker().processNextTask(); err != nil {
		t.Fatal(err)
	}
	reloaded := &Service{repo: repository.New(db), dataDir: svc.dataDir}
	finished, err := reloaded.VideoTransfer("transfer-owner", record.ID)
	if err != nil {
		t.Fatal(err)
	}
	state := finished.Branches[videotransfer.Fixed]
	if state.Status != "succeeded" || state.OutputResourceID == "" || state.Summary == nil || state.Summary.Frames != 30 {
		t.Fatalf("bad terminal state: %+v", state)
	}
	if finished.Revision != record.Revision {
		t.Fatal("progress changed configuration revision")
	}
	output, err := svc.repo.ResourceForUser("transfer-owner", state.OutputResourceID)
	if err != nil || output.Status != model.ResourceStatusReady {
		t.Fatalf("output=%+v %v", output, err)
	}
	if _, err = svc.repo.ResourceForUser("transfer-other", state.OutputResourceID); !errors.Is(err, gorm.ErrRecordNotFound) {
		t.Fatal("output crossed user boundary")
	}
	snapshot, err := svc.repo.ResourceReferenceSnapshot("transfer-owner", "", []string{source, state.OutputResourceID})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, doc := range snapshot.Documents {
		if doc.Kind == "视频搬运" && strings.Contains(doc.PrimaryJSON, state.OutputResourceID) {
			found = true
		}
	}
	if !found {
		t.Fatal("workflow output missing from deletion reference snapshot")
	}
	asset := model.Asset{ID: "transfer-output-asset", UserID: "transfer-owner", Title: "处理结果", PayloadJSON: `{"data":{"storageKey":"resource:` + state.OutputResourceID + `"}}`}
	if err = db.Create(&asset).Error; err != nil {
		t.Fatal(err)
	}
	if err = svc.DeleteUserAsset("transfer-owner", asset.ID); err == nil || !strings.Contains(err.Error(), "视频搬运") {
		t.Fatalf("workflow result deletion must be blocked: %v", err)
	}
	workers := 3
	changed, err := svc.UpdateVideoTransfer("transfer-owner", record.ID, VideoTransferPatch{ExpectedRevision: finished.Revision, Branches: map[string]videotransfer.BranchPatch{videotransfer.Moving: {Settings: videotransfer.SettingsPatch{Workers: &workers}}}})
	if err != nil || changed.Branches[videotransfer.Fixed].OutputResourceID != state.OutputResourceID {
		t.Fatalf("other branch output lost: %+v %v", changed, err)
	}
	if _, err = svc.RetryTask("transfer-owner", taskID); err == nil {
		t.Fatal("generic retry bypassed record validation")
	}
}

func TestVideoTransferFailureAndCancellation(t *testing.T) {
	for _, mode := range []string{"failed", "cancelled"} {
		t.Run(mode, func(t *testing.T) {
			svc, db := videoTransferTestService(t)
			source := videoTransferTestSource(t, svc, db)
			body := `printf '%s\n' 'ERROR: 水印区域缺少有效模板' >&2; exit 1`
			if mode == "cancelled" {
				body = `printf '%s\n' '{"progress":20,"stage":"等待取消"}'; sleep 30 & wait`
			}
			videoTransferTestRunner(t, body)
			record, err := svc.CreateVideoTransfer("transfer-owner", VideoTransferCreateRequest{SourceResourceID: source})
			if err != nil {
				t.Fatal(err)
			}
			record, err = svc.RunVideoTransfer(context.Background(), "transfer-owner", record.ID, VideoTransferRunRequest{Branch: videotransfer.Fixed, ExpectedRevision: record.Revision})
			if err != nil {
				t.Fatal(err)
			}
			done := make(chan error, 1)
			go func() { done <- svc.taskWorker().processNextTask() }()
			if mode == "cancelled" {
				deadline := time.Now().Add(5 * time.Second)
				for time.Now().Before(deadline) {
					task, readErr := svc.repo.TaskForUser("transfer-owner", record.Branches[videotransfer.Fixed].TaskID)
					if readErr == nil && strings.Contains(task.ResultJSON, "等待取消") {
						break
					}
					time.Sleep(10 * time.Millisecond)
				}
				if _, err = svc.CancelVideoTransfer(context.Background(), "transfer-owner", record.ID); err != nil {
					t.Fatal(err)
				}
			}
			select {
			case err = <-done:
				if err != nil {
					t.Fatal(err)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("subprocess tree did not stop")
			}
			finished, err := svc.VideoTransfer("transfer-owner", record.ID)
			if err != nil {
				t.Fatal(err)
			}
			if state := finished.Branches[videotransfer.Fixed]; state.Status != mode || state.OutputResourceID != "" {
				t.Fatalf("state=%+v", state)
			}
			if mode == "failed" {
				state := finished.Branches[videotransfer.Fixed]
				task, readErr := svc.repo.TaskForUser("transfer-owner", state.TaskID)
				if readErr != nil || !strings.Contains(state.Error, "水印区域缺少有效模板") || task.Error != state.Error {
					t.Fatalf("actual failure was not persisted: task=%+v state=%+v err=%v", task, state, readErr)
				}
			}
			entries, _ := os.ReadDir(filepath.Join(svc.dataDir, "tmp", "video-transfers"))
			if len(entries) != 0 {
				t.Fatal("temporary algorithm files were not cleaned")
			}
		})
	}
}

func TestVideoTransferMissingRuntimeCapabilities(t *testing.T) {
	t.Setenv("CANVAS_WATERMARK_PYTHON", filepath.Join(t.TempDir(), "missing-python"))
	caps := (&Service{}).GetVideoTransferCapabilities(context.Background())
	for _, branch := range caps.Branches {
		if branch.Available || branch.Reason == "" {
			t.Fatalf("missing runtime advertised: %+v", branch)
		}
	}
	if caps.Limits.MaxPixels < 3840*2160 {
		t.Fatal("4K capability regressed")
	}
}

func TestVideoTransferProbeHonorsConfiguredBinary(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX subprocess fixture")
	}
	bin := filepath.Join(t.TempDir(), "configured-ffprobe")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\nprintf '%s\\n' '{\"streams\":[{\"width\":320,\"height\":180}],\"format\":{\"duration\":\"1.5\"}}'\n"), 0700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("CANVAS_FFPROBE_PATH", bin)
	t.Setenv("PATH", t.TempDir())
	if !mediaProbeAvailable() {
		t.Fatal("configured ffprobe was not detected")
	}
	info, err := probeMediaFile(context.Background(), "input.mp4")
	if err != nil || info.Width != 320 || info.Height != 180 || info.DurationMs != 1500 {
		t.Fatalf("configured probe=%+v %v", info, err)
	}
	t.Setenv("CANVAS_FFPROBE_PATH", bin+"-missing")
	if _, err = probeMediaFile(context.Background(), "input.mp4"); !errors.Is(err, errMediaProbeUnavailable) {
		t.Fatalf("invalid configured probe should fail closed: %v", err)
	}
}

func TestVideoTransferUsesVideoDurationWithLongerAudio(t *testing.T) {
	requireFFmpegTools(t)
	path := filepath.Join(t.TempDir(), "longer-audio.mp4")
	output, err := exec.Command("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=160x90:r=30:d=2", "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", path).CombinedOutput()
	if err != nil {
		t.Fatalf("generate longer audio fixture: %v %s", err, output)
	}
	containerInfo, err := probeMediaFile(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	videoInfo, err := probeMediaFileDuration(context.Background(), path, true)
	if err != nil || videoInfo.DurationMs != 2000 || containerInfo.DurationMs != 3000 {
		t.Fatalf("video=%+v container=%+v err=%v", videoInfo, containerInfo, err)
	}
	unknown, err := parseMediaProbeOutputDuration([]byte(`{"streams":[{"width":160,"height":90,"duration":"N/A"}],"format":{"duration":"3"}}`), true)
	if err != nil || unknown.DurationMs != 3000 {
		t.Fatalf("unknown video duration must fall back to container: %+v %v", unknown, err)
	}
}

package app

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"yingce/backend/internal/model"
	"yingce/backend/internal/videotransfer"
)

func TestVideoTransferFixedRegionsRunnerContract(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 fixture")
	}
	svc, db := videoTransferTestService(t)
	source := videoTransferTestSource(t, svc, db)
	observedPath := filepath.Join(t.TempDir(), "request.json")
	t.Setenv("VIDEO_TRANSFER_OBSERVED_REQUEST", observedPath)
	videoTransferTestRunner(t, `python3 - "$2" "$4" <<'PY'
import json, os, shutil, sys
r=json.load(open(sys.argv[1])); output=sys.argv[2]
shutil.copyfile(sys.argv[1], os.environ['VIDEO_TRANSFER_OBSERVED_REQUEST'])
shutil.copyfile(r['input'], output)
json.dump({'frames':30,'changedPixels':0,'trackingCoverage':0,'timelineMode':'timestamp-resampled-cfr-v1','outputFps':30,'sourceTimestampOrigin':0,'startSeconds':0,'durationSeconds':1},open(output.replace('.mp4','.report.json'),'w'))
PY`)
	record, err := svc.CreateVideoTransfer("transfer-owner", VideoTransferCreateRequest{SourceResourceID: source})
	if err != nil {
		t.Fatal(err)
	}
	regions := []videotransfer.FixedRegion{
		{ROI: videotransfer.ROI{X: 1, Y: 2, Width: 20, Height: 10}, ReferenceSeconds: .2, DurationSeconds: .5},
		{ROI: videotransfer.ROI{X: 100, Y: 3, Width: 30, Height: 15}, ReferenceSeconds: .8, StartSeconds: .5},
	}
	record, err = svc.UpdateVideoTransfer("transfer-owner", record.ID, VideoTransferPatch{ExpectedRevision: record.Revision, Branches: map[string]videotransfer.BranchPatch{videotransfer.Fixed: {Settings: videotransfer.SettingsPatch{FixedRegions: &regions}}}})
	if err != nil {
		t.Fatal(err)
	}
	record, err = svc.RunVideoTransfer(context.Background(), "transfer-owner", record.ID, VideoTransferRunRequest{Branch: videotransfer.Fixed, ExpectedRevision: record.Revision})
	if err != nil {
		t.Fatal(err)
	}
	if err = svc.taskWorker().processNextTask(); err != nil {
		t.Fatal(err)
	}
	finished, err := svc.VideoTransfer("transfer-owner", record.ID)
	if err != nil {
		t.Fatal(err)
	}
	state := finished.Branches[videotransfer.Fixed]
	if state.Status != "succeeded" || state.OutputResourceID == "" || state.Summary.ChangedPixels != 0 {
		t.Fatalf("explicit inactive output failed: %+v", state)
	}
	data, err := os.ReadFile(observedPath)
	if err != nil {
		t.Fatal(err)
	}
	var observed struct {
		videotransfer.Settings
		StrategyVersion string `json:"strategyVersion"`
	}
	if err = json.Unmarshal(data, &observed); err != nil || !reflect.DeepEqual(observed.FixedRegions, regions) || observed.StrategyVersion != videoTransferStrategyVersion(videotransfer.Fixed) {
		t.Fatalf("runner request %+v: %v", observed, err)
	}
}

func TestVideoTransferOldStrategyCannotResume(t *testing.T) {
	svc, db := videoTransferTestService(t)
	source := videoTransferTestSource(t, svc, db)
	marker := filepath.Join(t.TempDir(), "called")
	t.Setenv("VIDEO_TRANSFER_UNEXPECTED_RUNNER", marker)
	videoTransferTestRunner(t, `touch "$VIDEO_TRANSFER_UNEXPECTED_RUNNER"; exit 1`)
	record, err := svc.CreateVideoTransfer("transfer-owner", VideoTransferCreateRequest{SourceResourceID: source})
	if err != nil {
		t.Fatal(err)
	}
	record, err = svc.RunVideoTransfer(context.Background(), "transfer-owner", record.ID, VideoTransferRunRequest{Branch: videotransfer.Fixed, ExpectedRevision: record.Revision})
	if err != nil {
		t.Fatal(err)
	}
	state := record.Branches[videotransfer.Fixed]
	task, err := svc.repo.TaskForUser("transfer-owner", state.TaskID)
	if err != nil {
		t.Fatal(err)
	}
	var input videoTransferTaskInput
	if err = json.Unmarshal([]byte(task.InputJSON), &input); err != nil {
		t.Fatal(err)
	}
	input.StrategyVersion = "fixed-r16-regions-v1"
	payload, _ := json.Marshal(input)
	state.Segments[0].Status, state.Segments[0].OutputResourceID = "succeeded", source
	state.Segments[0].Summary = &videotransfer.Summary{Frames: 30}
	result, _ := json.Marshal(map[string]any{"segments": state.Segments})
	if err = db.Model(&model.Task{}).Where("id = ?", task.ID).Updates(map[string]any{"input_json": string(payload), "result_json": string(result)}).Error; err != nil {
		t.Fatal(err)
	}
	if err = svc.taskWorker().processNextTask(); err != nil {
		t.Fatal(err)
	}
	finished, err := svc.VideoTransfer("transfer-owner", record.ID)
	if err != nil {
		t.Fatal(err)
	}
	got := finished.Branches[videotransfer.Fixed]
	if got.Status != "failed" || got.OutputResourceID != "" || !strings.Contains(got.Error, "处理算法已更新") || !reflect.DeepEqual(got.Segments, state.Segments) {
		t.Fatalf("obsolete task continued or discarded old output: %+v", got)
	}
	if _, err = os.Stat(marker); !os.IsNotExist(err) {
		t.Fatal("obsolete task invoked the algorithm")
	}
	if _, err = svc.repo.ResourceForUser("transfer-owner", source); err != nil {
		t.Fatalf("retained output removed: %v", err)
	}
}

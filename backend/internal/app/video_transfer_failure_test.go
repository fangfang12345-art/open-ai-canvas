package app

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"yingce/backend/internal/model"
	"yingce/backend/internal/videotransfer"
)

const videoTransferCalibrationFailure = "算法处理失败：未可靠识别位置固定的亮色水印。请检查所选角落；水印会移动时请选择移动 V6。低对比度灰色水印也可能无法标定。"

func TestVideoTransferFailureMessage(t *testing.T) {
	segments := make([]videotransfer.Segment, 145)
	for i := range segments {
		segments[i] = videotransfer.Segment{Index: i, Status: "failed", Error: videoTransferCalibrationFailure}
		if i < 16 {
			segments[i].Status, segments[i].Error, segments[i].OutputResourceID = "succeeded", "", "retained-output"
		}
	}
	before := append([]videotransfer.Segment(nil), segments...)
	message := videoTransferFailureMessage(videoTransferGenericFailure, segments)
	if strings.Count(message, videoTransferCalibrationFailure) != 1 || !strings.Contains(message, "已完成的处理结果已保留") || strings.Contains(message, "片段") {
		t.Fatalf("unexpected aggregate: %s", message)
	}
	if !reflect.DeepEqual(before, segments) {
		t.Fatal("error aggregation modified successful outputs")
	}
	for _, specific := range []string{"完整视频生成失败，已保留处理进度，请重试", "账号存储容量不足", "视频处理状态保存失败，请检查任务记录后重试"} {
		if got := videoTransferFailureMessage(specific, segments); got != specific {
			t.Fatalf("specific parent failure was hidden: %s", got)
		}
	}

	many := []videotransfer.Segment{}
	for _, prefix := range []string{"甲", "乙", "丙", "丁", "戊"} {
		many = append(many, videotransfer.Segment{Status: "failed", Error: prefix + strings.Repeat("错", 500)})
	}
	message = videoTransferFailureMessage("", many)
	if len([]rune(message)) > 700 || !strings.Contains(message, "另有其他处理错误") || strings.ContainsAny(message, "丁戊") {
		t.Fatalf("aggregate is not bounded: %s", message)
	}
}

func TestVideoTransferFailureSafety(t *testing.T) {
	for _, unsafe := range []string{
		"open /private/tmp/source.mp4: permission denied",
		`open C:\private\source.mp4: denied`,
		"request https://internal.example/token failed",
		"ValueError\n  File secret.py, line 2",
		"Traceback (most recent call last)",
		"goroutine 123 [running]:",
		"panic: internal state",
		"bad\x00value", "bad\u202evalue", "Bearer private-secret",
		strings.Repeat("x", 800) + "/private/secret",
	} {
		t.Run(unsafe[:min(20, len(unsafe))], func(t *testing.T) {
			if got := videoTransferSafeFailure(unsafe, 200); got != "" {
				t.Fatalf("unsafe diagnostic exposed: %q", got)
			}
			parts := []videotransfer.Segment{{Status: "failed", Error: unsafe}, {Status: "failed", Error: "算法处理失败：水印区域缺少有效模板"}}
			if got := videoTransferFailureMessage(videoTransferGenericFailure, parts); got != parts[1].Error {
				t.Fatalf("safe actionable reason lost: %q", got)
			}
			if got := videoTransferFailureMessage(unsafe, nil); got != videoTransferSafeFailureFallback {
				t.Fatalf("unsafe parent exposed: %q", got)
			}
		})
	}
	if got := videoTransferFailureMessage("", []videotransfer.Segment{{Status: "succeeded", Error: "stale"}, {Status: "running", Error: "stale"}}); got != videoTransferSafeFailureFallback {
		t.Fatalf("non-failed diagnostics included: %s", got)
	}
	for _, message := range []string{"ERROR: 水印区域缺少有效模板", `{"error":"ValueError","message":"水印区域缺少有效模板"}`} {
		if got := videoTransferAlgorithmError([]byte(message), "/unused"); got != "水印区域缺少有效模板" {
			t.Fatalf("runner reason lost: %s", got)
		}
	}
	if got := videoTransferAlgorithmError([]byte(`{"error":"ValueError","message":"unsafe\nTraceback: private.py"}`), "/unused"); strings.Contains(got, "Traceback") || strings.Contains(got, "private.py") {
		t.Fatalf("structured traceback exposed: %s", got)
	}
}

func TestVideoTransferFailureHydrationUsesOnlyFailedTaskResult(t *testing.T) {
	result, err := json.Marshal(map[string]any{"segments": []videotransfer.Segment{{Status: "failed", Error: videoTransferCalibrationFailure}}})
	if err != nil {
		t.Fatal(err)
	}
	for _, status := range []model.TaskStatus{model.TaskStatusRunning, model.TaskStatusCancelled, model.TaskStatusSucceeded} {
		task := &model.Task{Status: status, Error: "原始任务状态提示", ResultJSON: string(result)}
		got := videoTransferBranchFromTask(videotransfer.BranchState{}, task)
		if got.Error != task.Error || got.Status != string(status) {
			t.Fatalf("non-failed task changed: %+v", got)
		}
	}
	task := &model.Task{Status: model.TaskStatusFailed, Error: videoTransferGenericFailure, ResultJSON: "invalid json"}
	state := videotransfer.BranchState{Segments: []videotransfer.Segment{{Status: "failed", Error: "stale diagnostic"}}}
	if got := videoTransferBranchFromTask(state, task); got.Error != videoTransferGenericFailure {
		t.Fatalf("stale record diagnostics substituted for task result: %+v", got)
	}
}

func TestVideoTransferFailureReadDoesNotRewriteHistory(t *testing.T) {
	svc, db := videoTransferTestService(t)
	record, err := svc.CreateVideoTransfer("transfer-owner", VideoTransferCreateRequest{})
	if err != nil {
		t.Fatal(err)
	}
	parts := []videotransfer.Segment{
		{ID: "good", Index: 0, Status: "succeeded", OutputResourceID: "existing-output", Summary: &videotransfer.Summary{Frames: 600}},
		{ID: "bad", Index: 1, Status: "failed", Error: videoTransferCalibrationFailure},
		{ID: "unsafe", Index: 2, Status: "failed", Error: "open /private/tmp/file: denied"},
	}
	result, err := json.Marshal(map[string]any{"segments": parts, "summary": &videotransfer.Summary{Frames: 600}})
	if err != nil {
		t.Fatal(err)
	}
	task := model.Task{ID: "legacy-video-task", UserID: "transfer-owner", Type: videotransfer.TaskType, Status: model.TaskStatusFailed, Error: videoTransferGenericFailure, ResultJSON: string(result)}
	if err = db.Create(&task).Error; err != nil {
		t.Fatal(err)
	}
	state := record.Branches[videotransfer.Fixed]
	state.TaskID, state.Status, state.Error = task.ID, "failed", videoTransferGenericFailure
	record.Branches[videotransfer.Fixed] = state
	row, err := svc.repo.VideoTransferForUser("transfer-owner", record.ID)
	if err != nil {
		t.Fatal(err)
	}
	if err = encodeVideoTransfer(row, record); err != nil {
		t.Fatal(err)
	}
	if err = db.Save(row).Error; err != nil {
		t.Fatal(err)
	}
	before, err := svc.repo.VideoTransferForUser("transfer-owner", record.ID)
	if err != nil {
		t.Fatal(err)
	}
	beforeTask, err := svc.repo.TaskForUser("transfer-owner", task.ID)
	if err != nil {
		t.Fatal(err)
	}
	for _, readList := range []bool{false, true} {
		var hydrated *videotransfer.Record
		if readList {
			page, readErr := svc.VideoTransfers("transfer-owner", 1, 10, "")
			if readErr != nil || len(page.Records) != 1 {
				t.Fatalf("list: %+v %v", page, readErr)
			}
			hydrated = &page.Records[0]
		} else {
			hydrated, err = svc.VideoTransfer("transfer-owner", record.ID)
			if err != nil {
				t.Fatal(err)
			}
		}
		got := hydrated.Branches[videotransfer.Fixed]
		if !strings.Contains(got.Error, videoTransferCalibrationFailure) || strings.Contains(got.Error, "/private") || got.Status != "failed" || got.OutputResourceID != "" || got.Summary.Frames != 600 {
			t.Fatalf("historical error hydration: %+v", got)
		}
		if !reflect.DeepEqual(got.Segments[0], parts[0]) || strings.Contains(got.Segments[2].Error, "/private") {
			t.Fatalf("successful output changed or unsafe error exposed: %+v", got.Segments)
		}
	}
	after, err := svc.repo.VideoTransferForUser("transfer-owner", record.ID)
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatalf("read changed record: %v", err)
	}
	afterTask, err := svc.repo.TaskForUser("transfer-owner", task.ID)
	if err != nil || !reflect.DeepEqual(beforeTask, afterTask) {
		t.Fatalf("read changed task: %v", err)
	}
}

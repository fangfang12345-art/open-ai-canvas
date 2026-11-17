package app

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"time"

	"gorm.io/gorm"
	"yingce/backend/internal/model"
	"yingce/backend/internal/repository"
	"yingce/backend/internal/videotransfer"
)

type VideoTransferCreateRequest = videotransfer.CreateRequest
type VideoTransferPatch = videotransfer.Patch
type VideoTransferRunRequest = videotransfer.RunRequest
type VideoTransferRecord = videotransfer.Record
type VideoTransferCapabilities = videotransfer.Capabilities

type VideoTransferPage struct {
	Records  []videotransfer.Record `json:"records"`
	Total    int64                  `json:"total"`
	Page     int                    `json:"page"`
	PageSize int                    `json:"pageSize"`
	HasMore  bool                   `json:"hasMore"`
}

type videoTransferTaskInput struct {
	TransferID       string                 `json:"transferId"`
	SourceResourceID string                 `json:"sourceResourceId"`
	Branch           string                 `json:"branch"`
	Settings         videotransfer.Settings `json:"settings"`
	SegmentCount     int                    `json:"segmentCount"`
	StrategyVersion  string                 `json:"strategyVersion"`
}

func videoTransferStrategyVersion(branch string) string {
	if branch == videotransfer.Fixed {
		return "fixed-r16-regions-time-v2"
	}
	return "moving-v6-candidate5-time-v2"
}

func defaultVideoTransferBranches() map[string]videotransfer.BranchState {
	branches := videotransfer.DefaultBranches()
	for key, state := range branches {
		state.Settings.Workers = min(state.Settings.Workers, videoTransferLimits().MaxWorkers)
		branches[key] = state
	}
	return branches
}

func videoTransferError(err error) error {
	switch {
	case errors.Is(err, gorm.ErrRecordNotFound):
		return NewAppError(404, "视频搬运记录或视频资源不存在")
	case errors.Is(err, repository.ErrVideoTransferConflict):
		return NewAppError(409, "视频搬运记录已更新，请载入服务器版本后重试")
	case errors.Is(err, repository.ErrActiveTaskLimit):
		return NewAppError(429, "进行中的任务已达上限，请等待完成后重试")
	default:
		return err
	}
}

func decodeVideoTransfer(row *model.VideoTransfer) (*videotransfer.Record, error) {
	var record videotransfer.Record
	if err := json.Unmarshal([]byte(row.PayloadJSON), &record); err != nil {
		return nil, err
	}
	record.ID, record.Title, record.SourceResourceID = row.ID, row.Title, row.SourceResourceID
	record.Revision, record.CreatedAt, record.UpdatedAt = row.Revision, row.CreatedAt, row.UpdatedAt
	return &record, nil
}

func encodeVideoTransfer(row *model.VideoTransfer, record *videotransfer.Record) error {
	row.Title, row.SourceResourceID = record.Title, record.SourceResourceID
	payload, err := json.Marshal(record)
	row.PayloadJSON = string(payload)
	return err
}

func (s *Service) VideoTransfers(userID string, page, pageSize int, query string) (*VideoTransferPage, error) {
	if strings.TrimSpace(userID) == "" {
		return nil, Unauthorized("请先登录")
	}
	page, pageSize = max(page, 1), min(max(pageSize, 1), 100)
	if len([]rune(query)) > 240 {
		return nil, BadAuthRequest("搜索关键词不能超过 240 字")
	}
	rows, total, err := s.repo.VideoTransfers(userID, page, pageSize, query)
	if err != nil {
		return nil, err
	}
	result := &VideoTransferPage{Records: make([]videotransfer.Record, 0, len(rows)), Total: total, Page: page, PageSize: pageSize, HasMore: int64(page)*int64(pageSize) < total}
	for _, row := range rows {
		record, err := decodeVideoTransfer(&row)
		if err != nil {
			return nil, err
		}
		if err := s.hydrateVideoTransfer(userID, record); err != nil {
			return nil, err
		}
		result.Records = append(result.Records, *record)
	}
	return result, nil
}

func (s *Service) VideoTransfer(userID, id string) (*videotransfer.Record, error) {
	if strings.TrimSpace(userID) == "" {
		return nil, Unauthorized("请先登录")
	}
	row, err := s.repo.VideoTransferForUser(userID, id)
	if err != nil {
		return nil, videoTransferError(err)
	}
	record, err := decodeVideoTransfer(row)
	if err != nil {
		return nil, err
	}
	if err = s.hydrateVideoTransfer(userID, record); err != nil {
		return nil, err
	}
	return record, nil
}

func (s *Service) CreateVideoTransfer(userID string, req VideoTransferCreateRequest) (*videotransfer.Record, error) {
	if strings.TrimSpace(userID) == "" {
		return nil, Unauthorized("请先登录")
	}
	title := strings.TrimSpace(req.Title)
	if title == "" {
		title = "新视频搬运"
	}
	if len([]rune(title)) > 240 || len([]rune(req.SourceName)) > 240 {
		return nil, BadAuthRequest("名称不能超过 240 字")
	}
	branch := req.SelectedBranch
	if branch == "" {
		branch = videotransfer.Fixed
	}
	if !videotransfer.ValidBranch(branch) {
		return nil, BadAuthRequest("请选择有效的处理分支")
	}
	_, total, err := s.repo.VideoTransfers(userID, 1, 1, "")
	if err != nil {
		return nil, err
	}
	if total >= 1000 {
		return nil, QuotaExceeded("视频搬运记录已达到 1000 条上限")
	}
	now := time.Now()
	record := &videotransfer.Record{ID: newID(), Title: title, SelectedBranch: branch, Branches: defaultVideoTransferBranches(), Revision: 1, CreatedAt: now, UpdatedAt: now}
	if req.SourceResourceID != "" {
		if err = s.setVideoTransferSource(userID, record, req.SourceResourceID, req.SourceName); err != nil {
			return nil, err
		}
	}
	row := &model.VideoTransfer{ID: record.ID, UserID: userID, Revision: 1, CreatedAt: now, UpdatedAt: now}
	if err = encodeVideoTransfer(row, record); err != nil {
		return nil, err
	}
	if err = s.repo.CreateVideoTransfer(row); err != nil {
		return nil, err
	}
	return record, nil
}

func (s *Service) setVideoTransferSource(userID string, record *videotransfer.Record, id, name string) error {
	id = strings.TrimSpace(id)
	if id == "" {
		record.SourceResourceID = ""
		record.SourceName = ""
		record.SourceWidth = 0
		record.SourceHeight = 0
		record.SourceDuration = 0
		record.SourceFrameRate, record.SourceFrames = 0, 0
		record.Branches = defaultVideoTransferBranches()
		return nil
	}
	resource, err := s.repo.ResourceForUser(userID, id)
	if err != nil {
		return videoTransferError(err)
	}
	if resource.Status != model.ResourceStatusReady || !strings.HasPrefix(resource.MimeType, "video/") {
		return BadAuthRequest("请选择已完成上传的视频资源")
	}
	if resource.Size <= 0 || (videoTransferLimits().MaxSourceBytes > 0 && resource.Size > videoTransferLimits().MaxSourceBytes) {
		return BadAuthRequest("源视频超过服务器处理大小上限")
	}
	record.SourceResourceID = id
	record.SourceName = strings.TrimSpace(name)
	if record.SourceName == "" {
		record.SourceName = "视频素材"
	}
	record.SourceWidth, record.SourceHeight = resource.Width, resource.Height
	record.SourceDuration = float64(resource.DurationMs) / 1000
	record.SourceFrameRate, record.SourceFrames = 0, 0
	if info, probeErr := s.probeStoredResourceMedia(context.Background(), resource); probeErr == nil {
		record.SourceFrameRate, record.SourceFrames = info.FrameRate, info.Frames
		if info.VideoDurationSeconds > 0 {
			record.SourceDuration = info.VideoDurationSeconds
		}
	}
	record.Branches = defaultVideoTransferBranches()
	for branch, state := range record.Branches {
		state.Segments = videotransfer.SegmentsForTimeline(state.Settings, record.SourceDuration, record.SourceFrameRate)
		record.Branches[branch] = state
	}
	return nil
}

func (s *Service) UpdateVideoTransfer(userID, id string, req VideoTransferPatch) (*videotransfer.Record, error) {
	record, err := s.VideoTransfer(userID, id)
	if err != nil {
		return nil, err
	}
	if req.ExpectedRevision <= 0 || record.Revision != req.ExpectedRevision {
		return nil, videoTransferError(repository.ErrVideoTransferConflict)
	}
	if req.Title != nil {
		title := strings.TrimSpace(*req.Title)
		if title == "" || len([]rune(title)) > 240 {
			return nil, BadAuthRequest("名称必须为 1 到 240 字")
		}
		record.Title = title
	}
	if req.SourceName != nil && len([]rune(*req.SourceName)) > 240 {
		return nil, BadAuthRequest("名称不能超过 240 字")
	}
	if req.SelectedBranch != nil {
		if !videotransfer.ValidBranch(*req.SelectedBranch) {
			return nil, BadAuthRequest("请选择有效的处理分支")
		}
		record.SelectedBranch = *req.SelectedBranch
	}
	if req.SourceResourceID != nil && *req.SourceResourceID != record.SourceResourceID {
		for _, state := range record.Branches {
			if videotransfer.Active(state.Status) {
				return nil, NewAppError(409, "处理期间不能更换源视频")
			}
		}
		name := ""
		if req.SourceName != nil {
			name = *req.SourceName
		}
		if err = s.setVideoTransferSource(userID, record, *req.SourceResourceID, name); err != nil {
			return nil, err
		}
	} else if req.SourceName != nil {
		record.SourceName = strings.TrimSpace(*req.SourceName)
	}
	for branch, patch := range req.Branches {
		if !videotransfer.ValidBranch(branch) {
			return nil, BadAuthRequest("未知处理分支")
		}
		state := record.Branches[branch]
		settings := state.Settings.Merge(patch.Settings)
		if err = videotransfer.ValidateSettings(branch, settings); err != nil {
			return nil, BadAuthRequest(err.Error())
		}
		if !reflect.DeepEqual(settings, state.Settings) {
			if videotransfer.Active(state.Status) {
				return nil, NewAppError(409, "处理期间不能修改该分支参数")
			}
			state = videotransfer.BranchState{Settings: settings, Status: "idle", Segments: videotransfer.SegmentsForTimeline(settings, record.SourceDuration, record.SourceFrameRate)}
		}
		record.Branches[branch] = state
	}
	row := &model.VideoTransfer{ID: id, UserID: userID, Revision: record.Revision + 1, CreatedAt: record.CreatedAt, UpdatedAt: time.Now()}
	record.Revision, record.UpdatedAt = row.Revision, row.UpdatedAt
	if err = encodeVideoTransfer(row, record); err != nil {
		return nil, err
	}
	if err = s.repo.UpdateVideoTransfer(row, req.ExpectedRevision); err != nil {
		return nil, videoTransferError(err)
	}
	return record, nil
}

func (s *Service) RunVideoTransfer(ctx context.Context, userID, id string, req VideoTransferRunRequest) (*videotransfer.Record, error) {
	if s.IsDraining() {
		return nil, NewAppError(503, "服务正在维护，暂不接受新的处理任务")
	}
	record, err := s.VideoTransfer(userID, id)
	if err != nil {
		return nil, err
	}
	if req.ExpectedRevision <= 0 || record.Revision != req.ExpectedRevision {
		return nil, videoTransferError(repository.ErrVideoTransferConflict)
	}
	if !videotransfer.ValidBranch(req.Branch) {
		return nil, BadAuthRequest("请选择有效的处理分支")
	}
	state := record.Branches[req.Branch]
	if videotransfer.Active(state.Status) {
		return nil, NewAppError(409, "该分支已在处理，请等待完成或取消")
	}
	resource, err := s.repo.ResourceForUser(userID, record.SourceResourceID)
	if err != nil {
		return nil, videoTransferError(err)
	}
	if resource.Status != model.ResourceStatusReady || !strings.HasPrefix(resource.MimeType, "video/") || resource.Size <= 0 || (videoTransferLimits().MaxSourceBytes > 0 && resource.Size > videoTransferLimits().MaxSourceBytes) {
		return nil, BadAuthRequest("源视频不可用或超过服务器大小上限")
	}
	if err = videotransfer.ValidateRun(req.Branch, state.Settings, resource.Width, resource.Height, float64(resource.DurationMs)/1000, videoTransferLimits()); err != nil {
		return nil, BadAuthRequest(err.Error())
	}
	if record.SourceFrameRate > 0 {
		if err = videotransfer.ValidateFixedRegionFrames(state.Settings.FixedRegions, record.SourceDuration, record.SourceFrameRate); err != nil {
			return nil, BadAuthRequest(err.Error())
		}
	}
	caps := s.GetVideoTransferCapabilities(ctx)
	for _, capability := range caps.Branches {
		if capability.ID == req.Branch && !capability.Available {
			return nil, NewAppError(503, capability.Reason)
		}
	}
	policy, err := s.RuntimePolicy()
	if err != nil {
		return nil, err
	}
	if err = s.requireStoredFileCapacityForTask(userID, videotransfer.TaskType, policy); err != nil {
		return nil, err
	}
	segments := videotransfer.SegmentsForTimeline(state.Settings, record.SourceDuration, record.SourceFrameRate)
	if len(segments) == 0 {
		return nil, BadAuthRequest("处理范围不足一帧")
	}
	input := videoTransferTaskInput{TransferID: id, SourceResourceID: record.SourceResourceID, Branch: req.Branch, Settings: state.Settings, SegmentCount: len(segments), StrategyVersion: videoTransferStrategyVersion(req.Branch)}
	payload, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}
	task := &model.Task{ID: newID(), UserID: userID, Type: videotransfer.TaskType, Status: model.TaskStatusQueued, Stage: "等待队列调度", Progress: 1, Prompt: record.Title, Provider: "local", Model: req.Branch, InputJSON: string(payload)}
	for i := range segments {
		segments[i].Status = "queued"
	}
	if (state.Status == "failed" || state.Status == "cancelled") && state.TaskID != "" {
		previous, readErr := s.repo.TaskForUser(userID, state.TaskID)
		if readErr != nil && !errors.Is(readErr, gorm.ErrRecordNotFound) {
			return nil, readErr
		}
		if readErr == nil && previous.Type == videotransfer.TaskType {
			var oldInput videoTransferTaskInput
			var oldResult struct {
				Segments []videotransfer.Segment `json:"segments"`
			}
			if json.Unmarshal([]byte(previous.InputJSON), &oldInput) == nil && oldInput.StrategyVersion == input.StrategyVersion && oldInput.TransferID == id && oldInput.SourceResourceID == input.SourceResourceID && oldInput.Branch == input.Branch && reflect.DeepEqual(oldInput.Settings, input.Settings) && json.Unmarshal([]byte(previous.ResultJSON), &oldResult) == nil {
				s.restoreVideoTransferSegments(userID, segments, oldResult.Segments)
			}
		}
	}
	initialResult, err := json.Marshal(map[string]any{"segments": segments})
	if err != nil {
		return nil, err
	}
	task.ResultJSON = string(initialResult)
	record.Branches[req.Branch] = videotransfer.BranchState{Settings: state.Settings, Status: "queued", Progress: 1, Stage: task.Stage, TaskID: task.ID, Segments: segments}
	record.SelectedBranch = req.Branch
	row := &model.VideoTransfer{ID: id, UserID: userID, Revision: record.Revision + 1, CreatedAt: record.CreatedAt, UpdatedAt: time.Now()}
	record.Revision, record.UpdatedAt = row.Revision, row.UpdatedAt
	if err = encodeVideoTransfer(row, record); err != nil {
		return nil, err
	}
	s.storageMu.Lock()
	err = s.repo.Transaction(func(repo *repository.Repository) error {
		if err := repo.UpdateVideoTransfer(row, req.ExpectedRevision); err != nil {
			return err
		}
		return createTaskWithStorageQuotaRepository(repo, task, nil, policy)
	})
	s.storageMu.Unlock()
	if err != nil {
		return nil, videoTransferError(err)
	}
	s.wakeTaskDispatcher()
	_ = s.log(userID, task.ID, "info", "视频搬运任务已进入队列", req.Branch)
	return record, nil
}

func (s *Service) CancelVideoTransfer(ctx context.Context, userID, id string) (*videotransfer.Record, error) {
	record, err := s.VideoTransfer(userID, id)
	if err != nil {
		return nil, err
	}
	for _, branch := range []string{videotransfer.Fixed, videotransfer.Moving} {
		state := record.Branches[branch]
		if !videotransfer.Active(state.Status) || state.TaskID == "" {
			continue
		}
		if _, err = s.CancelTask(ctx, userID, state.TaskID); err != nil {
			latest, readErr := s.repo.TaskForUser(userID, state.TaskID)
			if readErr != nil || latest.Status == model.TaskStatusQueued || latest.Status == model.TaskStatusRunning {
				return nil, err
			}
		}
		if err = s.syncVideoTransferTask(userID, state.TaskID); err != nil {
			return nil, err
		}
	}
	return s.VideoTransfer(userID, id)
}

func videoTransferBranchFromTask(state videotransfer.BranchState, task *model.Task) videotransfer.BranchState {
	state.Status, state.Progress, state.Stage, state.Error = string(task.Status), task.Progress, task.Stage, task.Error
	if state.Error != "" {
		state.Error = videoTransferFailureMessage(state.Error, nil)
	}
	if task.StartedAt != nil {
		end := time.Now()
		if task.CompletedAt != nil {
			end = *task.CompletedAt
		}
		state.ElapsedSeconds = max(0, end.Sub(*task.StartedAt).Seconds())
	}
	if task.ResultJSON != "" {
		var out struct {
			ResourceID string                  `json:"resourceId"`
			Summary    *videotransfer.Summary  `json:"summary"`
			Segments   []videotransfer.Segment `json:"segments"`
		}
		if json.Unmarshal([]byte(task.ResultJSON), &out) == nil {
			state.OutputResourceID = out.ResourceID
			state.Summary = out.Summary
			if out.Segments != nil {
				state.Segments = out.Segments
				if task.Status == model.TaskStatusFailed {
					// Derive old generic failures from the task snapshot without rewriting it.
					state.Error = videoTransferFailureMessage(task.Error, out.Segments)
				}
			}
		}
	}
	for i := range state.Segments {
		if state.Segments[i].Error != "" {
			message := videoTransferSafeFailure(state.Segments[i].Error, 300)
			if message == "" {
				message = videoTransferSafeFailureFallback
			}
			state.Segments[i].Error = message
		}
	}
	if !videotransfer.Active(state.Status) && state.Status != "idle" {
		for i := range state.Segments {
			if videotransfer.Active(state.Segments[i].Status) {
				state.Segments[i].Status = state.Status
				state.Segments[i].Error = state.Error
			}
		}
	}
	return state
}

func (s *Service) hydrateVideoTransfer(userID string, record *videotransfer.Record) error {
	for branch, state := range record.Branches {
		if state.TaskID == "" {
			continue
		}
		task, err := s.repo.TaskForUser(userID, state.TaskID)
		if errors.Is(err, gorm.ErrRecordNotFound) {
			if videotransfer.Active(state.Status) {
				state.Status = "failed"
				state.Error = "任务记录已不存在，请重新运行"
				record.Branches[branch] = state
			}
			continue
		}
		if err != nil {
			return err
		}
		if task.Type != videotransfer.TaskType {
			return fmt.Errorf("invalid video transfer task type")
		}
		record.Branches[branch] = videoTransferBranchFromTask(state, task)
	}
	return nil
}

// 配置版本不随任务进度变化。使用整个旧快照作 CAS，避免终态回写覆盖并发配置。
func (s *Service) syncVideoTransferTask(userID, taskID string) error {
	task, err := s.repo.TaskForUser(userID, taskID)
	if err != nil {
		return err
	}
	var input videoTransferTaskInput
	if err = json.Unmarshal([]byte(task.InputJSON), &input); err != nil {
		return err
	}
	for attempt := 0; attempt < 3; attempt++ {
		row, err := s.repo.VideoTransferForUser(userID, input.TransferID)
		if err != nil {
			return err
		}
		previous := row.PayloadJSON
		record, err := decodeVideoTransfer(row)
		if err != nil {
			return err
		}
		state := record.Branches[input.Branch]
		if state.TaskID != taskID {
			return nil
		}
		record.Branches[input.Branch] = videoTransferBranchFromTask(state, task)
		if err = encodeVideoTransfer(row, record); err != nil {
			return err
		}
		changed, err := s.repo.RefreshVideoTransferSnapshot(row, previous)
		if err != nil {
			return err
		}
		if changed {
			return nil
		}
	}
	return repository.ErrVideoTransferConflict
}

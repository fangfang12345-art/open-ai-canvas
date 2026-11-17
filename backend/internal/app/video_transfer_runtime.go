package app

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"yingce/backend/internal/model"
	"yingce/backend/internal/repository"
	"yingce/backend/internal/videotransfer"
)

// 每个后端进程同时执行一批任务，批内分段受独立并发预算约束。
var videoTransferExecutionSlots = make(chan struct{}, 1)

func videoTransferLimits() videotransfer.Limits {
	limits := videotransfer.DefaultLimits()
	positive := func(key string, fallback int64) int64 {
		value, err := strconv.ParseInt(strings.TrimSpace(os.Getenv(key)), 10, 64)
		if err != nil || value <= 0 {
			return fallback
		}
		return value
	}
	limits.MaxSourceBytes = min(positive("CANVAS_WATERMARK_MAX_SOURCE_MB", limits.MaxSourceBytes>>20), 32768) << 20
	limits.MaxSourceDurationSeconds = float64(min(positive("CANVAS_WATERMARK_MAX_SOURCE_SECONDS", int64(limits.MaxSourceDurationSeconds)), 86400))
	limits.MaxDurationSeconds = float64(min(positive("CANVAS_WATERMARK_MAX_PROCESS_SECONDS", int64(limits.MaxDurationSeconds)), 86400))
	limits.MaxPixels = int(min(positive("CANVAS_WATERMARK_MAX_PIXELS", int64(limits.MaxPixels)), 16384*16384))
	limits.MaxWorkers = int(min(positive("CANVAS_WATERMARK_MAX_WORKERS", int64(limits.MaxWorkers)), 16))
	limits.MaxConcurrentSegments = int(min(positive("CANVAS_WATERMARK_CONCURRENT_SEGMENTS", 2), 4))
	return limits
}

func videoTransferRuntime() (string, string, error) {
	if runtime.GOOS == "windows" {
		return "", "", errors.New("视频处理执行器目前支持 Linux 和 macOS")
	}
	python := strings.TrimSpace(os.Getenv("CANVAS_WATERMARK_PYTHON"))
	if python == "" {
		python = "python3"
	}
	bin, err := exec.LookPath(python)
	if err != nil {
		return "", "", errors.New("视频处理环境缺少 Python 3.10 或更新版本")
	}
	runner := strings.TrimSpace(os.Getenv("CANVAS_WATERMARK_RUNNER"))
	if runner == "" {
		for _, candidate := range []string{"tools/watermark/runner.py", "backend/tools/watermark/runner.py", "/app/tools/watermark/runner.py"} {
			if info, err := os.Stat(candidate); err == nil && info.Mode().IsRegular() {
				runner = candidate
				break
			}
		}
	}
	if runner == "" {
		return "", "", errors.New("视频处理算法未安装，请配置 CANVAS_WATERMARK_RUNNER")
	}
	runner, err = filepath.Abs(runner)
	if err != nil {
		return "", "", errors.New("视频处理算法路径配置无效")
	}
	if info, err := os.Stat(runner); err != nil || !info.Mode().IsRegular() {
		return "", "", errors.New("视频处理算法入口不可用")
	}
	return bin, runner, nil
}

func (s *Service) GetVideoTransferCapabilities(ctx context.Context) videotransfer.Capabilities {
	result := videotransfer.Capabilities{Branches: []videotransfer.Capability{{ID: videotransfer.Fixed}, {ID: videotransfer.Moving}}, Limits: videoTransferLimits()}
	fail := func(reason string) videotransfer.Capabilities {
		for i := range result.Branches {
			result.Branches[i].Reason = reason
		}
		return result
	}
	python, runner, err := videoTransferRuntime()
	if err != nil {
		return fail(err.Error())
	}
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, python, runner, "--check")
	configureVideoTransferProcess(cmd)
	output := &videoTransferBoundedBuffer{limit: 64 << 10}
	cmd.Stdout = output
	cmd.Stderr = io.Discard
	if err = cmd.Run(); err != nil {
		return fail("视频处理环境检查失败，请检查 Python、NumPy、OpenCV 和 FFmpeg")
	}
	var check struct {
		Branches map[string]struct {
			Available bool   `json:"available"`
			Reason    string `json:"reason"`
		} `json:"branches"`
	}
	if json.Unmarshal(output.Bytes(), &check) != nil {
		return fail("视频处理算法返回了无效的环境检查结果")
	}
	for i := range result.Branches {
		branch, ok := check.Branches[result.Branches[i].ID]
		result.Branches[i].Available = ok && branch.Available
		if !result.Branches[i].Available {
			result.Branches[i].Reason = truncateRunes(branch.Reason, 300)
			if result.Branches[i].Reason == "" {
				result.Branches[i].Reason = "该处理分支暂不可用"
			}
		}
	}
	return result
}

type videoTransferBoundedBuffer struct {
	mu    sync.Mutex
	data  []byte
	limit int
}

func (b *videoTransferBoundedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	n := len(p)
	if len(p) >= b.limit {
		b.data = append(b.data[:0], p[len(p)-b.limit:]...)
	} else {
		extra := len(b.data) + len(p) - b.limit
		if extra > 0 {
			b.data = b.data[extra:]
		}
		b.data = append(b.data, p...)
	}
	return n, nil
}
func (b *videoTransferBoundedBuffer) Bytes() []byte {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]byte(nil), b.data...)
}

func (w *taskWorkerCoordinator) processVideoTransfer(task *model.Task, ctx context.Context) (returnErr error) {
	s := w.service
	defer func() {
		if returnErr != nil {
			if latest, err := s.repo.TaskForUser(task.UserID, task.ID); err == nil && latest.Status == model.TaskStatusCancelled {
				returnErr = nil
			}
		}
		if returnErr != nil {
			_, _ = s.repo.UpdateTaskTerminalState(task.ID, task.LeaseOwner, model.TaskStatusRunning, model.TaskStatusFailed, "处理失败", "视频处理状态保存失败，请检查任务记录后重试", time.Now())
		}
		if err := s.syncVideoTransferTask(task.UserID, task.ID); err != nil {
			_ = s.log(task.UserID, task.ID, "error", "同步视频搬运状态失败", "")
		}
	}()
	fail := func(stage, message string) error {
		message = videoTransferFailureMessage(message, nil)
		latest, err := s.repo.TaskForUser(task.UserID, task.ID)
		if err == nil && latest.Status == model.TaskStatusCancelled {
			return nil
		}
		done, err := s.repo.UpdateTaskTerminalState(task.ID, task.LeaseOwner, model.TaskStatusRunning, model.TaskStatusFailed, stage, message, time.Now())
		if err != nil {
			return err
		}
		if !done {
			return repository.ErrTaskStateConflict
		}
		return nil
	}
	select {
	case videoTransferExecutionSlots <- struct{}{}:
		defer func() { <-videoTransferExecutionSlots }()
	case <-ctx.Done():
		return fail("处理已中止", "处理超时或已取消")
	}
	python, runner, err := videoTransferRuntime()
	if err != nil {
		return fail("处理环境不可用", err.Error())
	}
	var input videoTransferTaskInput
	if err = json.Unmarshal([]byte(task.InputJSON), &input); err != nil || !videotransfer.ValidBranch(input.Branch) {
		return fail("处理输入无效", "任务缺少有效的处理参数")
	}
	if input.StrategyVersion != videoTransferStrategyVersion(input.Branch) {
		return fail("处理策略已更新", "处理算法已更新，请重新开始处理；旧策略结果不会被复用")
	}
	resource, body, err := s.OpenResource(task.UserID, input.SourceResourceID)
	if err != nil {
		return fail("读取源视频失败", "源视频不存在或无法读取")
	}
	defer body.Close()
	if resource.Status != model.ResourceStatusReady || !strings.HasPrefix(resource.MimeType, "video/") || resource.Size <= 0 || (videoTransferLimits().MaxSourceBytes > 0 && resource.Size > videoTransferLimits().MaxSourceBytes) {
		return fail("源视频无效", "源视频不可用或超过大小上限")
	}
	root := filepath.Join(s.dataDir, "tmp", "video-transfers")
	if err = os.MkdirAll(root, 0700); err != nil {
		return fail("创建工作目录失败", "服务器临时存储不可用")
	}
	dir, err := os.MkdirTemp(root, "task-*")
	if err != nil {
		return fail("创建工作目录失败", "服务器临时存储不可用")
	}
	defer os.RemoveAll(dir)
	copyDone := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			_ = body.Close()
		case <-copyDone:
		}
	}()
	defer close(copyDone)
	if err = w.progress(task, "准备源视频", 3); err != nil {
		return err
	}
	source := filepath.Join(dir, "source"+videoTransferExtension(resource.MimeType))
	file, err := os.OpenFile(source, os.O_CREATE|os.O_WRONLY|os.O_EXCL, 0600)
	if err != nil {
		return fail("准备源视频失败", "服务器临时存储不可用")
	}
	written, copyErr := io.Copy(file, io.LimitReader(body, resource.Size))
	closeErr := file.Close()
	if copyErr != nil || closeErr != nil || written != resource.Size || ctx.Err() != nil {
		return fail("准备源视频失败", "源视频传输中断或大小不一致")
	}
	info, err := probeMediaFileDuration(ctx, source, true)
	if err != nil {
		return fail("视频检测失败", "无法读取真实视频尺寸和时长，请检查文件和 FFprobe")
	}
	duration := float64(info.DurationMs) / 1000
	if info.VideoDurationSeconds > 0 {
		duration = info.VideoDurationSeconds
	}
	if err = videotransfer.ValidateRun(input.Branch, input.Settings, info.Width, info.Height, duration, videoTransferLimits()); err != nil {
		return fail("处理范围无效", err.Error())
	}
	if err = videotransfer.ValidateFixedRegionFrames(input.Settings.FixedRegions, duration, info.FrameRate); err != nil {
		return fail("处理范围无效", err.Error())
	}
	segments := videotransfer.SegmentsForTimeline(input.Settings, duration, info.FrameRate)
	var previous struct {
		Segments []videotransfer.Segment `json:"segments"`
	}
	_ = json.Unmarshal([]byte(task.ResultJSON), &previous)
	s.restoreVideoTransferSegments(task.UserID, segments, previous.Segments)
	concurrency := min(videoTransferLimits().MaxConcurrentSegments, input.Settings.Workers, len(segments))
	if concurrency < 1 {
		return fail("处理范围无效", "没有可处理的片段")
	}
	var mu sync.Mutex
	persist := func() error {
		payload, marshalErr := json.Marshal(map[string]any{"segments": segments})
		if marshalErr != nil {
			return marshalErr
		}
		return s.repo.UpdateVideoTransferTaskProgress(task.ID, task.LeaseOwner, "正在处理视频", videoTransferProcessingProgress(segments), string(payload))
	}
	if err = persist(); err != nil {
		return err
	}
	jobs := make(chan int)
	var wg sync.WaitGroup
	var persistErr error
	segmentCtx, cancelSegments := context.WithCancel(ctx)
	defer cancelSegments()
	for worker := 0; worker < concurrency; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for index := range jobs {
				if segments[index].Status == "succeeded" {
					continue
				}
				if segmentCtx.Err() != nil {
					continue
				}
				started := time.Now()
				segmentDir := filepath.Join(dir, segments[index].ID)
				segmentInput := input
				segmentInput.Settings.StartSeconds = segments[index].StartSeconds
				segmentInput.Settings.DurationSeconds = segments[index].DurationSeconds
				segmentInput.Settings.Workers = max(1, input.Settings.Workers/concurrency)
				update := func(stage string, progress int) error {
					mu.Lock()
					defer mu.Unlock()
					segments[index].Status, segments[index].Stage, segments[index].Progress = "running", stage, progress
					if updateErr := persist(); updateErr != nil {
						persistErr = updateErr
						cancelSegments()
						return updateErr
					}
					return nil
				}
				segmentErr := os.MkdirAll(segmentDir, 0700)
				var result *model.Resource
				var summary *videotransfer.Summary
				if segmentErr == nil {
					segmentErr = update("开始处理片段", 5)
				}
				if segmentErr == nil {
					partCtx, cancelPart := context.WithTimeout(segmentCtx, 2*time.Hour)
					result, summary, segmentErr = s.executeVideoTransferSegment(partCtx, task, segmentInput, source, segmentDir, python, runner, info, update)
					cancelPart()
				}
				mu.Lock()
				segment := &segments[index]
				segment.ElapsedSeconds = time.Since(started).Seconds()
				if segmentErr != nil {
					message := videoTransferSafeFailure(segmentErr.Error(), 300)
					if message == "" {
						message = videoTransferSafeFailureFallback
					}
					segment.Status, segment.Stage, segment.Error = "failed", "片段处理失败", message
					if segmentCtx.Err() != nil {
						segment.Status = "cancelled"
					}
				} else {
					segment.Status, segment.Progress, segment.Stage = "succeeded", 100, "处理完成"
					segment.OutputResourceID, segment.Summary = result.ID, summary
				}
				if updateErr := persist(); updateErr != nil {
					persistErr = updateErr
					cancelSegments()
					if result != nil {
						s.discardUnpublishedVideoTransferOutput(task.UserID, result)
					}
				}
				mu.Unlock()
			}
		}()
	}
	for index := range segments {
		select {
		case jobs <- index:
		case <-segmentCtx.Done():
		}
	}
	close(jobs)
	wg.Wait()
	if persistErr != nil {
		if latest, readErr := s.repo.TaskForUser(task.UserID, task.ID); readErr == nil && latest.Status == model.TaskStatusCancelled {
			return nil
		}
		return persistErr
	}
	if ctx.Err() != nil {
		return fail("处理已中止", "处理超时或已取消")
	}
	summary := &videotransfer.Summary{}
	failures := 0
	for _, segment := range segments {
		if segment.Status != "succeeded" {
			failures++
			continue
		}
		part := segment.Summary
		summary.Frames += part.Frames
		summary.TrackedFrames += part.TrackedFrames
		summary.UntrackedFrames += part.UntrackedFrames
		summary.ChangedPixels += part.ChangedPixels
		summary.TemporalRecoveredPixels += part.TemporalRecoveredPixels
		summary.InversePixels += part.InversePixels
		summary.SpatialFallbackPixels += part.SpatialFallbackPixels
		summary.CalibratedInversePixels += part.CalibratedInversePixels
		if len(summary.Limitations) == 0 {
			summary.Limitations = part.Limitations
		}
	}
	if summary.Frames > 0 {
		summary.TrackingCoverage = float64(summary.TrackedFrames) / float64(summary.Frames)
	}
	payload, err := json.Marshal(map[string]any{"branch": input.Branch, "summary": summary, "segments": segments})
	if err != nil {
		return err
	}
	task.ResultJSON = string(payload)
	if failures > 0 {
		if err := s.repo.UpdateVideoTransferTaskProgress(task.ID, task.LeaseOwner, "处理失败", videoTransferProcessingProgress(segments), task.ResultJSON); err != nil {
			return err
		}
		return fail("处理失败", videoTransferFailureMessage(videoTransferGenericFailure, segments))
	}
	outputID := segments[0].OutputResourceID
	var merged *model.Resource
	if len(segments) > 1 {
		update := func(stage string, progress int) error {
			return s.repo.UpdateVideoTransferTaskProgress(task.ID, task.LeaseOwner, stage, progress, task.ResultJSON)
		}
		merged, err = s.mergeVideoTransferOutputs(ctx, task, input, source, dir, info, segments, update)
		if err != nil {
			return fail("生成完整视频失败", err.Error())
		}
		outputID = merged.ID
	}
	// The merged resource is private until the lease-fenced terminal commit succeeds.
	// Cancellation or a replacement worker must not leave an unreferenced output.
	if merged != nil {
		defer func() {
			if returnErr != nil || ctx.Err() != nil {
				s.discardUnpublishedVideoTransferOutput(task.UserID, merged)
			}
		}()
	}
	payload, err = json.Marshal(map[string]any{"resourceId": outputID, "branch": input.Branch, "summary": summary, "segments": segments})
	if err != nil {
		return err
	}
	if ctx.Err() != nil {
		return fail("处理已中止", "处理超时或已取消")
	}
	task.ResultJSON = string(payload)
	task.Status, task.Stage, task.Progress = model.TaskStatusSucceeded, "处理完成", 100
	completed := time.Now()
	task.CompletedAt = &completed
	return s.repo.SaveTaskCompletion(task, model.TaskStatusRunning, nil)
}

func (s *Service) executeVideoTransferSegment(ctx context.Context, task *model.Task, input videoTransferTaskInput, source, dir, python, runner string, info mediaProbeInfo, reportProgress func(string, int) error) (*model.Resource, *videotransfer.Summary, error) {
	fail := func(stage, message string) (*model.Resource, *videotransfer.Summary, error) {
		return nil, nil, fmt.Errorf("%s：%s", stage, message)
	}
	var err error
	request := struct {
		Input              string `json:"input"`
		Branch             string `json:"branch"`
		ValidationCacheDir string `json:"validationCacheDir"`
		StrategyVersion    string `json:"strategyVersion"`
		videotransfer.Settings
	}{Input: source, Branch: input.Branch, ValidationCacheDir: filepath.Dir(source), StrategyVersion: input.StrategyVersion, Settings: input.Settings}
	requestJSON, err := json.Marshal(request)
	if err != nil {
		return fail("处理输入无效", "无法生成处理参数")
	}
	requestPath := filepath.Join(dir, "request.json")
	if err = os.WriteFile(requestPath, requestJSON, 0600); err != nil {
		return fail("处理输入无效", "无法写入处理参数")
	}
	outputPath := filepath.Join(dir, "output.mp4")
	cmd := exec.CommandContext(ctx, python, runner, "--request", requestPath, "--output", outputPath)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1", "OMP_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2")
	configureVideoTransferProcess(cmd)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fail("启动处理失败", "无法建立算法输出通道")
	}
	stderr := &videoTransferBoundedBuffer{limit: 16 << 10}
	cmd.Stderr = stderr
	if err = cmd.Start(); err != nil {
		return fail("启动处理失败", "无法启动视频处理算法")
	}
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 4096), 64<<10)
	lastProgress := 5
	lastWrite := time.Time{}
	for scanner.Scan() {
		var update struct {
			Progress float64 `json:"progress"`
			Stage    string  `json:"stage"`
		}
		if json.Unmarshal(scanner.Bytes(), &update) != nil {
			continue
		}
		progress := max(lastProgress, min(94, 5+int(update.Progress*0.89)))
		if progress == lastProgress && time.Since(lastWrite) < time.Second {
			continue
		}
		stage := truncateRunes(strings.TrimSpace(update.Stage), 70)
		if stage == "" {
			stage = "正在处理视频"
		}
		if err = reportProgress(stage, progress); err != nil {
			_ = cmd.Cancel()
			_ = cmd.Wait()
			return nil, nil, err
		}
		lastProgress, lastWrite = progress, time.Now()
	}
	if scanner.Err() != nil {
		_ = cmd.Cancel()
	}
	runErr := cmd.Wait()
	if ctx.Err() != nil {
		return fail("处理已中止", "处理超时或已取消")
	}
	if runErr != nil || scanner.Err() != nil {
		return fail("算法处理失败", videoTransferAlgorithmError(stderr.Bytes(), dir))
	}
	if err = reportProgress("检查处理结果", 96); err != nil {
		return nil, nil, err
	}
	out, err := os.Open(outputPath)
	if err != nil {
		return fail("处理结果无效", "算法未生成视频文件")
	}
	defer out.Close()
	stat, err := out.Stat()
	if err != nil || !stat.Mode().IsRegular() || stat.Size() <= 0 {
		return fail("处理结果无效", "算法输出的视频为空")
	}
	outInfo, err := probeMediaFileDuration(ctx, outputPath, true)
	if err != nil || outInfo.Width != info.Width || outInfo.Height != info.Height || outInfo.DurationMs <= 0 {
		return fail("处理结果无效", "处理结果的尺寸或时长无效")
	}
	reportFile, err := os.Open(strings.TrimSuffix(outputPath, filepath.Ext(outputPath)) + ".report.json")
	if err != nil {
		return fail("处理报告缺失", "算法未生成有效的处理摘要")
	}
	reportBytes, reportErr := io.ReadAll(io.LimitReader(reportFile, 256<<10))
	_ = reportFile.Close()
	var report videoTransferRunnerReport
	if reportErr != nil || json.Unmarshal(reportBytes, &report) != nil || report.Frames <= 0 || report.ChangedPixels < 0 || report.TrackingCoverage < 0 || report.TrackingCoverage > 1 {
		return fail("处理报告无效", "无法验证算法的处理摘要")
	}
	if err = validateVideoTransferTimeline(report, input.Settings, info, outInfo); err != nil {
		return fail("处理报告无效", err.Error())
	}
	summary := report.Summary
	if len(summary.Limitations) > 20 {
		summary.Limitations = summary.Limitations[:20]
	}
	for i := range summary.Limitations {
		summary.Limitations[i] = truncateRunes(summary.Limitations[i], 300)
	}
	policy, err := s.RuntimePolicy()
	if err != nil {
		return fail("读取存储限制失败", "无法读取账号存储设置")
	}
	if stat.Size() > megabytes(policy.Resource.GeneratedFileMB) {
		return fail("结果超过大小限制", fmt.Sprintf("处理结果超过 %dMB 生成文件上限", policy.Resource.GeneratedFileMB))
	}
	if ctx.Err() != nil {
		return fail("处理已中止", "处理超时或已取消")
	}
	if err = reportProgress("保存处理结果", 98); err != nil {
		return nil, nil, err
	}
	result, err := s.storeVideoTransferOutput(task.UserID, "video-transfer-"+input.Branch+".mp4", stat.Size(), outInfo, out, "video-transfer:"+task.ID+":"+filepath.Base(filepath.Dir(dir))+":"+filepath.Base(dir))
	if err != nil {
		return fail("保存处理结果失败", renderFailureReason(err))
	}
	return result, &summary, nil
}

func videoTransferExtension(mime string) string {
	switch mime {
	case "video/webm":
		return ".webm"
	case "video/quicktime":
		return ".mov"
	case "video/x-matroska":
		return ".mkv"
	default:
		return ".mp4"
	}
}

func videoTransferAlgorithmError(raw []byte, dir string) string {
	// 仅公开 runner 的简短末行错误，不返回 Python 堆栈和服务器路径。
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if line == "" {
			continue
		}
		var reported struct {
			Error   string `json:"error"`
			Message string `json:"message"`
		}
		if json.Unmarshal([]byte(line), &reported) == nil && reported.Error != "" && reported.Message != "" {
			line = "ERROR: " + reported.Message
		}
		if strings.HasPrefix(line, "ERROR:") {
			line = strings.TrimSpace(strings.TrimPrefix(line, "ERROR:"))
			line = strings.ReplaceAll(line, dir, "[临时目录]")
			if message := videoTransferSafeFailure(line, 300); message != "" {
				return message
			}
		}
		break
	}
	return "视频处理算法失败，请检查处理时间范围、水印区域和运行环境"
}

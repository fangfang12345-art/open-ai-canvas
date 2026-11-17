package app

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"yingce/backend/internal/model"
	"yingce/backend/internal/videotransfer"
)

func (s *Service) restoreVideoTransferSegments(userID string, segments, previous []videotransfer.Segment) {
	for i := range segments {
		segments[i].Status = "queued"
		for _, old := range previous {
			if old.Index != segments[i].Index || old.StartSeconds != segments[i].StartSeconds || old.DurationSeconds != segments[i].DurationSeconds || old.Status != "succeeded" || old.Summary == nil || old.Summary.Frames <= 0 {
				continue
			}
			if output, err := s.repo.ResourceForUser(userID, old.OutputResourceID); err == nil && output.Status == model.ResourceStatusReady && output.Size > 0 && strings.HasPrefix(output.MimeType, "video/") {
				// Retain only the generated segment identity, never a persisted filesystem name.
				id := segments[i].ID
				segments[i] = old
				segments[i].ID, segments[i].Progress = id, 100
			}
			break
		}
	}
}

func videoTransferProcessingProgress(segments []videotransfer.Segment) int {
	var weighted, duration float64
	for _, part := range segments {
		if part.DurationSeconds > 0 {
			duration += part.DurationSeconds
			weighted += part.DurationSeconds * float64(max(0, min(100, part.Progress)))
		}
	}
	if duration == 0 {
		return 3
	}
	// Reserve the last tenth for assembly, validation, storage and terminal publication.
	return max(3, min(90, 3+int(87*weighted/(100*duration))))
}

// orderVideoTransferSegments validates one contiguous normalized output interval.
// Duration is derived from absolute frame boundaries, including fractional rates.
func orderVideoTransferSegments(segments []videotransfer.Segment, info mediaProbeInfo) ([]videotransfer.Segment, int, int, error) {
	if len(segments) == 0 || info.FrameRate <= 0 || math.IsNaN(info.FrameRate) || math.IsInf(info.FrameRate, 0) {
		return nil, 0, 0, errors.New("完整视频缺少有效的帧率或处理结果")
	}
	ordered := append([]videotransfer.Segment(nil), segments...)
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].StartSeconds == ordered[j].StartSeconds {
			return ordered[i].Index < ordered[j].Index
		}
		return ordered[i].StartSeconds < ordered[j].StartSeconds
	})
	first, end := 0, 0
	for i, part := range ordered {
		if part.Status != "succeeded" || part.OutputResourceID == "" || part.Summary == nil || part.DurationSeconds <= 0 || math.IsNaN(part.StartSeconds) || math.IsNaN(part.DurationSeconds) || math.IsInf(part.StartSeconds, 0) || math.IsInf(part.DurationSeconds, 0) {
			return nil, 0, 0, errors.New("视频处理结果尚不完整")
		}
		start := int(math.Round(part.StartSeconds * info.FrameRate))
		last := int(math.Round((part.StartSeconds + part.DurationSeconds) * info.FrameRate))
		if start < 0 || last <= start || part.Summary.Frames != last-start || (i > 0 && start != end) {
			return nil, 0, 0, errors.New("视频处理结果的时间顺序或帧数不一致")
		}
		if i == 0 {
			first = start
		}
		end = last
	}
	return ordered, first, end - first, nil
}

func (s *Service) mergeVideoTransferOutputs(ctx context.Context, task *model.Task, input videoTransferTaskInput, source, dir string, info mediaProbeInfo, segments []videotransfer.Segment, progress func(string, int) error) (*model.Resource, error) {
	ordered, first, frames, err := orderVideoTransferSegments(segments, info)
	if err != nil {
		return nil, err
	}
	if err = progress("正在生成完整视频", 90); err != nil {
		return nil, err
	}
	mergeDir := filepath.Join(dir, "complete")
	if err = os.MkdirAll(mergeDir, 0700); err != nil {
		return nil, errors.New("服务器临时存储不可用")
	}
	var manifest strings.Builder
	manifest.WriteString("ffconcat version 1.0\n")
	for i, part := range ordered {
		path, resourceErr := s.videoTransferMergeInput(ctx, task.UserID, part.OutputResourceID, filepath.Join(mergeDir, fmt.Sprintf("input-%d.mp4", i)))
		if resourceErr != nil {
			return nil, resourceErr
		}
		partInfo, probeErr := probeMediaFileDuration(ctx, path, true)
		if probeErr != nil || partInfo.Width != info.Width || partInfo.Height != info.Height || partInfo.Frames != part.Summary.Frames || math.Abs(partInfo.FrameRate-info.FrameRate) > .001 {
			return nil, errors.New("处理结果的尺寸、帧率或帧数不一致")
		}
		if strings.ContainsAny(path, "\r\n") {
			return nil, errors.New("服务器临时存储路径无效")
		}
		// Explicit video duration prevents AAC padding in each MP4 from shifting
		// subsequent video frames. Audio is read once from the original below.
		fmt.Fprintf(&manifest, "file '%s'\nduration %.12f\n", strings.ReplaceAll(path, "'", "'\\''"), float64(part.Summary.Frames)/info.FrameRate)
	}
	listPath := filepath.Join(mergeDir, "inputs.ffconcat")
	if err = os.WriteFile(listPath, []byte(manifest.String()), 0600); err != nil {
		return nil, errors.New("无法准备完整视频")
	}
	outputPath := filepath.Join(mergeDir, "output.mp4")
	duration := float64(frames) / info.FrameRate
	if err = concatenateVideoTransfer(ctx, listPath, source, outputPath, info.VideoStartSeconds+float64(first)/info.FrameRate, duration, progress); err != nil {
		return nil, err
	}
	if err = progress("检查完整视频", 97); err != nil {
		return nil, err
	}
	outInfo, err := probeMediaFileDuration(ctx, outputPath, true)
	if err != nil || outInfo.Width != info.Width || outInfo.Height != info.Height || outInfo.Frames != frames || math.Abs(outInfo.FrameRate-info.FrameRate) > .001 || math.Abs(outInfo.VideoDurationSeconds-duration) > .005 {
		return nil, errors.New("完整视频的尺寸、帧数或时长校验失败")
	}
	out, err := os.Open(outputPath)
	if err != nil {
		return nil, errors.New("无法读取完整视频")
	}
	defer out.Close()
	stat, err := out.Stat()
	if err != nil || stat.Size() <= 0 {
		return nil, errors.New("完整视频文件为空")
	}
	if err = progress("保存完整视频", 98); err != nil {
		return nil, err
	}
	if ctx.Err() != nil {
		return nil, errors.New("处理超时或已取消")
	}
	result, err := s.storeVideoTransferOutput(task.UserID, "video-transfer-"+input.Branch+".mp4", stat.Size(), outInfo, out, "video-transfer:"+task.ID+":"+filepath.Base(dir)+":complete", true)
	if err != nil {
		return nil, fmt.Errorf("保存完整视频失败：%s", renderFailureReason(err))
	}
	return result, nil
}

// Local objects can be concatenated in place. Remote objects are streamed to a
// bounded temporary file, with cancellation closing the body to interrupt reads.
func (s *Service) videoTransferMergeInput(ctx context.Context, userID, id, destination string) (string, error) {
	if ctx.Err() != nil {
		return "", ctx.Err()
	}
	resource, body, err := s.OpenResource(userID, id)
	if err != nil {
		return "", errors.New("无法读取已保存的处理结果")
	}
	defer body.Close()
	if resource.Status != model.ResourceStatusReady || resource.Size <= 0 || !strings.HasPrefix(resource.MimeType, "video/") {
		return "", errors.New("已保存的处理结果不可用")
	}
	if local, ok := body.(*os.File); ok {
		return filepath.Abs(local.Name())
	}
	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case <-ctx.Done():
			_ = body.Close()
		case <-done:
		}
	}()
	file, err := os.OpenFile(destination, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return "", errors.New("服务器临时存储不可用")
	}
	n, copyErr := io.Copy(file, io.LimitReader(body, resource.Size))
	closeErr := file.Close()
	if copyErr != nil || closeErr != nil || n != resource.Size || ctx.Err() != nil {
		return "", errors.New("读取处理结果中断或大小不一致")
	}
	return filepath.Abs(destination)
}

func concatenateVideoTransfer(ctx context.Context, list, source, output string, start, duration float64, progress func(string, int) error) error {
	bin := strings.TrimSpace(os.Getenv("CANVAS_FFMPEG_PATH"))
	if bin == "" {
		bin = "ffmpeg"
	}
	bin, err := exec.LookPath(bin)
	if err != nil {
		return errors.New("视频处理环境缺少 FFmpeg")
	}
	cmd := exec.CommandContext(ctx, bin, "-nostdin", "-v", "error", "-n", "-f", "concat", "-safe", "0", "-i", list,
		"-seek_timestamp", "1", "-ss", strconv.FormatFloat(start, 'f', 12, 64), "-i", source,
		"-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
		"-t", strconv.FormatFloat(duration, 'f', 12, 64), "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", output)
	configureVideoTransferProcess(cmd)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return errors.New("无法读取完整视频生成进度")
	}
	cmd.Stderr = &videoTransferBoundedBuffer{limit: 16 << 10}
	if err = cmd.Start(); err != nil {
		return errors.New("无法启动完整视频生成")
	}
	scanner := bufio.NewScanner(stdout)
	last := 90
	for scanner.Scan() {
		if value, ok := strings.CutPrefix(scanner.Text(), "out_time_us="); ok {
			elapsed, _ := strconv.ParseFloat(value, 64)
			next := max(last, min(96, 90+int(6*elapsed/(duration*1e6))))
			if next > last {
				if err = progress("正在生成完整视频", next); err != nil {
					_ = cmd.Cancel()
					_ = cmd.Wait()
					return err
				}
				last = next
			}
		}
	}
	if scanner.Err() != nil {
		_ = cmd.Cancel()
	}
	runErr := cmd.Wait()
	if ctx.Err() != nil {
		return errors.New("处理超时或已取消")
	}
	if runErr != nil || scanner.Err() != nil {
		return errors.New("完整视频生成失败，已保留处理进度，请重试")
	}
	return nil
}

package app

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"yingce/backend/internal/model"
)

// 媒体探测的唯一 ffprobe 入口：宽高（按显示旋转换算）、时长与容器级标签一次读出。
// 二进制优先使用 CANVAS_FFPROBE_PATH，空值才通过 PATH 查找；未安装时调用方按“未知”降级。

const (
	// mediaProbeTimeout 是单次 ffprobe 的上限；探测只读容器头，正常远小于该值。
	mediaProbeTimeout = 20 * time.Second
	// mediaProbeMaxDimension 过滤损坏容器报出的离谱尺寸，避免把垃圾值写进资源。
	mediaProbeMaxDimension = 16384
	// mediaProbeRemoteHeadBytes 是对象存储资源探测时下载的文件头上限。moov 在文件头（faststart）
	// 的视频足够探测；moov 在文件尾的视频会探测失败并保持尺寸未知，不为补尺寸下载整段视频。
	mediaProbeRemoteHeadBytes int64 = 8 << 20
	// mediaProbeRemoteReadTimeout 防止对象存储读取卡住时占住请求或回填任务。
	mediaProbeRemoteReadTimeout = 60 * time.Second
)

// errMediaProbeUnavailable 表示宿主机没有 ffprobe；它不代表媒体损坏。
var errMediaProbeUnavailable = errors.New("媒体探测依赖未安装（需要 ffprobe）")

// mediaProbeInfo 是 ffprobe 对一个本地媒体文件的探测结果。没有视频流（纯音频）时宽高为 0；
// Tags 是容器级（format）标签，键保持 ffprobe 输出的原样大小写。
type mediaProbeInfo struct {
	Width                int
	Height               int
	DurationMs           int64
	VideoDurationSeconds float64
	VideoStartSeconds    float64
	FrameRate            float64
	Frames               int
	Tags                 map[string]string
}

func mediaProbeAvailable() bool {
	_, err := mediaProbeBinary()
	return err == nil
}

func mediaProbeBinary() (string, error) {
	bin := strings.TrimSpace(os.Getenv("CANVAS_FFPROBE_PATH"))
	if bin == "" {
		bin = "ffprobe"
	}
	return exec.LookPath(bin)
}

// probeMediaFile 运行一次 ffprobe。ctx 取消或超过 mediaProbeTimeout 时终止子进程。
func probeMediaFile(ctx context.Context, path string) (mediaProbeInfo, error) {
	return probeMediaFileDuration(ctx, path, false)
}

// 水印算法按视频流时间轴裁切，容器里的较长音轨不能扩展可处理画面范围。
func probeMediaFileDuration(ctx context.Context, path string, preferVideoDuration bool) (mediaProbeInfo, error) {
	if strings.TrimSpace(path) == "" {
		return mediaProbeInfo{}, errors.New("媒体路径为空")
	}
	bin, err := mediaProbeBinary()
	if err != nil {
		return mediaProbeInfo{}, errMediaProbeUnavailable
	}
	if ctx == nil {
		ctx = context.Background()
	}
	ctx, cancel := context.WithTimeout(ctx, mediaProbeTimeout)
	defer cancel()
	output, err := exec.CommandContext(ctx, bin,
		"-v", "error", "-select_streams", "v:0",
		"-show_entries", "stream=width,height,duration,duration_ts,time_base,start_time,avg_frame_rate,nb_frames:stream_tags=DURATION:stream_side_data=rotation:format=duration,format_name:format_tags",
		"-of", "json", path).Output()
	if err != nil {
		return mediaProbeInfo{}, fmt.Errorf("媒体探测失败：%w", err)
	}
	return parseMediaProbeOutputDuration(output, preferVideoDuration)
}

// parseMediaProbeOutput 解析 ffprobe 的 JSON 输出。显示旋转 ±90° 的视频交换宽高，
// 与 ffmpeg 默认 autorotate 后实际参与滤镜的画面一致。
func parseMediaProbeOutput(raw []byte) (mediaProbeInfo, error) {
	return parseMediaProbeOutputDuration(raw, false)
}

func parseMediaProbeOutputDuration(raw []byte, preferVideoDuration bool) (mediaProbeInfo, error) {
	var document struct {
		Streams []struct {
			Width        int               `json:"width"`
			Height       int               `json:"height"`
			Duration     string            `json:"duration"`
			StartTime    string            `json:"start_time"`
			DurationTS   int64             `json:"duration_ts"`
			TimeBase     string            `json:"time_base"`
			Tags         map[string]string `json:"tags"`
			FrameRate    string            `json:"avg_frame_rate"`
			Frames       string            `json:"nb_frames"`
			SideDataList []struct {
				Rotation *float64 `json:"rotation"`
			} `json:"side_data_list"`
		} `json:"streams"`
		Format struct {
			Duration string            `json:"duration"`
			Name     string            `json:"format_name"`
			Tags     map[string]string `json:"tags"`
		} `json:"format"`
	}
	if err := json.Unmarshal(raw, &document); err != nil {
		return mediaProbeInfo{}, fmt.Errorf("媒体探测结果无效：%w", err)
	}
	info := mediaProbeInfo{Tags: document.Format.Tags}
	if len(document.Streams) > 0 {
		stream := document.Streams[0]
		if seconds, err := strconv.ParseFloat(stream.StartTime, 64); err == nil && !math.IsInf(seconds, 0) && !math.IsNaN(seconds) {
			info.VideoStartSeconds = seconds
		}
		if seconds, err := strconv.ParseFloat(stream.Duration, 64); err == nil && seconds > 0 && !math.IsInf(seconds, 0) && !math.IsNaN(seconds) {
			info.VideoDurationSeconds = seconds
		}
		if info.VideoDurationSeconds == 0 && stream.DurationTS > 0 {
			if parts := strings.Split(stream.TimeBase, "/"); len(parts) == 2 {
				n, _ := strconv.ParseFloat(parts[0], 64)
				d, _ := strconv.ParseFloat(parts[1], 64)
				seconds := float64(stream.DurationTS) * n / d
				if seconds > 0 && !math.IsInf(seconds, 0) && !math.IsNaN(seconds) {
					info.VideoDurationSeconds = seconds
				}
			}
		}
		if info.VideoDurationSeconds == 0 {
			seconds, _ := strconv.ParseFloat(document.Format.Duration, 64)
			if strings.Contains(document.Format.Name, "matroska") || strings.Contains(document.Format.Name, "webm") {
				// Matroska's DURATION tag and format fallback are absolute ends.
				if parts := strings.Split(mediaProbeTag(stream.Tags, "DURATION"), ":"); len(parts) == 3 {
					h, hErr := strconv.ParseFloat(parts[0], 64)
					m, mErr := strconv.ParseFloat(parts[1], 64)
					s, sErr := strconv.ParseFloat(parts[2], 64)
					if hErr == nil && mErr == nil && sErr == nil {
						seconds = h*3600 + m*60 + s
					}
				}
				seconds -= info.VideoStartSeconds
			} else if info.VideoStartSeconds != 0 {
				seconds = 0
			}
			if seconds > 0 && !math.IsInf(seconds, 0) && !math.IsNaN(seconds) {
				info.VideoDurationSeconds = seconds
			}
		}
		if parts := strings.Split(stream.FrameRate, "/"); len(parts) == 2 {
			n, _ := strconv.ParseFloat(parts[0], 64)
			d, _ := strconv.ParseFloat(parts[1], 64)
			if n > 0 && d > 0 && n/d < 1000 {
				info.FrameRate = n / d
			}
		}
		info.Frames, _ = strconv.Atoi(stream.Frames)
		if preferVideoDuration {
			if info.VideoDurationSeconds <= 0 && info.VideoStartSeconds != 0 {
				return mediaProbeInfo{}, errors.New("视频流缺少可确认的时长，无法确定非零起点的处理范围")
			}
			if info.VideoDurationSeconds > 0 {
				document.Format.Duration = strconv.FormatFloat(info.VideoDurationSeconds, 'f', -1, 64)
			}
		}
		if stream.Width > 0 && stream.Height > 0 && stream.Width <= mediaProbeMaxDimension && stream.Height <= mediaProbeMaxDimension {
			info.Width, info.Height = stream.Width, stream.Height
			for _, side := range stream.SideDataList {
				if side.Rotation != nil && int(math.Abs(math.Round(*side.Rotation)))%180 == 90 {
					info.Width, info.Height = info.Height, info.Width
					break
				}
			}
		}
	}
	if seconds, err := strconv.ParseFloat(strings.TrimSpace(document.Format.Duration), 64); err == nil && seconds > 0 && !math.IsNaN(seconds) && !math.IsInf(seconds, 0) {
		info.DurationMs = int64(math.Ceil(seconds * 1000))
	}
	return info, nil
}

// probeMediaDurationMs 只取时长；探测不可用或失败时返回 0（按秒结算会转人工或回退上游用量）。
func probeMediaDurationMs(path string) int64 {
	info, err := probeMediaFile(context.Background(), path)
	if err != nil {
		return 0
	}
	return info.DurationMs
}

// mediaProbeTag 按大小写不敏感读取容器标签；mp4 的 mdta 键与 Matroska 标签大小写不一。
func mediaProbeTag(tags map[string]string, key string) string {
	if value, ok := tags[key]; ok {
		return value
	}
	for name, value := range tags {
		if strings.EqualFold(name, key) {
			return value
		}
	}
	return ""
}

// probeStoredResourceMedia 探测已保存资源：本地存储直接读文件；对象存储只下载文件头
// （mediaProbeRemoteHeadBytes），并在读取卡住时由看门狗关闭读取流。
func (s *Service) probeStoredResourceMedia(ctx context.Context, resource *model.Resource) (mediaProbeInfo, error) {
	if resource == nil || resource.Status != model.ResourceStatusReady || strings.TrimSpace(resource.ObjectKey) == "" {
		return mediaProbeInfo{}, errors.New("资源未就绪")
	}
	if !mediaProbeAvailable() {
		return mediaProbeInfo{}, errMediaProbeUnavailable
	}
	if resource.Provider == "local" {
		return probeMediaFile(ctx, filepath.Join(s.dataDir, "resources", filepath.FromSlash(resource.ObjectKey)))
	}
	stream, err := s.openResourceRange(resource.UserID, resource, fmt.Sprintf("bytes=0-%d", mediaProbeRemoteHeadBytes-1))
	if err != nil {
		return mediaProbeInfo{}, err
	}
	var stalled atomic.Bool
	watchdog := time.AfterFunc(mediaProbeRemoteReadTimeout, func() {
		stalled.Store(true)
		_ = stream.Body.Close()
	})
	defer watchdog.Stop()
	defer stream.Body.Close()
	file, err := os.CreateTemp("", "yingce-probe-*")
	if err != nil {
		return mediaProbeInfo{}, err
	}
	defer os.Remove(file.Name())
	_, copyErr := io.Copy(file, io.LimitReader(stream.Body, mediaProbeRemoteHeadBytes))
	closeErr := file.Close()
	if copyErr != nil {
		if stalled.Load() {
			return mediaProbeInfo{}, errors.New("读取媒体文件头超时")
		}
		return mediaProbeInfo{}, copyErr
	}
	if closeErr != nil {
		return mediaProbeInfo{}, closeErr
	}
	return probeMediaFile(ctx, file.Name())
}

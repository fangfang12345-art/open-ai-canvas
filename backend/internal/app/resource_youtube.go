package app

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

	"yingce/backend/internal/model"
)

const youtubeImportTimeout = 5 * time.Minute
const youtubeMaxDurationMs = 20 * 60 * 1000
const youtubeMaxBytes int64 = 500 << 20

var youtubeVideoIDPattern = regexp.MustCompile(`^[A-Za-z0-9_-]{11}$`)

// A process-wide bound also protects multiple Service instances in one server.
var youtubeImportSlots = make(chan struct{}, 2)

type YouTubeImportResult struct {
	Resource  *model.Resource `json:"resource"`
	Title     string          `json:"title"`
	SourceURL string          `json:"sourceUrl"`
	VideoID   string          `json:"videoId"`
}

func parseYouTubeURL(raw string) (string, string, error) {
	invalid := func() (string, string, error) {
		return "", "", BadAuthRequest("请输入有效的 YouTube 视频链接，支持 watch、Shorts 和 youtu.be 链接")
	}
	raw = strings.TrimSpace(raw)
	if len(raw) == 0 || len(raw) > 4096 {
		return invalid()
	}
	if !strings.Contains(raw, "://") {
		raw = "https://" + raw
	}
	u, err := url.Parse(raw)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.User != nil || u.Opaque != "" || u.Port() != "" || strings.Contains(u.Host, ":") {
		return invalid()
	}
	query, err := url.ParseQuery(u.RawQuery)
	if err != nil {
		return invalid()
	}
	var id string
	switch strings.ToLower(u.Host) {
	case "youtu.be", "www.youtu.be":
		id = strings.TrimPrefix(u.Path, "/")
	case "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com":
		if u.Path == "/watch" && len(query["v"]) == 1 {
			id = query.Get("v")
		} else {
			parts := strings.Split(strings.Trim(u.Path, "/"), "/")
			if len(parts) == 2 && (parts[0] == "shorts" || parts[0] == "embed") {
				id = parts[1]
			}
		}
	default:
		return invalid()
	}
	if !youtubeVideoIDPattern.MatchString(id) {
		return invalid()
	}
	// Only this canonical URL reaches the extractor. User query strings, redirect
	// targets, credentials, playlists and downloader options never reach yt-dlp.
	return id, "https://www.youtube.com/watch?v=" + id, nil
}

func (s *Service) ImportYouTubeVideo(ctx context.Context, userID, rawURL string, uploadIdentity ...string) (*YouTubeImportResult, error) {
	if strings.TrimSpace(userID) == "" {
		return nil, Unauthorized("请先登录")
	}
	id, sourceURL, err := parseYouTubeURL(rawURL)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, youtubeContextError(err)
	}
	policy, err := s.RuntimePolicy()
	if err != nil {
		return nil, err
	}
	maxBytes := min(megabytes(policy.Resource.ResourceUploadMB)-1, youtubeMaxBytes)
	if maxBytes <= 0 {
		return nil, BadAuthRequest("当前账号不允许导入视频")
	}
	select {
	case youtubeImportSlots <- struct{}{}:
		defer func() { <-youtubeImportSlots }()
	default:
		return nil, NewAppError(429, "已有视频正在下载，请稍后重试")
	}
	bin := strings.TrimSpace(os.Getenv("CANVAS_YTDLP_PATH"))
	if bin == "" {
		bin = "yt-dlp"
	}
	bin, err = exec.LookPath(bin)
	if err != nil {
		return nil, NewAppError(503, "视频下载工具未安装，请安装 yt-dlp 后重试")
	}
	ffmpeg, err := renderFfmpegBinary()
	if err != nil || !mediaProbeAvailable() {
		return nil, NewAppError(503, "视频处理工具未安装，请安装 ffmpeg 和 ffprobe 后重试")
	}
	if _, err := exec.LookPath(ffmpeg); err != nil {
		return nil, NewAppError(503, "视频处理工具不可用，请检查 ffmpeg 配置")
	}
	root := filepath.Join(s.dataDir, "tmp", "youtube-imports")
	if err := os.MkdirAll(root, 0700); err != nil {
		return nil, NewAppError(500, "无法创建视频下载目录，请检查本地存储")
	}
	dir, err := os.MkdirTemp(root, "import-*")
	if err != nil {
		return nil, NewAppError(500, "无法创建视频下载目录，请检查本地存储")
	}
	defer os.RemoveAll(dir)
	ctx, cancel := context.WithTimeout(ctx, youtubeImportTimeout)
	defer cancel()
	metadata, err := downloadYouTubeVideo(ctx, bin, ffmpeg, sourceURL, dir, maxBytes)
	if err != nil {
		return nil, err
	}
	if metadata.ID != id {
		return nil, NewAppError(502, "下载的视频与链接不一致，请重新导入")
	}
	path := filepath.Join(dir, "video.mp4")
	file, err := os.Open(path)
	if err != nil {
		return nil, NewAppError(502, "未取得视频文件，请确认链接为可公开访问的视频")
	}
	defer file.Close()
	stat, err := file.Stat()
	if err != nil || !stat.Mode().IsRegular() || stat.Size() <= 0 {
		return nil, NewAppError(502, "下载的视频文件无效，请重试")
	}
	if stat.Size() > maxBytes {
		return nil, BadAuthRequest(fmt.Sprintf("视频文件超过导入上限（%d MB）", (maxBytes+1)>>20))
	}
	probe, err := probeMediaFile(ctx, path)
	if ctx.Err() != nil {
		return nil, youtubeContextError(ctx.Err())
	}
	if err != nil || probe.Width <= 0 || probe.Height <= 0 || probe.DurationMs <= 0 {
		return nil, NewAppError(502, "下载的视频无法解析，请换一个视频链接")
	}
	if probe.DurationMs > youtubeMaxDurationMs {
		return nil, BadAuthRequest("当前支持导入 20 分钟以内的视频")
	}
	mimeType, err := detectUploadedMimeType(file)
	if err != nil || !strings.HasPrefix(mimeType, "video/") {
		return nil, NewAppError(502, "下载结果不是有效的视频文件")
	}
	// Existing resource storage owns MIME detection, user quota, idempotency and
	// physical persistence. Never create a canvas reference to a temporary file.
	resource, err := s.UploadResourceFile(userID, id+".mp4", stat.Size(), "video", probe.Width, probe.Height, probe.DurationMs, file, uploadIdentity...)
	if err != nil {
		return nil, err
	}
	title := strings.TrimSpace(strings.Map(func(r rune) rune {
		if unicode.IsControl(r) {
			return -1
		}
		return r
	}, metadata.Title))
	if title == "" {
		title = "YouTube 视频 " + id
	}
	if runes := []rune(title); len(runes) > 240 {
		title = string(runes[:240])
	}
	return &YouTubeImportResult{Resource: resource, Title: title, SourceURL: sourceURL, VideoID: id}, nil
}

type youtubeDownloadMetadata struct {
	ID    string `json:"id"`
	Title string `json:"title"`
}

func youtubeDownloadArgs(ffmpeg, sourceURL, dir string, maxBytes int64) []string {
	return []string{
		"--ignore-config", "--no-plugin-dirs", "--no-playlist", "--use-extractors", "Youtube",
		"--no-cache-dir", "--no-progress", "--quiet", "--no-simulate",
		"--no-js-runtimes", "--js-runtimes", "node",
		"--socket-timeout", "15", "--retries", "2", "--fragment-retries", "2", "--extractor-retries", "2",
		"--break-match-filters", "!is_live & duration <= 1200",
		"--max-filesize", strconv.FormatInt(maxBytes, 10),
		"--format", "bv*[height<=1080][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/b[height<=1080][ext=mp4][vcodec^=avc1]/bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]",
		"--merge-output-format", "mp4", "--remux-video", "mp4", "--ffmpeg-location", ffmpeg,
		"--output", filepath.Join(dir, "video.%(ext)s"),
		"--print", `after_move:{"id":%(id)j,"title":%(title)j}`,
		"--", sourceURL,
	}
}

// yt-dlp diagnostics may contain signed CDN URLs. Keep a bounded copy solely
// for classifying public errors; never return or log the original output.
type youtubeOutput struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (w *youtubeOutput) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	n := len(p)
	if remaining := (32 << 10) - w.b.Len(); remaining > 0 {
		_, _ = w.b.Write(p[:min(len(p), remaining)])
	}
	return n, nil
}

func (w *youtubeOutput) String() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.b.String()
}

func downloadYouTubeVideo(ctx context.Context, bin, ffmpeg, sourceURL, dir string, maxBytes int64) (youtubeDownloadMetadata, error) {
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	cmd := exec.CommandContext(ctx, bin, youtubeDownloadArgs(ffmpeg, sourceURL, dir, maxBytes)...)
	configureYouTubeProcess(cmd)
	cmd.Dir = dir
	cmd.WaitDelay = 2 * time.Second
	var stdout, stderr youtubeOutput
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Start(); err != nil {
		return youtubeDownloadMetadata{}, NewAppError(503, "无法启动视频下载工具，请检查 yt-dlp 配置")
	}
	// Merging temporarily holds video, audio and the final MP4. Bound that disk
	// use even when the upstream does not advertise a trustworthy Content-Length.
	oversize := errors.New("youtube temporary files exceed limit")
	done := make(chan struct{})
	go func() {
		ticker := time.NewTicker(250 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ctx.Done():
				return
			case <-ticker.C:
				var total int64
				_ = filepath.WalkDir(dir, func(path string, entry os.DirEntry, err error) error {
					if err == nil && !entry.IsDir() {
						if info, err := entry.Info(); err == nil {
							total += info.Size()
						}
					}
					return nil
				})
				if total > 2*maxBytes+(8<<20) {
					cancel(oversize)
					return
				}
			}
		}
	}()
	err := cmd.Wait()
	close(done)
	if errors.Is(context.Cause(ctx), oversize) {
		return youtubeDownloadMetadata{}, BadAuthRequest("视频文件过大，请选择较短的视频")
	}
	if ctx.Err() != nil {
		return youtubeDownloadMetadata{}, youtubeContextError(ctx.Err())
	}
	if err != nil {
		return youtubeDownloadMetadata{}, youtubeDownloadError(stderr.String())
	}
	var metadata youtubeDownloadMetadata
	if err := json.NewDecoder(io.LimitReader(strings.NewReader(stdout.String()), 32<<10)).Decode(&metadata); err != nil {
		return youtubeDownloadMetadata{}, youtubeDownloadError(stderr.String())
	}
	return metadata, nil
}

func youtubeContextError(err error) error {
	if errors.Is(err, context.DeadlineExceeded) {
		return NewAppError(504, "视频下载超时，请检查网络后重试或选择较短的视频")
	}
	return NewAppError(408, "已取消视频下载")
}

func youtubeDownloadError(output string) error {
	text := strings.ToLower(output)
	switch {
	case strings.Contains(text, "does not pass filter"), strings.Contains(text, "rejected by filter"):
		return BadAuthRequest("当前支持 20 分钟以内的公开视频，暂不支持直播")
	case strings.Contains(text, "max-filesize"), strings.Contains(text, "larger than max"), strings.Contains(text, "file is larger"):
		return BadAuthRequest("视频文件过大，请选择较短的视频")
	case strings.Contains(text, "sign in"), strings.Contains(text, "confirm you"), strings.Contains(text, "private video"), strings.Contains(text, "members-only"):
		return NewAppError(502, "YouTube 要求登录或验证，当前仅支持无需登录即可下载的公开视频")
	case strings.Contains(text, "video unavailable"), strings.Contains(text, "not available"), strings.Contains(text, "removed"):
		return NewAppError(502, "该视频不可用，请确认链接有效且视频可公开访问")
	default:
		return NewAppError(502, "YouTube 视频下载失败，请检查本机网络后重试")
	}
}

package app

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"yingce/backend/internal/model"
)

func TestYouTubeURLCanonicalization(t *testing.T) {
	for _, raw := range []string{
		"https://www.youtube.com/watch?v=BaW_jenozKc&list=ignored&redirect=http://127.0.0.1/",
		"http://youtube.com/watch?v=BaW_jenozKc#ignored",
		"https://m.youtube.com/shorts/BaW_jenozKc?si=ignored",
		"youtube.com/embed/BaW_jenozKc",
		" https://youtu.be/BaW_jenozKc?t=1 ",
	} {
		id, canonical, err := parseYouTubeURL(raw)
		if err != nil || id != "BaW_jenozKc" || canonical != "https://www.youtube.com/watch?v=BaW_jenozKc" {
			t.Fatalf("parse %q = %q %q %v", raw, id, canonical, err)
		}
	}
	for _, raw := range []string{
		"", "--exec=evil", "https://youtube.com.evil.test/watch?v=BaW_jenozKc",
		"https://youtube.com@127.0.0.1/watch?v=BaW_jenozKc", "https://user@youtube.com/watch?v=BaW_jenozKc",
		"https://youtube.com:443/watch?v=BaW_jenozKc", "https://youtube.com:/watch?v=BaW_jenozKc",
		"https://youtube.com./watch?v=BaW_jenozKc", "file://youtube.com/watch?v=BaW_jenozKc",
		"https://youtu.be/BaW_jenozKc/other", "https://youtube.com/redirect?q=https://example.com",
		"https://youtube.com/watch?v=BaW_jenozKc&v=BaW_jenozKc", "https://youtube.com/playlist?list=BaW_jenozKc",
		"https://youtube.com/watch?v=../../local", "https://youtube.com/watch?v=BaW_jenozKc%0a--exec=evil",
		"https://youtube.com/watch?v=BaW_jenozKc;other", "https://youtube.com/watch?v=123",
	} {
		if _, _, err := parseYouTubeURL(raw); err == nil {
			t.Fatalf("accepted unsafe URL %q", raw)
		}
	}
}

func youtubeTestScript(t *testing.T, body string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("test downloader fixture requires a POSIX shell")
	}
	path := filepath.Join(t.TempDir(), "yt-dlp-fixture")
	if err := os.WriteFile(path, []byte("#!/bin/sh\nset -eu\n"+body+"\n"), 0700); err != nil {
		t.Fatal(err)
	}
	return path
}

func youtubeTestDownloader(t *testing.T) {
	t.Helper()
	requireFFmpegTools(t)
	video := filepath.Join(t.TempDir(), "fixture.mp4")
	writeTestVideo(t, video, 160, 90, nil)
	t.Setenv("YOUTUBE_TEST_VIDEO", video)
	t.Setenv("CANVAS_YTDLP_PATH", youtubeTestScript(t, `
output=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--output" ]; then shift; output="$1"; fi
  shift
done
cp "$YOUTUBE_TEST_VIDEO" "$(dirname "$output")/video.mp4"
printf '%s\n' '{"id":"BaW_jenozKc","title":"A test clip"}'
`))
}

func assertYouTubeTempEmpty(t *testing.T, svc *Service) {
	t.Helper()
	entries, err := os.ReadDir(filepath.Join(svc.dataDir, "tmp", "youtube-imports"))
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("left %d temporary download directories", len(entries))
	}
}

func assertYouTubeStatus(t *testing.T, err error, status int) {
	t.Helper()
	var appErr *AppError
	if !errors.As(err, &appErr) || appErr.Status != status {
		t.Fatalf("error = %v, want status %d", err, status)
	}
}

func TestYouTubeImportPersistsMeasuredVideoAndScopesIdempotency(t *testing.T) {
	youtubeTestDownloader(t)
	svc := newResourceTestService(t)
	result, err := svc.ImportYouTubeVideo(context.Background(), "user", "https://youtu.be/BaW_jenozKc?t=1", "youtube-identity")
	if err != nil {
		t.Fatal(err)
	}
	resource := result.Resource
	if result.Title != "A test clip" || result.VideoID != "BaW_jenozKc" || result.SourceURL != "https://www.youtube.com/watch?v=BaW_jenozKc" {
		t.Fatalf("metadata = %+v", result)
	}
	if resource.UserID != "user" || resource.Kind != "video" || resource.Status != model.ResourceStatusReady || resource.Width != 160 || resource.Height != 90 || resource.DurationMs < 1000 || resource.Size <= 0 {
		t.Fatalf("resource = %+v", resource)
	}
	if _, err := os.Stat(filepath.Join(svc.dataDir, "resources", filepath.FromSlash(resource.ObjectKey))); err != nil {
		t.Fatal("resource was not persisted", err)
	}
	assertYouTubeTempEmpty(t, svc)
	day := time.Now().UTC().Format("2006-01-02")
	used, err := svc.repo.DailyUploadBytes("user", day)
	if err != nil || used != resource.Size {
		t.Fatalf("quota = %d err=%v", used, err)
	}
	retry, err := svc.ImportYouTubeVideo(context.Background(), "user", result.SourceURL, "youtube-identity")
	if err != nil || retry.Resource.ID != resource.ID {
		t.Fatalf("idempotent retry = %+v err=%v", retry, err)
	}
	used, err = svc.repo.DailyUploadBytes("user", day)
	if err != nil || used != resource.Size {
		t.Fatalf("retry consumed quota again: %d err=%v", used, err)
	}
	other, err := svc.ImportYouTubeVideo(context.Background(), "other-user", result.SourceURL, "youtube-identity")
	if err != nil || other.Resource.ID == resource.ID || other.Resource.UserID != "other-user" {
		t.Fatalf("cross-user import = %+v err=%v", other, err)
	}
	assertYouTubeTempEmpty(t, svc)
}

func TestYouTubeImportQuotaFailureLeavesNoFileOrResource(t *testing.T) {
	youtubeTestDownloader(t)
	svc := newResourceTestService(t)
	day := time.Now().UTC().Format("2006-01-02")
	limit := megabytes(defaultRuntimePolicy().Resource.DailyUploadMB)
	if err := svc.repo.ReserveDailyUpload("user", day, limit-1, limit); err != nil {
		t.Fatal(err)
	}
	_, err := svc.ImportYouTubeVideo(context.Background(), "user", "https://youtu.be/BaW_jenozKc")
	assertYouTubeStatus(t, err, 403)
	assertYouTubeTempEmpty(t, svc)
	resources, err := svc.repo.Resources("user", 10)
	if err != nil || len(resources) != 0 {
		t.Fatalf("failed import created resource: %v %v", resources, err)
	}
	used, err := svc.repo.DailyUploadBytes("user", day)
	if err != nil || used != limit-1 || svc.pendingStorage["user"] != 0 {
		t.Fatalf("failed import changed quota: used=%d pending=%d err=%v", used, svc.pendingStorage["user"], err)
	}
}

func TestYouTubeImportFailureIsSanitizedAndCleansTemp(t *testing.T) {
	requireFFmpegTools(t)
	t.Setenv("CANVAS_YTDLP_PATH", youtubeTestScript(t, `
printf '%s\n' 'ERROR: network failure https://cdn.invalid/?secret=do-not-expose /private/path' >&2
exit 1
`))
	svc := newResourceTestService(t)
	_, err := svc.ImportYouTubeVideo(context.Background(), "user", "https://youtu.be/BaW_jenozKc")
	assertYouTubeStatus(t, err, 502)
	if strings.Contains(err.Error(), "secret") || strings.Contains(err.Error(), "/private") || strings.Contains(err.Error(), "cdn.invalid") {
		t.Fatalf("diagnostics leaked: %v", err)
	}
	assertYouTubeTempEmpty(t, svc)
	resources, listErr := svc.repo.Resources("user", 10)
	if listErr != nil || len(resources) != 0 {
		t.Fatalf("failed downloader wrote resources: %v %v", resources, listErr)
	}
}

func TestYouTubeImportCancellationStopsDownloaderAndCleansTemp(t *testing.T) {
	requireFFmpegTools(t)
	t.Setenv("CANVAS_YTDLP_PATH", youtubeTestScript(t, "sleep 30 &\nwait"))
	svc := newResourceTestService(t)
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err := svc.ImportYouTubeVideo(ctx, "user", "https://youtu.be/BaW_jenozKc")
	assertYouTubeStatus(t, err, 504)
	if elapsed := time.Since(start); elapsed > 3*time.Second {
		t.Fatalf("cancel took %s", elapsed)
	}
	assertYouTubeTempEmpty(t, svc)
}

func TestYouTubeImportRejectsUnavailableToolAndBusyProcess(t *testing.T) {
	svc := newResourceTestService(t)
	t.Setenv("CANVAS_YTDLP_PATH", filepath.Join(t.TempDir(), "missing"))
	_, err := svc.ImportYouTubeVideo(context.Background(), "user", "https://youtu.be/BaW_jenozKc")
	assertYouTubeStatus(t, err, 503)
	if len(youtubeImportSlots) != 0 {
		t.Fatal("tool failure leaked a concurrency slot")
	}
	for i := 0; i < cap(youtubeImportSlots); i++ {
		youtubeImportSlots <- struct{}{}
	}
	defer func() {
		for i := 0; i < cap(youtubeImportSlots); i++ {
			<-youtubeImportSlots
		}
	}()
	_, err = svc.ImportYouTubeVideo(context.Background(), "user", "https://youtu.be/BaW_jenozKc")
	assertYouTubeStatus(t, err, 429)
}

func TestYouTubeImportRejectsInvalidInputBeforeDownloader(t *testing.T) {
	svc := &Service{}
	_, err := svc.ImportYouTubeVideo(context.Background(), "", "https://youtu.be/BaW_jenozKc")
	assertYouTubeStatus(t, err, 401)
	_, err = svc.ImportYouTubeVideo(context.Background(), "user", "https://127.0.0.1/private")
	assertYouTubeStatus(t, err, 400)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = svc.ImportYouTubeVideo(ctx, "user", "https://youtu.be/BaW_jenozKc")
	assertYouTubeStatus(t, err, 408)
}

func TestYouTubeDownloadErrorClassification(t *testing.T) {
	for _, tc := range []struct {
		output string
		status int
	}{
		{"does not pass filter (!is_live & duration <= 1200)", 400},
		{"File is larger than max-filesize", 400},
		{"Sign in to confirm you're not a bot https://signed.invalid/?token=secret", 502},
		{"Video unavailable", 502},
	} {
		err := youtubeDownloadError(tc.output)
		assertYouTubeStatus(t, err, tc.status)
		if strings.Contains(err.Error(), "token=") {
			t.Fatal("classification leaked upstream URL")
		}
	}
}

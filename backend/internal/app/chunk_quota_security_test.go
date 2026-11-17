package app

import (
	"bytes"
	"testing"
	"time"
)

func TestChunkQuotaRejectsSingleFileLimit(t *testing.T) {
	svc := newResourceTestService(t)
	if _, err := svc.reserveChunkedUploadQuota("user-1", megabytes(defaultRuntimePolicy().Resource.ResourceUploadMB)); err == nil {
		t.Fatal("chunked upload accepted a file at the exclusive single-file limit")
	}
}

func TestVideoTransferUploadBypassesSingleFileLimitButReservesAccountQuota(t *testing.T) {
	svc := newResourceTestService(t)
	if err := svc.ReserveVideoTransferUploadSession("user", "video", 60<<20, time.Now().Add(time.Hour), 32); err != nil {
		t.Fatal(err)
	}
	if err := svc.ReserveChunkUploadSession("user", "generic", 60<<20, time.Now().Add(time.Hour), 32); err == nil {
		t.Fatal("generic upload limit bypassed")
	}
	if err := svc.ReserveVideoTransferUploadSession("user", "too-large-for-account", megabytes(defaultRuntimePolicy().Resource.DailyUploadMB), time.Now().Add(time.Hour), 32); err == nil {
		t.Fatal("daily quota bypassed")
	}
	if _, err := svc.UploadVideoTransferSourceFile("user", "video", "invalid.mp4", 60<<20, "video", 1, 1, 1, bytes.NewReader([]byte("invalid"))); err == nil {
		t.Fatal("invalid video accepted")
	}
	if used, _ := svc.repo.DailyUploadBytes("user", time.Now().UTC().Format("2006-01-02")); used != 0 {
		t.Fatalf("failed validation retained quota=%d", used)
	}
}

func TestChunkQuotaReservedUploadConsumesOnce(t *testing.T) {
	svc := newResourceTestService(t)
	if err := svc.ReserveChunkUploadSession("user", "session", 7, time.Now().Add(time.Hour), 32); err != nil {
		t.Fatal(err)
	}
	day := time.Now().UTC().Format("2006-01-02")
	if used, _ := svc.repo.DailyUploadBytes("user", day); used != 7 {
		t.Fatalf("quota before receiving bytes=%d", used)
	}
	resource, err := svc.UploadReservedResourceFile("user", "session", "test.txt", 7, "file", 0, 0, 0, bytes.NewReader([]byte("payload")), "logical-key")
	if err != nil {
		t.Fatal(err)
	}
	if resource.Size != 7 {
		t.Fatalf("size=%d", resource.Size)
	}
	if used, _ := svc.repo.DailyUploadBytes("user", day); used != 7 {
		t.Fatalf("quota after completion=%d", used)
	}
	if err := svc.ReserveChunkUploadSession("user", "retry", 7, time.Now().Add(time.Hour), 32); err != nil {
		t.Fatal(err)
	}
	reused, err := svc.UploadReservedResourceFile("user", "retry", "test.txt", 7, "file", 0, 0, 0, bytes.NewReader([]byte("payload")), "logical-key")
	if err != nil || reused.ID != resource.ID {
		t.Fatalf("retry=%v err=%v", reused, err)
	}
	if used, _ := svc.repo.DailyUploadBytes("user", day); used != 7 {
		t.Fatalf("quota after replay=%d", used)
	}
}

func TestChunkQuotaFailedCompletionReleases(t *testing.T) {
	svc := newResourceTestService(t)
	if err := svc.ReserveChunkUploadSession("user", "failed", 7, time.Now().Add(time.Hour), 32); err != nil {
		t.Fatal(err)
	}
	if _, err := svc.UploadReservedResourceFile("user", "failed", "test.txt", 7, "file", 0, 0, 0, nil); err == nil {
		t.Fatal("nil file accepted")
	}
	if used, _ := svc.repo.DailyUploadBytes("user", time.Now().UTC().Format("2006-01-02")); used != 0 {
		t.Fatalf("failed upload retained quota=%d", used)
	}
}

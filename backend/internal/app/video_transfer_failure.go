package app

import (
	"strings"
	"unicode"
	"unicode/utf8"

	"yingce/backend/internal/videotransfer"
)

const videoTransferGenericFailure = "视频处理失败，已保留处理进度，请重试"
const videoTransferSafeFailureFallback = "视频处理失败，请检查运行环境后重试"

// Only concise diagnostics may cross the worker/API boundary. Inspect the whole
// message before truncation so a long prefix cannot conceal a path or traceback.
func videoTransferSafeFailure(message string, limit int) string {
	if !utf8.ValidString(message) || strings.ContainsAny(message, "/\\") {
		return ""
	}
	for _, r := range message {
		if unicode.IsControl(r) || unicode.Is(unicode.Cf, r) {
			return ""
		}
	}
	lower := strings.ToLower(message)
	for _, marker := range []string{"traceback", "stack trace", "goroutine ", "panic:", "authorization:", "bearer ", "api_key=", "token="} {
		if strings.Contains(lower, marker) {
			return ""
		}
	}
	return truncateRunes(strings.TrimSpace(message), limit)
}

func videoTransferFailureMessage(parent string, segments []videotransfer.Segment) string {
	safeParent := videoTransferSafeFailure(parent, 700)
	switch safeParent {
	case "", videoTransferGenericFailure, "处理失败", "视频处理失败", "部分片段处理失败":
	default:
		// A final merge/storage failure must not be replaced by an earlier diagnostic.
		return safeParent
	}
	reasons := make([]string, 0, 3)
	seen := make(map[string]bool, 3)
	retained, more := false, false
	for _, segment := range segments {
		if segment.Status == "succeeded" && segment.OutputResourceID != "" {
			retained = true
		}
		if segment.Status != "failed" {
			continue
		}
		reason := videoTransferSafeFailure(segment.Error, 200)
		if reason == "" || seen[reason] {
			continue
		}
		if len(reasons) == cap(reasons) {
			more = true
			continue
		}
		seen[reason] = true
		reasons = append(reasons, reason)
	}
	if len(reasons) == 0 {
		if safeParent != "" {
			return safeParent
		}
		return videoTransferSafeFailureFallback
	}
	message := strings.Join(reasons, "；")
	if more {
		message += "；另有其他处理错误。"
	}
	if retained {
		message += " 已完成的处理结果已保留。"
	}
	return message
}

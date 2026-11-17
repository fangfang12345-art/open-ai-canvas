package app

import (
	"context"
	"io"
	"math"
	"os"
	"strings"

	"yingce/backend/internal/model"
)

// 专用上传会话免单文件上限，但必须以真实视频探测结果入库，仍消费预留的账号额度。
func (s *Service) UploadVideoTransferSourceFile(userID, reservationID, fileName string, size int64, kind string, width, height int, durationMs int64, file io.ReadSeeker, uploadIdentity ...string) (resource *model.Resource, returnErr error) {
	defer func() {
		if returnErr != nil {
			_ = s.ReleaseChunkUploadSession(userID, reservationID)
		}
	}()
	if file == nil {
		return nil, BadAuthRequest("视频上传文件不可读取")
	}
	mime, err := detectUploadedMimeType(file)
	if err != nil || !strings.HasPrefix(mime, "video/") {
		return nil, BadAuthRequest("视频搬运只接受有效的视频文件")
	}
	local, ok := file.(*os.File)
	if !ok {
		return nil, BadAuthRequest("视频上传文件不可读取")
	}
	info, err := probeMediaFileDuration(context.Background(), local.Name(), true)
	if err != nil || info.Width <= 0 || info.Height <= 0 || info.DurationMs <= 0 {
		return nil, BadAuthRequest("无法读取视频尺寸和时长，请检查文件与 FFprobe")
	}
	return s.UploadReservedResourceFile(userID, reservationID, fileName, size, "video", info.Width, info.Height, info.DurationMs, file, uploadIdentity...)
}

// 取消或租约交接可能发生在文件入库和片段结果提交之间，只清理未发布的独立输出。
func (s *Service) discardUnpublishedVideoTransferOutput(userID string, resource *model.Resource) {
	snapshot, err := s.repo.ResourceReferenceSnapshot(userID, "", []string{resource.ID})
	if err != nil {
		return
	}
	if len(snapshot.Direct) > 0 {
		return
	}
	ids := map[string]struct{}{resource.ID: {}}
	for _, doc := range snapshot.Documents {
		if documentReferencesResources(doc.PrimaryJSON, ids) || documentReferencesResources(doc.SecondaryJSON, ids) {
			return
		}
	}
	if err = s.deleteStoredResourceObject(userID, resource); err == nil {
		err = s.repo.DeleteResource(userID, resource.ID)
	}
	if err != nil {
		_ = s.log(userID, "", "error", "清理未发布的视频片段失败", "")
	}
}

func (s *Service) storeVideoTransferOutput(userID, fileName string, size int64, info mediaProbeInfo, file io.ReadSeeker, identity string, complete ...bool) (*model.Resource, error) {
	reserve := s.reserveGeneratedResourceQuota
	if len(complete) > 0 && complete[0] {
		reserve = s.reserveVideoTransferCompleteQuota
	}
	day, err := reserve(userID, size)
	if err != nil {
		return nil, err
	}
	resource, stored, err := s.storeResource(userID, "video", fileName, "video/mp4", size, info.Width, info.Height, info.DurationMs, file, normalizedResourceUploadKey([]string{identity}), false)
	if err != nil || !stored {
		s.releaseUserUploadQuota(userID, day, size)
	} else {
		s.commitUserUploadQuota(userID, size)
	}
	return resource, err
}

func (s *Service) reserveVideoTransferCompleteQuota(userID string, size int64) (string, error) {
	// The final video may be arbitrarily longer than one processing interval.
	// Keep account capacity and daily reservations without a per-file cap.
	policy, err := s.RuntimePolicy()
	if err != nil {
		return "", err
	}
	limit, err := s.storedFileLimit(userID, policy)
	if err != nil {
		return "", err
	}
	return s.reserveUserStoredFileQuota(userID, size, math.MaxInt64, megabytes(policy.Resource.DailyUploadMB), limit, "完整视频文件大小无效")
}

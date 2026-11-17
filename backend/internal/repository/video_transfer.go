package repository

import (
	"errors"
	"strings"
	"time"
	"yingce/backend/internal/model"
)

var ErrVideoTransferConflict = errors.New("视频搬运记录已更新，请刷新后重试")

func (r *Repository) UpdateVideoTransferTaskProgress(id, owner, stage string, progress int, resultJSON string) error {
	result := taskLeaseWriter(r.db.Model(&model.Task{}), owner).
		Where("id = ? AND type = ? AND status = ?", id, "video_transfer", model.TaskStatusRunning).
		Updates(map[string]any{"stage": stage, "progress": progress, "result_json": resultJSON, "updated_at": time.Now()})
	if result.Error != nil {
		return result.Error
	}
	if result.RowsAffected != 1 {
		return ErrTaskStateConflict
	}
	return nil
}

func (r *Repository) VideoTransferForUser(userID, id string) (*model.VideoTransfer, error) {
	var row model.VideoTransfer
	err := r.db.Where("user_id = ? AND id = ?", userID, id).First(&row).Error
	return &row, err
}

func (r *Repository) VideoTransfers(userID string, page, pageSize int, query string) ([]model.VideoTransfer, int64, error) {
	var rows []model.VideoTransfer
	var total int64
	q := r.db.Model(&model.VideoTransfer{}).Where("user_id = ?", userID)
	if query = strings.TrimSpace(query); query != "" {
		q = q.Where("LOWER(title) LIKE ?", "%"+strings.ToLower(query)+"%")
	}
	if err := q.Count(&total).Error; err != nil {
		return nil, 0, err
	}
	err := q.Order("updated_at DESC, id DESC").Offset((page - 1) * pageSize).Limit(pageSize).Find(&rows).Error
	return rows, total, err
}

func (r *Repository) CreateVideoTransfer(row *model.VideoTransfer) error {
	return r.db.Create(row).Error
}

func (r *Repository) UpdateVideoTransfer(row *model.VideoTransfer, expectedRevision int64) error {
	result := r.db.Model(&model.VideoTransfer{}).Where("id = ? AND user_id = ? AND revision = ?", row.ID, row.UserID, expectedRevision).
		Updates(map[string]any{"title": row.Title, "source_resource_id": row.SourceResourceID, "payload_json": row.PayloadJSON, "revision": row.Revision, "updated_at": row.UpdatedAt})
	if result.Error != nil {
		return result.Error
	}
	if result.RowsAffected != 1 {
		return ErrVideoTransferConflict
	}
	return nil
}

func (r *Repository) RefreshVideoTransferSnapshot(row *model.VideoTransfer, previousPayload string) (bool, error) {
	result := r.db.Model(&model.VideoTransfer{}).Where("id = ? AND user_id = ? AND revision = ? AND payload_json = ?", row.ID, row.UserID, row.Revision, previousPayload).UpdateColumn("payload_json", row.PayloadJSON)
	return result.RowsAffected == 1, result.Error
}

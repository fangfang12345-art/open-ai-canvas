package model

import "time"

// VideoTransfer 保存独立视频搬运工作流；执行状态来自关联的持久化 Task。
type VideoTransfer struct {
	ID               string `gorm:"primaryKey;size:36"`
	UserID           string `gorm:"size:36;index:idx_video_transfers_user_updated,priority:1"`
	Title            string `gorm:"size:240"`
	SourceResourceID string `gorm:"size:36;index"`
	PayloadJSON      string `gorm:"type:text;not null"`
	Revision         int64  `gorm:"not null;default:1"`
	CreatedAt        time.Time
	UpdatedAt        time.Time `gorm:"index:idx_video_transfers_user_updated,priority:2"`
}

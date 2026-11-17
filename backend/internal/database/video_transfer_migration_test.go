package database

import (
	"reflect"
	"testing"
	"time"

	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
	"yingce/backend/internal/model"
)

func TestVideoTransferMigration100PreservesExistingData(t *testing.T) {
	db, err := gorm.Open(sqlite.Dialector{DriverName: "sqlite", DSN: ":memory:"}, &gorm.Config{})
	if err != nil {
		t.Fatal(err)
	}
	connection, err := db.DB()
	if err != nil {
		t.Fatal(err)
	}
	connection.SetMaxOpenConns(1)
	t.Cleanup(func() { _ = connection.Close() })
	if err = db.AutoMigrate(&schemaMigration{}, &model.SystemSetting{}); err != nil {
		t.Fatal(err)
	}
	var before []schemaMigration
	for _, item := range schemaMigrations {
		if item.version >= 100 {
			break
		}
		before = append(before, schemaMigration{Version: item.version, Name: item.name, Checksum: item.checksum, AppliedAt: time.Unix(1000, 0)})
	}
	if err = db.Create(&before).Error; err != nil {
		t.Fatal(err)
	}
	if err = db.Create(&model.SystemSetting{Key: "keep", ValueJSON: "preserved"}).Error; err != nil {
		t.Fatal(err)
	}
	for range 2 {
		if err = MigrateSchema(db); err != nil {
			t.Fatal(err)
		}
	}
	if err = RequireSchemaVersion(db); err != nil {
		t.Fatal(err)
	}
	var after []schemaMigration
	if err = db.Where("version < 100").Order("version").Find(&after).Error; err != nil || !reflect.DeepEqual(before, after) {
		t.Fatalf("published migrations changed: %v", err)
	}
	var saved string
	if err = db.Raw("SELECT value_json FROM system_settings WHERE key = 'keep'").Scan(&saved).Error; err != nil || saved != "preserved" {
		t.Fatalf("existing settings changed: %q %v", saved, err)
	}
	for _, index := range []string{"idx_video_transfers_user_updated", "idx_video_transfers_source_resource_id"} {
		if !db.Migrator().HasIndex(&model.VideoTransfer{}, index) {
			t.Fatalf("missing workflow index %s", index)
		}
	}
	record := model.VideoTransfer{ID: "transfer", UserID: "owner", Title: "clip", PayloadJSON: "{}", Revision: 1}
	if err = db.Create(&record).Error; err != nil {
		t.Fatal(err)
	}
}

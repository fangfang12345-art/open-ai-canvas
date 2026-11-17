package handler

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
	"yingce/backend/internal/auth"
	"yingce/backend/internal/model"
	"yingce/backend/internal/repository"
	"yingce/backend/internal/service"
)

func TestYouTubeImportHTTPRejectsAnonymousAndUnsafeInput(t *testing.T) {
	db, err := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{})
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, _ := db.DB()
	sqlDB.SetMaxOpenConns(1)
	t.Cleanup(func() { _ = sqlDB.Close() })
	if err := db.AutoMigrate(&model.User{}, &model.AuthSession{}, &model.SystemSetting{}, &model.UserDailyActivity{}); err != nil {
		t.Fatal(err)
	}
	if err := db.Create(&model.User{ID: "youtube-user", Username: "youtube-user", Status: model.UserStatusActive}).Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Create(&model.AuthSession{ID: "youtube-session", UserID: "youtube-user", TokenHash: auth.HashToken("fixture-token"), ExpiresAt: time.Now().Add(time.Hour)}).Error; err != nil {
		t.Fatal(err)
	}
	svc := service.New(repository.New(db), t.TempDir())
	t.Cleanup(func() { _ = svc.Close() })
	previous := runtimeService
	ConfigureRuntime(svc)
	t.Cleanup(func() { runtimeService = previous })
	gin.SetMode(gin.TestMode)
	router := gin.New()
	RegisterUserDataRoutes(router.Group("/api"), svc)
	for _, tc := range []struct {
		name string
		body string
		auth bool
		code int
	}{
		{"anonymous", `{"url":"https://youtu.be/BaW_jenozKc"}`, false, 401},
		{"private-host", `{"url":"http://127.0.0.1/private"}`, true, 400},
		{"spoofed-host", `{"url":"https://youtube.com.attacker.test/watch?v=BaW_jenozKc"}`, true, 400},
		{"malformed-json", `{"url":`, true, 400},
		{"oversized-body", `{"url":"` + strings.Repeat("x", 9<<10) + `"}`, true, 400},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/api/resources/import-youtube", strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")
			if tc.auth {
				req.AddCookie(&http.Cookie{Name: service.SessionCookieName, Value: "youtube-session.fixture-token"})
			}
			out := httptest.NewRecorder()
			router.ServeHTTP(out, req)
			if out.Code != tc.code {
				t.Fatalf("status=%d want=%d body=%s", out.Code, tc.code, out.Body.String())
			}
			if strings.Contains(out.Body.String(), "attacker.test") || strings.Contains(out.Body.String(), "/private") {
				t.Fatalf("input leaked into error: %s", out.Body.String())
			}
		})
	}
}

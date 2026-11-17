package handler

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"yingce/backend/internal/service"
)

func TestVideoTransferStrictWriteJSON(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.PATCH("/record", func(c *gin.Context) {
		var patch service.VideoTransferPatch
		if bindVideoTransferJSON(c, &patch) {
			c.Status(http.StatusNoContent)
		}
	})
	for _, tc := range []struct {
		name string
		body string
		code int
	}{
		{"settings", `{"expectedRevision":1,"branches":{"moving-v6":{"settings":{"roi":{"x":1,"y":2,"width":10,"height":10}}}}}`, 204},
		{"fixed-regions", `{"expectedRevision":1,"branches":{"fixed-r16":{"settings":{"fixedRegions":[{"roi":{"x":10,"y":10,"width":30,"height":10},"referenceSeconds":12,"startSeconds":5,"durationSeconds":20}]}}}}`, 204},
		{"clear-fixed-regions", `{"expectedRevision":1,"branches":{"fixed-r16":{"settings":{"fixedRegions":[]}}}}`, 204},
		{"missing-region-reference", `{"expectedRevision":1,"branches":{"fixed-r16":{"settings":{"fixedRegions":[{"roi":{"x":10,"y":10,"width":30,"height":10},"startSeconds":0,"durationSeconds":0}]}}}}`, 400},
		{"injected-region-path", `{"expectedRevision":1,"branches":{"fixed-r16":{"settings":{"fixedRegions":[{"roi":{"x":10,"y":10,"width":30,"height":10},"input":"/etc/passwd"}]}}}}`, 400},
		{"injected-strategy", `{"expectedRevision":1,"branches":{"fixed-r16":{"settings":{"strategyVersion":"old"}}}}`, 400},
		{"injected-status", `{"expectedRevision":1,"branches":{"fixed-r16":{"status":"succeeded"}}}`, 400},
		{"injected-result", `{"expectedRevision":1,"branches":{"fixed-r16":{"outputResourceId":"foreign"}}}`, 400},
		{"injected-path", `{"expectedRevision":1,"branches":{"fixed-r16":{"settings":{"input":"/etc/passwd"}}}}`, 400},
		{"injected-metadata", `{"expectedRevision":1,"sourceWidth":1}`, 400},
		{"trailing-json", `{"expectedRevision":1}{}`, 400},
		{"oversized", `{"expectedRevision":1,"title":"` + strings.Repeat("x", 33<<10) + `"}`, 400},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out := httptest.NewRecorder()
			router.ServeHTTP(out, httptest.NewRequest(http.MethodPatch, "/record", strings.NewReader(tc.body)))
			if out.Code != tc.code {
				t.Fatalf("status=%d want=%d body=%s", out.Code, tc.code, out.Body.String())
			}
		})
	}
}

func TestVideoTransferRoutesRequireAuthentication(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	RegisterVideoTransferRoutes(router.Group("/api"), &service.Service{})
	for _, route := range []struct{ method, path string }{
		{http.MethodGet, "/api/video-transfers"},
		{http.MethodPost, "/api/video-transfers"},
		{http.MethodGet, "/api/video-transfers/capabilities"},
		{http.MethodGet, "/api/video-transfers/id"},
		{http.MethodPatch, "/api/video-transfers/id"},
		{http.MethodPost, "/api/video-transfers/id/run"},
		{http.MethodPost, "/api/video-transfers/id/cancel"},
	} {
		out := httptest.NewRecorder()
		router.ServeHTTP(out, httptest.NewRequest(route.method, route.path, strings.NewReader("{}")))
		if out.Code != http.StatusUnauthorized {
			t.Fatalf("%s %s status=%d body=%s", route.method, route.path, out.Code, out.Body.String())
		}
	}
}

package handler

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"yingce/backend/internal/service"
)

// 仅接收显式配置合同，禁止客户端写入任务状态、结果或服务器路径。
func bindVideoTransferJSON(c *gin.Context, target any) bool {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 32<<10)
	decoder := json.NewDecoder(c.Request.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		fail(c, http.StatusBadRequest, errors.New("视频搬运请求格式无效"))
		return false
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		fail(c, http.StatusBadRequest, errors.New("视频搬运请求格式无效"))
		return false
	}
	return true
}

func RegisterVideoTransferRoutes(r *gin.RouterGroup, svc *service.Service) {
	r.GET("/video-transfers/capabilities", func(c *gin.Context) {
		if _, err := currentUser(c, svc); err != nil {
			failService(c, err)
			return
		}
		ok(c, svc.GetVideoTransferCapabilities(c.Request.Context()))
	})
	r.GET("/video-transfers", func(c *gin.Context) {
		user, err := currentUser(c, svc)
		if err != nil {
			failService(c, err)
			return
		}
		page, size, err := parsePaginationQuery(c, 24)
		if err != nil {
			fail(c, http.StatusBadRequest, err)
			return
		}
		result, err := svc.VideoTransfers(user.ID, page, size, c.Query("q"))
		if err != nil {
			failService(c, err)
			return
		}
		ok(c, result)
	})
	r.POST("/video-transfers", func(c *gin.Context) {
		user, err := currentUser(c, svc)
		if err != nil {
			failService(c, err)
			return
		}
		if !enforceRateLimit(c, "video-transfer-write:"+user.ID, 60, time.Minute) {
			return
		}
		var req service.VideoTransferCreateRequest
		if !bindVideoTransferJSON(c, &req) {
			return
		}
		record, err := svc.CreateVideoTransfer(user.ID, req)
		if err != nil {
			failService(c, err)
			return
		}
		ok(c, gin.H{"record": record})
	})
	r.GET("/video-transfers/:id", func(c *gin.Context) {
		user, err := currentUser(c, svc)
		if err != nil {
			failService(c, err)
			return
		}
		record, err := svc.VideoTransfer(user.ID, c.Param("id"))
		if err != nil {
			failService(c, err)
			return
		}
		c.Header("Cache-Control", "no-store")
		ok(c, gin.H{"record": record})
	})
	r.PATCH("/video-transfers/:id", func(c *gin.Context) {
		user, err := currentUser(c, svc)
		if err != nil {
			failService(c, err)
			return
		}
		if !enforceRateLimit(c, "video-transfer-write:"+user.ID, 60, time.Minute) {
			return
		}
		var req service.VideoTransferPatch
		if !bindVideoTransferJSON(c, &req) {
			return
		}
		record, err := svc.UpdateVideoTransfer(user.ID, c.Param("id"), req)
		if err != nil {
			failService(c, err)
			return
		}
		ok(c, gin.H{"record": record})
	})
	r.POST("/video-transfers/:id/run", func(c *gin.Context) {
		user, err := currentUser(c, svc)
		if err != nil {
			failService(c, err)
			return
		}
		policy, available := loadRuntimePolicy(c, svc)
		if !available || !enforceRateLimit(c, "tasks:"+user.ID, policy.Request.TaskCreatePerMinute, time.Minute) {
			return
		}
		var req service.VideoTransferRunRequest
		if !bindVideoTransferJSON(c, &req) {
			return
		}
		record, err := svc.RunVideoTransfer(c.Request.Context(), user.ID, c.Param("id"), req)
		if err != nil {
			failService(c, err)
			return
		}
		ok(c, gin.H{"record": record})
	})
	r.POST("/video-transfers/:id/cancel", func(c *gin.Context) {
		user, err := currentUser(c, svc)
		if err != nil {
			failService(c, err)
			return
		}
		record, err := svc.CancelVideoTransfer(c.Request.Context(), user.ID, c.Param("id"))
		if err != nil {
			failService(c, err)
			return
		}
		ok(c, gin.H{"record": record})
	})
}

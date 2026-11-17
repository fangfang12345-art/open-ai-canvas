//go:build unix

package app

import (
	"os"
	"os/exec"
	"syscall"
)

func configureYouTubeProcess(cmd *exec.Cmd) {
	// ffmpeg and the JS runtime are children of yt-dlp; cancellation must stop
	// the entire owned process group before its temporary directory is removed.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		if err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL); err == syscall.ESRCH {
			return os.ErrProcessDone
		} else {
			return err
		}
	}
}

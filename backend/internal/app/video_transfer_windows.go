//go:build windows

package app

import (
	"os/exec"
	"time"
)

func configureVideoTransferProcess(cmd *exec.Cmd) { cmd.WaitDelay = 3 * time.Second }

//go:build !unix

package app

import "os/exec"

func configureYouTubeProcess(cmd *exec.Cmd) {}

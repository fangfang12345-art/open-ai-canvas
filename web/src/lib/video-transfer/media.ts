export function readVideoTransferMetadata(file: File, signal: AbortSignal): Promise<{ width: number; height: number; duration: number }> {
    return new Promise((resolve, reject) => {
        if (signal.aborted) return reject(new DOMException("已取消", "AbortError"));
        const video = document.createElement("video");
        const url = URL.createObjectURL(file);
        const cleanup = () => {
            clearTimeout(timeout);
            signal.removeEventListener("abort", abort);
            video.onloadedmetadata = null;
            video.onerror = null;
            video.removeAttribute("src");
            video.load();
            URL.revokeObjectURL(url);
        };
        const fail = (error: Error) => {
            cleanup();
            reject(error);
        };
        const abort = () => fail(new DOMException("已取消", "AbortError"));
        const timeout = setTimeout(() => fail(new Error("读取视频信息超时，请检查视频是否能正常播放")), 15_000);
        signal.addEventListener("abort", abort, { once: true });
        video.preload = "metadata";
        video.onloadedmetadata = () => {
            const { videoWidth: width, videoHeight: height, duration } = video;
            if (!width || !height || !Number.isFinite(duration) || duration <= 0) return fail(new Error("无法读取视频尺寸或时长"));
            cleanup();
            resolve({ width, height, duration });
        };
        video.onerror = () => fail(new Error("浏览器无法读取该视频，请使用 MP4 或 WebM 文件"));
        video.src = url;
    });
}

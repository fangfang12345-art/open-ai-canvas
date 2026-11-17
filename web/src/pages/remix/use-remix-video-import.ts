import { useCallback, useEffect, useRef, useState } from "react";
import { getActiveUserScope } from "@/lib/user-scope";
import { importYouTubeVideo, type YouTubeVideoImport } from "@/services/api/resources";
import { useUserStore } from "@/stores/use-user-store";

export type RemixVideoImportState = {
    url: string;
    setUrl: (value: string) => void;
    busy: boolean;
    error: string;
    onImport: () => Promise<void>;
    onCancel: () => void;
};

export function useRemixVideoImport(canvasId: string, onImported: (video: YouTubeVideoImport, signal: AbortSignal) => Promise<void>): RemixVideoImportState {
    const [url, setUrl] = useState("");
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState("");
    const requestRef = useRef<AbortController | null>(null);
    const userId = useUserStore((state) => state.user?.id);

    useEffect(() => {
        setUrl("");
        setError("");
        setBusy(false);
        return () => {
            requestRef.current?.abort();
            requestRef.current = null;
        };
    }, [canvasId, userId]);

    const onCancel = useCallback(() => {
        requestRef.current?.abort();
        requestRef.current = null;
        setBusy(false);
        setError("");
    }, []);

    const onImport = useCallback(async () => {
        if (requestRef.current) return;
        if (!url.trim()) {
            setError("请先粘贴 YouTube 视频链接");
            return;
        }
        const request = new AbortController();
        const scope = getActiveUserScope();
        requestRef.current = request;
        setBusy(true);
        setError("");
        try {
            const video = await importYouTubeVideo(url.trim(), request.signal);
            if (request.signal.aborted || requestRef.current !== request || scope !== getActiveUserScope()) return;
            await onImported(video, request.signal);
            if (requestRef.current === request) setUrl("");
        } catch (cause) {
            if (!request.signal.aborted && requestRef.current === request && scope === getActiveUserScope()) {
                setError(cause instanceof Error ? cause.message : "视频下载失败，请稍后重试");
            }
        } finally {
            if (requestRef.current === request) {
                requestRef.current = null;
                setBusy(false);
            }
        }
    }, [onImported, url]);

    return { url, setUrl, busy, error, onImport, onCancel };
}

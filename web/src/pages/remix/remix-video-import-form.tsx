import { useId } from "react";
import { canvasThemes } from "@/lib/canvas-theme";
import { useActiveTheme } from "@/stores/canvas/use-canvas-theme-store";
import type { RemixVideoImportState } from "./use-remix-video-import";

export function RemixVideoImportForm({ state, compact = false }: { state: RemixVideoImportState; compact?: boolean }) {
    const theme = canvasThemes[useActiveTheme()];
    const inputId = useId();
    const errorId = `${inputId}-error`;

    return (
        <form
            className="w-full text-left"
            aria-label="从 YouTube 导入视频"
            aria-busy={state.busy}
            data-canvas-no-zoom
            onPointerDown={(event) => event.stopPropagation()}
            onDoubleClick={(event) => event.stopPropagation()}
            onKeyDown={(event) => event.stopPropagation()}
            onKeyUp={(event) => event.stopPropagation()}
            onSubmit={(event) => {
                event.preventDefault();
                event.stopPropagation();
                if (!state.busy) void state.onImport();
            }}
            style={{ color: theme.node.text }}
        >
            <label htmlFor={inputId} className="mb-2 block text-sm font-medium">
                YouTube 视频链接
            </label>
            <div className="flex items-center gap-2">
                <input
                    id={inputId}
                    type="text"
                    inputMode="url"
                    autoComplete="off"
                    spellCheck={false}
                    value={state.url}
                    readOnly={state.busy}
                    onChange={(event) => state.setUrl(event.target.value)}
                    placeholder="https://www.youtube.com/shorts/..."
                    aria-invalid={Boolean(state.error)}
                    aria-describedby={state.error ? errorId : undefined}
                    className={`min-w-0 flex-1 rounded-lg border px-3 text-sm outline-none focus-visible:outline-2 focus-visible:outline-offset-2 ${compact ? "h-10" : "h-12"}`}
                    style={{ background: theme.node.panel, borderColor: state.error ? "var(--color-red-600)" : theme.node.stroke }}
                />
                <button
                    type="submit"
                    disabled={state.busy || !state.url.trim()}
                    className={`shrink-0 rounded-lg px-4 text-sm font-semibold transition-opacity hover:opacity-80 focus-visible:outline-2 focus-visible:outline-offset-4 disabled:cursor-not-allowed disabled:opacity-50 ${compact ? "h-10" : "h-12"}`}
                    style={{ background: "var(--color-red-600)", color: "var(--color-white)" }}
                >
                    {state.busy ? "正在下载…" : "下载并导入"}
                </button>
            </div>
            {state.busy ? (
                <div className="mt-2 flex items-center justify-between gap-3 text-xs" style={{ color: theme.node.muted }}>
                    <span role="status">正在下载原视频，完成后自动加入画布</span>
                    <button type="button" onClick={state.onCancel} className="shrink-0 rounded px-1 py-1 underline underline-offset-4 hover:opacity-70 focus-visible:outline-2 focus-visible:outline-offset-2" style={{ color: theme.node.text }}>
                        取消
                    </button>
                </div>
            ) : null}
            {state.error ? (
                <p id={errorId} role="alert" className="mt-2 break-words text-xs leading-5" style={{ color: "var(--color-red-600)" }}>
                    {state.error}
                </p>
            ) : null}
        </form>
    );
}

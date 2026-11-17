import { RemixIcon } from "@/components/icons/remix-icon";
import { canvasThemes } from "@/lib/canvas-theme";
import { useActiveTheme } from "@/stores/canvas/use-canvas-theme-store";
import { RemixVideoImportForm } from "./remix-video-import-form";
import type { RemixVideoImportState } from "./use-remix-video-import";

export function RemixEmptyState({ videoImport, onAddText }: { videoImport: RemixVideoImportState; onAddText: () => void }) {
    const theme = canvasThemes[useActiveTheme()];
    return (
        <div className="pointer-events-none absolute inset-0 z-20 grid place-items-center px-6 pb-20 pt-24">
            <section className="pointer-events-auto w-full max-w-lg text-center" data-canvas-no-zoom aria-label="二创短视频画布" style={{ color: theme.node.text }}>
                <RemixIcon className="mx-auto mb-5 size-10 opacity-60" />
                <h1 className="text-2xl font-semibold">二创短视频</h1>
                <p className="mt-3 text-sm leading-6" style={{ color: theme.node.muted }}>
                    粘贴 YouTube 链接，把原视频下载到本地画布。
                </p>
                <div className="mt-7">
                    <RemixVideoImportForm state={videoImport} />
                </div>
                <div className="mt-5 flex justify-center">
                    <button
                        type="button"
                        onClick={onAddText}
                        className="rounded-lg border px-5 py-3 text-sm font-medium transition-opacity hover:opacity-70 focus-visible:outline-2 focus-visible:outline-offset-4"
                        style={{ background: theme.toolbar.panel, borderColor: theme.toolbar.border }}
                    >
                        添加改编文本
                    </button>
                </div>
                <p className="mt-5 text-xs leading-5" style={{ color: theme.node.muted }}>
                    也可以直接拖入视频、图片或音频
                </p>
            </section>
        </div>
    );
}

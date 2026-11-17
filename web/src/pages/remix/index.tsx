import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { WorkspaceErrorState } from "@/components/layout/workspace-state";
import { WorkspaceRouteLoader } from "@/components/ui/aceternity/full-screen-loader";
import { remixCanvasPath, resolveRemixCanvas } from "@/lib/canvas/remix-workspace";
import { canvasDraftNavigationState, useCanvasStore } from "@/stores/canvas/use-canvas-store";
import { useUserStore } from "@/stores/use-user-store";

export default function RemixWorkspacePage() {
    const navigate = useNavigate();
    const hydrated = useCanvasStore((state) => state.hydrated);
    const sessionHydrated = useUserStore((state) => state.hydrated);
    const userId = useUserStore((state) => state.user?.id);
    const [error, setError] = useState("");
    const [attempt, setAttempt] = useState(0);

    useEffect(() => {
        if (!hydrated || !sessionHydrated) return;
        let cancelled = false;
        setError("");
        void resolveRemixCanvas()
            .then((id) => {
                if (!cancelled) navigate(remixCanvasPath(id), { replace: true, state: canvasDraftNavigationState(id) });
            })
            .catch((cause) => {
                if (!cancelled) setError(cause instanceof Error ? cause.message : "无法打开二创短视频画布");
            });
        return () => {
            cancelled = true;
        };
    }, [attempt, hydrated, navigate, sessionHydrated, userId]);

    if (error) return <WorkspaceErrorState title="无法打开二创短视频" description={error} onRetry={() => setAttempt((value) => value + 1)} />;
    return <WorkspaceRouteLoader label="正在打开二创短视频画布" />;
}

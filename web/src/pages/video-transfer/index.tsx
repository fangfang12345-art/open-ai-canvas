import { useDeferredValue, useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { App, Button, Input, Table } from "antd";
import { useNavigate } from "react-router";
import { PageHeader, PaginationBar, TableSurface, WorkspacePage } from "@/components/layout/workspace-page";
import { WorkspaceErrorState, WorkspaceForbiddenState, WorkspaceNoResultsState, WorkspaceState } from "@/components/layout/workspace-state";
import { AppModal } from "@/components/ui/product/app-modal";
import { RelativeTime } from "@/components/ui/product/relative-time";
import { VIDEO_TRANSFER_BRANCHES, VIDEO_TRANSFER_BRANCH_LABELS, VIDEO_TRANSFER_STATUS_LABELS, isVideoTransferActive, videoTransferKeys } from "@/lib/video-transfer/workflow";
import { createVideoTransfer, listVideoTransfers, type VideoTransferBranch, type VideoTransferRecord } from "@/services/api/video-transfer";
import { useUserStore } from "@/stores/use-user-store";
import "./video-transfer.css";

export default function VideoTransferListPage() {
    const navigate = useNavigate();
    const queryClient = useQueryClient();
    const { message } = App.useApp();
    const userId = useUserStore((state) => state.user?.id);
    const [keyword, setKeyword] = useState("");
    const search = useDeferredValue(keyword.trim());
    const [page, setPage] = useState(1);
    const [pageSize, setPageSize] = useState(20);
    const [createOpen, setCreateOpen] = useState(false);
    const [title, setTitle] = useState("");
    const [branch, setBranch] = useState<VideoTransferBranch>("fixed-r16");
    const [creating, setCreating] = useState(false);
    const requestRef = useRef<AbortController | null>(null);

    useEffect(() => setPage(1), [search, pageSize, userId]);
    useEffect(
        () => () => {
            requestRef.current?.abort();
        },
        [userId],
    );

    const query = useQuery({
        queryKey: videoTransferKeys.list(userId || "", { page, pageSize, q: search }),
        enabled: Boolean(userId),
        queryFn: ({ signal }) => listVideoTransfers({ page, pageSize, q: search }, signal),
        refetchInterval: (query) => (query.state.data?.records.some(isVideoTransferActive) ? 3000 : false),
    });

    async function createRecord() {
        if (!userId || requestRef.current || !title.trim()) return;
        const request = new AbortController();
        requestRef.current = request;
        setCreating(true);
        try {
            const record = await createVideoTransfer({ title: title.trim(), selectedBranch: branch }, request.signal);
            if (request.signal.aborted || useUserStore.getState().user?.id !== userId) return;
            queryClient.setQueryData(videoTransferKeys.record(userId, record.id), record);
            void queryClient.invalidateQueries({ queryKey: videoTransferKeys.all(userId) });
            navigate(`/video-transfer/${encodeURIComponent(record.id)}`);
        } catch (error) {
            if (!request.signal.aborted && useUserStore.getState().user?.id === userId) message.error(error instanceof Error ? error.message : "创建失败");
        } finally {
            if (requestRef.current === request) {
                requestRef.current = null;
                setCreating(false);
            }
        }
    }

    if (!userId)
        return (
            <WorkspacePage>
                <WorkspaceForbiddenState title="登录后使用视频搬运" description="登录后可保存视频、处理设置与处理结果。" />
            </WorkspacePage>
        );

    return (
        <WorkspacePage className="video-transfer-list">
            <PageHeader
                title="视频搬运"
                description="上传原视频，选择固定或移动水印算法，在画布中查看处理进度与完整结果。"
                actions={
                    <Button
                        type="primary"
                        onClick={() => {
                            setTitle("");
                            setCreateOpen(true);
                        }}
                    >
                        新建搬运
                    </Button>
                }
            />
            <div className="video-transfer-list-toolbar">
                <Input.Search allowClear value={keyword} aria-label="搜索搬运记录" placeholder="搜索记录名称" onChange={(event) => setKeyword(event.target.value)} />
                <span>{query.data ? `共 ${query.data.total} 条记录` : "视频与处理结果独立保存"}</span>
            </div>
            {query.isError ? (
                <WorkspaceErrorState description={query.error instanceof Error ? query.error.message : "搬运记录加载失败"} onRetry={() => void query.refetch()} />
            ) : (
                <TableSurface>
                    <Table<VideoTransferRecord>
                        rowKey="id"
                        loading={query.isLoading}
                        pagination={false}
                        dataSource={query.data?.records || []}
                        scroll={{ x: 820 }}
                        locale={{
                            emptyText: search ? (
                                <WorkspaceNoResultsState compact query={search} onClear={() => setKeyword("")} />
                            ) : (
                                <WorkspaceState
                                    compact
                                    title="还没有搬运记录"
                                    description="新建一条记录，上传原视频后选择处理算法。"
                                    action={
                                        <Button type="primary" onClick={() => setCreateOpen(true)}>
                                            新建搬运
                                        </Button>
                                    }
                                />
                            ),
                        }}
                        columns={[
                            {
                                title: "记录",
                                key: "title",
                                render: (_, record) => (
                                    <div className="video-transfer-record-name">
                                        <button type="button" onClick={() => navigate(`/video-transfer/${encodeURIComponent(record.id)}`)}>
                                            {record.title}
                                        </button>
                                        <span>{record.sourceName || "尚未上传原视频"}</span>
                                    </div>
                                ),
                            },
                            { title: "当前算法", width: 156, render: (_, record) => VIDEO_TRANSFER_BRANCH_LABELS[record.selectedBranch] },
                            {
                                title: "处理状态",
                                width: 200,
                                render: (_, record) => (
                                    <div className="video-transfer-table-status">
                                        {VIDEO_TRANSFER_BRANCHES.map((branch) => (
                                            <span key={branch}>
                                                <span className={`video-transfer-status-dot is-${record.branches[branch].status}`} />
                                                <span>{branch === "fixed-r16" ? "R16" : "V6"}</span>
                                                {VIDEO_TRANSFER_STATUS_LABELS[record.branches[branch].status]}
                                            </span>
                                        ))}
                                    </div>
                                ),
                            },
                            { title: "更新时间", width: 150, render: (_, record) => <RelativeTime value={record.updatedAt} /> },
                            { title: "操作", key: "actions", width: 124, fixed: "right", render: (_, record) => <Button onClick={() => navigate(`/video-transfer/${encodeURIComponent(record.id)}`)}>进入画布</Button> },
                        ]}
                    />
                </TableSurface>
            )}
            {query.data ? (
                <PaginationBar
                    current={page}
                    pageSize={pageSize}
                    total={query.data.total}
                    itemLabel="条"
                    onChange={(next, size) => {
                        setPage(next);
                        setPageSize(size);
                    }}
                />
            ) : null}
            <AppModal
                title="新建视频搬运"
                open={createOpen}
                confirmLoading={creating}
                okText="创建并进入画布"
                okButtonProps={{ disabled: !title.trim() }}
                cancelButtonProps={{ disabled: creating }}
                maskClosable={!creating}
                closable={!creating}
                onCancel={() => setCreateOpen(false)}
                onOk={() => void createRecord()}
            >
                <div className="video-transfer-create-form">
                    <label htmlFor="video-transfer-title">记录名称</label>
                    <Input id="video-transfer-title" value={title} maxLength={120} autoFocus placeholder="例如：产品介绍片" disabled={creating} onChange={(event) => setTitle(event.target.value)} onPressEnter={() => void createRecord()} />
                    <span>默认处理算法</span>
                    <div className="video-transfer-branch-switch">
                        {VIDEO_TRANSFER_BRANCHES.map((value) => (
                            <button key={value} type="button" aria-pressed={branch === value} disabled={creating} onClick={() => setBranch(value)}>
                                {VIDEO_TRANSFER_BRANCH_LABELS[value]}
                            </button>
                        ))}
                    </div>
                    <p className="video-transfer-hint">可先创建记录，稍后在画布里上传原视频或切换算法。</p>
                </div>
            </AppModal>
        </WorkspacePage>
    );
}

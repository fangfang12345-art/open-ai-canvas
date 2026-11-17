export function VideoTransferIcon({ className, strokeWidth = 1.8 }: { className?: string; strokeWidth?: number }) {
    return (
        <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <rect x="3" y="4" width="13" height="13" rx="3" />
            <path d="m8 8 4 2.5L8 13V8Zm1 13h9a3 3 0 0 0 3-3v-7m-3 3 3-3 3 3" />
        </svg>
    );
}

export function RemixIcon({ className, strokeWidth = 1.8 }: { className?: string; strokeWidth?: number }) {
    return (
        <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <rect x="3" y="4" width="18" height="16" rx="3" />
            <path d="M3 9h18M8 4l3 5m4-5 3 5m-8 3v5l5-2.5z" />
        </svg>
    );
}

'use client'

import { useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { X, ExternalLink, MapPin, Clock, Code2, Music, Video, FileText, File, Image } from 'lucide-react'
import { QueryResult } from '@/app/page'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'

interface Props {
  chunk:   QueryResult
  onClose: () => void
}

const TYPE_COLORS: Record<string, string> = {
  image:    '#60a5fa',
  video:    '#a78bfa',
  audio:    '#34d399',
  document: '#fbbf24',
  code:     '#f87171',
}

function TypeIcon({ type }: { type: string }) {
  const c = TYPE_COLORS[type] || '#505068'
  const s = { color: c }
  if (type === 'image')    return <Image    size={11} style={s} />
  if (type === 'video')    return <Video    size={11} style={s} />
  if (type === 'audio')    return <Music    size={11} style={s} />
  if (type === 'document') return <FileText size={11} style={s} />
  if (type === 'code')     return <Code2    size={11} style={s} />
  return <File size={11} style={s} />
}

export default function PreviewPane({ chunk, onClose }: Props) {
  const filename = chunk.source_path.split(/[\\/]/).pop() || chunk.source_path
  const color    = TYPE_COLORS[chunk.file_type] || '#505068'

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: '#050507' }}>

      {/* Header */}
      <div style={{
        flexShrink: 0, padding: '0 14px', height: 44,
        borderBottom: '1px solid #1a1a2e',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        gap: 8,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7, minWidth: 0 }}>
          <TypeIcon type={chunk.file_type} />
          <span style={{ fontSize: 12, color: '#a0a0b8', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {filename}
          </span>
        </div>
        <div style={{ display: 'flex', gap: 2, flexShrink: 0 }}>
          <button
            title="Open file location"
            style={{
              width: 26, height: 26, borderRadius: 6, border: 'none', background: 'transparent',
              cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: '#383850', transition: 'color 0.12s',
            }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.color = '#a0a0b8' }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.color = '#383850' }}
          >
            <ExternalLink size={11} />
          </button>
          <button
            onClick={onClose}
            style={{
              width: 26, height: 26, borderRadius: 6, border: 'none', background: 'transparent',
              cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: '#383850', transition: 'color 0.12s',
            }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.color = '#a0a0b8' }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.color = '#383850' }}
          >
            <X size={12} />
          </button>
        </div>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '14px' }}>
        <motion.div
          key={chunk.chunk_id}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.18 }}
        >
          {/* Type badge */}
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 5,
            padding: '3px 8px 3px 6px', borderRadius: 6, marginBottom: 14,
            background: `${color}10`,
            border: `1px solid ${color}25`,
          }}>
            <TypeIcon type={chunk.file_type} />
            <span style={{ fontSize: 10, color, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
              {chunk.file_type}
            </span>
          </div>

          {chunk.file_type === 'image'    && <ImagePreview    chunk={chunk} />}
          {chunk.file_type === 'document' && <TextPreview     chunk={chunk} />}
          {chunk.file_type === 'code'     && <CodePreview     chunk={chunk} />}
          {chunk.file_type === 'audio'    && <AudioPreview    chunk={chunk} />}
          {chunk.file_type === 'video'    && <VideoPreview    chunk={chunk} />}
        </motion.div>
      </div>

      {/* Metadata footer */}
      <div style={{
        flexShrink: 0, borderTop: '1px solid #1a1a2e',
        padding: '10px 14px',
        background: 'rgba(5,5,7,0.8)',
      }}>
        <MetadataTable metadata={chunk.metadata} sourcePath={chunk.source_path} />
      </div>
    </div>
  )
}

// ── Image ──────────────────────────────────────────────────────────────────────

function ImagePreview({ chunk }: { chunk: QueryResult }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ borderRadius: 10, overflow: 'hidden', border: '1px solid #1a1a2e' }}>
        <img
          src={`${API}/chunk/${chunk.chunk_id}/raw`}
          alt=""
          style={{ width: '100%', display: 'block' }}
          loading="lazy"
        />
      </div>
      {chunk.metadata?.gps && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <MapPin size={10} color="#60a5fa" />
          <span style={{ fontSize: 10, color: '#505068', fontFamily: 'JetBrains Mono, monospace' }}>
            {(chunk.metadata.gps as any).lat?.toFixed(4)}, {(chunk.metadata.gps as any).lng?.toFixed(4)}
          </span>
        </div>
      )}
    </div>
  )
}

// ── Document ──────────────────────────────────────────────────────────────────

function TextPreview({ chunk }: { chunk: QueryResult }) {
  if (!chunk.text) return <Empty />
  return (
    <p style={{ fontSize: 13, color: '#a0a0b8', lineHeight: 1.75, whiteSpace: 'pre-wrap' }}>
      {chunk.text}
    </p>
  )
}

// ── Code ──────────────────────────────────────────────────────────────────────

function CodePreview({ chunk }: { chunk: QueryResult }) {
  if (!chunk.text) return <Empty />
  const language   = chunk.metadata?.language   as string | undefined
  const symbolName = chunk.metadata?.symbol_name as string | undefined
  const symbolType = chunk.metadata?.symbol_type as string | undefined
  const startLine  = chunk.metadata?.start_line  as number | undefined
  const endLine    = chunk.metadata?.end_line    as number | undefined

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {(symbolName || language) && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
          {language && (
            <span style={{
              fontSize: 9, padding: '2px 6px', borderRadius: 4,
              background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.2)',
              color: '#f87171', letterSpacing: '0.1em', textTransform: 'uppercase',
            }}>{language}</span>
          )}
          {symbolType && symbolName && (
            <span style={{ fontSize: 11, color: '#505068' }}>
              {symbolType}{' '}
              <span style={{ color: '#7c6af7', fontFamily: 'JetBrains Mono, monospace' }}>{symbolName}</span>
            </span>
          )}
          {startLine != null && (
            <span style={{ fontSize: 10, color: '#383850', fontFamily: 'JetBrains Mono, monospace' }}>
              L{startLine}{endLine != null && endLine !== startLine ? `–${endLine}` : ''}
            </span>
          )}
        </div>
      )}
      <pre style={{
        fontSize: 11, color: '#a0a0b8',
        fontFamily: 'JetBrains Mono, monospace',
        background: 'rgba(10,10,15,0.8)',
        border: '1px solid #1a1a2e',
        borderRadius: 10, padding: '12px 14px',
        overflowX: 'auto', lineHeight: 1.6,
        whiteSpace: 'pre',
      }}>
        <code>{chunk.text}</code>
      </pre>
    </div>
  )
}

// ── Audio ─────────────────────────────────────────────────────────────────────

function AudioPreview({ chunk }: { chunk: QueryResult }) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const start    = chunk.metadata?.start_seconds as number | undefined
  const end      = chunk.metadata?.end_seconds   as number | undefined

  useEffect(() => {
    if (audioRef.current && start != null && start > 0) {
      audioRef.current.currentTime = start
    }
  }, [chunk.chunk_id, start])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <audio
        ref={audioRef}
        controls
        style={{ width: '100%', borderRadius: 8 }}
        src={`${API}/chunk/${chunk.chunk_id}/raw`}
      />
      {start != null && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <Clock size={10} color="#34d399" />
          <span style={{ fontSize: 10, color: '#505068', fontFamily: 'JetBrains Mono, monospace' }}>
            {fmtTime(start)} – {end != null ? fmtTime(end) : '…'}
            {chunk.metadata?.language ? ` · ${chunk.metadata.language}` : ''}
          </span>
        </div>
      )}
      {chunk.text && (
        <div>
          <div style={{ fontSize: 9, color: '#383850', letterSpacing: '0.15em', textTransform: 'uppercase', marginBottom: 8 }}>Transcript</div>
          <p style={{ fontSize: 12, color: '#a0a0b8', lineHeight: 1.7 }}>{chunk.text}</p>
        </div>
      )}
    </div>
  )
}

// ── Video ─────────────────────────────────────────────────────────────────────

function VideoPreview({ chunk }: { chunk: QueryResult }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const start    = chunk.metadata?.start_seconds     as number | undefined
  const end      = chunk.metadata?.end_seconds       as number | undefined
  const ts       = chunk.metadata?.timestamp_seconds as number | undefined

  useEffect(() => {
    const seek = start ?? ts
    if (videoRef.current && seek != null && seek > 0) {
      videoRef.current.currentTime = seek
    }
  }, [chunk.chunk_id, start, ts])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <video
        ref={videoRef}
        controls
        style={{ width: '100%', borderRadius: 10, border: '1px solid #1a1a2e' }}
        src={`${API}/chunk/${chunk.chunk_id}/raw`}
      />
      {(start != null || ts != null) && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <Clock size={10} color="#a78bfa" />
          <span style={{ fontSize: 10, color: '#505068', fontFamily: 'JetBrains Mono, monospace' }}>
            {start != null
              ? `${fmtTime(start)} – ${end != null ? fmtTime(end) : '…'}`
              : `Keyframe @ ${fmtTime(ts!)}`}
            {chunk.metadata?.language ? ` · ${chunk.metadata.language}` : ''}
          </span>
        </div>
      )}
      {chunk.text && (
        <div>
          <div style={{ fontSize: 9, color: '#383850', letterSpacing: '0.15em', textTransform: 'uppercase', marginBottom: 8 }}>Transcript</div>
          <p style={{ fontSize: 12, color: '#a0a0b8', lineHeight: 1.7 }}>{chunk.text}</p>
        </div>
      )}
    </div>
  )
}

// ── Metadata ──────────────────────────────────────────────────────────────────

function MetadataTable({ metadata, sourcePath }: { metadata: Record<string, unknown>; sourcePath: string }) {
  const rows: [string, string][] = []

  const filename = sourcePath.split(/[\\/]/).pop() || sourcePath
  const dir      = sourcePath.replace(/[\\/][^\\/]*$/, '') || sourcePath

  rows.push(['File', filename])
  if (dir !== filename) rows.push(['Path', dir])

  if (metadata?.created_at)
    rows.push(['Created', new Date(metadata.created_at as string).toLocaleDateString()])
  if (metadata?.exif_datetime)
    rows.push(['Taken', metadata.exif_datetime as string])
  if (metadata?.dimensions) {
    const d = metadata.dimensions as { w: number; h: number }
    rows.push(['Size', `${d.w} × ${d.h}`])
  }
  if (metadata?.device)
    rows.push(['Device', metadata.device as string])
  if (metadata?.duration_seconds) {
    const d = metadata.duration_seconds as number
    rows.push(['Duration', `${Math.floor(d / 60)}:${Math.floor(d % 60).toString().padStart(2, '0')}`])
  }
  if (metadata?.language)
    rows.push(['Language', metadata.language as string])
  if (metadata?.symbol_name && metadata?.symbol_type)
    rows.push(['Symbol', `${metadata.symbol_type} ${metadata.symbol_name}`])
  if (metadata?.start_line != null) {
    const range = metadata.end_line != null
      ? `L${metadata.start_line}–${metadata.end_line}`
      : `L${metadata.start_line}`
    rows.push(['Lines', range])
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      {rows.map(([k, v]) => (
        <div key={k} style={{ display: 'flex', gap: 10, fontSize: 10 }}>
          <span style={{ color: '#383850', width: 52, flexShrink: 0 }}>{k}</span>
          <span style={{ color: '#505068', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontFamily: 'JetBrains Mono, monospace', fontSize: 10 }}>{v}</span>
        </div>
      ))}
    </div>
  )
}

function Empty() {
  return <p style={{ fontSize: 12, color: '#383850' }}>No content available.</p>
}

function fmtTime(s: number) {
  return `${Math.floor(s / 60)}:${Math.floor(s % 60).toString().padStart(2, '0')}`
}

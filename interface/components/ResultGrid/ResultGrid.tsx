'use client'

import { motion } from 'framer-motion'
import { FileText, Code2, Music, Video, File, Play, Clock, MapPin, Brain, Trash2 } from 'lucide-react'
import { QueryResult } from '@/app/page'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'

interface Props {
  results:  QueryResult[]
  onSelect: (chunk: QueryResult) => void
  selected: QueryResult | null
  onDelete?: (sourcePath: string) => void
  deleting?: string | null  // source_path currently being deleted
}

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.035 } },
}
const item = {
  hidden: { opacity: 0, y: 8 },
  show:   { opacity: 1, y: 0, transition: { duration: 0.22, ease: [0.16, 1, 0.3, 1] as any } },
}

export default function ResultGrid({ results, onSelect, selected, onDelete, deleting }: Props) {
  if (!results.length) {
    return (
      <div style={{ padding: '20px 0', textAlign: 'center' }}>
        <p style={{ fontSize: 12, color: '#383850' }}>No matching memories.</p>
      </div>
    )
  }

  const images       = results.filter((r) => r.file_type === 'image')
  const videos       = results.filter((r) => r.file_type === 'video')
  const observations = results.filter((r) => r.file_type === 'observation')
  const nonVisual    = results.filter((r) => r.file_type !== 'image' && r.file_type !== 'video' && r.file_type !== 'observation')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* Images — masonry-style grid */}
      {images.length > 0 && (
        <section>
          <SectionLabel label="Photos" count={images.length} color="#60a5fa" />
          <motion.div
            variants={container} initial="hidden" animate="show"
            style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))', gap: 6 }}
          >
            {images.map((r) => (
              <motion.button
                key={r.chunk_id}
                variants={item}
                whileHover={{ scale: 1.03, zIndex: 2 }}
                whileTap={{ scale: 0.97 }}
                onClick={() => onSelect(r)}
                style={{
                  position: 'relative', aspectRatio: '1',
                  borderRadius: 10, overflow: 'hidden',
                  border: `1px solid ${selected?.chunk_id === r.chunk_id ? 'rgba(124,106,247,0.6)' : '#1a1a2e'}`,
                  background: '#0a0a0f', cursor: 'pointer',
                  boxShadow: selected?.chunk_id === r.chunk_id ? '0 0 0 2px rgba(124,106,247,0.3)' : 'none',
                  transition: 'border-color 0.15s, box-shadow 0.15s',
                }}
              >
                {r.thumbnail_url ? (
                  <img
                    src={`${API}${r.thumbnail_url}`}
                    alt=""
                    style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                    loading="lazy"
                  />
                ) : (
                  <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#0f0f18' }}>
                    <File size={18} color="#252540" />
                  </div>
                )}
                {/* Hover overlay */}
                <div style={{
                  position: 'absolute', inset: 0,
                  background: 'linear-gradient(to top, rgba(0,0,0,0.7) 0%, transparent 50%)',
                  opacity: 0, transition: 'opacity 0.15s',
                }}
                  onMouseEnter={(e) => { (e.currentTarget as HTMLDivElement).style.opacity = '1' }}
                  onMouseLeave={(e) => { (e.currentTarget as HTMLDivElement).style.opacity = '0' }}
                />
                {/* Score */}
                <MatchBadge score={r.score} />
                {/* GPS indicator */}
                {r.metadata?.gps && (
                  <div style={{ position: 'absolute', bottom: 4, left: 4 }}>
                    <MapPin size={8} color="rgba(255,255,255,0.6)" />
                  </div>
                )}
                {/* Delete */}
                {onDelete && (
                  <DeleteOverlayBtn
                    isDeleting={deleting === r.source_path}
                    onClick={(e) => { e.stopPropagation(); onDelete(r.source_path) }}
                  />
                )}
              </motion.button>
            ))}
          </motion.div>
        </section>
      )}

      {/* Videos */}
      {videos.length > 0 && (
        <section>
          <SectionLabel label="Video" count={videos.length} color="#a78bfa" />
          <motion.div
            variants={container} initial="hidden" animate="show"
            style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 6 }}
          >
            {videos.map((r) => {
              const ts = (r.metadata?.timestamp_seconds ?? r.metadata?.start_seconds) as number | undefined
              return (
                <motion.button
                  key={r.chunk_id}
                  variants={item}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={() => onSelect(r)}
                  style={{
                    position: 'relative', aspectRatio: '16/9',
                    borderRadius: 10, overflow: 'hidden',
                    border: `1px solid ${selected?.chunk_id === r.chunk_id ? 'rgba(124,106,247,0.6)' : '#1a1a2e'}`,
                    background: '#0a0a0f', cursor: 'pointer',
                    boxShadow: selected?.chunk_id === r.chunk_id ? '0 0 0 2px rgba(124,106,247,0.3)' : 'none',
                    transition: 'border-color 0.15s, box-shadow 0.15s',
                  }}
                >
                  {r.thumbnail_url ? (
                    <img src={`${API}${r.thumbnail_url}`} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} loading="lazy" />
                  ) : (
                    <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#0f0f18' }}>
                      <Video size={20} color="#252540" />
                    </div>
                  )}
                  {/* Play */}
                  <div style={{
                    position: 'absolute', inset: 0,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    background: 'rgba(0,0,0,0.2)',
                  }}>
                    <div style={{
                      width: 28, height: 28, borderRadius: '50%',
                      background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(4px)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      border: '1px solid rgba(255,255,255,0.1)',
                    }}>
                      <Play size={10} color="white" fill="white" style={{ marginLeft: 1 }} />
                    </div>
                  </div>
                  {ts != null && (
                    <div style={{
                      position: 'absolute', bottom: 5, left: 5,
                      display: 'flex', alignItems: 'center', gap: 3,
                      padding: '2px 5px', borderRadius: 4,
                      background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(4px)',
                    }}>
                      <Clock size={8} color="rgba(255,255,255,0.6)" />
                      <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.7)', fontFamily: 'JetBrains Mono, monospace' }}>
                        {Math.floor(ts / 60)}:{Math.floor(ts % 60).toString().padStart(2, '0')}
                      </span>
                    </div>
                  )}
                  <MatchBadge score={r.score} />
                  {onDelete && (
                    <DeleteOverlayBtn
                      isDeleting={deleting === r.source_path}
                      onClick={(e) => { e.stopPropagation(); onDelete(r.source_path) }}
                    />
                  )}
                </motion.button>
              )
            })}
          </motion.div>
        </section>
      )}

      {/* Agent observations */}
      {observations.length > 0 && (
        <section>
          <SectionLabel label="Agent Memory" count={observations.length} color="#a78bfa" />
          <motion.div variants={container} initial="hidden" animate="show" style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {observations.map((r) => {
              const isSelected = selected?.chunk_id === r.chunk_id
              const agentName  = (r.metadata?.agent_name as string) || 'Agent'
              return (
                <motion.button
                  key={r.chunk_id}
                  variants={item}
                  whileTap={{ scale: 0.998 }}
                  onClick={() => onSelect(r)}
                  style={{
                    display: 'flex', alignItems: 'flex-start', gap: 12,
                    padding: '10px 12px', borderRadius: 10, textAlign: 'left',
                    border: `1px solid ${isSelected ? 'rgba(167,139,250,0.4)' : 'rgba(124,106,247,0.12)'}`,
                    background: isSelected ? 'rgba(124,106,247,0.08)' : 'rgba(124,106,247,0.03)',
                    cursor: 'pointer', width: '100%',
                    transition: 'all 0.15s',
                  }}
                >
                  {/* Brain icon */}
                  <div style={{
                    width: 30, height: 30, borderRadius: 8, flexShrink: 0,
                    background: 'rgba(124,106,247,0.1)', border: '1px solid rgba(124,106,247,0.2)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}>
                    <Brain size={13} color="#a78bfa" strokeWidth={1.4} />
                  </div>

                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
                      {/* Agent badge */}
                      <span style={{
                        fontSize: 9, padding: '2px 6px', borderRadius: 4,
                        background: 'rgba(124,106,247,0.12)', border: '1px solid rgba(124,106,247,0.2)',
                        color: '#a78bfa', letterSpacing: '0.06em', textTransform: 'uppercase',
                        fontWeight: 500,
                      }}>
                        {agentName}
                      </span>
                      {r.metadata?.source && r.metadata.source !== 'agent' && (
                        <span style={{ fontSize: 10, color: '#505068' }}>via {r.metadata.source as string}</span>
                      )}
                    </div>
                    {r.text && (
                      <p style={{
                        fontSize: 12, color: '#c4b5fd', lineHeight: 1.5,
                        overflow: 'hidden', display: '-webkit-box',
                        WebkitLineClamp: 3, WebkitBoxOrient: 'vertical',
                      }}>
                        {r.text}
                      </p>
                    )}
                  </div>

                  {r.score > 0 && (
                    <div style={{ flexShrink: 0, display: 'flex', alignItems: 'center', gap: 4 }}>
                      <div style={{ width: 28, height: 3, borderRadius: 2, overflow: 'hidden', background: '#1a1a2e' }}>
                        <div style={{
                          height: '100%', borderRadius: 2,
                          width: `${Math.round(r.score * 100)}%`,
                          background: `hsl(${140 + (1 - r.score) * (-100)}, 60%, 55%)`,
                        }} />
                      </div>
                      <span style={{ fontSize: 10, color: '#383850', fontFamily: 'JetBrains Mono, monospace' }}>
                        {Math.round(r.score * 100)}
                      </span>
                    </div>
                  )}
                  {onDelete && (
                    <button
                      onClick={(e) => { e.stopPropagation(); onDelete(r.source_path) }}
                      title="Delete from index"
                      style={{
                        flexShrink: 0, width: 24, height: 24, borderRadius: 6,
                        border: '1px solid transparent', background: 'transparent',
                        cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
                        color: deleting === r.source_path ? '#f87171' : '#383850',
                        transition: 'color 0.12s, border-color 0.12s',
                      }}
                      onMouseEnter={(e) => {
                        const b = e.currentTarget as HTMLButtonElement
                        b.style.color = '#f87171'
                        b.style.borderColor = 'rgba(248,113,113,0.2)'
                      }}
                      onMouseLeave={(e) => {
                        if (deleting !== r.source_path) {
                          const b = e.currentTarget as HTMLButtonElement
                          b.style.color = '#383850'
                          b.style.borderColor = 'transparent'
                        }
                      }}
                    >
                      <Trash2 size={11} />
                    </button>
                  )}
                </motion.button>
              )
            })}
          </motion.div>
        </section>
      )}

      {/* Non-visual files */}
      {nonVisual.length > 0 && (
        <section>
          {(images.length > 0 || videos.length > 0) && (
            <SectionLabel label="Files" count={nonVisual.length} color="#fbbf24" />
          )}
          <motion.div variants={container} initial="hidden" animate="show" style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {nonVisual.map((r) => {
              const isSelected = selected?.chunk_id === r.chunk_id
              return (
                <motion.button
                  key={r.chunk_id}
                  variants={item}
                  whileTap={{ scale: 0.998 }}
                  onClick={() => onSelect(r)}
                  style={{
                    display: 'flex', alignItems: 'flex-start', gap: 12,
                    padding: '10px 12px', borderRadius: 10, textAlign: 'left',
                    border: `1px solid ${isSelected ? 'rgba(124,106,247,0.35)' : '#1a1a2e'}`,
                    background: isSelected ? 'rgba(124,106,247,0.05)' : 'rgba(10,10,15,0.6)',
                    cursor: 'pointer', width: '100%',
                    boxShadow: isSelected ? '0 0 0 1px rgba(124,106,247,0.15)' : 'none',
                    transition: 'all 0.15s',
                  }}
                  onMouseEnter={(e) => {
                    if (!isSelected) {
                      (e.currentTarget as HTMLButtonElement).style.borderColor = '#252540'
                      ;(e.currentTarget as HTMLButtonElement).style.background = 'rgba(14,14,22,0.8)'
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!isSelected) {
                      (e.currentTarget as HTMLButtonElement).style.borderColor = '#1a1a2e'
                      ;(e.currentTarget as HTMLButtonElement).style.background = 'rgba(10,10,15,0.6)'
                    }
                  }}
                >
                  {/* Icon */}
                  <div style={{
                    width: 30, height: 30, borderRadius: 8, flexShrink: 0,
                    background: 'rgba(15,15,24,0.8)',
                    border: '1px solid #1a1a2e',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}>
                    <FileTypeIcon type={r.file_type} />
                  </div>

                  {/* Content */}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3, flexWrap: 'wrap' }}>
                      <span style={{ fontSize: 12, color: '#e8e8f0', fontWeight: 500 }}>
                        {r.source_path.split(/[\\/]/).pop()}
                      </span>
                      {r.file_type === 'code' && r.metadata?.language && (
                        <span style={{
                          fontSize: 9, padding: '1px 6px', borderRadius: 4,
                          background: 'rgba(15,15,24,0.8)', border: '1px solid #1a1a2e',
                          color: '#505068', letterSpacing: '0.08em', textTransform: 'uppercase',
                        }}>
                          {r.metadata.language as string}
                        </span>
                      )}
                      {r.file_type === 'code' && r.metadata?.symbol_name && (
                        <span style={{ fontSize: 11, color: '#7c6af7', fontFamily: 'JetBrains Mono, monospace' }}>
                          {r.metadata.symbol_name as string}
                        </span>
                      )}
                    </div>
                    {r.text && (
                      <p style={{
                        fontSize: 11, color: '#505068', lineHeight: 1.5,
                        overflow: 'hidden', display: '-webkit-box',
                        WebkitLineClamp: 2, WebkitBoxOrient: 'vertical',
                        fontFamily: r.file_type === 'code' ? 'JetBrains Mono, monospace' : 'inherit',
                      }}>
                        {r.text.slice(0, 200)}
                      </p>
                    )}
                  </div>

                  {/* Score — only shown for vector search results */}
                  {r.score > 0 && (
                    <div style={{ flexShrink: 0, display: 'flex', alignItems: 'center', gap: 4 }}>
                      <div style={{
                        width: 28, height: 3, borderRadius: 2, overflow: 'hidden',
                        background: '#1a1a2e',
                      }}>
                        <div style={{
                          height: '100%', borderRadius: 2,
                          width: `${Math.round(r.score * 100)}%`,
                          background: `hsl(${140 + (1 - r.score) * (-100)}, 60%, 55%)`,
                        }} />
                      </div>
                      <span style={{ fontSize: 10, color: '#383850', fontFamily: 'JetBrains Mono, monospace' }}>
                        {Math.round(r.score * 100)}
                      </span>
                    </div>
                  )}
                  {/* Inline delete for list rows */}
                  {onDelete && (
                    <button
                      onClick={(e) => { e.stopPropagation(); onDelete(r.source_path) }}
                      title="Delete from index"
                      style={{
                        flexShrink: 0, width: 24, height: 24, borderRadius: 6,
                        border: '1px solid transparent', background: 'transparent',
                        cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
                        color: deleting === r.source_path ? '#f87171' : '#383850',
                        transition: 'color 0.12s, border-color 0.12s',
                      }}
                      onMouseEnter={(e) => {
                        const b = e.currentTarget as HTMLButtonElement
                        b.style.color = '#f87171'
                        b.style.borderColor = 'rgba(248,113,113,0.2)'
                      }}
                      onMouseLeave={(e) => {
                        if (deleting !== r.source_path) {
                          const b = e.currentTarget as HTMLButtonElement
                          b.style.color = '#383850'
                          b.style.borderColor = 'transparent'
                        }
                      }}
                    >
                      <Trash2 size={11} />
                    </button>
                  )}
                </motion.button>
              )
            })}
          </motion.div>
        </section>
      )}
    </div>
  )
}

function SectionLabel({ label, count, color }: { label: string; count: number; color: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
      <div style={{ width: 3, height: 10, borderRadius: 2, background: color, opacity: 0.7 }} />
      <span style={{ fontSize: 10, color: '#505068', letterSpacing: '0.15em', textTransform: 'uppercase' }}>{label}</span>
      <span style={{ fontSize: 10, color: '#252540', fontFamily: 'JetBrains Mono, monospace' }}>{count}</span>
    </div>
  )
}

function MatchBadge({ score }: { score: number }) {
  if (score < 0.5) return null
  return (
    <div style={{
      position: 'absolute', top: 5, right: 5,
      padding: '1px 5px', borderRadius: 4,
      background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(4px)',
      fontSize: 9, color: 'rgba(255,255,255,0.6)',
      fontFamily: 'JetBrains Mono, monospace',
    }}>
      {Math.round(score * 100)}%
    </div>
  )
}

function FileTypeIcon({ type }: { type: string }) {
  const style = { color: '#383850' }
  if (type === 'document')    return <FileText size={12} style={style} />
  if (type === 'code')        return <Code2    size={12} style={{ color: '#f87171' }} />
  if (type === 'audio')       return <Music    size={12} style={{ color: '#34d399' }} />
  if (type === 'video')       return <Video    size={12} style={{ color: '#a78bfa' }} />
  if (type === 'observation') return <Brain    size={12} style={{ color: '#a78bfa' }} />
  return <File size={12} style={style} />
}

function DeleteOverlayBtn({ isDeleting, onClick }: { isDeleting: boolean; onClick: (e: React.MouseEvent) => void }) {
  return (
    <button
      onClick={onClick}
      title="Delete from index"
      style={{
        position: 'absolute', top: 5, left: 5,
        width: 22, height: 22, borderRadius: 6,
        border: '1px solid rgba(248,113,113,0.3)',
        background: isDeleting ? 'rgba(248,113,113,0.2)' : 'rgba(0,0,0,0.65)',
        backdropFilter: 'blur(4px)',
        cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: '#f87171', opacity: isDeleting ? 1 : 0,
        transition: 'opacity 0.15s',
        zIndex: 10,
      }}
      onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.opacity = '1' }}
      onMouseLeave={(e) => { if (!isDeleting) (e.currentTarget as HTMLButtonElement).style.opacity = '0' }}
    >
      <Trash2 size={10} />
    </button>
  )
}

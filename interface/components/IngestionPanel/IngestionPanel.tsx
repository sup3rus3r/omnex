'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  FolderOpen, HardDrive, File, CheckCircle2, AlertCircle,
  Loader2, ArrowRight, Zap, Database, Image, Video, Music,
  FileText, Code2, Activity, Upload, StopCircle, Trash2, ChevronDown, ChevronRight
} from 'lucide-react'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'

type Scope = 'file' | 'folder' | 'drive'

interface IngestionStatus {
  status:      string
  total_files: number
  processed:   number
  indexed:     number
  skipped:     number
  errors:      number
  by_type?:    Record<string, number>
}

interface Props {
  onDone: () => void
}

export default function IngestionPanel({ onDone }: Props) {
  const [scope,      setScope]     = useState<Scope>('folder')
  const [path,       setPath]      = useState('')
  const [uploadFiles, setUploadFiles] = useState<FileList | null>(null)
  const [running,    setRunning]   = useState(false)
  const [status,     setStatus]    = useState<IngestionStatus | null>(null)
  const [error,      setError]     = useState<string | null>(null)
  const [log,        setLog]       = useState<string[]>([])
  const [dragging,   setDragging]  = useState(false)
  const pollRef    = useRef<ReturnType<typeof setInterval> | null>(null)
  const logRef     = useRef<HTMLDivElement>(null)
  const fileInputRef   = useRef<HTMLInputElement>(null)
  const folderInputRef = useRef<HTMLInputElement>(null)

  function stopPolling() {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }
  useEffect(() => () => stopPolling(), [])

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [log])

  async function start() {
    if (!path.trim() && !uploadFiles) return
    setError(null); setRunning(true); setStatus(null); setLog([])
    const label = uploadFiles ? (uploadFiles.length === 1 ? uploadFiles[0].name : `${uploadFiles.length} files`) : path.trim()
    addLog(`Starting ingestion: ${label}`)
    addLog(`Scope: ${scope} · Workers: 4`)

    try {
      let ingestPath = path.trim()

      if (uploadFiles && uploadFiles.length > 0) {
        // Upload files directly — browser can't provide full paths
        addLog('Uploading files to server…')
        const form = new FormData()
        Array.from(uploadFiles).forEach(f => form.append('files', f, f.webkitRelativePath || f.name))
        form.append('workers', '4')
        const up = await fetch(`${API}/ingest/upload`, { method: 'POST', body: form })
        if (!up.ok) {
          const err = await up.json().catch(() => ({ detail: up.statusText }))
          throw new Error(err.detail || 'Upload failed')
        }
        const upData = await up.json()
        ingestPath = upData.path
        addLog(`Uploaded ${upData.files} file(s) — pipeline started…`)
      } else {
        await fetch(`${API}/ingest/trigger`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: ingestPath, workers: 4 }),
        })
        addLog('Pipeline started — scanning files…')
      }

      pollRef.current = setInterval(async () => {
        try {
          const res  = await fetch(`${API}/ingest/status?path=${encodeURIComponent(ingestPath)}`)
          const data = await res.json()
          setStatus(data)
          if (data.processed > 0 && data.total_files > 0) {
            addLog(`Processed ${data.processed}/${data.total_files} · ${data.indexed} indexed`)
          }
          if (data.status === 'complete' || data.status === 'error') {
            stopPolling()
            setRunning(false)
            if (data.status === 'complete') {
              addLog(`✓ Complete — ${data.indexed} memories built`)
            } else {
              addLog('! Ingestion ended with errors')
            }
          }
        } catch (_) { /* keep polling */ }
      }, 2000)
    } catch (e: any) {
      setError(e.message || 'Failed to start ingestion')
      setRunning(false)
      addLog(`Error: ${e.message}`)
    }
  }

  async function cancelIngestion() {
    try {
      await fetch(`${API}/ingest/cancel`, { method: 'POST' })
      addLog('Cancel requested — stopping after current file…')
      stopPolling()
      setRunning(false)
    } catch (_) {
      addLog('Failed to send cancel request')
    }
  }

  function addLog(msg: string) {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
    setLog((l) => [...l.slice(-50), `${time}  ${msg}`])
  }

  function extractPath(files: FileList): string {
    if (!files.length) return ''
    const f = files[0]
    // Electron / some Chromium builds expose the full path
    if ((f as any).path) return (f as any).path

    // webkitRelativePath: "FolderName/sub/file.txt" → strip to root folder name
    // We can't get the absolute path from the browser, so we return what we have
    // and let the user correct it if needed
    if (f.webkitRelativePath) {
      const parts = f.webkitRelativePath.split('/')
      return parts[0] // root folder name only — user may need to prepend full path
    }
    return f.name
  }

  function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files
    if (!files?.length) return
    setUploadFiles(files)
    setPath('') // clear manual path — we'll use upload instead
    const label = files.length === 1 ? files[0].name : `${files.length} files selected`
    setPath(label) // display only — actual upload uses FileList
    setScope(files[0].webkitRelativePath ? 'folder' : 'file')
    e.target.value = ''
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const files = e.dataTransfer.files
    if (!files.length) return
    setUploadFiles(files)
    const label = files.length === 1 ? files[0].name : `${files.length} files`
    setPath(label)
    setScope(files.length > 1 ? 'folder' : 'file')
  }, [])

  const pct  = status?.total_files ? Math.round((status.processed / status.total_files) * 100) : 0
  const done = status?.status === 'complete'

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

      {/* Header */}
      <div style={{
        flexShrink: 0, padding: '0 24px', height: 44,
        borderBottom: '1px solid #1a1a2e',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        background: 'rgba(5,5,7,0.8)',
        backdropFilter: 'blur(12px)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Database size={12} color="#505068" />
          <span style={{ fontSize: 12, color: '#505068' }}>Ingest data into memory</span>
        </div>
        {done && (
          <button
            onClick={onDone}
            style={{
              display: 'flex', alignItems: 'center', gap: 5,
              padding: '4px 12px', borderRadius: 6,
              background: 'rgba(124,106,247,0.1)',
              border: '1px solid rgba(124,106,247,0.25)',
              color: '#a78bfa', fontSize: 11, cursor: 'pointer',
              fontFamily: 'inherit',
            }}
          >
            Start recalling →
          </button>
        )}
      </div>

      {/* Body */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <div style={{ flex: 1, overflowY: 'auto', padding: '24px' }}>
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            style={{ maxWidth: 800, margin: '0 auto' }}
          >

            {/* Title */}
            <div style={{ marginBottom: 28 }}>
              <h2 style={{ fontSize: 22, fontWeight: 300, color: '#e8e8f0', letterSpacing: '-0.02em', marginBottom: 6 }}>
                Build your memory
              </h2>
              <p style={{ fontSize: 13, color: '#505068', lineHeight: 1.6 }}>
                Point Omnex at your data. It will extract meaning from every file — photos, documents, code, audio, video — and make them permanently searchable by what they contain.
              </p>
            </div>

            {/* Source selector */}
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontSize: 10, letterSpacing: '0.15em', color: '#505068', textTransform: 'uppercase', marginBottom: 10 }}>Source type</div>
              <div style={{ display: 'flex', gap: 8 }}>
                {([
                  { id: 'file',   icon: <File size={13} />,       label: 'Single file',  desc: 'Index one file' },
                  { id: 'folder', icon: <FolderOpen size={13} />, label: 'Folder',       desc: 'Recursive scan' },
                  { id: 'drive',  icon: <HardDrive size={13} />,  label: 'Full drive',   desc: 'Everything' },
                ] as const).map(({ id, icon, label, desc }) => (
                  <motion.button
                    key={id}
                    whileTap={{ scale: 0.97 }}
                    onClick={() => setScope(id)}
                    style={{
                      flex: 1, padding: '12px', borderRadius: 12,
                      border: `1px solid ${scope === id ? 'rgba(124,106,247,0.4)' : '#1a1a2e'}`,
                      background: scope === id ? 'rgba(124,106,247,0.06)' : 'rgba(10,10,15,0.6)',
                      cursor: 'pointer', textAlign: 'left',
                      transition: 'all 0.15s',
                    }}
                  >
                    <div style={{ color: scope === id ? '#a78bfa' : '#505068', marginBottom: 6 }}>{icon}</div>
                    <div style={{ fontSize: 13, color: scope === id ? '#e8e8f0' : '#a0a0b8', fontFamily: 'inherit', fontWeight: scope === id ? 500 : 400, marginBottom: 2 }}>{label}</div>
                    <div style={{ fontSize: 11, color: '#383850', fontFamily: 'inherit' }}>{desc}</div>
                  </motion.button>
                ))}
              </div>
            </div>

            {/* Path input + browse */}
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 10, letterSpacing: '0.15em', color: '#505068', textTransform: 'uppercase', marginBottom: 10 }}>Location</div>

              {/* Drag-drop zone */}
              <div
                onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
                onDragLeave={() => setDragging(false)}
                onDrop={handleDrop}
                style={{
                  border: `1px dashed ${dragging ? 'rgba(124,106,247,0.6)' : '#252540'}`,
                  borderRadius: 12, padding: '14px 16px', marginBottom: 10,
                  background: dragging ? 'rgba(124,106,247,0.04)' : 'rgba(10,10,15,0.4)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                  cursor: 'default', transition: 'all 0.15s',
                }}
              >
                <Upload size={14} color={dragging ? '#7c6af7' : '#383850'} />
                <span style={{ fontSize: 12, color: dragging ? '#a78bfa' : '#383850' }}>
                  Drop a file or folder here
                </span>
                <span style={{ fontSize: 11, color: '#252540' }}>or</span>
                <button
                  onClick={() => scope === 'file' ? fileInputRef.current?.click() : folderInputRef.current?.click()}
                  style={{
                    padding: '4px 10px', borderRadius: 6,
                    border: '1px solid #252540',
                    background: 'rgba(15,15,24,0.8)',
                    color: '#a0a0b8', fontSize: 11, cursor: 'pointer',
                    fontFamily: 'inherit', transition: 'all 0.12s',
                  }}
                  onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(124,106,247,0.4)'; (e.currentTarget as HTMLButtonElement).style.color = '#e8e8f0' }}
                  onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.borderColor = '#252540'; (e.currentTarget as HTMLButtonElement).style.color = '#a0a0b8' }}
                >
                  Browse {scope === 'file' ? 'file' : 'folder'}
                </button>
                {/* Hidden inputs */}
                <input ref={fileInputRef} type="file" multiple style={{ display: 'none' }} onChange={handleFileSelect} />
                <input ref={folderInputRef} type="file" style={{ display: 'none' }}
                  // @ts-ignore — webkitdirectory is not in the React typings
                  webkitdirectory="" directory="" onChange={handleFileSelect}
                />
              </div>

              {/* Path text field + start button */}
              <div style={{ display: 'flex', gap: 8 }}>
                <input
                  type="text"
                  value={path}
                  onChange={(e) => { setPath(e.target.value); setUploadFiles(null) }}
                  onKeyDown={(e) => e.key === 'Enter' && start()}
                  placeholder={
                    scope === 'file'   ? 'C:\\Users\\you\\Documents\\report.pdf' :
                    scope === 'folder' ? 'C:\\Users\\you\\Documents' :
                    'D:\\'
                  }
                  style={{
                    flex: 1, padding: '10px 14px',
                    background: 'rgba(10,10,15,0.8)',
                    border: '1px solid #1a1a2e',
                    borderRadius: 10,
                    color: '#e8e8f0', fontSize: 13,
                    fontFamily: 'JetBrains Mono, monospace',
                    outline: 'none', transition: 'border-color 0.15s',
                  }}
                  onFocus={(e) => { e.currentTarget.style.borderColor = 'rgba(124,106,247,0.4)' }}
                  onBlur={(e) => { e.currentTarget.style.borderColor = '#1a1a2e' }}
                />
                <motion.button
                  whileTap={{ scale: 0.97 }}
                  onClick={start}
                  disabled={(!path.trim() && !uploadFiles) || running}
                  style={{
                    padding: '10px 20px', borderRadius: 10,
                    background: (!path.trim() || running)
                      ? 'rgba(124,106,247,0.1)'
                      : 'linear-gradient(135deg, #7c6af7, #6b5ce7)',
                    border: 'none', cursor: ((!path.trim() && !uploadFiles) || running) ? 'not-allowed' : 'pointer',
                    color: 'white', fontSize: 13, fontFamily: 'inherit',
                    display: 'flex', alignItems: 'center', gap: 6,
                    boxShadow: ((!path.trim() && !uploadFiles) || running) ? 'none' : '0 0 20px rgba(124,106,247,0.35)',
                    opacity: ((!path.trim() && !uploadFiles) || running) ? 0.5 : 1,
                    transition: 'all 0.15s',
                    flexShrink: 0,
                  }}
                >
                  {running ? <Loader2 size={13} style={{ animation: 'spin 1s linear infinite' }} /> : <Zap size={13} />}
                  {running ? 'Running…' : 'Start ingestion'}
                </motion.button>
                {running && (
                  <motion.button
                    whileTap={{ scale: 0.97 }}
                    onClick={cancelIngestion}
                    style={{
                      padding: '10px 16px', borderRadius: 10,
                      background: 'rgba(248,113,113,0.08)',
                      border: '1px solid rgba(248,113,113,0.2)',
                      cursor: 'pointer', color: '#f87171',
                      fontSize: 13, fontFamily: 'inherit',
                      display: 'flex', alignItems: 'center', gap: 6,
                      transition: 'all 0.15s', flexShrink: 0,
                    }}
                  >
                    <StopCircle size={13} />
                    Cancel
                  </motion.button>
                )}
              </div>

              {/* Browser path note */}
              <p style={{ fontSize: 10, color: '#252540', marginTop: 6 }}>
                Browser security limits full path access — if Browse shows only a folder name, type or paste the full path above.
              </p>
            </div>

            {error && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#f87171', fontSize: 13, marginBottom: 16, padding: '10px 14px', background: 'rgba(248,113,113,0.05)', border: '1px solid rgba(248,113,113,0.15)', borderRadius: 10 }}>
                <AlertCircle size={14} /><span>{error}</span>
              </div>
            )}

            {/* Progress */}
            <AnimatePresence>
              {status && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  style={{ marginBottom: 20 }}
                >
                  {/* Progress bar */}
                  <div style={{
                    padding: '16px', borderRadius: 12,
                    background: 'rgba(10,10,15,0.8)',
                    border: '1px solid #1a1a2e',
                    marginBottom: 12,
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                      <span style={{ fontSize: 12, color: '#a0a0b8' }}>
                        {done ? 'Ingestion complete' : `${status.processed?.toLocaleString()} of ${status.total_files?.toLocaleString()} files`}
                      </span>
                      <span style={{ fontSize: 12, color: '#7c6af7', fontFamily: 'JetBrains Mono, monospace', fontWeight: 500 }}>{pct}%</span>
                    </div>
                    <div style={{ height: 3, background: '#1a1a2e', borderRadius: 2, overflow: 'hidden', position: 'relative' }}>
                      <motion.div
                        style={{ height: '100%', borderRadius: 2, background: 'linear-gradient(90deg, #7c6af7, #a78bfa)', position: 'relative' }}
                        animate={{ width: `${pct}%` }}
                        transition={{ duration: 0.4, ease: 'easeOut' }}
                      />
                      {running && (
                        <div
                          style={{
                            position: 'absolute', inset: 0,
                            background: 'linear-gradient(90deg, transparent, rgba(167,139,250,0.5), transparent)',
                            animation: 'dataStream 1.5s ease-in-out infinite',
                          }}
                        />
                      )}
                    </div>
                  </div>

                  {/* Stats */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, marginBottom: 12 }}>
                    {[
                      { label: 'Indexed',  value: status.indexed,  color: '#34d399', icon: <CheckCircle2 size={13} /> },
                      { label: 'Skipped',  value: status.skipped,  color: '#505068', icon: <ArrowRight size={13} /> },
                      { label: 'Errors',   value: status.errors,   color: status.errors > 0 ? '#f87171' : '#505068', icon: <AlertCircle size={13} /> },
                    ].map(({ label, value, color, icon }) => (
                      <div key={label} style={{
                        padding: '12px', borderRadius: 10,
                        background: 'rgba(10,10,15,0.6)',
                        border: '1px solid #1a1a2e',
                        textAlign: 'center',
                      }}>
                        <div style={{ color, marginBottom: 4, display: 'flex', justifyContent: 'center' }}>{icon}</div>
                        <div style={{ fontSize: 20, fontWeight: 300, color, fontFamily: 'JetBrains Mono, monospace' }}>{(value || 0).toLocaleString()}</div>
                        <div style={{ fontSize: 10, color: '#383850', marginTop: 2, textTransform: 'uppercase', letterSpacing: '0.1em' }}>{label}</div>
                      </div>
                    ))}
                  </div>

                  {/* By type breakdown */}
                  {status.by_type && Object.keys(status.by_type).length > 0 && (
                    <div style={{ padding: '12px 14px', borderRadius: 10, background: 'rgba(10,10,15,0.6)', border: '1px solid #1a1a2e' }}>
                      <div style={{ fontSize: 10, letterSpacing: '0.12em', color: '#383850', textTransform: 'uppercase', marginBottom: 10 }}>By type</div>
                      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                        {Object.entries(status.by_type).map(([type, count]) => (
                          <div key={type} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                            <TypeIcon type={type} />
                            <span style={{ fontSize: 11, color: '#a0a0b8', fontFamily: 'JetBrains Mono, monospace' }}>{count.toLocaleString()}</span>
                            <span style={{ fontSize: 10, color: '#383850' }}>{type}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Activity log */}
            {log.length > 0 && (
              <div style={{ borderRadius: 10, border: '1px solid #1a1a2e', overflow: 'hidden' }}>
                <div style={{
                  padding: '8px 12px',
                  background: 'rgba(10,10,15,0.9)',
                  borderBottom: '1px solid #1a1a2e',
                  display: 'flex', alignItems: 'center', gap: 6,
                }}>
                  <Activity size={11} color="#383850" />
                  <span style={{ fontSize: 10, color: '#383850', letterSpacing: '0.12em', textTransform: 'uppercase' }}>Activity log</span>
                  {running && (
                    <div style={{ width: 4, height: 4, borderRadius: '50%', background: '#34d399', boxShadow: '0 0 4px #34d399', marginLeft: 2 }} />
                  )}
                </div>
                <div
                  ref={logRef}
                  style={{
                    maxHeight: 160, overflowY: 'auto',
                    background: 'rgba(5,5,7,0.8)',
                    padding: '8px 12px',
                    display: 'flex', flexDirection: 'column', gap: 2,
                  }}
                >
                  {log.map((line, i) => (
                    <div key={i} style={{ fontSize: 11, color: '#383850', fontFamily: 'JetBrains Mono, monospace', lineHeight: 1.5 }}>
                      {line}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Empty hint */}
            {!status && !running && (
              <div style={{ marginTop: 32, padding: '16px', borderRadius: 12, border: '1px dashed #1a1a2e', textAlign: 'center' }}>
                <p style={{ fontSize: 12, color: '#383850', lineHeight: 1.8 }}>
                  Start with a folder of documents or photos.<br />
                  Full drive indexing runs in background — query while it works.
                </p>
              </div>
            )}

            {/* Indexed sources manager */}
            <IndexedSources api={API} />
          </motion.div>
        </div>
      </div>
    </div>
  )
}

function TypeIcon({ type }: { type: string }) {
  const style = { color: '#505068' }
  if (type === 'image')    return <Image    size={11} style={style} />
  if (type === 'video')    return <Video    size={11} style={style} />
  if (type === 'audio')    return <Music    size={11} style={style} />
  if (type === 'document') return <FileText size={11} style={style} />
  if (type === 'code')     return <Code2    size={11} style={style} />
  return <File size={11} style={style} />
}

/* ── Indexed Sources Manager ──────────────────────────────────────────────── */
function IndexedSources({ api }: { api: string }) {
  const [open,     setOpen]     = useState(false)
  const [sources,  setSources]  = useState<{source_path: string, count: number, status: string}[]>([])
  const [loading,  setLoading]  = useState(false)
  const [deleting, setDeleting] = useState<string | null>(null)

  async function load() {
    setLoading(true)
    try {
      const res  = await fetch(`${api}/ingest/status`)
      const data = await res.json()
      // Also get chunk counts per source
      const statsRes  = await fetch(`${api}/stats`)
      const statsData = await statsRes.json()

      const ingestion: any[] = data.ingestion || []
      setSources(ingestion.map((r: any) => ({
        source_path: r.source_path,
        count:       r.indexed || 0,
        status:      r.status  || 'unknown',
      })))
    } catch {}
    setLoading(false)
  }

  async function deleteSource(sourcePath: string) {
    if (!confirm(`Remove all indexed data from:\n${sourcePath}\n\nThis cannot be undone.`)) return
    setDeleting(sourcePath)
    try {
      const res = await fetch(`${api}/ingest/source?source_path=${encodeURIComponent(sourcePath)}`, { method: 'DELETE' })
      if (res.ok) setSources(s => s.filter(x => x.source_path !== sourcePath))
    } catch {}
    setDeleting(null)
  }

  function toggle() {
    if (!open) load()
    setOpen(o => !o)
  }

  return (
    <div style={{ marginTop: 28 }}>
      <button
        onClick={toggle}
        style={{
          display: 'flex', alignItems: 'center', gap: 6, width: '100%',
          background: 'transparent', border: 'none', cursor: 'pointer',
          padding: '10px 0', color: '#505068', fontSize: 11,
          letterSpacing: '0.12em', textTransform: 'uppercase', fontFamily: 'inherit',
        }}
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        Manage indexed sources
      </button>

      {open && (
        <div style={{ border: '1px solid #1a1a2e', borderRadius: 10, overflow: 'hidden' }}>
          {loading ? (
            <div style={{ padding: '16px', fontSize: 12, color: '#505068', display: 'flex', alignItems: 'center', gap: 6 }}>
              <Loader2 size={12} style={{ animation: 'spin 1s linear infinite' }} /> Loading…
            </div>
          ) : sources.length === 0 ? (
            <div style={{ padding: '16px', fontSize: 12, color: '#383850', textAlign: 'center' }}>
              No ingestion records found.
            </div>
          ) : (
            sources.map((s, i) => (
              <div
                key={s.source_path}
                style={{
                  display: 'flex', alignItems: 'center', gap: 10,
                  padding: '10px 14px',
                  borderBottom: i < sources.length - 1 ? '1px solid #1a1a2e' : 'none',
                  background: 'rgba(10,10,15,0.6)',
                }}
              >
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 12, color: '#e8e8f0', fontFamily: 'JetBrains Mono, monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {s.source_path}
                  </div>
                  <div style={{ fontSize: 10, color: '#505068', marginTop: 2 }}>
                    {s.count.toLocaleString()} chunks · {s.status}
                  </div>
                </div>
                <button
                  onClick={() => deleteSource(s.source_path)}
                  disabled={deleting === s.source_path}
                  title="Remove from index"
                  style={{
                    display: 'flex', alignItems: 'center', gap: 4,
                    padding: '5px 10px', borderRadius: 6, flexShrink: 0,
                    background: 'rgba(248,113,113,0.06)',
                    border: '1px solid rgba(248,113,113,0.15)',
                    color: deleting === s.source_path ? '#383850' : '#f87171',
                    fontSize: 11, cursor: deleting === s.source_path ? 'not-allowed' : 'pointer',
                    fontFamily: 'inherit',
                  }}
                >
                  {deleting === s.source_path
                    ? <Loader2 size={11} style={{ animation: 'spin 1s linear infinite' }} />
                    : <Trash2 size={11} />
                  }
                  Remove
                </button>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  )
}

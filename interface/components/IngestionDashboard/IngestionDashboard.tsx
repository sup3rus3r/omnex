'use client'

import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { FolderOpen, HardDrive, File, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'

type Scope = 'file' | 'folder' | 'drive'

interface IngestionStatus {
  status:       string
  total_files:  number
  processed:    number
  indexed:      number
  skipped:      number
  errors:       number
}

interface Props {
  onDone: () => void
}

export default function IngestionDashboard({ onDone }: Props) {
  const [scope,   setScope]   = useState<Scope>('folder')
  const [path,    setPath]    = useState('')
  const [running, setRunning] = useState(false)
  const [status,  setStatus]  = useState<IngestionStatus | null>(null)
  const [error,   setError]   = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  function stopPolling() {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }
  useEffect(() => () => stopPolling(), [])

  async function start() {
    if (!path.trim()) return
    setError(null); setRunning(true); setStatus(null)
    try {
      await fetch(`${API}/ingest/trigger`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: path.trim(), workers: 4 }),
      })
      pollRef.current = setInterval(async () => {
        try {
          const res  = await fetch(`${API}/ingest/status?path=${encodeURIComponent(path.trim())}`)
          const data = await res.json()
          setStatus(data)
          if (data.status === 'complete' || data.status === 'error') {
            stopPolling(); setRunning(false)
          }
        } catch { /* keep polling */ }
      }, 2000)
    } catch (e: any) {
      setError(e.message || 'Failed to start ingestion'); setRunning(false)
    }
  }

  const pct  = status?.total_files ? Math.round((status.processed / status.total_files) * 100) : 0
  const done = status?.status === 'complete'

  return (
    <div className="max-w-2xl mx-auto px-6 py-10 w-full">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}>
        <h2 className="text-xl font-light text-omnex-text mb-1">Add memory</h2>
        <p className="text-sm text-omnex-muted mb-8">Point Omnex at your files and it will remember them — forever.</p>

        {/* Scope */}
        <div className="flex gap-2 mb-6">
          {(['file', 'folder', 'drive'] as Scope[]).map((s) => (
            <button key={s} onClick={() => setScope(s)}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm border transition-all ${
                scope === s
                  ? 'border-omnex-accent text-omnex-accent bg-omnex-accent/5'
                  : 'border-omnex-border text-omnex-muted hover:border-omnex-border-2 hover:text-omnex-text'
              }`}
            >
              {s === 'file'   && <File size={13} />}
              {s === 'folder' && <FolderOpen size={13} />}
              {s === 'drive'  && <HardDrive size={13} />}
              <span className="capitalize">{s}</span>
            </button>
          ))}
        </div>

        {/* Path input */}
        <div className="flex gap-3 mb-6">
          <input type="text" value={path} onChange={(e) => setPath(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && start()}
            placeholder={scope === 'file' ? '/path/to/file.pdf' : scope === 'folder' ? '/path/to/folder' : 'D:\\ or /mnt/drive'}
            className="flex-1 bg-omnex-surface border border-omnex-border rounded-xl px-4 py-2.5 text-sm text-omnex-text placeholder-omnex-muted focus:outline-none focus:border-omnex-accent transition-colors font-mono"
          />
          <motion.button whileTap={{ scale: 0.97 }} onClick={start}
            disabled={!path.trim() || running}
            className="px-5 py-2.5 rounded-xl bg-omnex-accent text-white text-sm font-medium disabled:opacity-30 disabled:cursor-not-allowed hover:bg-omnex-accent-hover transition-colors flex items-center gap-2"
          >
            {running && <Loader2 size={13} className="animate-spin" />}
            {running ? 'Indexing…' : 'Start'}
          </motion.button>
        </div>

        {error && (
          <div className="flex items-center gap-2 text-red-400 text-sm mb-4">
            <AlertCircle size={14} /><span>{error}</span>
          </div>
        )}

        <AnimatePresence>
          {status && (
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs text-omnex-muted">
                    {done ? 'Complete' : `Remembering ${status.processed?.toLocaleString()} of ${status.total_files?.toLocaleString()}`}
                  </span>
                  <span className="text-xs text-omnex-muted font-mono">{pct}%</span>
                </div>
                <div className="h-0.5 bg-omnex-surface-2 rounded-full overflow-hidden">
                  <motion.div className="h-full bg-omnex-accent rounded-full"
                    animate={{ width: `${pct}%` }} transition={{ duration: 0.4, ease: 'easeOut' }} />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-3">
                {[
                  { label: 'Indexed', value: status.indexed,  color: 'text-emerald-400' },
                  { label: 'Skipped', value: status.skipped,  color: 'text-omnex-muted' },
                  { label: 'Errors',  value: status.errors,   color: status.errors > 0 ? 'text-red-400' : 'text-omnex-muted' },
                ].map(({ label, value, color }) => (
                  <div key={label} className="bg-omnex-surface border border-omnex-border rounded-xl p-3 text-center">
                    <p className={`text-lg font-light ${color}`}>{(value || 0).toLocaleString()}</p>
                    <p className="text-xs text-omnex-muted mt-0.5">{label}</p>
                  </div>
                ))}
              </div>

              {done && (
                <motion.div initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }}
                  className="flex flex-col items-center gap-3 py-6 text-center"
                >
                  <CheckCircle2 size={28} className="text-emerald-400" />
                  <div>
                    <p className="text-omnex-text text-sm font-medium">Memory built.</p>
                    <p className="text-omnex-muted text-xs mt-1">{status.indexed?.toLocaleString()} items indexed and ready to recall.</p>
                  </div>
                  <button onClick={onDone}
                    className="mt-2 px-5 py-2 rounded-xl bg-omnex-accent text-white text-sm hover:bg-omnex-accent-hover transition-colors">
                    Ask me anything →
                  </button>
                </motion.div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {!status && !running && (
          <p className="text-xs text-omnex-muted text-center mt-8 leading-relaxed">
            Start with a single folder. A few hundred files takes seconds.<br />
            Full drive indexing runs in the background — query while it works.
          </p>
        )}
      </motion.div>
    </div>
  )
}

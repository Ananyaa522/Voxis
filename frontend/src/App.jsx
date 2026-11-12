import { useEffect, useRef, useState } from 'react'
import './App.css'

const LANGS = [
  { id: 'en', label: 'English' },
  { id: 'hi', label: 'हिन्दी (Hindi)' },
]

export default function App() {
  const [lang, setLang] = useState('en')
  const [tab, setTab] = useState('upload')
  const [blob, setBlob] = useState(null)
  const [fileName, setFileName] = useState('')
  const [previewUrl, setPreviewUrl] = useState(null)
  const [recording, setRecording] = useState(false)
  const [seconds, setSeconds] = useState(0)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [ready, setReady] = useState(null) // null = checking, false = models loading, true = ready
  const [drag, setDrag] = useState(false)

  const recRef = useRef(null)
  const chunksRef = useRef([])
  const timerRef = useRef(null)
  const fileInput = useRef(null)

  // Poll the backend until the models have finished loading.
  useEffect(() => {
    let stop = false
    const check = async () => {
      try {
        const r = await fetch('/api/health')
        const j = await r.json()
        if (j.error) setError('Server error: ' + j.error)
        if (!stop) setReady(Boolean(j.ready))
        if (!j.ready && !j.error && !stop) setTimeout(check, 3000)
      } catch {
        if (!stop) { setReady(false); setTimeout(check, 3000) }
      }
    }
    check()
    return () => { stop = true }
  }, [])

  function chooseAudio(b, name) {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setBlob(b); setFileName(name); setPreviewUrl(URL.createObjectURL(b))
    setResult(null); setError('')
  }

  async function startRecording() {
    setError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4']
        .find(m => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || ''
      const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : {})
      chunksRef.current = []
      rec.ondataavailable = e => e.data.size && chunksRef.current.push(e.data)
      rec.onstop = () => {
        stream.getTracks().forEach(t => t.stop())
        clearInterval(timerRef.current)
        const type = rec.mimeType || 'audio/webm'
        chooseAudio(new Blob(chunksRef.current, { type }), 'recording.' + (type.includes('mp4') ? 'mp4' : 'webm'))
        setRecording(false)
      }
      rec.start()
      recRef.current = rec
      setSeconds(0); setRecording(true)
      timerRef.current = setInterval(() => setSeconds(s => s + 1), 1000)
    } catch (e) {
      setError('Microphone unavailable: ' + e.message + ' (the mic needs localhost or HTTPS)')
    }
  }

  function stopRecording() { recRef.current && recRef.current.state === 'recording' && recRef.current.stop() }

  async function transcribe() {
    if (!blob) return
    setBusy(true); setError(''); setResult(null)
    const fd = new FormData()
    fd.append('file', blob, fileName || 'audio.webm')
    fd.append('language', lang)
    try {
      const r = await fetch('/api/transcribe', { method: 'POST', body: fd })
      const j = await r.json()
      if (!r.ok) throw new Error(j.detail || r.statusText)
      setResult(j)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const copy = () => result && navigator.clipboard.writeText(result.text)

  return (
    <main className="wrap">
      <h1>Voxis</h1>
      <p className="sub">English and Hindi speech to text, powered by fine-tuned wav2vec2 models.</p>

      {ready === false && !error && <div className="banner">Models are loading, this can take a minute after a restart…</div>}

      <section className="card">
        <label className="lbl">Language</label>
        <div className="row">
          {LANGS.map(l => (
            <button key={l.id} className={'pill' + (lang === l.id ? ' on' : '')} onClick={() => setLang(l.id)}>{l.label}</button>
          ))}
        </div>
      </section>

      <section className="card">
        <div className="row" style={{ marginBottom: 16 }}>
          <button className={'pill' + (tab === 'upload' ? ' on' : '')} onClick={() => setTab('upload')}>Upload audio</button>
          <button className={'pill' + (tab === 'record' ? ' on' : '')} onClick={() => setTab('record')}>Record with mic</button>
        </div>

        {tab === 'upload' && (
          <div
            className={'drop' + (drag ? ' over' : '')}
            onClick={() => fileInput.current.click()}
            onDragOver={e => { e.preventDefault(); setDrag(true) }}
            onDragLeave={() => setDrag(false)}
            onDrop={e => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f) chooseAudio(f, f.name) }}
          >
            Drop an audio file here, or click to choose
            <small>wav · mp3 · m4a · flac · ogg · webm</small>
            <input ref={fileInput} type="file" accept="audio/*" hidden
              onChange={e => { const f = e.target.files[0]; if (f) chooseAudio(f, f.name) }} />
          </div>
        )}

        {tab === 'record' && (
          <div className="row">
            <button className={'big' + (recording ? ' rec' : '')} onClick={recording ? stopRecording : startRecording}>
              {recording ? 'Stop recording' : blob ? 'Record again' : 'Start recording'}
            </button>
            {recording && <span className="meta"><span className="dot" />{seconds} s</span>}
          </div>
        )}

        {previewUrl && <audio controls src={previewUrl} className="player" />}

        <div className="row" style={{ marginTop: 16 }}>
          <button className="big" disabled={!blob || busy || ready === false} onClick={transcribe}>
            {busy ? 'Transcribing…' : 'Transcribe'}
          </button>
          {error && <span className="meta err">{error}</span>}
        </div>
      </section>

      <section className="card">
        <div className="row spread">
          <label className="lbl" style={{ margin: 0 }}>Transcript</label>
          <button className="ghost" onClick={copy} disabled={!result}>Copy</button>
        </div>
        <div className="out" dir="auto">{result ? (result.text || '(no speech detected)') : ''}</div>
        {result && (
          <div className="meta">
            {result.audio_seconds}s of audio · processed in {result.processing_seconds}s ({result.rtf}× real-time) · {result.device}
          </div>
        )}
      </section>
    </main>
  )
}

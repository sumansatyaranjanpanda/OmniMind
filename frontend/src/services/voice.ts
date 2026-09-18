import { getAuthToken } from './api';
import { CitationItem } from '../types';

/**
 * Browser end of the voice lane.
 *
 * Three moving parts:
 *   capture   mic → AudioWorklet → Int16 PCM @16kHz → WebSocket binary frames
 *   playback  WebSocket binary frames → Int16 PCM @24kHz → scheduled AudioBuffers
 *   meter     an AnalyserNode on each side, polled once per animation frame,
 *             producing the 0–1 level the orb shader consumes
 *
 * The sample rates are not arbitrary — the Live API requires 16kHz in and
 * emits 24kHz out, so capture and playback need two separate AudioContexts.
 */

const INPUT_SAMPLE_RATE = 16000;
const OUTPUT_SAMPLE_RATE = 24000;

/** 32ms of audio per frame. Small enough that barge-in feels immediate, large
 *  enough not to flood the socket with 125 messages a second. */
const FRAME_SAMPLES = 512;

export type VoiceState = 'connecting' | 'listening' | 'thinking' | 'speaking' | 'error' | 'closed';

export interface VoiceCallbacks {
  onState: (state: VoiceState) => void;
  onLevel: (level: number) => void;
  onUserTranscript: (text: string) => void;
  onAgentTranscript: (text: string) => void;
  onTurnComplete: (citations: CitationItem[]) => void;
  onToolStart: (tool: string, query: string) => void;
  onToolEnd: (tool: string, status: string) => void;
  onError: (message: string) => void;
}

/** The worklet runs on the audio thread, so it can't close over anything —
 *  it's compiled from source at runtime and handed over as a blob URL. This
 *  avoids shipping a separate static file that Vite would have to copy and
 *  that could fall out of sync with this module. */
const CAPTURE_WORKLET = `
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Float32Array(${FRAME_SAMPLES});
    this._n = 0;
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0];
    for (let i = 0; i < ch.length; i++) {
      this._buf[this._n++] = ch[i];
      if (this._n === ${FRAME_SAMPLES}) {
        // Float32 [-1,1] -> Int16 PCM, which is what the Live API expects.
        const pcm = new Int16Array(${FRAME_SAMPLES});
        for (let j = 0; j < ${FRAME_SAMPLES}; j++) {
          const s = Math.max(-1, Math.min(1, this._buf[j]));
          pcm[j] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        this.port.postMessage(pcm.buffer, [pcm.buffer]);
        this._n = 0;
      }
    }
    return true;
  }
}
registerProcessor('capture-processor', CaptureProcessor);
`;

function rms(data: Uint8Array): number {
  let sum = 0;
  for (let i = 0; i < data.length; i++) {
    const v = (data[i] - 128) / 128;
    sum += v * v;
  }
  return Math.sqrt(sum / data.length);
}

export class VoiceClient {
  private ws: WebSocket | null = null;
  private inCtx: AudioContext | null = null;
  private outCtx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private worklet: AudioWorkletNode | null = null;
  private inAnalyser: AnalyserNode | null = null;
  private outAnalyser: AnalyserNode | null = null;
  private outGain: GainNode | null = null;

  private sources = new Set<AudioBufferSourceNode>();
  private playCursor = 0;
  private meterFrame = 0;
  private state: VoiceState = 'connecting';
  private closedByUser = false;

  constructor(private cb: VoiceCallbacks) {}

  private setState(next: VoiceState) {
    if (this.state === next) return;
    this.state = next;
    this.cb.onState(next);
  }

  async start(threadId: string | null, voice?: string): Promise<void> {
    const token = getAuthToken();
    if (!token) {
      this.cb.onError('Your session has expired. Please sign in again.');
      this.setState('error');
      return;
    }

    try {
      // Echo cancellation is load-bearing, not a nicety: without it the mic
      // hears the model's own voice through the speakers and the server-side
      // VAD treats it as the user interrupting, so the agent talks over itself.
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
    } catch {
      this.cb.onError('Microphone access was denied. Enable it in your browser to use voice mode.');
      this.setState('error');
      return;
    }

    // start() is async and fire-and-forget, so stop() can land in the middle of
    // it — and every await below is a place where that happens. React 18
    // StrictMode makes it happen on EVERY mount in development: effect runs →
    // cleanup → effect runs again, all while the first getUserMedia is still
    // pending. Without these bail-outs the aborted client sails past stop(),
    // opens its WebSocket afterwards, and you end up with two live Gemini
    // sessions talking over each other out of the same speakers.
    if (this.abandoned()) return;

    await this.setupPlayback();
    if (this.abandoned()) return;

    await this.setupCapture();
    if (this.abandoned()) return;

    this.connect(token, threadId, voice);
    this.startMeter();
  }

  /** True once stop() has run — the caller must unwind instead of continuing. */
  private abandoned(): boolean {
    if (!this.closedByUser) return false;
    this.releaseResources();
    return true;
  }

  // ── Transport ─────────────────────────────────────────────────

  private connect(token: string, threadId: string | null, voice?: string) {
    const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${scheme}://${window.location.host}/voice/stream`);
    ws.binaryType = 'arraybuffer';
    this.ws = ws;

    ws.onopen = () => {
      // The token goes in the first frame rather than the URL — a query-string
      // bearer token ends up in access logs, proxy logs and browser history.
      ws.send(JSON.stringify({ token, thread_id: threadId, voice }));
      this.setState('listening');
    };

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        this.enqueueAudio(new Int16Array(event.data));
        return;
      }
      try {
        this.handleEvent(JSON.parse(event.data));
      } catch {
        /* non-JSON control frame — nothing actionable */
      }
    };

    ws.onerror = () => {
      if (!this.closedByUser) {
        this.cb.onError('Voice connection failed.');
        this.setState('error');
      }
    };

    ws.onclose = (event) => {
      if (this.closedByUser) return;
      if (event.code === 1008) {
        this.cb.onError(event.reason || 'Voice mode is unavailable.');
        this.setState('error');
      } else {
        this.setState('closed');
      }
    };
  }

  private handleEvent(msg: Record<string, unknown>) {
    switch (msg.event) {
      case 'user_transcript':
        this.cb.onUserTranscript(String(msg.text ?? ''));
        break;
      case 'agent_transcript':
        this.cb.onAgentTranscript(String(msg.text ?? ''));
        this.setState('speaking');
        break;
      case 'tool_start':
        this.setState('thinking');
        this.cb.onToolStart(String(msg.tool ?? ''), String(msg.query ?? ''));
        break;
      case 'tool_end':
        this.cb.onToolEnd(String(msg.tool ?? ''), String(msg.status ?? 'ok'));
        break;
      case 'interrupted':
        // The user talked over the model. Anything already scheduled has to be
        // dropped now, or the browser keeps playing a reply that was cut off
        // server-side several hundred milliseconds ago.
        this.stopPlayback();
        this.setState('listening');
        break;
      case 'turn_complete':
        this.cb.onTurnComplete((msg.citations as CitationItem[]) ?? []);
        this.setState('listening');
        break;
      case 'error':
        this.cb.onError(String(msg.error ?? 'Voice error'));
        this.setState('error');
        break;
      default:
        break;
    }
  }

  // ── Capture ───────────────────────────────────────────────────

  private async setupCapture() {
    const ctx = new AudioContext({ sampleRate: INPUT_SAMPLE_RATE });
    this.inCtx = ctx;

    const blobUrl = URL.createObjectURL(new Blob([CAPTURE_WORKLET], { type: 'application/javascript' }));
    try {
      await ctx.audioWorklet.addModule(blobUrl);
    } finally {
      URL.revokeObjectURL(blobUrl);
    }

    const source = ctx.createMediaStreamSource(this.stream!);
    const node = new AudioWorkletNode(ctx, 'capture-processor');
    node.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(event.data);
      }
    };

    const analyser = ctx.createAnalyser();
    analyser.fftSize = 256;

    source.connect(analyser);
    source.connect(node);
    // The worklet emits no audio; connecting it to the destination only keeps
    // the node alive in the graph, which some browsers require to keep pulling.
    node.connect(ctx.destination);

    this.worklet = node;
    this.inAnalyser = analyser;
  }

  // ── Playback ──────────────────────────────────────────────────

  private async setupPlayback() {
    const ctx = new AudioContext({ sampleRate: OUTPUT_SAMPLE_RATE });
    const gain = ctx.createGain();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 256;

    gain.connect(analyser);
    analyser.connect(ctx.destination);

    this.outCtx = ctx;
    this.outGain = gain;
    this.outAnalyser = analyser;
  }

  private enqueueAudio(pcm: Int16Array) {
    const ctx = this.outCtx;
    if (!ctx || !this.outGain || pcm.length === 0) return;

    const buffer = ctx.createBuffer(1, pcm.length, OUTPUT_SAMPLE_RATE);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.outGain);

    // Chunks arrive faster than real time, so each one is scheduled to start
    // where the previous one ends. Without this shared cursor they'd all fire
    // at currentTime and overlap into noise. The 60ms floor is a jitter buffer
    // for when the network briefly falls behind playback.
    const now = ctx.currentTime;
    if (this.playCursor < now) this.playCursor = now + 0.06;
    source.start(this.playCursor);
    this.playCursor += buffer.duration;

    this.sources.add(source);
    source.onended = () => this.sources.delete(source);
  }

  private stopPlayback() {
    for (const source of this.sources) {
      try {
        source.stop();
      } catch {
        /* already ended */
      }
    }
    this.sources.clear();
    this.playCursor = 0;
  }

  // ── Level meter ───────────────────────────────────────────────

  private startMeter() {
    const inData = new Uint8Array(128);
    const outData = new Uint8Array(128);

    const tick = () => {
      this.meterFrame = requestAnimationFrame(tick);

      let level = 0;
      if (this.state === 'speaking' && this.outAnalyser) {
        this.outAnalyser.getByteTimeDomainData(outData);
        level = rms(outData);
      } else if (this.inAnalyser) {
        this.inAnalyser.getByteTimeDomainData(inData);
        level = rms(inData);
      }

      // Speech RMS rarely exceeds ~0.25, so it's scaled up before clamping —
      // otherwise the orb barely moves during normal conversation.
      this.cb.onLevel(Math.min(1, level * 3.2));
    };
    tick();
  }

  // ── Teardown ──────────────────────────────────────────────────

  stop() {
    this.closedByUser = true;
    this.releaseResources();
    this.setState('closed');
  }

  /** Tears down every resource acquired so far. Safe to call repeatedly, and
   *  safe to call mid-`start()` when only some of them exist yet. */
  private releaseResources() {
    cancelAnimationFrame(this.meterFrame);
    this.stopPlayback();

    this.worklet?.port.close();
    this.worklet?.disconnect();
    this.worklet = null;

    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;

    if (this.ws && this.ws.readyState <= WebSocket.OPEN) this.ws.close();
    this.ws = null;

    void this.inCtx?.close().catch(() => {});
    void this.outCtx?.close().catch(() => {});
    this.inCtx = null;
    this.outCtx = null;
  }
}

import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  AlertTriangle,
  Check,
  Clipboard,
  Download,
  FileText,
  Loader2,
  Play,
  RefreshCcw,
  Sparkles
} from 'lucide-react';
import './styles.css';

type ModelOption = {
  name: string;
  provider: string;
  model_id: string;
  temperature: number;
  requests_per_minute: number;
  max_output_tokens: number;
};

type DemoConfig = {
  models: ModelOption[];
  mixing_modes: string[];
  ratios: number[];
  max_text_chars: number;
  default_model: string;
};

type SampleText = {
  id: string;
  title: string;
  category: 'academic' | 'news' | 'web' | 'essay';
  text: string;
};

type DemoResult = {
  ok: boolean;
  error: string;
  original_text: string;
  mixed_text: string;
  sentences: string[];
  mixed_sentences: string[];
  selected_indices: number[];
  rewrites: Record<string, string>;
  sentence_labels: number[];
  sentence_pairs: Array<{
    index: number;
    original: string;
    rewritten: string;
    selected: boolean;
    label: number;
  }>;
  labels: Record<string, number | null>;
  model: string;
  provider_model_id: string;
  mixing_mode: string;
  target_ratio: number;
  input_tokens: number;
  output_tokens: number;
  steps: Array<{
    id: string;
    label: string;
    status: 'complete' | 'failed';
    detail: string;
  }>;
  dataset_record: Record<string, unknown>;
};

const FALLBACK_TEXT =
  'Online learning gives students more control over when and where they study. This flexibility can be especially helpful for people who have jobs, family responsibilities, or long commutes. At the same time, virtual classes require a high level of self-discipline. Students may fall behind if they do not manage deadlines carefully or ask questions when they are confused. In my view, online learning works best when schools combine recorded material with regular live discussion. That balance preserves flexibility while still giving students a sense of structure and community.';

const MODEL_DISPLAY_NAMES: Record<string, string> = {
  'llama4-fast:latest': 'Llama 4',
  gemma4: 'Gemma 4',
  'gemini-3.1-flash-lite-preview': 'Gemini 3.1 Flash Lite',
  'MiniMax-M2.7': 'MiniMax M2.7',
  'qwen3.5-plus': 'Qwen 3.5 Plus',
  'qwen3.6-plus': 'Qwen 3.6 Plus',
  'qwen3.5-flash': 'Qwen 3.5 Flash',
  'qwen3.6-plus-preview-free': 'Qwen 3.6 Plus',
  'claude-haiku-4.5': 'Claude Haiku 4.5',
  'gpt-5.4': 'GPT-5.4',
  'DeepSeek-V3.2': 'DeepSeek V3.2',
  'doubao-seed-2-0-pro': 'Doubao Seed 2.0 Pro',
  'mimo-v2.5-pro': 'MiMo V2.5 Pro'
};

function formatModelName(modelName: string) {
  return MODEL_DISPLAY_NAMES[modelName] ?? modelName;
}

function App() {
  const [config, setConfig] = useState<DemoConfig | null>(null);
  const [samples, setSamples] = useState<SampleText[]>([]);
  const [text, setText] = useState(FALLBACK_TEXT);
  const [model, setModel] = useState('');
  const [mixingMode, setMixingMode] = useState('block_replace');
  const [ratio, setRatio] = useState(0.4);
  const [seed, setSeed] = useState(42);
  const [result, setResult] = useState<DemoResult | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    async function loadInitialData() {
      const [configResponse, samplesResponse] = await Promise.all([
        fetch('/api/config'),
        fetch('/api/samples')
      ]);
      const loadedConfig = await readJson<DemoConfig>(configResponse);
      const loadedSamples = await readJson<{ samples: SampleText[] }>(samplesResponse);
      setConfig(loadedConfig);
      setSamples(loadedSamples.samples ?? []);
      setModel(loadedConfig.default_model);
      setMixingMode(loadedConfig.mixing_modes?.[0] ?? 'block_replace');
      setRatio(loadedConfig.ratios?.[1] ?? 0.4);
    }

    loadInitialData().catch((err) => {
      setError(`Failed to load demo configuration: ${String(err)}`);
    });
  }, []);

  const selectedModel = useMemo(
    () => config?.models.find((item) => item.name === model),
    [config, model]
  );

  async function runRewrite() {
    setLoading(true);
    setError('');
    setCopied(false);
    try {
      const response = await fetch('/api/rewrite', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          model,
          mixing_mode: mixingMode,
          target_ratio: ratio,
          seed,
          language_hint: 'English'
        })
      });
      const payload = await readJson<DemoResult | { detail?: string }>(response);
      if (!response.ok) {
        const detail = 'detail' in payload ? payload.detail : undefined;
        throw new Error(detail ?? 'Request failed');
      }
      const rewriteResult = payload as DemoResult;
      setResult(rewriteResult);
      if (!rewriteResult.ok) {
        setError(rewriteResult.error || 'The model call failed; local construction steps are still shown.');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  async function copyDatasetJson() {
    if (!result) return;
    await navigator.clipboard.writeText(JSON.stringify(result.dataset_record, null, 2));
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  }

  function downloadDatasetJson() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result.dataset_record, null, 2)], {
      type: 'application/json'
    });
    const href = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = href;
    link.download = `${String(result.dataset_record.id ?? 'pact-demo')}.json`;
    link.click();
    URL.revokeObjectURL(href);
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <aside className="control-panel" aria-label="Demo controls">
          <div className="brand-row">
            <div className="brand-mark" aria-hidden="true">
              <Sparkles size={22} />
            </div>
            <div>
              <h1>PACT Dataset Demo</h1>
              <p>Interactive human-AI mixed text construction</p>
            </div>
          </div>

          <section className="panel-section">
            <div className="section-title">
              <FileText size={18} />
              <h2>Sample Texts</h2>
            </div>
            <div className="sample-grid">
              {samples.map((sample) => (
                <button
                  type="button"
                  className="sample-button"
                  key={sample.id}
                  onClick={() => setText(sample.text)}
                >
                  <span>{sample.title}</span>
                  <small>{sample.category}</small>
                </button>
              ))}
            </div>
          </section>

          <section className="panel-section">
            <label className="field-label" htmlFor="source-text">
              Source Text
            </label>
            <textarea
              id="source-text"
              value={text}
              maxLength={config?.max_text_chars ?? 6000}
              onChange={(event) => setText(event.target.value)}
            />
            <div className="field-meta">
              <span>{text.length} / {config?.max_text_chars ?? 6000}</span>
              <button type="button" className="text-button" onClick={() => setText('')}>
                Clear
              </button>
            </div>
          </section>

          <section className="panel-section control-grid">
            <label>
              <span className="field-label">Rewrite Model</span>
              <select value={model} onChange={(event) => setModel(event.target.value)}>
                {(config?.models ?? []).map((item) => (
                  <option value={item.name} key={item.name}>
                    {formatModelName(item.name)}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span className="field-label">Mixing Mode</span>
              <select
                value={mixingMode}
                onChange={(event) => setMixingMode(event.target.value)}
              >
                {(config?.mixing_modes ?? ['block_replace', 'random_scatter']).map((mode) => (
                  <option value={mode} key={mode}>
                    {mode}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span className="field-label">AI Sentence Ratio</span>
              <select value={ratio} onChange={(event) => setRatio(Number(event.target.value))}>
                {(config?.ratios ?? [0.2, 0.4, 0.6, 0.8, 1.0]).map((item) => (
                  <option value={item} key={item}>
                    {Math.round(item * 100)}%
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span className="field-label">Seed</span>
              <input
                type="number"
                min={0}
                max={1000000}
                value={seed}
                onChange={(event) => setSeed(Number(event.target.value))}
              />
            </label>
          </section>

          {selectedModel && (
            <div className="model-meta">
              <span>{selectedModel.provider}</span>
              <span>{selectedModel.model_id}</span>
            </div>
          )}

          <button
            type="button"
            className="run-button"
            onClick={runRewrite}
            disabled={loading || !config}
          >
            {loading ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
            <span>{loading ? 'Running pipeline...' : 'Run PACT Flow'}</span>
          </button>

          {error && (
            <div className="alert" role="alert">
              <AlertTriangle size={18} />
              <span>{error}</span>
            </div>
          )}
        </aside>

        <section className="result-panel" aria-label="Demo result">
          {!result ? (
            <EmptyState loading={loading} />
          ) : (
            <>
              <ResultHeader result={result} />
              <Timeline result={result} />
              <Metrics result={result} />
              <MixedText result={result} />
              <SentenceComparison result={result} />
              <section className="output-section">
                <div className="output-header">
                  <h2>Dataset JSON</h2>
                  <div className="button-row">
                    <button type="button" className="icon-button" onClick={copyDatasetJson}>
                      {copied ? <Check size={17} /> : <Clipboard size={17} />}
                      <span>{copied ? 'Copied' : 'Copy'}</span>
                    </button>
                    <button type="button" className="icon-button" onClick={downloadDatasetJson}>
                      <Download size={17} />
                      <span>Export</span>
                    </button>
                  </div>
                </div>
                <pre>{JSON.stringify(result.dataset_record, null, 2)}</pre>
              </section>
            </>
          )}
        </section>
      </section>
    </main>
  );
}

async function readJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  const contentType = response.headers.get('content-type') ?? '';
  if (!contentType.includes('application/json')) {
    const startsWithHtml = text.trimStart().startsWith('<');
    const message = startsWithHtml
      ? 'The public tunnel returned an HTML error page instead of the API response. The anonymous tunnel URL may have expired; reload using the latest deployment URL.'
      : `Expected JSON but received ${contentType || 'an unknown response type'}.`;
    throw new Error(message);
  }
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new Error('The server returned malformed JSON.');
  }
}

function EmptyState({ loading }: { loading: boolean }) {
  return (
    <div className="empty-state">
      <RefreshCcw className={loading ? 'spin' : ''} size={34} />
      <h2>{loading ? 'Constructing the mixed sample' : 'Run the flow to inspect a PACT sample'}</h2>
      <p>
        The result view will show sentence selection, rewritten sentences, continuous labels, and
        the final JSON record.
      </p>
    </div>
  );
}

function ResultHeader({ result }: { result: DemoResult }) {
  return (
    <div className="result-header">
      <div>
        <h2>{result.ok ? 'Mixed Text Generated' : 'Local Steps Preserved'}</h2>
        <p>
          {formatModelName(result.model)} · {result.mixing_mode} · {Math.round(result.target_ratio * 100)}% target
        </p>
      </div>
      <div className={result.ok ? 'status-pill success' : 'status-pill warning'}>
        {result.ok ? 'Complete' : 'Model failed'}
      </div>
    </div>
  );
}

function Timeline({ result }: { result: DemoResult }) {
  return (
    <section className="output-section">
      <h2>Data Flow</h2>
      <div className="timeline">
        {result.steps.map((step, index) => (
          <div className={`timeline-item ${step.status}`} key={`${step.id}-${index}`}>
            <div className="timeline-dot">
              {step.status === 'complete' ? <Check size={15} /> : <AlertTriangle size={15} />}
            </div>
            <div>
              <strong>{step.label}</strong>
              <p>{step.detail}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function Metrics({ result }: { result: DemoResult }) {
  const metrics = [
    ['LIR', result.labels.lir],
    ['Jaccard Distance', result.labels.jaccard_distance],
    ['Sentence Jaccard', result.labels.sentence_jaccard],
    ['Cosine Distance', result.labels.cosine_distance]
  ];
  return (
    <section className="metric-grid" aria-label="Computed labels">
      {metrics.map(([label, value]) => (
        <div className="metric-card" key={label as string}>
          <span>{label}</span>
          <strong>{typeof value === 'number' ? value.toFixed(4) : 'N/A'}</strong>
        </div>
      ))}
      <div className="metric-card">
        <span>Sentence Labels</span>
        <strong>{result.sentence_labels.join(' ')}</strong>
      </div>
      <div className="metric-card">
        <span>API Tokens</span>
        <strong>{result.input_tokens} / {result.output_tokens}</strong>
      </div>
    </section>
  );
}

function MixedText({ result }: { result: DemoResult }) {
  return (
    <section className="output-section">
      <h2>Mixed Text</h2>
      <div className="mixed-text">
        {result.mixed_sentences.map((sentence, index) => (
          <span
            className={result.sentence_labels[index] === 1 ? 'sentence ai' : 'sentence human'}
            key={`${index}-${sentence}`}
          >
            {sentence}{' '}
          </span>
        ))}
      </div>
    </section>
  );
}

function SentenceComparison({ result }: { result: DemoResult }) {
  return (
    <section className="output-section">
      <h2>Sentence Comparison</h2>
      <div className="comparison-list">
        {result.sentence_pairs.map((pair) => (
          <article className={pair.selected ? 'sentence-row selected' : 'sentence-row'} key={pair.index}>
            <div className="sentence-index">{pair.index + 1}</div>
            <div>
              <label>Original</label>
              <p>{pair.original}</p>
            </div>
            <div>
              <label>{pair.label === 1 ? 'Rewritten' : 'Unchanged'}</label>
              <p>{pair.rewritten}</p>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

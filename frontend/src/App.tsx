import { useEffect, useState } from "react";

import PitchChart from "./PitchChart";
import type { AudioCatalog, PitchAnalysis } from "./types";

function formatDuration(seconds: number) {
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return `${minutes}:${remainder.toString().padStart(2, "0")}`;
}

function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : "An unexpected error occurred.";
}

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as {
      detail?: string;
    } | null;
    throw new Error(body?.detail ?? `Request failed with status ${response.status}.`);
  }
  return response.json() as Promise<T>;
}

export default function App() {
  const [catalog, setCatalog] = useState<AudioCatalog | null>(null);
  const [selectedFile, setSelectedFile] = useState("");
  const [analysis, setAnalysis] = useState<PitchAnalysis | null>(null);
  const [catalogError, setCatalogError] = useState("");
  const [analysisError, setAnalysisError] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    getJson<AudioCatalog>("/api/audio-files", controller.signal)
      .then((result) => {
        setCatalog(result);
        const shortestFile = result.files.reduce<
          AudioCatalog["files"][number] | undefined
        >(
          (shortest, file) =>
            !shortest || file.duration_seconds < shortest.duration_seconds
              ? file
              : shortest,
          undefined,
        );
        setSelectedFile(shortestFile?.name ?? "");
      })
      .catch((error: unknown) => {
        if ((error as Error).name !== "AbortError") {
          setCatalogError(getErrorMessage(error));
        }
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setAnalysis(null);
      return;
    }

    const controller = new AbortController();
    setIsAnalyzing(true);
    setAnalysisError("");
    setAnalysis(null);
    getJson<PitchAnalysis>(
      `/api/pitch?filename=${encodeURIComponent(selectedFile)}`,
      controller.signal,
    )
      .then(setAnalysis)
      .catch((error: unknown) => {
        if ((error as Error).name !== "AbortError") {
          setAnalysisError(getErrorMessage(error));
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setIsAnalyzing(false);
      });

    return () => controller.abort();
  }, [selectedFile]);

  const selectedMetadata = catalog?.files.find(
    (file) => file.name === selectedFile,
  );
  const voicedFrames =
    analysis?.points.filter((point) => point.f0_hz !== null).length ?? 0;
  const voicedRatio = analysis?.points.length
    ? Math.round((voicedFrames / analysis.points.length) * 100)
    : 0;

  return (
    <main>
      <header className="masthead">
        <div className="brand-lockup">
          <span className="eyebrow">Audio Pitch Core / 01</span>
          <h1>Signal Desk</h1>
        </div>
        <div className="system-status">
          <span className="status-dot" />
          local analysis
        </div>
      </header>

      <section className="intro">
        <p className="kicker">Frame-level evidence, without guesswork.</p>
        <p className="intro-copy">
          Select a WAV from the working library and inspect the raw
          autocorrelation pitch track alongside estimator confidence.
        </p>
      </section>

      <section className="workspace">
        <aside className="control-panel">
          <div>
            <span className="section-number">01</span>
            <h2>Source</h2>
          </div>

          <label htmlFor="audio-file">Audio file</label>
          <div className="select-wrap">
            <select
              id="audio-file"
              value={selectedFile}
              onChange={(event) => setSelectedFile(event.target.value)}
              disabled={!catalog || catalog.files.length === 0}
            >
              {!catalog && <option>Loading WAV library...</option>}
              {catalog?.files.length === 0 && <option>No WAV files found</option>}
              {catalog?.files.map((file) => (
                <option value={file.name} key={file.name}>
                  {file.name}
                </option>
              ))}
            </select>
          </div>

          {catalogError && <p className="error-message">{catalogError}</p>}
          <p className="format-rule">
            <span>Accepted</span>
            WAV / linear PCM
          </p>
          <p className="library-path" title={catalog?.directory}>
            {catalog?.directory ?? "Resolving audio library..."}
          </p>

          {selectedMetadata && (
            <dl className="source-facts">
              <div>
                <dt>Duration</dt>
                <dd>{formatDuration(selectedMetadata.duration_seconds)}</dd>
              </div>
              <div>
                <dt>Format</dt>
                <dd>WAV</dd>
              </div>
            </dl>
          )}
        </aside>

        <section className="analysis-panel">
          <div className="panel-heading">
            <div>
              <span className="section-number">02</span>
              <h2>Pitch window</h2>
            </div>
            {analysis && (
              <div className="analysis-stamp">
                <span>{analysis.sample_rate.toLocaleString()} Hz</span>
                <span>{analysis.points.length.toLocaleString()} frames</span>
                <span>{voicedRatio}% voiced</span>
              </div>
            )}
          </div>

          {isAnalyzing && (
            <div className="chart-state" aria-live="polite">
              <div className="analysis-pulse" />
              <p>Reading periodic structure across the file...</p>
              <span>Long recordings may take a moment.</span>
            </div>
          )}
          {analysisError && (
            <div className="chart-state error-state" role="alert">
              <p>Analysis could not be completed.</p>
              <span>{analysisError}</span>
            </div>
          )}
          {!isAnalyzing && !analysisError && analysis && (
            <>
              <PitchChart analysis={analysis} />
              <div className="chart-legend">
                <span>
                  <i className="legend-dot" /> pitch estimate
                </span>
                <span>
                  <i className="legend-line" /> confidence
                </span>
                <span className="method-label">
                  autocorr / {analysis.frame_size_ms} ms frame /{" "}
                  {analysis.hop_size_ms} ms hop
                </span>
              </div>
            </>
          )}
          {!selectedFile && !catalogError && (
            <div className="chart-state">
              <p>No valid WAV files found.</p>
              <span>MP3 files are intentionally excluded from this dashboard.</span>
            </div>
          )}
        </section>
      </section>

      <footer>
        <span>Built for honest abstention</span>
        <span>Spectrogram / notes / diagnostics next</span>
      </footer>
    </main>
  );
}

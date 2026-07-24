import { useEffect, useRef, useState } from "react";

import type { PitchAnalysis, PitchPoint } from "./types";

type PitchChartProps = {
  analysis: PitchAnalysis;
};

const HEIGHT = 430;
const LEFT = 64;
const RIGHT = 22;
const TOP = 22;
const PITCH_HEIGHT = 230;
const GAP = 54;
const CONFIDENCE_HEIGHT = 78;
const PITCH_MAX_HZ = 200;

function formatTime(seconds: number) {
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`;
}

function reducePoints(points: PitchPoint[], width: number) {
  const targetCount = Math.max(1, Math.floor(width * 1.5));
  const stride = Math.max(1, Math.ceil(points.length / targetCount));
  return points.filter((_, index) => index % stride === 0);
}

export default function PitchChart({ analysis }: PitchChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(900);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;

    const observer = new ResizeObserver(([entry]) => {
      setWidth(Math.max(360, Math.floor(entry.contentRect.width)));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const plotWidth = width - LEFT - RIGHT;
  const confidenceTop = TOP + PITCH_HEIGHT + GAP;
  const x = (time: number) =>
    LEFT + (time / Math.max(analysis.duration_seconds, 0.001)) * plotWidth;
  const pitchY = (frequency: number) =>
    TOP + PITCH_HEIGHT - (frequency / PITCH_MAX_HZ) * PITCH_HEIGHT;
  const confidenceY = (confidence: number) =>
    confidenceTop + CONFIDENCE_HEIGHT - confidence * CONFIDENCE_HEIGHT;
  const visiblePoints = reducePoints(analysis.points, plotWidth);
  const confidencePath = visiblePoints
    .map(
      (point, index) =>
        `${index === 0 ? "M" : "L"} ${x(point.time_seconds).toFixed(2)} ${confidenceY(point.confidence).toFixed(2)}`,
    )
    .join(" ");
  const timeTicks = Array.from({ length: 6 }, (_, index) => {
    const ratio = index / 5;
    return {
      value: analysis.duration_seconds * ratio,
      x: LEFT + plotWidth * ratio,
    };
  });

  return (
    <div className="chart-shell" ref={containerRef}>
      <svg
        className="pitch-chart"
        width={width}
        height={HEIGHT}
        viewBox={`0 0 ${width} ${HEIGHT}`}
        role="img"
        aria-label={`Pitch estimates and confidence over time for ${analysis.filename}`}
      >
        <defs>
          <clipPath id="pitch-clip">
            <rect x={LEFT} y={TOP} width={plotWidth} height={PITCH_HEIGHT} />
          </clipPath>
        </defs>

        {[0, 50, 100, 150, 200].map((frequency) => {
          const y = pitchY(frequency);
          return (
            <g key={frequency}>
              <line
                className="grid-line"
                x1={LEFT}
                x2={width - RIGHT}
                y1={y}
                y2={y}
              />
              <text className="axis-label" x={LEFT - 12} y={y + 4} textAnchor="end">
                {frequency}
              </text>
            </g>
          );
        })}

        <text
          className="axis-title"
          x={16}
          y={TOP + PITCH_HEIGHT / 2}
          textAnchor="middle"
          transform={`rotate(-90 16 ${TOP + PITCH_HEIGHT / 2})`}
        >
          f0 (Hz)
        </text>

        <g clipPath="url(#pitch-clip)">
          {visiblePoints.map((point) =>
            point.f0_hz === null ? null : (
              <circle
                className="pitch-point"
                key={point.time_seconds}
                cx={x(point.time_seconds)}
                cy={pitchY(point.f0_hz)}
                r={2.3}
              />
            ),
          )}
        </g>

        {[0, 0.5, 1].map((confidence) => {
          const y = confidenceY(confidence);
          return (
            <g key={confidence}>
              <line
                className="grid-line"
                x1={LEFT}
                x2={width - RIGHT}
                y1={y}
                y2={y}
              />
              <text className="axis-label" x={LEFT - 12} y={y + 4} textAnchor="end">
                {confidence.toFixed(1)}
              </text>
            </g>
          );
        })}

        <path className="confidence-line" d={confidencePath} />
        <text
          className="axis-title"
          x={16}
          y={confidenceTop + CONFIDENCE_HEIGHT / 2}
          textAnchor="middle"
          transform={`rotate(-90 16 ${confidenceTop + CONFIDENCE_HEIGHT / 2})`}
        >
          confidence
        </text>

        {timeTicks.map((tick) => (
          <g key={tick.value}>
            <line
              className="tick-mark"
              x1={tick.x}
              x2={tick.x}
              y1={confidenceTop + CONFIDENCE_HEIGHT}
              y2={confidenceTop + CONFIDENCE_HEIGHT + 7}
            />
            <text
              className="time-label"
              x={tick.x}
              y={confidenceTop + CONFIDENCE_HEIGHT + 24}
              textAnchor="middle"
            >
              {formatTime(tick.value)}
            </text>
          </g>
        ))}
        <text
          className="axis-caption"
          x={LEFT + plotWidth / 2}
          y={HEIGHT - 4}
          textAnchor="middle"
        >
          time
        </text>
      </svg>
    </div>
  );
}


export type AudioFileSummary = {
  name: string;
  size_bytes: number;
  duration_seconds: number;
};

export type AudioCatalog = {
  directory: string;
  files: AudioFileSummary[];
};

export type PitchPoint = {
  time_seconds: number;
  f0_hz: number | null;
  confidence: number;
};

export type PitchAnalysis = {
  filename: string;
  duration_seconds: number;
  sample_rate: number;
  frame_size_ms: number;
  hop_size_ms: number;
  f_min_hz: number;
  f_max_hz: number;
  points: PitchPoint[];
};


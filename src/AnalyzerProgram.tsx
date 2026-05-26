import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import type { CSSProperties, FormEvent, SyntheticEvent } from "react";
import type { User } from "@supabase/supabase-js";
import SwingGradeEngine from "./analysis";
import type { SwingDrill } from "./analysis";
import { isSupabaseConfigured, requireSupabase } from "./lib/supabase";

const ANALYZER_BACKDROP = new URL("./assets/course-backgrounds/course-12-cypress-point-style.jpg", import.meta.url).href;
const MAX_SAVED_VIDEO_BYTES = 50 * 1024 * 1024;
const MAX_SAVED_VIDEO_LABEL = "50 MB";

type ModelState = "idle" | "loading" | "ready" | "error";
type ViewAngle = "face-on" | "down-the-line";
type WorkflowStep = "capture" | "analyze" | "practice";

type StoredReport = {
  id: string;
  user_id: string;
  title: string;
  source_name: string;
  storage_path: string;
  handedness: "right" | "left";
  swing_angle: ViewAngle;
  overall_score: number;
  analysis: any;
  created_at: string;
  signedUrl?: string;
};

type PracticePlanItem = {
  drillId: string;
  name: string;
  time: string;
  reps: string;
  focus: string;
  checked: boolean;
  cue?: string;
};

type QualityNote = {
  id: string;
  label: string;
  severity: "good" | "info" | "warn" | "bad";
  detail: string;
};

type VideoClipWindow = {
  start: number;
  end: number;
  confidence: "high" | "medium" | "low";
  source: "motion" | "full-clip";
  reason: string;
};

type MotionSample = {
  time: number;
  motion: number;
  brightness: number;
};

const ICONS: Record<string, JSX.Element> = {
  upload: (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 16V4" />
      <path d="m7 9 5-5 5 5" />
      <path d="M5 20h14" />
    </svg>
  ),
  camera: (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3Z" />
      <circle cx="12" cy="13" r="3.5" />
    </svg>
  ),
  analyze: (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M4 19V5" />
      <path d="M4 19h16" />
      <path d="m7 15 3-4 4 3 5-7" />
    </svg>
  ),
  download: (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 3v12" />
      <path d="m7 10 5 5 5-5" />
      <path d="M5 21h14" />
    </svg>
  ),
  play: (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="currentColor">
      <path d="M8 5v14l11-7L8 5Z" />
    </svg>
  ),
  stop: (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="currentColor">
      <path d="M7 7h10v10H7z" />
    </svg>
  ),
  reset: (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M3 12a9 9 0 1 0 3-6.7" />
      <path d="M3 4v6h6" />
    </svg>
  ),
  save: (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2Z" />
      <path d="M17 21v-8H7v8" />
      <path d="M7 3v5h8" />
    </svg>
  ),
  user: (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M20 21a8 8 0 0 0-16 0" />
      <circle cx="12" cy="7" r="4" />
    </svg>
  ),
  trash: (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M3 6h18" />
      <path d="M8 6V4h8v2" />
      <path d="M19 6l-1 15H6L5 6" />
      <path d="M10 11v6" />
      <path d="M14 11v6" />
    </svg>
  ),
  trophy: (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M8 21h8" />
      <path d="M12 17v4" />
      <path d="M7 4h10v5a5 5 0 0 1-10 0V4Z" />
      <path d="M5 5H3v2a4 4 0 0 0 4 4" />
      <path d="M19 5h2v2a4 4 0 0 1-4 4" />
    </svg>
  ),
  previous: (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="m15 18-6-6 6-6" />
    </svg>
  ),
  next: (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="m9 18 6-6-6-6" />
    </svg>
  ),
};

function formatTime(seconds: number) {
  if (!Number.isFinite(seconds)) return "0:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${mins}:${secs}`;
}

function roundNumber(value: number | null | undefined, places = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) return null;
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
}

function clampNumber(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function waitForEvent(target: EventTarget, eventName: string) {
  return new Promise<void>((resolve) => {
    target.addEventListener(eventName, () => resolve(), { once: true });
  });
}

function normalizePoseForFrame(pose: any, width: number, height: number) {
  if (!pose?.keypoints?.length) return pose ?? null;

  const visibleKeypoints = pose.keypoints.filter((item: any) => Number.isFinite(item.x) && Number.isFinite(item.y));
  const maxX = Math.max(...visibleKeypoints.map((item: any) => Math.abs(item.x)), 0);
  const maxY = Math.max(...visibleKeypoints.map((item: any) => Math.abs(item.y)), 0);
  const looksNormalized = maxX <= 1.5 && maxY <= 1.5;

  return {
    ...pose,
    keypoints: pose.keypoints.map((item: any) => ({
      ...item,
      x: looksNormalized ? item.x * width : item.x,
      y: looksNormalized ? item.y * height : item.y,
    })),
  };
}

function estimateFrameBrightness(ctx: CanvasRenderingContext2D, width: number, height: number) {
  const sampleSize = 64;
  const sampleCanvas = document.createElement("canvas");
  sampleCanvas.width = sampleSize;
  sampleCanvas.height = Math.max(36, Math.round(sampleSize * (height / width)));
  const sampleCtx = sampleCanvas.getContext("2d");
  if (!sampleCtx) return null;
  sampleCtx.drawImage(ctx.canvas, 0, 0, sampleCanvas.width, sampleCanvas.height);
  const pixels = sampleCtx.getImageData(0, 0, sampleCanvas.width, sampleCanvas.height).data;
  let total = 0;
  for (let index = 0; index < pixels.length; index += 4) {
    total += pixels[index] * 0.2126 + pixels[index + 1] * 0.7152 + pixels[index + 2] * 0.0722;
  }
  return total / (pixels.length / 4);
}

function readMotionSample(ctx: CanvasRenderingContext2D, width: number, height: number, previousPixels: Uint8ClampedArray | null) {
  const rgba = ctx.getImageData(0, 0, width, height).data;
  const pixels = new Uint8ClampedArray(width * height);
  let brightness = 0;
  let difference = 0;

  for (let rgbaIndex = 0, pixelIndex = 0; rgbaIndex < rgba.length; rgbaIndex += 4, pixelIndex += 1) {
    const luminance = rgba[rgbaIndex] * 0.2126 + rgba[rgbaIndex + 1] * 0.7152 + rgba[rgbaIndex + 2] * 0.0722;
    pixels[pixelIndex] = luminance;
    brightness += luminance;
    if (previousPixels) difference += Math.abs(luminance - previousPixels[pixelIndex]);
  }

  const pixelCount = Math.max(pixels.length, 1);
  return {
    pixels,
    brightness: brightness / pixelCount,
    motion: previousPixels ? difference / (pixelCount * 255) : 0,
  };
}

function chooseImportantVideoWindow(samples: MotionSample[], rawDuration: number): VideoClipWindow {
  const fallbackEnd = Math.min(rawDuration, 14);
  const fallback: VideoClipWindow = {
    start: 0,
    end: fallbackEnd,
    confidence: rawDuration <= 14 ? "high" : "low",
    source: "full-clip",
    reason: rawDuration <= 14 ? "Clip already fits the analysis window." : "No clear motion cluster was found.",
  };

  if (rawDuration <= 14 || samples.length < 8) return fallback;

  const rawScores = samples.map((sample) => sample.motion);
  const smoothedScores = rawScores.map((score, index) => {
    const nearby = [rawScores[index - 1], score, rawScores[index + 1]].filter((value) => Number.isFinite(value)) as number[];
    return nearby.reduce((total, value) => total + value, 0) / Math.max(nearby.length, 1);
  });
  const sortedScores = [...smoothedScores].sort((a, b) => a - b);
  const baseline = sortedScores[Math.floor(sortedScores.length * 0.45)] ?? 0;
  const maxScore = Math.max(...smoothedScores, 0);

  if (maxScore < 0.007) return fallback;

  const threshold = Math.max(0.012, baseline * 2.4, maxScore * 0.22);
  const activeIndexes = smoothedScores
    .map((score, index) => (score >= threshold ? index : -1))
    .filter((index) => index > 0);

  if (!activeIndexes.length) return fallback;

  const clusters: Array<{ start: number; end: number; peak: number; energy: number; count: number }> = [];
  let current = {
    start: activeIndexes[0],
    end: activeIndexes[0],
    peak: activeIndexes[0],
    energy: smoothedScores[activeIndexes[0]],
    count: 1,
  };

  for (const index of activeIndexes.slice(1)) {
    if (index <= current.end + 3) {
      current.end = index;
      current.energy += smoothedScores[index];
      current.count += 1;
      if (smoothedScores[index] > smoothedScores[current.peak]) current.peak = index;
    } else {
      clusters.push(current);
      current = { start: index, end: index, peak: index, energy: smoothedScores[index], count: 1 };
    }
  }
  clusters.push(current);

  const bestCluster = [...clusters].sort((a, b) => b.energy - a.energy)[0];
  if (!bestCluster) return fallback;

  const startAnchor = samples[Math.max(0, bestCluster.start - 1)]?.time ?? 0;
  const endAnchor = samples[Math.min(samples.length - 1, bestCluster.end + 1)]?.time ?? rawDuration;
  const peakTime = samples[bestCluster.peak]?.time ?? (startAnchor + endAnchor) / 2;
  const maxWindow = Math.min(14, rawDuration);
  const minWindow = Math.min(4.2, rawDuration);
  let start = Math.max(0, startAnchor - 2.4);
  let end = Math.min(rawDuration, endAnchor + 1.6);

  if (end - start > maxWindow) {
    start = clampNumber(peakTime - 4.2, 0, Math.max(0, rawDuration - maxWindow));
    end = Math.min(rawDuration, start + maxWindow);
  }

  if (end - start < minWindow) {
    start = clampNumber(peakTime - 2.3, 0, Math.max(0, rawDuration - minWindow));
    end = Math.min(rawDuration, start + minWindow);
  }

  return {
    start,
    end,
    confidence: bestCluster.count >= 4 && maxScore > threshold * 1.6 ? "high" : "medium",
    source: "motion",
    reason: `Motion cluster detected near ${formatTime(peakTime)}.`,
  };
}

function scoreTone(score: number) {
  if (score >= 82) return "good";
  if (score >= 66) return "warn";
  return "bad";
}

function escapeHtml(value: unknown) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function toneForQuality(severity: QualityNote["severity"]) {
  if (severity === "good") return "good";
  if (severity === "bad") return "bad";
  return "warn";
}

function modelConfidenceCopy(label?: string) {
  if (label === "high") return "Strong read";
  if (label === "low") return "Low confidence";
  if (label === "medium") return "Usable read";
  return "Model status";
}

function qualityBadgeCopy(note: QualityNote) {
  if (note.id === "pose-confidence") return modelConfidenceCopy(note.label.startsWith("high") ? "high" : note.label.startsWith("low") ? "low" : "medium");
  if (note.id === "full-body") return note.severity === "good" ? "Full body visible" : "Feet hidden";
  if (note.id === "hands") return "Club/hands hidden";
  if (note.id === "view-angle") return note.label.includes("Face-on") ? "Better from down-the-line" : "Down-the-line view";
  if (note.id === "lighting") return note.severity === "good" ? "Lighting usable" : "Low light";
  if (note.id === "duration-long") return "Auto-trimmed";
  return note.label;
}

function estimatePracticeMinutes(items: PracticePlanItem[]) {
  return items.reduce((total, item) => total + (Number.parseInt(item.time, 10) || 0), 0);
}

function planItemFromDrill(drill: SwingDrill): PracticePlanItem {
  return {
    drillId: drill.id,
    name: drill.name,
    time: drill.time,
    reps: drill.reps,
    focus: drill.target,
    cue: drill.cue,
    checked: false,
  };
}

function planItemsFromAnalysis(result: any): PracticePlanItem[] {
  const drillsById = new Map<string, SwingDrill>((result?.drills ?? []).map((drill: SwingDrill) => [drill.id, drill]));

  return (result?.practicePlan?.items ?? []).map((item: any) => {
    const drill = drillsById.get(item.drillId);

    return {
      drillId: item.drillId,
      name: item.name,
      time: item.time,
      reps: item.reps,
      focus: item.focus ?? drill?.target ?? "Practice",
      cue: drill?.cue,
      checked: Boolean(item.checked),
    };
  });
}

function categoryFrom(report: any, id: string) {
  return report?.analysis?.categoryScores?.find((item: any) => item.id === id);
}

function categoryDeltaRows(currentAnalysis: any, compareReport?: StoredReport | null) {
  if (!currentAnalysis || !compareReport?.analysis?.categoryScores) return [];

  return currentAnalysis.categoryScores.map((score: any) => {
    const previous = categoryFrom(compareReport, score.id);
    return {
      id: score.id,
      label: score.label,
      current: score.score,
      previous: previous?.score ?? null,
      delta: previous ? score.score - previous.score : null,
    };
  });
}

function bestCategoryImprovement(previous: StoredReport | undefined | null, analysis: any) {
  const deltas = categoryDeltaRows(analysis, previous);
  if (!deltas.length) return null;
  return [...deltas].filter((item) => item.delta !== null).sort((a: any, b: any) => b.delta - a.delta)[0] ?? null;
}

function biggestCategoryRegression(previous: StoredReport | undefined | null, analysis: any) {
  const deltas = categoryDeltaRows(analysis, previous);
  if (!deltas.length) return null;
  return [...deltas].filter((item) => item.delta !== null).sort((a: any, b: any) => a.delta - b.delta)[0] ?? null;
}

function strongestCategory(reportOrAnalysis: any) {
  const scores = reportOrAnalysis?.analysis?.categoryScores ?? reportOrAnalysis?.categoryScores ?? [];
  return [...scores].sort((a: any, b: any) => b.score - a.score)[0] ?? null;
}

function weakestCategory(reportOrAnalysis: any) {
  const scores = reportOrAnalysis?.analysis?.categoryScores ?? reportOrAnalysis?.categoryScores ?? [];
  return [...scores].sort((a: any, b: any) => a.score - b.score)[0] ?? null;
}

function phaseComparisonRows(currentAnalysis: any, compareReport?: StoredReport | null) {
  const previousPhases = compareReport?.analysis?.phases ?? [];
  if (!currentAnalysis?.phases?.length || !previousPhases.length) return [];

  return currentAnalysis.phases
    .map((phase: any) => {
      const previous = previousPhases.find((item: any) => item.id === phase.id);
      if (!previous) return null;
      return {
        id: phase.id,
        label: phase.label,
        currentTime: phase.time,
        previousTime: previous.time,
        timeDelta: phase.time - previous.time,
        currentIssues: phase.issues?.length ?? 0,
        previousIssues: previous.issues?.length ?? 0,
      };
    })
    .filter(Boolean)
    .sort((a: any, b: any) => Math.abs(b.timeDelta) - Math.abs(a.timeDelta))
    .slice(0, 4);
}

function App() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const detectorRef = useRef<any>(null);
  const cancelAnalysisRef = useRef(false);

  const [sourceUrl, setSourceUrl] = useState("");
  const [sourceBlob, setSourceBlob] = useState<Blob | null>(null);
  const [sourceName, setSourceName] = useState("");
  const [dragging, setDragging] = useState(false);
  const [recording, setRecording] = useState(false);
  const [modelState, setModelState] = useState<ModelState>("idle");
  const [modelMessage, setModelMessage] = useState("Pose model loads on first analysis");
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [analysis, setAnalysis] = useState<any>(null);
  const [videoClipWindow, setVideoClipWindow] = useState<VideoClipWindow | null>(null);
  const [activePhaseIndex, setActivePhaseIndex] = useState(0);
  const [selectedFrameIndex, setSelectedFrameIndex] = useState(0);
  const [handedness, setHandedness] = useState<"right" | "left">("right");
  const [viewAngle, setViewAngle] = useState<ViewAngle>("face-on");
  const [user, setUser] = useState<User | null>(null);
  const [authMode, setAuthMode] = useState<"login" | "signup">("signup");
  const [authName, setAuthName] = useState("");
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authError, setAuthError] = useState("");
  const [history, setHistory] = useState<StoredReport[]>([]);
  const [leaderboard, setLeaderboard] = useState<any>(null);
  const [saveMessage, setSaveMessage] = useState("");
  const [reanalyzingId, setReanalyzingId] = useState("");
  const [deletingId, setDeletingId] = useState("");
  const [pendingReanalysisId, setPendingReanalysisId] = useState("");
  const [customPracticePlan, setCustomPracticePlan] = useState<PracticePlanItem[]>([]);
  const [comparisonReportId, setComparisonReportId] = useState("");

  const activePhase = analysis?.phases?.[activePhaseIndex];
  const activeFrame = analysis?.frames?.[selectedFrameIndex] ?? activePhase?.frame;
  const isReportReady = Boolean(analysis && !sourceUrl && !recording);
  const progressLabel = processing ? "Processing" : analysis ? "Report" : "Model";
  const progressValue = processing
    ? progress
    : analysis
      ? 100
      : modelState === "ready"
        ? 100
        : modelState === "error"
          ? 0
          : modelState === "loading"
            ? 38
            : 0;
  const currentStep: WorkflowStep = analysis ? "practice" : sourceUrl || recording || processing ? "analyze" : "capture";

  useEffect(() => {
    fetchLeaderboard();

    if (!isSupabaseConfigured) {
      setAuthError("Add Supabase environment variables to enable accounts and saving.");
      return () => stopStream();
    }

    const supabase = requireSupabase();
    supabase.auth.getSession().then(({ data }) => {
      const currentUser = data.session?.user ?? null;
      setUser(currentUser);
      if (currentUser) fetchHistory(currentUser);
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      const currentUser = session?.user ?? null;
      setUser(currentUser);
      if (currentUser) fetchHistory(currentUser);
      else setHistory([]);
    });

    return () => {
      subscription.unsubscribe();
      stopStream();
    };
  }, []);

  useEffect(() => {
    if (!pendingReanalysisId || !sourceUrl || processing) return;

    const timer = window.setTimeout(() => {
      setSaveMessage("Re-analyzing saved swing");
      setPendingReanalysisId("");
      void analyzeVideo();
    }, 350);

    return () => window.clearTimeout(timer);
  }, [pendingReanalysisId, sourceUrl, processing]);

  useEffect(() => {
    if (!comparisonReportId && history.length) {
      setComparisonReportId(history[0].id);
    }
    if (comparisonReportId && history.length && !history.some((report) => report.id === comparisonReportId)) {
      setComparisonReportId(history[0].id);
    }
  }, [comparisonReportId, history]);

  async function handleAuth(event: FormEvent) {
    event.preventDefault();
    setAuthError("");

    if (!isSupabaseConfigured) {
      setAuthError("Supabase is not configured. Copy .env.example to .env.local and add your project keys.");
      return;
    }

    const supabase = requireSupabase();

    try {
      const response =
        authMode === "signup"
          ? await supabase.auth.signUp({
              email: authEmail,
              password: authPassword,
              options: { data: { name: authName } },
            })
          : await supabase.auth.signInWithPassword({ email: authEmail, password: authPassword });

      if (response.error) throw response.error;

      const sessionUser = response.data.session?.user ?? null;
      setUser(sessionUser);
      setAuthPassword("");
      setSaveMessage("");
      if (!sessionUser && authMode === "signup") {
        setAuthError("Check your email to confirm your account, then log in.");
      }
      if (sessionUser) await fetchHistory(sessionUser);
      await fetchLeaderboard();
    } catch (error) {
      setAuthError(error instanceof Error ? error.message : "Sign in failed.");
    }
  }

  async function logout() {
    if (isSupabaseConfigured) {
      await requireSupabase().auth.signOut();
    }

    setUser(null);
    setHistory([]);
    setSaveMessage("");
  }

  function syncPracticePlanFromAnalysis(result: any) {
    setCustomPracticePlan(planItemsFromAnalysis(result).slice(0, 2));
  }

  function addDrillToPracticePlan(drill: SwingDrill) {
    setCustomPracticePlan((items) => {
      if (items.some((item) => item.drillId === drill.id)) return items;
      return [...items, planItemFromDrill(drill)];
    });
  }

  function removeDrillFromPracticePlan(drillId: string) {
    setCustomPracticePlan((items) => items.filter((item) => item.drillId !== drillId));
  }

  function togglePracticePlanItem(drillId: string) {
    setCustomPracticePlan((items) =>
      items.map((item) => (item.drillId === drillId ? { ...item, checked: !item.checked } : item)),
    );
  }

  function compactAnalysis(result = analysis) {
    if (!result) return null;
    const currentPlan = customPracticePlan.length ? customPracticePlan : planItemsFromAnalysis(result);

    return {
      overall: result.overall,
      viewAngle: result.viewAngle ?? viewAngle,
      analysisWindow: result.analysisWindow,
      trimWindow: result.trimWindow,
      captureQuality: result.captureQuality,
      qualityNotes: result.qualityNotes,
      modelConfidence: result.modelConfidence,
      categoryScores: result.categoryScores,
      mistakes: result.mistakes,
      drills: result.drills,
      practicePlan: currentPlan,
      customPracticePlan: currentPlan,
      practiceFocus: result.practicePlan?.focus,
      tempoRatio: result.tempoRatio,
      sequenceScore: result.sequenceScore,
      angleNotes: result.angleNotes,
      modelNotes: result.modelNotes,
      generatedAt: result.generatedAt,
      phases: result.phases?.map((phase: any) => ({
        id: phase.id,
        label: phase.label,
        time: phase.time,
        confidence: phase.confidence,
        confidenceLabel: phase.confidenceLabel,
        positives: phase.positives,
        issues: phase.issues,
        suggestion: phase.suggestion,
      })),
    };
  }

  function safeFileName(name: string) {
    return name.toLowerCase().replace(/[^a-z0-9.]+/g, "-").replace(/^-+|-+$/g, "") || "swing-video.webm";
  }

  async function fetchHistory(currentUser = user) {
    if (!currentUser || !isSupabaseConfigured) return;
    const supabase = requireSupabase();

    const { data, error } = await supabase
      .from("swing_videos")
      .select("*")
      .eq("user_id", currentUser.id)
      .order("created_at", { ascending: false });

    if (error) {
      setAuthError(error.message);
      return;
    }

    const rowsWithUrls = await Promise.all(
      (data ?? []).map(async (row: StoredReport) => {
        const { data: signedData } = await supabase.storage.from("swing-videos").createSignedUrl(row.storage_path, 60 * 60);
        return { ...row, signedUrl: signedData?.signedUrl };
      }),
    );

    setHistory(rowsWithUrls);
  }

  async function fetchLeaderboard() {
    if (!isSupabaseConfigured) {
      setLeaderboard(null);
      return;
    }

    try {
      const supabase = requireSupabase();
      const { data, error } = await supabase
        .from("swing_leaderboard_entries")
        .select("id,player_key,title,overall_score,tempo_score,impact_score,created_at")
        .order("created_at", { ascending: false })
        .limit(100);

      if (error) throw error;

      const rows = data ?? [];
      const latestByUser = new Map<string, any>();
      for (const row of rows) {
        if (!latestByUser.has(row.player_key)) latestByUser.set(row.player_key, row);
      }

      const latest = Array.from(latestByUser.values());
      const labelFor = (row: any) => row.title || `Golfer ${String(row.player_key).slice(0, 6)}`;
      const decorate = (row: any) => ({
        userName: labelFor(row),
        reportId: row.id,
        overall: row.overall_score,
        tempo: row.tempo_score ?? 0,
        impact: row.impact_score ?? 0,
        createdAt: row.created_at,
      });

      const reportsByUser = rows.reduce((map: Map<string, any[]>, row: any) => {
        const list = map.get(row.player_key) ?? [];
        list.push(row);
        map.set(row.player_key, list);
        return map;
      }, new Map<string, any[]>());

      const mostImproved = Array.from(reportsByUser.values())
        .filter((reports) => reports.length >= 2)
        .map((reports) => {
          const sorted = [...reports].sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
          const previous = sorted[sorted.length - 2];
          const current = sorted[sorted.length - 1];
          return {
            userName: labelFor(current),
            reportId: current.id,
            improvement: current.overall_score - previous.overall_score,
            currentOverall: current.overall_score,
          };
        })
        .filter((item) => item.improvement > 0)
        .sort((a, b) => b.improvement - a.improvement)
        .slice(0, 10);

      setLeaderboard({
        topOverall: latest.map(decorate).sort((a, b) => b.overall - a.overall).slice(0, 10),
        mostImproved,
        bestTempo: latest.map(decorate).sort((a, b) => b.tempo - a.tempo).slice(0, 10),
        bestImpact: latest.map(decorate).sort((a, b) => b.impact - a.impact).slice(0, 10),
      });
    } catch {
      setLeaderboard(null);
    }
  }

  async function saveCurrentReport() {
    if (!analysis) return;
    if (!isSupabaseConfigured) {
      setAuthError("Supabase is not configured. Add your URL and anon key to .env.local first.");
      return;
    }
    if (!user) {
      setAuthError("Create an account or sign in before saving swings.");
      return;
    }
    if (!sourceBlob) {
      setSaveMessage("Upload or record a real video before saving.");
      return;
    }
    if (sourceBlob.size > MAX_SAVED_VIDEO_BYTES) {
      setSaveMessage(`This video is too large to save. Keep saved swings under ${MAX_SAVED_VIDEO_LABEL}.`);
      return;
    }
    if (sourceBlob.type && !sourceBlob.type.startsWith("video/")) {
      setSaveMessage("Only video files can be saved.");
      return;
    }

    const supabase = requireSupabase();
    const fileName = safeFileName(sourceName);
    const storagePath = `${user.id}/${Date.now()}-${fileName}`;
    const title = sourceName ? sourceName.replace(/\.[^/.]+$/, "") : `Swing ${new Date().toLocaleDateString()}`;

    setSaveMessage("Uploading video");
    try {
      const { error: uploadError } = await supabase.storage.from("swing-videos").upload(storagePath, sourceBlob, {
        cacheControl: "3600",
        contentType: sourceBlob.type || "video/webm",
        upsert: false,
      });

      if (uploadError) throw uploadError;

      setSaveMessage("Saving analysis");
      const { error: insertError } = await supabase.from("swing_videos").insert({
        user_id: user.id,
        title,
        source_name: sourceName || fileName,
        storage_path: storagePath,
        handedness,
        swing_angle: analysis.viewAngle ?? viewAngle,
        overall_score: analysis.overall,
        analysis: compactAnalysis(),
      });

      if (insertError) throw insertError;

      setSaveMessage("Swing saved");
      await fetchHistory(user);
      await fetchLeaderboard();
    } catch (error) {
      setSaveMessage("");
      setAuthError(error instanceof Error ? error.message : "Could not save report.");
    }
  }

  async function reAnalyzeSavedSwing(report: StoredReport) {
    if (processing) {
      setSaveMessage("Finish the current analysis before loading another swing.");
      return;
    }
    if (!isSupabaseConfigured) {
      setAuthError("Supabase is not configured. Add your URL and anon key to .env.local first.");
      return;
    }

    setAuthError("");
    setSaveMessage(`Loading ${report.title || report.source_name}`);
    setReanalyzingId(report.id);

    try {
      const supabase = requireSupabase();
      const { data, error } = await supabase.storage.from("swing-videos").createSignedUrl(report.storage_path, 60 * 60);
      if (error) throw error;

      const videoUrl = data?.signedUrl ?? report.signedUrl;
      if (!videoUrl) throw new Error("Could not create a signed URL for this swing.");

      const response = await fetch(videoUrl);
      if (!response.ok) throw new Error("Could not load the saved swing video.");

      const blob = await response.blob();
      const fileName = report.source_name || `${report.title || "saved-swing"}.webm`;
      setHandedness(report.handedness);
      setViewAngle(report.swing_angle);
      setPendingReanalysisId(report.id);
      setVideoSource(URL.createObjectURL(blob), `Saved: ${fileName}`, blob);
    } catch (error) {
      setSaveMessage("");
      setAuthError(error instanceof Error ? error.message : "Could not re-analyze this saved swing.");
    } finally {
      setReanalyzingId("");
    }
  }

  async function deleteSavedSwing(report: StoredReport) {
    if (!isSupabaseConfigured) {
      setAuthError("Supabase is not configured. Add your URL and anon key to .env.local first.");
      return;
    }
    if (!user) {
      setAuthError("Sign in before deleting saved swings.");
      return;
    }

    const label = report.title || report.source_name || "this swing";
    const shouldDelete = window.confirm(`Delete "${label}" from your profile? This removes the saved report and video file.`);
    if (!shouldDelete) return;

    setAuthError("");
    setSaveMessage(`Deleting ${label}`);
    setDeletingId(report.id);

    try {
      const supabase = requireSupabase();
      const { error: deleteError } = await supabase
        .from("swing_videos")
        .delete()
        .eq("id", report.id)
        .eq("user_id", user.id);

      if (deleteError) throw deleteError;

      setHistory((items) => items.filter((item) => item.id !== report.id));

      const { error: storageError } = await supabase.storage.from("swing-videos").remove([report.storage_path]);
      if (storageError) {
        setSaveMessage("Removed from profile. Video cleanup needs another try.");
        await fetchHistory(user);
        return;
      }

      setSaveMessage("Saved swing deleted");
      await fetchLeaderboard();
    } catch (error) {
      setSaveMessage("");
      setAuthError(error instanceof Error ? error.message : "Could not delete this saved swing.");
      await fetchHistory(user);
    } finally {
      setDeletingId("");
    }
  }

  async function loadDetector() {
    if (detectorRef.current) return detectorRef.current;
    try {
      setModelState("loading");
      setModelMessage("Loading pose detector");
      if (!window.isSecureContext && location.hostname !== "localhost" && location.hostname !== "127.0.0.1") {
        setModelMessage("Camera recording needs localhost or HTTPS; upload still works.");
      }
      const [tf, poseDetection] = await Promise.all([
        import("@tensorflow/tfjs"),
        import("@tensorflow-models/pose-detection"),
        import("@tensorflow/tfjs-backend-webgl").then(() => null),
      ]).then(([tfModule, poseModule]) => [tfModule, poseModule] as const);
      await tf.setBackend("webgl");
      await tf.ready();
      if (import.meta.env.VITE_POSE_MODEL === "movenet") {
        try {
          detectorRef.current = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
            modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
            enableSmoothing: true,
          });
        } catch (moveNetError) {
          console.warn("MoveNet failed to load, using PoseNet fallback.", moveNetError);
        }
      }

      if (!detectorRef.current) {
        detectorRef.current = await poseDetection.createDetector(poseDetection.SupportedModels.PoseNet, {
          architecture: "MobileNetV1",
          outputStride: 16,
          inputResolution: { width: 257, height: 257 },
          multiplier: 0.75,
        });
      }
      setModelState("ready");
      setModelMessage("Pose model ready");
      return detectorRef.current;
    } catch (error) {
      console.error(error);
      setModelState("error");
      setModelMessage("Pose model could not load. Check the browser console or network access.");
      throw error;
    }
  }

  function stopStream() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }

  function setVideoSource(url: string, name: string, blob: Blob | null = null) {
    if (sourceUrl) URL.revokeObjectURL(sourceUrl);
    stopStream();
    setSourceUrl(url);
    setSourceBlob(blob);
    setSourceName(name);
    setAnalysis(null);
    setVideoClipWindow(null);
    setProgress(0);
    setActivePhaseIndex(0);
    setSelectedFrameIndex(0);
    setCustomPracticePlan([]);
    setSaveMessage("");
    window.setTimeout(() => {
      if (videoRef.current) {
        videoRef.current.srcObject = null;
        videoRef.current.src = url;
        videoRef.current.load();
      }
    }, 0);
  }

  function handleFiles(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    if (!file.type.startsWith("video/")) {
      setModelMessage("Choose a video file for analysis.");
      return;
    }
    setVideoSource(URL.createObjectURL(file), file.name, file);
    if (file.size > MAX_SAVED_VIDEO_BYTES) {
      setSaveMessage(`You can analyze this clip locally, but saved swings must be under ${MAX_SAVED_VIDEO_LABEL}.`);
    }
  }

  async function toggleRecording() {
    if (recording) {
      recorderRef.current?.stop();
      setRecording(false);
      return;
    }

    try {
      stopStream();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" },
        audio: false,
      });
      streamRef.current = stream;
      chunksRef.current = [];
      if (videoRef.current) {
        videoRef.current.src = "";
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;
        await videoRef.current.play();
      }
      const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
        ? "video/webm;codecs=vp9"
        : "video/webm";
      const recorder = new MediaRecorder(stream, { mimeType });
      recorderRef.current = recorder;
      recorder.ondataavailable = (event) => {
        if (event.data.size) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType });
        const url = URL.createObjectURL(blob);
        setVideoSource(url, `Recorded swing ${new Date().toLocaleTimeString()}.webm`, blob);
        stopStream();
      };
      recorder.start();
      setRecording(true);
      setSourceName("Recording");
      setAnalysis(null);
      setVideoClipWindow(null);
      setSaveMessage("");
    } catch (error) {
      console.error(error);
      setModelMessage("Camera recording was not available. Upload a video instead.");
    }
  }

  async function seekVideo(video: HTMLVideoElement, time: number) {
    const clamped = Math.max(0, Math.min(time, Math.max(video.duration - 0.02, 0)));
    if (Math.abs(video.currentTime - clamped) < 0.015) return;
    const done = waitForEvent(video, "seeked");
    video.currentTime = clamped;
    await done;
  }

  async function detectVideoClipWindow(video: HTMLVideoElement, rawDuration: number) {
    if (!Number.isFinite(rawDuration) || rawDuration <= 14) {
      return chooseImportantVideoWindow([], Number.isFinite(rawDuration) ? rawDuration : 4);
    }

    setModelMessage("Finding swing in the long clip");
    const videoRatio = video.videoWidth && video.videoHeight ? video.videoHeight / video.videoWidth : 9 / 16;
    const scanCanvas = document.createElement("canvas");
    scanCanvas.width = 112;
    scanCanvas.height = Math.max(64, Math.round(scanCanvas.width * videoRatio));
    const scanCtx = scanCanvas.getContext("2d", { willReadFrequently: true });
    if (!scanCtx) return chooseImportantVideoWindow([], rawDuration);

    const sampleCount = Math.max(28, Math.min(120, Math.round(rawDuration * 2.1)));
    const samples: MotionSample[] = [];
    let previousPixels: Uint8ClampedArray | null = null;

    for (let index = 0; index < sampleCount; index += 1) {
      if (cancelAnalysisRef.current) return null;
      const time = (rawDuration * index) / Math.max(sampleCount - 1, 1);
      await seekVideo(video, time);
      scanCtx.drawImage(video, 0, 0, scanCanvas.width, scanCanvas.height);
      const sample = readMotionSample(scanCtx, scanCanvas.width, scanCanvas.height, previousPixels);
      previousPixels = sample.pixels;
      samples.push({ time, motion: sample.motion, brightness: sample.brightness });
      setProgress(Math.min(24, Math.round(((index + 1) / sampleCount) * 24)));
      if (index % 10 === 0) await new Promise((resolve) => window.requestAnimationFrame(resolve));
    }

    return chooseImportantVideoWindow(samples, rawDuration);
  }

  function handleVideoPlay(event: SyntheticEvent<HTMLVideoElement>) {
    if (!videoClipWindow || recording) return;
    const video = event.currentTarget;
    if (video.currentTime < videoClipWindow.start || video.currentTime > videoClipWindow.end) {
      video.currentTime = Math.max(0, Math.min(videoClipWindow.start, Math.max(video.duration - 0.02, 0)));
    }
  }

  function handleVideoTimeUpdate(event: SyntheticEvent<HTMLVideoElement>) {
    if (!videoClipWindow || recording) return;
    const video = event.currentTarget;
    if (video.currentTime > videoClipWindow.end) {
      video.pause();
      video.currentTime = Math.max(0, Math.min(videoClipWindow.start, Math.max(video.duration - 0.02, 0)));
    }
  }

  async function analyzeVideo() {
    const video = videoRef.current;
    if (!video || !sourceUrl || processing) return;

    cancelAnalysisRef.current = false;
    setProcessing(true);
    setProgress(0);
    setModelMessage("Preparing video");
    try {
      if (video.readyState < 1) await waitForEvent(video, "loadedmetadata");
      video.pause();
      const rawDuration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 4;
      const analysisWindow = await detectVideoClipWindow(video, rawDuration);
      if (!analysisWindow || cancelAnalysisRef.current) {
        setModelMessage("Analysis canceled");
        setProgress(0);
        return;
      }

      setVideoClipWindow(analysisWindow);
      if (analysisWindow.source === "motion") {
        video.currentTime = Math.max(0, Math.min(analysisWindow.start, Math.max(video.duration - 0.02, 0)));
      }
      setModelMessage(
        analysisWindow.source === "motion"
          ? `Auto-trimmed to ${formatTime(analysisWindow.start)} to ${formatTime(analysisWindow.end)}`
          : "Sampling swing frames",
      );

      const detector = await loadDetector();
      const duration = Math.max(1, analysisWindow.end - analysisWindow.start);
      const sampleCount = Math.max(36, Math.min(96, Math.round(duration * 10)));
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Canvas context unavailable");
      const videoRatio = video.videoWidth && video.videoHeight ? video.videoHeight / video.videoWidth : 9 / 16;
      canvas.width = 720;
      canvas.height = Math.round(canvas.width * videoRatio);
      const frames = [];

      for (let i = 0; i < sampleCount; i += 1) {
        if (cancelAnalysisRef.current) {
          setModelMessage("Analysis canceled");
          setProgress(0);
          return;
        }
        const time = analysisWindow.start + (duration * i) / Math.max(sampleCount - 1, 1);
        await seekVideo(video, time);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        // Each sample is a seeked frame, so stale MoveNet crop tracking can drift away from the golfer.
        detector.reset?.();
        const poses = await detector.estimatePoses(canvas, { maxPoses: 1, flipHorizontal: false });
        const pose = normalizePoseForFrame(poses?.[0] ?? null, canvas.width, canvas.height);
        frames.push({
          index: i,
          time,
          image: canvas.toDataURL("image/jpeg", 0.76),
          width: canvas.width,
          height: canvas.height,
          brightness: estimateFrameBrightness(ctx, canvas.width, canvas.height) ?? undefined,
          pose,
        });
        setProgress(Math.min(100, 25 + Math.round(((i + 1) / sampleCount) * 75)));
        if (i % 8 === 0) await new Promise((resolve) => window.requestAnimationFrame(resolve));
      }

      const result = SwingGradeEngine.analyzeSwing(frames, { handedness, viewAngle, videoDuration: rawDuration, analysisWindow });
      result.analysisWindow = analysisWindow;
      setAnalysis(result);
      syncPracticePlanFromAnalysis(result);
      setActivePhaseIndex(0);
      setSelectedFrameIndex(result.phases?.[0]?.index ?? 0);
      if (videoRef.current && analysisWindow.source === "motion") {
        videoRef.current.currentTime = Math.max(0, Math.min(analysisWindow.start, Math.max(videoRef.current.duration - 0.02, 0)));
      }
      setModelMessage(
        analysisWindow.source === "motion"
          ? `Analysis complete. Trimmed playback to ${formatTime(analysisWindow.start)} to ${formatTime(analysisWindow.end)}.`
          : "Analysis complete",
      );
      setSaveMessage("");
    } catch (error) {
      console.error(error);
      setModelMessage("Analysis stopped. Try a shorter clip with the full body visible.");
    } finally {
      setProcessing(false);
    }
  }

  function cancelAnalysis() {
    cancelAnalysisRef.current = true;
    setModelMessage("Canceling analysis");
  }

  function selectPhase(index: number) {
    const phase = analysis?.phases?.[index];
    setActivePhaseIndex(index);
    if (phase) setSelectedFrameIndex(phase.index);
  }

  function selectFrame(index: number) {
    if (!analysis?.frames?.length) return;
    const nextIndex = Math.max(0, Math.min(index, analysis.frames.length - 1));
    setSelectedFrameIndex(nextIndex);
    const nearestPhaseIndex = analysis.phases.reduce((nearest: number, phase: any, phaseIndex: number) => {
      const currentGap = Math.abs(analysis.phases[nearest].index - nextIndex);
      const nextGap = Math.abs(phase.index - nextIndex);
      return nextGap < currentGap ? phaseIndex : nearest;
    }, 0);
    setActivePhaseIndex(nearestPhaseIndex);
  }

  function loadDemoReport() {
    const demo = SwingGradeEngine.createDemoAnalysis();
    demo.viewAngle = viewAngle;
    setAnalysis(demo);
    setVideoClipWindow(null);
    syncPracticePlanFromAnalysis(demo);
    setActivePhaseIndex(0);
    setSelectedFrameIndex(demo.phases?.[0]?.index ?? 0);
    setSourceName("Sample movement report");
    setProgress(100);
    setModelMessage("Sample report loaded");
    setSaveMessage("");
  }

  function resetAll() {
    stopStream();
    if (sourceUrl) URL.revokeObjectURL(sourceUrl);
    setSourceUrl("");
    setSourceBlob(null);
    setSourceName("");
    setAnalysis(null);
    setVideoClipWindow(null);
    setProgress(0);
    setActivePhaseIndex(0);
    setSelectedFrameIndex(0);
    setCustomPracticePlan([]);
    setSaveMessage("");
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.removeAttribute("src");
      videoRef.current.load();
    }
  }

  function exportReport() {
    if (!analysis) return;
    const plan = customPracticePlan.length ? customPracticePlan : planItemsFromAnalysis(analysis);
    const compareReport = history.find((report) => report.id === comparisonReportId) ?? history[0];
    const improvement = bestCategoryImprovement(compareReport, analysis);
    const regression = biggestCategoryRegression(compareReport, analysis);
    const analyzedWindow = analysis.analysisWindow;
    const phaseRows = analysis.phases
      .map(
        (phase: any) => `
          <tr>
            <td>${escapeHtml(phase.label)}</td>
            <td>${escapeHtml(formatTime(phase.time))}</td>
            <td>${escapeHtml(phase.confidenceLabel ?? "n/a")}</td>
            <td>${escapeHtml(phase.positives.join("; "))}</td>
            <td>${escapeHtml(phase.issues.join("; "))}</td>
            <td>${escapeHtml(phase.suggestion)}</td>
          </tr>`,
      )
      .join("");
    const categoryRows = analysis.categoryScores
      .map(
        (item: any) =>
          `<tr><td>${escapeHtml(item.label)}</td><td>${escapeHtml(item.score)}</td><td>${escapeHtml(item.note)}</td><td>${escapeHtml(item.reason ?? "")}</td></tr>`,
      )
      .join("");
    const mistakeRows =
      analysis.mistakes
        .map((item: any) => `<li><strong>${escapeHtml(item.label)}</strong>: ${escapeHtml(item.evidence)}</li>`)
        .join("") || "<li>No major faults flagged.</li>";
    const drillRows = analysis.drills
      .map((item: any) => `<li><strong>${escapeHtml(item.name)}</strong>: ${escapeHtml(item.cue)} ${escapeHtml(item.reps)}.</li>`)
      .join("");
    const planRows = plan
      .map(
        (item) =>
          `<li><strong>${item.checked ? "Done: " : ""}${escapeHtml(item.name)}</strong>: ${escapeHtml(item.time)}, ${escapeHtml(item.reps)}. ${escapeHtml(item.cue ?? "")}</li>`,
      )
      .join("");
    const qualityRows = (analysis.qualityNotes ?? [])
      .map((item: QualityNote) => `<li><strong>${escapeHtml(item.label)}</strong> (${escapeHtml(item.severity)}): ${escapeHtml(item.detail)}</li>`)
      .join("");
    const modelRows = [...(analysis.angleNotes ?? []), ...(analysis.modelNotes ?? [])]
      .map((note: string) => `<li>${escapeHtml(note)}</li>`)
      .join("");
    const phaseCompareRows = phaseComparisonRows(analysis, compareReport)
      .map(
        (row: any) =>
          `<tr><td>${escapeHtml(row.label)}</td><td>${escapeHtml(formatTime(row.previousTime))}</td><td>${escapeHtml(formatTime(row.currentTime))}</td><td>${escapeHtml(roundNumber(row.timeDelta, 2))}s</td><td>${escapeHtml(row.previousIssues)} to ${escapeHtml(row.currentIssues)}</td></tr>`,
      )
      .join("");
    const comparisonBlock = compareReport
      ? `<h2>Before and after</h2>
          <p><strong>Compared against:</strong> ${escapeHtml(compareReport.title || compareReport.source_name)} from ${escapeHtml(new Date(compareReport.created_at).toLocaleDateString())}</p>
          <p><strong>Overall:</strong> ${escapeHtml(compareReport.overall_score)} to ${escapeHtml(analysis.overall)} (${analysis.overall - compareReport.overall_score >= 0 ? "+" : ""}${escapeHtml(analysis.overall - compareReport.overall_score)})</p>
          <p><strong>Biggest improvement:</strong> ${escapeHtml(improvement?.label ?? "n/a")} ${improvement?.delta !== null && improvement?.delta !== undefined ? `(${improvement.delta >= 0 ? "+" : ""}${escapeHtml(improvement.delta)})` : ""}</p>
          <p><strong>Biggest regression:</strong> ${escapeHtml(regression?.label ?? "n/a")} ${regression?.delta !== null && regression?.delta !== undefined ? `(${regression.delta >= 0 ? "+" : ""}${escapeHtml(regression.delta)})` : ""}</p>
          ${phaseCompareRows ? `<table><thead><tr><th>Phase</th><th>Saved</th><th>Current</th><th>Timing change</th><th>Issue count</th></tr></thead><tbody>${phaseCompareRows}</tbody></table>` : ""}`
      : "";
    const html = `<!doctype html>
      <html>
        <head>
          <meta charset="utf-8" />
          <title>SwingGrade Report</title>
          <style>
            body{background:oklch(97.3% 0.008 185);font-family:Arial,sans-serif;color:oklch(24% 0.018 165);line-height:1.48;margin:32px;max-width:1040px}
            h1{font-size:34px;margin-bottom:6px} h2{border-top:1px solid oklch(88% 0.012 190);margin-top:28px;padding-top:18px}
            table{border-collapse:collapse;width:100%;margin:14px 0 24px;background:oklch(99% 0.006 165)}
            td,th{border:1px solid oklch(88% 0.012 190);padding:9px;vertical-align:top} th{background:oklch(95% 0.01 185);text-align:left}
            .summary{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:18px 0}
            .box{background:oklch(99% 0.006 165);border:1px solid oklch(88% 0.012 190);border-radius:8px;padding:12px}
            .box span{color:oklch(48% 0.018 165);display:block;font-size:12px;font-weight:700;text-transform:uppercase}.box strong{font-size:22px}
            li{margin-bottom:7px}
          </style>
        </head>
        <body>
          <h1>SwingGrade Report</h1>
          <p>Coach-friendly analysis summary generated from 2D pose landmarks and rule-based movement checks.</p>
          <div class="summary">
            <div class="box"><span>Overall</span><strong>${analysis.overall}/100</strong></div>
            <div class="box"><span>Model confidence</span><strong>${escapeHtml(analysis.modelConfidence?.label ?? "n/a")}</strong></div>
            <div class="box"><span>Plan items</span><strong>${plan.length}</strong></div>
            <div class="box"><span>Fault flags</span><strong>${analysis.mistakes.length}</strong></div>
          </div>
          <p><strong>View angle:</strong> ${escapeHtml(analysis.viewAngle)} | <strong>Analyzed window:</strong> ${escapeHtml(
            analyzedWindow ? `${formatTime(analyzedWindow.start)} to ${formatTime(analyzedWindow.end)}` : "Full clip",
          )} | <strong>Detected swing:</strong> ${escapeHtml(
            analysis.trimWindow ? `${formatTime(analysis.trimWindow.start)} to ${formatTime(analysis.trimWindow.end)}` : "Not available",
          )}</p>
          <p><strong>Video:</strong> ${escapeHtml(sourceName || "Uploaded swing")} | <strong>Generated:</strong> ${escapeHtml(new Date(
            analysis.generatedAt,
          ).toLocaleString())}</p>
          <h2>Category scores</h2>
          <table><thead><tr><th>Category</th><th>Score</th><th>Basis</th><th>Reason</th></tr></thead><tbody>${categoryRows}</tbody></table>
          <h2>Phase checkpoints</h2>
          <table><thead><tr><th>Phase</th><th>Time</th><th>Confidence</th><th>Did well</th><th>Needs work</th><th>Suggestion</th></tr></thead><tbody>${phaseRows}</tbody></table>
          <h2>Detected mistakes</h2><ul>${mistakeRows}</ul>
          ${comparisonBlock}
          <h2>Custom practice plan</h2><ul>${planRows || "<li>No custom drills selected.</li>"}</ul>
          <h2>Recommended drills</h2><ul>${drillRows}</ul>
          <h2>Video and camera quality</h2><ul>${qualityRows}</ul>
          <h2>Model limitations</h2><ul>${modelRows}</ul>
        </body>
      </html>`;
    const blob = new Blob([html], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "swinggrade-report.html";
    anchor.click();
    URL.revokeObjectURL(url);
  }

  const statusBadge = useMemo(() => {
    if (processing) return <span className="badge warn"><span className="pulse-dot" />Analyzing</span>;
    if (modelState === "ready") return <span className="badge good">Model ready</span>;
    if (modelState === "error") return <span className="badge bad">Model issue</span>;
    if (modelState === "loading") return <span className="badge warn"><span className="pulse-dot" />Loading</span>;
    return <span className="badge warn">Model on demand</span>;
  }, [modelState, processing]);

  return (
    <main
      className="app-shell analyzer-luxury"
      style={{ "--analyzer-backdrop": `url(${ANALYZER_BACKDROP})` } as CSSProperties}
    >
      <header className="app-header">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="brand-mark">
              <span className="text-lg font-black">SG</span>
            </div>
            <div>
              <h1 className="text-2xl font-black tracking-normal text-ink">SwingGrade</h1>
              <p className="text-sm font-semibold text-slate-500">Pose-based golf swing analysis</p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <Link className="secondary-button analyzer-home-link" to="/">
              Home
            </Link>
            {statusBadge}
            <div className="segmented" aria-label="Golfer handedness">
              <button className={handedness === "right" ? "active" : ""} onClick={() => setHandedness("right")}>
                Right
              </button>
              <button className={handedness === "left" ? "active" : ""} onClick={() => setHandedness("left")}>
                Left
              </button>
            </div>
            <button className="secondary-button" onClick={exportReport} disabled={!analysis} title="Export report">
              <span className="h-5 w-5">{ICONS.download}</span>
              Export
            </button>
            <button className="secondary-button" onClick={saveCurrentReport} disabled={!analysis || !user} title="Save swing">
              <span className="h-5 w-5">{ICONS.save}</span>
              Save Swing
            </button>
            <AccountPanel
              user={user}
              authMode={authMode}
              setAuthMode={setAuthMode}
              authName={authName}
              setAuthName={setAuthName}
              authEmail={authEmail}
              setAuthEmail={setAuthEmail}
              authPassword={authPassword}
              setAuthPassword={setAuthPassword}
              authError={authError}
              handleAuth={handleAuth}
              logout={logout}
              history={history}
              saveMessage={saveMessage}
              onReanalyze={reAnalyzeSavedSwing}
              onDelete={deleteSavedSwing}
              reanalyzingId={reanalyzingId}
              deletingId={deletingId}
            />
          </div>
        </div>
      </header>

      <section className="analyzer-workspace mx-auto grid max-w-7xl gap-5 px-4 py-5 sm:px-6 lg:grid-cols-[minmax(0,1fr)_22rem] lg:px-8">
        <div className="tool-panel workspace-panel p-4">
          <WorkflowStepper currentStep={currentStep} />
          <div
            className={`drop-zone grid place-items-center p-4 ${dragging ? "dragging" : ""} ${isReportReady ? "report-ready" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              handleFiles(event.dataTransfer.files);
            }}
          >
            {sourceUrl || recording ? (
              <div className="grid w-full gap-3">
                <div className="video-frame aspect-video">
                  <video
                    ref={videoRef}
                    controls={!recording}
                    playsInline
                    muted={recording}
                    onPlay={handleVideoPlay}
                    onTimeUpdate={handleVideoTimeUpdate}
                  />
                </div>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-black text-ink">{sourceName || "Swing video"}</p>
                    <p className="text-xs font-semibold text-slate-500">{modelMessage}</p>
                    {videoClipWindow?.source === "motion" ? (
                      <p className="clip-window-note">
                        Auto-trimmed playback: {formatTime(videoClipWindow.start)} to {formatTime(videoClipWindow.end)}
                      </p>
                    ) : null}
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <button className="secondary-button" onClick={() => fileInputRef.current?.click()}>
                      <span className="h-5 w-5">{ICONS.upload}</span>
                      Replace
                    </button>
                    <button className="ghost-button" onClick={resetAll} title="Reset">
                      <span className="h-5 w-5">{ICONS.reset}</span>
                      Reset
                    </button>
                  </div>
                </div>
              </div>
            ) : analysis ? (
              <ReportReadyPanel
                analysis={analysis}
                activePhase={activePhase}
                onUpload={() => fileInputRef.current?.click()}
                onRecord={toggleRecording}
                onReset={resetAll}
                onSave={saveCurrentReport}
                user={user}
              />
            ) : (
              <div className="grid max-w-xl justify-items-center gap-4 text-center">
                <div className="upload-icon">
                  <span className="h-8 w-8">{ICONS.upload}</span>
                </div>
                <div>
                  <h2 className="text-2xl font-black tracking-normal text-ink">Upload or record a swing</h2>
                  <p className="mt-1 text-sm font-semibold text-slate-500">Full body in frame gives the cleanest keypoints.</p>
                </div>
                <CameraSetupGuide viewAngle={viewAngle} setViewAngle={setViewAngle} />
                <div className="flex flex-wrap justify-center gap-2">
                  <button className="primary-button" onClick={() => fileInputRef.current?.click()}>
                    <span className="h-5 w-5">{ICONS.upload}</span>
                    Upload
                  </button>
                  <button className="secondary-button" onClick={toggleRecording}>
                    <span className="h-5 w-5">{ICONS.camera}</span>
                    Record
                  </button>
                  <button className="ghost-button" onClick={loadDemoReport}>
                    Sample report
                  </button>
                </div>
              </div>
            )}
          </div>

          <div className="action-bar mt-4">
            <input
              ref={fileInputRef}
              className="hidden"
              type="file"
              accept="video/*"
              onChange={(event) => handleFiles(event.target.files)}
            />
            <div className="flex flex-wrap items-center gap-2">
              <button className="primary-button" onClick={processing ? cancelAnalysis : analyzeVideo} disabled={(!sourceUrl && !processing) || modelState === "error"}>
                <span className="h-5 w-5">{processing ? ICONS.stop : ICONS.analyze}</span>
                {processing ? "Cancel analysis" : "Analyze swing"}
              </button>
              <button className="secondary-button" onClick={toggleRecording}>
                <span className="h-5 w-5">{recording ? ICONS.stop : ICONS.camera}</span>
                {recording ? "Stop" : "Record"}
              </button>
            </div>
            <div className="min-w-[12rem] flex-1 sm:max-w-sm">
              <div className="mb-1 flex justify-between text-xs font-black uppercase text-slate-500">
                <span>{progressLabel}</span>
                <span>{progressValue}%</span>
              </div>
              <div className="meter">
                <span style={{ width: `${progressValue}%` }} />
              </div>
            </div>
          </div>
        </div>

        <aside className="grid content-start gap-5">
          <GradePanel analysis={analysis} modelState={modelState} modelMessage={modelMessage} processing={processing} />
          <ProgressDashboard history={history} />
        </aside>
      </section>

      {analysis ? (
        <>
          <PracticePlanFocus
            analysis={analysis}
            customPracticePlan={customPracticePlan}
            removeDrillFromPracticePlan={removeDrillFromPracticePlan}
            togglePracticePlanItem={togglePracticePlanItem}
          />
          <Timeline
            analysis={analysis}
            activePhaseIndex={activePhaseIndex}
            selectedFrameIndex={selectedFrameIndex}
            selectPhase={selectPhase}
            selectFrame={selectFrame}
          />
          <Comparison phase={activePhase} activeFrame={activeFrame} analysis={analysis} />
          <SwingReport
            analysis={analysis}
            activePhase={activePhase}
            history={history}
            leaderboard={leaderboard}
            saveCurrentReport={saveCurrentReport}
            user={user}
            customPracticePlan={customPracticePlan}
            addDrillToPracticePlan={addDrillToPracticePlan}
            removeDrillFromPracticePlan={removeDrillFromPracticePlan}
            togglePracticePlanItem={togglePracticePlanItem}
            comparisonReportId={comparisonReportId}
            setComparisonReportId={setComparisonReportId}
          />
        </>
      ) : (
        <EmptyDashboard modelMessage={modelMessage} />
      )}
    </main>
  );
}

function WorkflowStepper({ currentStep }: { currentStep: WorkflowStep }) {
  const steps: Array<{ id: WorkflowStep; label: string; detail: string }> = [
    { id: "capture", label: "Capture", detail: "Frame the body and club" },
    { id: "analyze", label: "Analyze", detail: "Sample pose and phases" },
    { id: "practice", label: "Practice", detail: "Use the next drill" },
  ];
  const currentIndex = steps.findIndex((step) => step.id === currentStep);

  return (
    <nav className="workflow-stepper" aria-label="Swing analysis workflow">
      {steps.map((step, index) => {
        const state = index === currentIndex ? "active" : index < currentIndex ? "complete" : "upcoming";
        return (
          <div className={`workflow-step ${state}`} key={step.id} aria-current={state === "active" ? "step" : undefined}>
            <span className="workflow-index">{index + 1}</span>
            <span>
              <strong>{step.label}</strong>
              <small>{step.detail}</small>
            </span>
          </div>
        );
      })}
    </nav>
  );
}

function CameraSetupGuide({ viewAngle, setViewAngle }: { viewAngle: ViewAngle; setViewAngle: (value: ViewAngle) => void }) {
  const checks = ["Full body visible", "Stable camera", "Good lighting", "Club and hands visible", "Waist-height camera"];
  const angleCopy =
    viewAngle === "face-on"
      ? "Best for weight shift, head movement, spine angle, and balance."
      : "Best for swing plane, early extension, posture loss, and path direction.";

  return (
    <div className="camera-guide">
      <div className="guide-topline">
        <span className="text-xs font-black uppercase text-slate-500">Camera setup</span>
        <div className="segmented compact" aria-label="Swing camera angle">
          <button className={viewAngle === "face-on" ? "active" : ""} onClick={() => setViewAngle("face-on")}>
            Face-on
          </button>
          <button className={viewAngle === "down-the-line" ? "active" : ""} onClick={() => setViewAngle("down-the-line")}>
            Down-the-line
          </button>
        </div>
      </div>
      <p className="text-sm font-semibold leading-6 text-slate-600">{angleCopy}</p>
      <div className="checklist">
        {checks.map((item) => (
          <span key={item}>{item}</span>
        ))}
      </div>
    </div>
  );
}

function AccountPanel({
  user,
  authMode,
  setAuthMode,
  authName,
  setAuthName,
  authEmail,
  setAuthEmail,
  authPassword,
  setAuthPassword,
  authError,
  handleAuth,
  logout,
  history,
  saveMessage,
  onReanalyze,
  onDelete,
  reanalyzingId,
  deletingId,
}: any) {
  const [open, setOpen] = useState(false);
  const profileName = user?.user_metadata?.name || user?.email || "Profile";

  return (
    <div className="profile-menu">
      <button className="profile-trigger" onClick={() => setOpen((value) => !value)} aria-expanded={open} aria-label="Open profile">
        <span className="profile-avatar">
          <span className="h-4 w-4">{ICONS.user}</span>
        </span>
        <span className="profile-trigger-copy">
          <strong>{user ? "Profile" : "Sign in"}</strong>
          <small>{user ? `${history.length} saved` : "Save swings"}</small>
        </span>
      </button>

      {open ? (
        <section className="profile-popover" aria-label="Profile menu">
          <div className="profile-popover-head">
            <div>
              <h2>Profile</h2>
              <p>{user ? `${history.length} saved swings` : "Sign in to save swing history"}</p>
            </div>
            <button className="ghost-button compact-button" onClick={() => setOpen(false)}>
              Close
            </button>
          </div>

          {user ? (
            <div className="grid gap-3">
              <div className="account-user">
                <strong>{profileName}</strong>
                <span>{user.email}</span>
              </div>
              {saveMessage ? <p className="text-sm font-black text-moss">{saveMessage}</p> : null}
              {authError ? <p className="text-sm font-semibold text-clay">{authError}</p> : null}
              <div className="history-heading">
                <strong>My Swings</strong>
                <span>{history.length ? "Play or re-analyze saved videos" : "No saved swings yet"}</span>
              </div>
              <div className="history-list">
                {history.map((report: StoredReport) => (
                  <div key={report.id} className="saved-swing-row">
                    <div className="saved-swing-topline">
                      <div className="saved-swing-title">
                        <strong>{report.title || report.source_name || "Saved swing"}</strong>
                        <span>
                          {new Date(report.created_at).toLocaleDateString()} · {report.swing_angle}
                        </span>
                      </div>
                      <span className={`badge ${scoreTone(report.overall_score)}`}>{report.overall_score}</span>
                    </div>
                    {report.signedUrl ? <video className="saved-swing-video" src={report.signedUrl} controls /> : null}
                    <div className="saved-swing-actions">
                      <button
                        className="secondary-button compact-button"
                        onClick={() => {
                          onReanalyze(report);
                          setOpen(false);
                        }}
                        disabled={reanalyzingId === report.id}
                      >
                        <span className="h-4 w-4">{ICONS.analyze}</span>
                        {reanalyzingId === report.id ? "Loading" : "Re-analyze"}
                      </button>
                      <button
                        className="danger-button compact-button"
                        onClick={() => onDelete(report)}
                        disabled={deletingId === report.id || reanalyzingId === report.id}
                      >
                        <span className="h-4 w-4">{ICONS.trash}</span>
                        {deletingId === report.id ? "Deleting" : "Delete"}
                      </button>
                    </div>
                  </div>
                ))}
                {!history.length ? <p className="text-sm font-semibold text-slate-500">Saved swing videos will appear here after you analyze and save.</p> : null}
              </div>
              <button className="ghost-button justify-self-start" onClick={logout}>
                Sign out
              </button>
            </div>
          ) : (
            <form className="grid gap-2" onSubmit={handleAuth}>
              <div className="segmented account-tabs" aria-label="Account mode">
                <button type="button" className={authMode === "signup" ? "active" : ""} onClick={() => setAuthMode("signup")}>
                  Sign up
                </button>
                <button type="button" className={authMode === "login" ? "active" : ""} onClick={() => setAuthMode("login")}>
                  Login
                </button>
              </div>
              {authMode === "signup" ? (
                <label className="field-label">
                  Name
                  <input value={authName} onChange={(event) => setAuthName(event.target.value)} placeholder="Your name" />
                </label>
              ) : null}
              <label className="field-label">
                Email
                <input value={authEmail} onChange={(event) => setAuthEmail(event.target.value)} placeholder="you@example.com" type="email" />
              </label>
              <label className="field-label">
                Password
                <input
                  value={authPassword}
                  onChange={(event) => setAuthPassword(event.target.value)}
                  placeholder="6+ characters"
                  type="password"
                />
              </label>
              {authError ? <p className="text-sm font-semibold text-clay">{authError}</p> : null}
              <button className="primary-button" type="submit">
                {authMode === "signup" ? "Create account" : "Login"}
              </button>
            </form>
          )}
        </section>
      ) : null}
    </div>
  );
}

function ReportReadyPanel({ analysis, activePhase, onUpload, onRecord, onReset, onSave, user }: any) {
  const weakest = [...analysis.categoryScores].sort((a: any, b: any) => a.score - b.score)[0];
  const best = [...analysis.categoryScores].sort((a: any, b: any) => b.score - a.score)[0];

  return (
    <div className="analysis-ready-panel w-full">
      <div>
        <span className="badge good">Report loaded</span>
        <h2 className="mt-3 text-2xl font-black tracking-normal text-ink">Sample swing analysis</h2>
        <p className="mt-2 max-w-2xl text-sm font-semibold leading-6 text-slate-600">
          Review the checkpoint timeline below, or upload a real swing to replace this sample with your own pose data.
        </p>
      </div>
      <div className="insight-grid">
        <div className="insight-tile">
          <span>Current phase</span>
          <strong>{activePhase?.label ?? "Setup"}</strong>
          <small>{formatTime(activePhase?.time ?? 0)}</small>
        </div>
        <div className="insight-tile">
          <span>Best category</span>
          <strong>{best?.label ?? "Impact position"}</strong>
          <small>{best?.score ?? "--"}/100</small>
        </div>
        <div className="insight-tile priority">
          <span>Practice priority</span>
          <strong>{weakest?.label ?? "Tempo"}</strong>
          <small>{weakest?.score ?? "--"}/100</small>
        </div>
      </div>
      <div className="flex flex-wrap gap-2">
        <button className="primary-button" onClick={onUpload}>
          <span className="h-5 w-5">{ICONS.upload}</span>
          Upload swing
        </button>
        <button className="secondary-button" onClick={onRecord}>
          <span className="h-5 w-5">{ICONS.camera}</span>
          Record swing
        </button>
        <button className="secondary-button" onClick={onSave} disabled={!user}>
          <span className="h-5 w-5">{ICONS.save}</span>
          Save Swing
        </button>
        <button className="ghost-button" onClick={onReset}>
          <span className="h-5 w-5">{ICONS.reset}</span>
          Reset
        </button>
      </div>
    </div>
  );
}

function PracticePlanFocus({ analysis, customPracticePlan, removeDrillFromPracticePlan, togglePracticePlanItem }: any) {
  const plan: PracticePlanItem[] = customPracticePlan.length ? customPracticePlan : planItemsFromAnalysis(analysis).slice(0, 2);
  const biggestIssue = analysis?.mistakes?.[0];
  const weakest = weakestCategory(analysis);
  const strongest = strongestCategory(analysis);
  const practiceMinutes = estimatePracticeMinutes(plan);

  return (
    <section className="mx-auto max-w-7xl px-4 pb-5 sm:px-6 lg:px-8">
      <div className="practice-focus-card">
        <div>
          <span className="badge good">Practice plan</span>
          <h2>Do this next</h2>
          <p>
            {biggestIssue?.label ?? weakest?.label ?? "No major fault flag"} is the priority. Your strongest pattern is{" "}
            {strongest?.label?.toLowerCase() ?? "still being measured"}.
          </p>
          <div className="practice-focus-stats">
            <div>
              <span>Top issue</span>
              <strong>{biggestIssue?.label ?? weakest?.label ?? "No major issue"}</strong>
            </div>
            <div>
              <span>Best strength</span>
              <strong>{strongest ? `${strongest.label} ${strongest.score}` : "--"}</strong>
            </div>
            <div>
              <span>Estimated time</span>
              <strong>{practiceMinutes ? `${practiceMinutes} min` : "--"}</strong>
            </div>
          </div>
        </div>
        <div className="practice-focus-list">
          {plan.slice(0, 2).map((item: PracticePlanItem, index: number) => (
            <div className={`plan-row ${index < 2 ? "priority-plan" : ""} ${item.checked ? "checked" : ""}`} key={`${item.drillId}-${item.name}-focus`}>
              <label className="plan-check">
                <input type="checkbox" checked={item.checked} onChange={() => togglePracticePlanItem(item.drillId)} />
                <span>{item.time}</span>
              </label>
              <strong>{item.name}</strong>
              <small>{item.reps}</small>
              <button className="ghost-button compact-button" onClick={() => removeDrillFromPracticePlan(item.drillId)}>
                Remove
              </button>
            </div>
          ))}
          {!plan.length ? (
            <p className="text-sm font-semibold text-slate-600">Add drills from the recommendations below to build a focused session.</p>
          ) : null}
        </div>
      </div>
    </section>
  );
}

function GradePanel({
  analysis,
  modelState,
  modelMessage,
  processing,
}: {
  analysis: any;
  modelState: ModelState;
  modelMessage: string;
  processing: boolean;
}) {
  const score = analysis?.overall ?? 0;
  const background = `conic-gradient(oklch(55% 0.12 165) ${score * 3.6}deg, oklch(89% 0.012 190) 0deg)`;
  const strongest = strongestCategory(analysis);
  const weakest = weakestCategory(analysis);
  const biggestIssue = analysis?.mistakes?.[0];
  const nextDrill = analysis?.drills?.[0];
  const confidenceLabel = analysis?.modelConfidence?.label
    ? modelConfidenceCopy(analysis.modelConfidence.label)
    : processing
      ? "Reading movement"
      : modelState === "ready"
        ? "Model ready"
        : modelState === "error"
          ? "Model issue"
          : "Model on demand";
  const confidenceTone = analysis?.modelConfidence?.label
    ? analysis.modelConfidence.label === "high"
      ? "good"
      : analysis.modelConfidence.label === "low"
        ? "bad"
        : "warn"
    : modelState === "ready"
      ? "good"
      : modelState === "error"
        ? "bad"
        : "warn";
  const summaryRows = [
    {
      label: "Biggest issue",
      value: biggestIssue?.label ?? weakest?.label ?? "Awaiting analysis",
      detail: biggestIssue?.evidence ?? (weakest ? `${weakest.score}/100, review this first.` : modelMessage),
      tone: biggestIssue?.severity === "high" || (weakest?.score ?? 100) < 66 ? "bad" : "warn",
    },
    {
      label: "Strongest category",
      value: strongest ? strongest.label : "Not scored yet",
      detail: strongest ? `${strongest.score}/100, keep this pattern.` : "Your best category will appear here.",
      tone: "good",
    },
    {
      label: "Next drill",
      value: nextDrill?.name ?? "Capture a swing first",
      detail: nextDrill ? `${nextDrill.time}, ${nextDrill.reps}` : "The first recommended drill becomes the practice priority.",
      tone: nextDrill ? "good" : "warn",
    },
  ];
  const qualityBadges: Array<{ label: string; tone: string }> =
    analysis?.qualityNotes?.length
      ? analysis.qualityNotes.slice(0, 4).map((note: QualityNote) => ({ label: qualityBadgeCopy(note), tone: toneForQuality(note.severity) }))
      : [
          { label: "Full body visible", tone: "good" },
          { label: "Stable camera", tone: "good" },
          { label: "Better from down-the-line", tone: "warn" },
        ];

  return (
    <aside className="tool-panel grade-summary grid content-start gap-4 p-4">
      <div className="grade-summary-head">
        <div className="score-ring" style={{ background }}>
          <strong className="text-4xl font-black text-ink">{analysis ? score : "--"}</strong>
          <span className="-mt-3 text-xs font-black uppercase text-slate-500">Grade</span>
        </div>
        <div>
          <span className={`badge ${confidenceTone}`}>{confidenceLabel}</span>
          <h2 className="mt-2 text-xl font-black text-ink">Analysis summary</h2>
          <p className="text-sm font-semibold leading-6 text-slate-500">
            {analysis ? "The next practice decision, not just the score." : "Upload or record a swing to turn this into a coach-ready read."}
          </p>
        </div>
      </div>
      <div className="grade-status-row">
        {qualityBadges.map((badge) => (
          <span className={`badge ${badge.tone}`} key={badge.label}>
            {badge.label}
          </span>
        ))}
      </div>
      <div className="grade-decision-list">
        {summaryRows.map((row) => (
          <div className={`grade-decision ${row.tone}`} key={row.label}>
            <span>{row.label}</span>
            <strong>{row.value}</strong>
            <small>{row.detail}</small>
          </div>
        ))}
      </div>
      <div className="grade-mini-scores">
        {(analysis?.categoryScores ?? []).slice(0, 4).map((item: any) => (
          <div key={item.id}>
            <span>{item.label}</span>
            <strong>{item.score}</strong>
            <div className="meter">
              <span style={{ width: `${item.score}%` }} />
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}

function ProgressDashboard({ history }: { history: StoredReport[] }) {
  const latest = history[0];
  const previous = history[1];
  const latestStrongest = strongestCategory(latest);
  const latestWeakest = weakestCategory(latest);
  const bestDelta = bestCategoryImprovement(previous, latest?.analysis);
  const trend = history.slice(0, 6).reverse();

  return (
    <aside className="tool-panel grid content-start gap-3 p-4">
      <div>
        <h2 className="text-xl font-black text-ink">Progress</h2>
        <p className="text-sm font-semibold text-slate-500">
          {latest ? "Recent saved swing history" : "Save swings to build a progress view"}
        </p>
      </div>

      {latest ? (
        <>
          <div className="progress-stat-grid">
            <div>
              <span>Latest</span>
              <strong>{latest.overall_score}</strong>
            </div>
            <div>
              <span>Change</span>
              <strong>
                {previous ? `${latest.overall_score - previous.overall_score >= 0 ? "+" : ""}${latest.overall_score - previous.overall_score}` : "--"}
              </strong>
            </div>
          </div>
          <div className="comparison-stack">
            <div className="comparison-line">
              <span>Strongest</span>
              <strong>{latestStrongest ? `${latestStrongest.label} ${latestStrongest.score}` : "--"}</strong>
            </div>
            <div className="comparison-line">
              <span>Weakest</span>
              <strong>{latestWeakest ? `${latestWeakest.label} ${latestWeakest.score}` : "--"}</strong>
            </div>
            <div className="comparison-line current">
              <span>Best gain</span>
              <strong>{bestDelta && bestDelta.delta !== null ? `${bestDelta.label} ${bestDelta.delta >= 0 ? "+" : ""}${bestDelta.delta}` : "--"}</strong>
            </div>
          </div>
          {trend.length >= 2 ? (
            <div>
              <p className="mb-2 text-xs font-black uppercase text-slate-500">Recent trend</p>
              <div className="trend-bars">
                {trend.map((report) => (
                  <span
                    key={report.id}
                    title={`${report.title}: ${report.overall_score}`}
                    style={{ height: `${Math.max(18, report.overall_score)}%` }}
                  />
                ))}
              </div>
            </div>
          ) : null}
        </>
      ) : (
        <p className="text-sm font-semibold leading-6 text-slate-600">
          After two saved swings, this panel will show latest score, change, strongest category, weakest category, and trend.
        </p>
      )}
    </aside>
  );
}

function EmptyDashboard({ modelMessage }: { modelMessage: string }) {
  const captureChecks = ["Face-on", "Down-the-line", "Full body visible", "Stable camera", "Good lighting", "Club and hands visible"];
  return (
    <section className="mx-auto grid max-w-7xl gap-5 px-4 pb-8 sm:px-6 lg:px-8">
      <div className="empty-guidance">
        <div className="empty-guidance-copy">
          <div>
            <span className="badge good">Before you record</span>
            <h2>Make the first read easier for the model.</h2>
            <p>
              SwingGrade works best when the full body, club, and hands stay visible. A clean capture gives more reliable phase timing and fewer camera warnings.
            </p>
          </div>
          <div className="system-status-note">
            <span>Model status</span>
            <strong>{modelMessage}</strong>
          </div>
        </div>
        <div className="capture-check-grid">
          {captureChecks.map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
      </div>
    </section>
  );
}

function Timeline({ analysis, activePhaseIndex, selectedFrameIndex, selectPhase, selectFrame }: any) {
  return (
    <section className="mx-auto max-w-7xl px-4 pb-5 sm:px-6 lg:px-8">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="text-xl font-black text-ink">Swing phases</h2>
        <p className="text-sm font-semibold text-slate-500">{analysis.frames.length} sampled frames</p>
      </div>
      <div className="timeline-rail mb-3" />
      <div className="flex gap-2 overflow-x-auto pb-2">
        {analysis.phases.map((phase: any, index: number) => {
          const categoryScore = analysis.categoryScores[Math.min(analysis.categoryScores.length - 1, activeCategoryIndex(phase.id))]?.score ?? 0;
          const confidenceTone = phase.confidenceLabel === "high" ? "good" : phase.confidenceLabel === "low" ? "bad" : "warn";
          return (
            <button
              key={phase.id}
              className={`phase-button ${index === activePhaseIndex ? "active" : ""}`}
              onClick={() => selectPhase(index)}
            >
              <span className="phase-button-head">
                <span className="text-xs font-black uppercase opacity-70">{formatTime(phase.time)}</span>
                <span className={`phase-score ${scoreTone(categoryScore)}`}>{categoryScore}</span>
              </span>
              <span className="phase-button-label">{phase.label}</span>
              <span className={`phase-confidence ${confidenceTone}`}>{phase.confidenceLabel ?? "model read"}</span>
            </button>
          );
        })}
      </div>
      <div className="scrubber-panel mt-3">
        <div>
          <p className="text-xs font-black uppercase text-slate-500">Frame scrubber</p>
          <p className="text-sm font-semibold text-slate-600">
            Frame {selectedFrameIndex + 1} of {analysis.frames.length}, {formatTime(analysis.frames[selectedFrameIndex]?.time ?? 0)}
          </p>
        </div>
        <div className="scrubber-controls">
          <button className="icon-button" onClick={() => selectFrame(selectedFrameIndex - 1)} title="Previous frame" aria-label="Previous frame">
            <span className="h-5 w-5">{ICONS.previous}</span>
          </button>
          <input
            aria-label="Frame scrubber"
            type="range"
            min="0"
            max={Math.max(analysis.frames.length - 1, 0)}
            value={selectedFrameIndex}
            onChange={(event) => selectFrame(Number(event.target.value))}
            onInput={(event) => selectFrame(Number(event.currentTarget.value))}
          />
          <button className="icon-button" onClick={() => selectFrame(selectedFrameIndex + 1)} title="Next frame" aria-label="Next frame">
            <span className="h-5 w-5">{ICONS.next}</span>
          </button>
        </div>
      </div>
    </section>
  );
}

function Comparison({ phase, activeFrame, analysis }: { phase: any; activeFrame: any; analysis: any }) {
  if (!phase) return null;
  return (
    <section className="mx-auto grid max-w-7xl gap-5 px-4 pb-5 sm:px-6 lg:grid-cols-2 lg:px-8">
      <div className="phase-card p-4">
        <div className="mb-3 flex items-center justify-between gap-3">
          <div>
            <h2 className="text-xl font-black text-ink">User frame</h2>
            <p className="text-sm font-semibold text-slate-500">
              {phase.label} at {formatTime(phase.time)}
            </p>
          </div>
          <span className={`badge ${scoreTone(analysis.categoryScores[Math.min(analysis.categoryScores.length - 1, activeCategoryIndex(phase.id))]?.score ?? 72)}`}>
            Frame {phase.index + 1}
          </span>
        </div>
        <PoseStage frame={activeFrame ?? phase.frame} edges={analysis.skeletonEdges} />
      </div>
      <div className="phase-card p-4">
        <div className="mb-3">
          <h2 className="text-xl font-black text-ink">Pro pattern</h2>
          <p className="text-sm font-semibold text-slate-500">{phase.referenceCue}</p>
        </div>
        <div className="pro-stage grid aspect-video place-items-center">
          <ProSkeleton phaseId={phase.id} analysis={analysis} />
        </div>
        <p className="mt-3 text-sm font-semibold text-slate-600">{phase.deltaSummary}</p>
      </div>
    </section>
  );
}

function activeCategoryIndex(phaseId: string) {
  const map: Record<string, number> = {
    setup: 0,
    takeaway: 1,
    backswing: 2,
    top: 3,
    downswing: 4,
    impact: 5,
    follow: 6,
    finish: 6,
  };
  return map[phaseId] ?? 0;
}

function PoseStage({ frame, edges }: { frame: any; edges: string[][] }) {
  return (
    <div className="pose-stage aspect-video">
      <img src={frame.image} alt="Selected swing frame" />
      <svg className="pose-overlay" viewBox={`0 0 ${frame.width} ${frame.height}`} preserveAspectRatio="xMidYMid meet">
        <g stroke="oklch(73% 0.14 135)" strokeWidth="5" strokeLinecap="round" opacity="0.92">
          {edges.map(([a, b]) => {
            const pa = point(frame.pose, a);
            const pb = point(frame.pose, b);
            if (!pa || !pb) return null;
            return <line key={`${a}-${b}`} x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y} />;
          })}
        </g>
        <g fill="oklch(98% 0.006 165)" stroke="oklch(55% 0.12 165)" strokeWidth="3">
          {(frame.pose?.keypoints ?? [])
            .filter((item: any) => (item.score ?? 1) > 0.22)
            .map((item: any) => (
              <circle key={item.name || item.part} cx={item.x} cy={item.y} r="5" />
            ))}
        </g>
      </svg>
    </div>
  );
}

function point(pose: any, name: string) {
  const item = pose?.keypoints?.find((keypoint: any) => keypoint.name === name || keypoint.part === name);
  if (!item || (item.score ?? 1) < 0.18) return null;
  return item;
}

function ProSkeleton({ phaseId, analysis }: { phaseId: string; analysis: any }) {
  const skeleton = analysis.referencePhases[phaseId]?.skeleton ?? analysis.referencePhases.setup.skeleton;
  const edges = analysis.skeletonEdges;
  const get = (name: string) => skeleton[name];
  return (
    <svg viewBox="0 0 100 100" className="h-full w-full max-w-[34rem]" aria-label="Normalized pro reference skeleton">
      <rect x="5" y="92" width="90" height="2" rx="1" fill="rgba(248,252,248,.18)" />
      <g stroke="oklch(73% 0.14 135)" strokeWidth="1.9" strokeLinecap="round">
        {edges.map(([a, b]: string[]) => {
          const pa = get(a);
          const pb = get(b);
          if (!pa || !pb) return null;
          return <line key={`${a}-${b}`} x1={pa[0]} y1={pa[1]} x2={pb[0]} y2={pb[1]} />;
        })}
      </g>
      <g fill="oklch(98% 0.006 165)" stroke="oklch(55% 0.12 165)" strokeWidth="1">
        {Object.entries(skeleton).map(([name, value]: any) => (
          <circle key={name} cx={value[0]} cy={value[1]} r="1.9" />
        ))}
      </g>
    </svg>
  );
}

function SwingReport({
  analysis,
  activePhase,
  history,
  leaderboard,
  saveCurrentReport,
  user,
  customPracticePlan,
  addDrillToPracticePlan,
  removeDrillFromPracticePlan,
  togglePracticePlanItem,
  comparisonReportId,
  setComparisonReportId,
}: any) {
  const selectedComparison = history.find((report: StoredReport) => report.id === comparisonReportId) ?? history?.[0];
  const bestDelta = bestCategoryImprovement(selectedComparison, analysis);
  const biggestRegression = biggestCategoryRegression(selectedComparison, analysis);
  const categoryDeltas = categoryDeltaRows(analysis, selectedComparison);
  const phaseDeltas = phaseComparisonRows(analysis, selectedComparison);
  const trim = analysis.trimWindow;
  const analyzedWindow = analysis.analysisWindow;
  const plan = customPracticePlan.length ? customPracticePlan : planItemsFromAnalysis(analysis);
  return (
    <section className="mx-auto grid max-w-7xl gap-5 px-4 pb-10 sm:px-6 lg:grid-cols-[1fr_24rem] lg:px-8">
      <div className="grid gap-5">
        <div className="phase-card p-4">
          <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="text-xl font-black text-ink">{activePhase.label} checkpoint</h2>
              <p className="text-sm font-semibold text-slate-500">Key angles and normalized movement proxies</p>
            </div>
            <span className="badge good">Torso-normalized</span>
          </div>
          {trim ? (
            <div className="trim-note mb-4">
              <strong>{analyzedWindow?.source === "motion" ? "Auto-trimmed analysis" : "Detected swing window"}</strong>
              <span>
                {analyzedWindow?.source === "motion"
                  ? `${formatTime(analyzedWindow.start)} to ${formatTime(analyzedWindow.end)}; swing ${formatTime(trim.start)} to ${formatTime(trim.end)}`
                  : `${formatTime(trim.start)} to ${formatTime(trim.end)}`}{" "}
                ({trim.confidence} confidence)
              </span>
            </div>
          ) : null}
          <div className="table-grid overflow-hidden rounded-lg border border-line">
            {activePhase.metrics.map((item: any) => (
              <div className="contents" key={item.label}>
                <div className="text-sm font-black text-ink">{item.label}</div>
                <div className="text-sm font-semibold text-slate-700">{item.value}</div>
                <div className="text-sm font-semibold text-slate-500">{item.ideal}</div>
              </div>
            ))}
          </div>
          <div className="mt-5 grid gap-4 md:grid-cols-3">
            <FeedbackBlock title="Did well" items={activePhase.positives} tone="good" />
            <FeedbackBlock title="Incorrect" items={activePhase.issues} tone="warn" />
            <div>
              <span className="badge good">Improve</span>
              <p className="mt-3 text-sm font-semibold leading-6 text-slate-700">{activePhase.suggestion}</p>
            </div>
          </div>
        </div>

        <div className="phase-card p-4">
          <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="text-xl font-black text-ink">Video quality and confidence</h2>
              <p className="text-sm font-semibold text-slate-500">
                {analysis.modelConfidence?.summary ?? "Camera setup notes for this analysis."}
              </p>
            </div>
            <span className={`badge ${analysis.modelConfidence?.label === "high" ? "good" : analysis.modelConfidence?.label === "low" ? "bad" : "warn"}`}>
              {analysis.modelConfidence?.label ?? "unknown"} confidence
            </span>
          </div>
          <div className="quality-grid">
            {(analysis.qualityNotes ?? []).map((note: QualityNote) => (
              <div key={note.id} className="quality-note">
                <span className={`badge ${toneForQuality(note.severity)}`}>{note.severity}</span>
                <strong>{note.label}</strong>
                <p>{note.detail}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-2">
          {analysis.categoryScores.map((item: any) => (
            <div className="metric-card p-4" key={item.id}>
              <div className="mb-2 flex items-center justify-between gap-2">
                <h3 className="font-black text-ink">{item.label}</h3>
                <span className={`badge ${scoreTone(item.score)}`}>{item.score}</span>
              </div>
              <p className="text-sm font-semibold leading-6 text-slate-600">{item.note}</p>
              {item.reason ? <p className="score-reason mt-2">Reason: {item.reason}</p> : null}
              <div className="meter mt-3">
                <span style={{ width: `${item.score}%` }} />
              </div>
            </div>
          ))}
        </div>
      </div>

      <aside className="grid content-start gap-5">
        <div className="phase-card p-4">
          <div className="mb-3 flex items-center justify-between gap-3">
            <h2 className="text-xl font-black text-ink">Before and after</h2>
            <span className="badge warn">Improvement</span>
          </div>
          {history.length ? (
            <div className="comparison-stack">
              <label className="field-label">
                Compare against
                <select value={selectedComparison?.id ?? ""} onChange={(event) => setComparisonReportId(event.target.value)}>
                  {history.map((report: StoredReport) => (
                    <option key={report.id} value={report.id}>
                      {report.title || report.source_name} ({report.overall_score})
                    </option>
                  ))}
                </select>
              </label>
              <div className="comparison-line">
                <span>Saved swing</span>
                <strong>{selectedComparison?.overall_score ?? "--"}</strong>
              </div>
              <div className="comparison-line current">
                <span>Current swing</span>
                <strong>{analysis.overall}</strong>
              </div>
              <p className="text-sm font-semibold leading-6 text-slate-600">
                {bestDelta && (bestDelta.delta ?? 0) > 0
                  ? `Improved ${bestDelta.label.toLowerCase()} by ${bestDelta.delta} points.`
                  : "Save this report, then compare it against your next swing."}
              </p>
              {biggestRegression && (biggestRegression.delta ?? 0) < 0 ? (
                <p className="text-sm font-semibold leading-6 text-slate-600">
                  Watch {biggestRegression.label.toLowerCase()}: {biggestRegression.delta} points versus the saved swing.
                </p>
              ) : null}
              <div className="comparison-mini-list">
                {categoryDeltas.slice(0, 4).map((item: any) => (
                  <div key={item.id}>
                    <span>{item.label}</span>
                    <strong>{item.previous ?? "--"} to {item.current} {item.delta !== null ? `(${item.delta >= 0 ? "+" : ""}${item.delta})` : ""}</strong>
                  </div>
                ))}
              </div>
              {phaseDeltas.length ? (
                <div className="phase-compare-list">
                  <p className="text-xs font-black uppercase text-slate-500">Matching phases</p>
                  {phaseDeltas.map((row: any) => (
                    <div key={row.id}>
                      <span>{row.label}</span>
                      <strong>
                        {formatTime(row.previousTime)} to {formatTime(row.currentTime)}
                      </strong>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          ) : (
            <div className="grid gap-3">
              <p className="text-sm font-semibold leading-6 text-slate-600">
                Save this report to make the next analysis show last swing, current swing, and the biggest category change.
              </p>
              <button className="secondary-button justify-self-start" onClick={saveCurrentReport} disabled={!user}>
                <span className="h-5 w-5">{ICONS.save}</span>
                Save baseline
              </button>
            </div>
          )}
        </div>

        <div className="phase-card p-4">
          <h2 className="text-xl font-black text-ink">Fault flags</h2>
          <div className="mt-3 grid gap-3">
            {analysis.mistakes.length ? (
              analysis.mistakes.map((mistake: any) => (
                <div key={mistake.id} className="border-b border-line pb-3 last:border-b-0 last:pb-0">
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <h3 className="text-sm font-black text-ink">{mistake.label}</h3>
                    <span className={`badge ${mistake.severity === "high" ? "bad" : "warn"}`}>{mistake.severity}</span>
                  </div>
                  <p className="text-sm font-semibold leading-6 text-slate-600">{mistake.evidence}</p>
                </div>
              ))
            ) : (
              <p className="text-sm font-semibold text-slate-600">No major fault flags were triggered.</p>
            )}
          </div>
        </div>

        <div className="phase-card p-4">
          <h2 className="text-xl font-black text-ink">Practice drills</h2>
          <div className="mt-3 grid gap-3">
            {analysis.drills.map((drill: any) => (
              <div key={drill.id} className="drill-card p-3">
                <div className="drill-media">Guide</div>
                <div className="mt-3 flex items-start justify-between gap-3">
                  <div>
                    <p className="text-sm font-black text-ink">{drill.name}</p>
                    <p className="mt-1 text-xs font-black uppercase text-moss">{drill.target}</p>
                  </div>
                  <span className="badge warn">{drill.difficulty ?? "drill"}</span>
                </div>
                <p className="mt-2 text-sm font-semibold leading-6 text-slate-600">{drill.cue}</p>
                <div className="drill-meta">
                  <span>{drill.time}</span>
                  <span>{drill.reps}</span>
                </div>
                <p className="mt-2 text-xs font-black uppercase text-slate-500">Fixes: {(drill.fixes ?? []).join(", ")}</p>
                {plan.some((item: PracticePlanItem) => item.drillId === drill.id) ? (
                  <button className="secondary-button compact-button mt-2" onClick={() => removeDrillFromPracticePlan(drill.id)}>
                    Remove from plan
                  </button>
                ) : (
                  <button className="ghost-button mt-2" onClick={() => addDrillToPracticePlan(drill)}>
                    Add to practice plan
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>

        <LeaderboardPanel leaderboard={leaderboard} />

        <div className="phase-card p-4">
          <h2 className="text-xl font-black text-ink">Model notes</h2>
          <ul className="mt-3 grid gap-2">
            {[...(analysis.angleNotes ?? []), ...analysis.modelNotes].map((note: string) => (
              <li key={note} className="text-sm font-semibold leading-6 text-slate-600">
                {note}
              </li>
            ))}
          </ul>
        </div>
      </aside>
    </section>
  );
}

function LeaderboardPanel({ leaderboard }: { leaderboard: any }) {
  const tabs = [
    { id: "topOverall", label: "Top overall", value: "overall" },
    { id: "mostImproved", label: "Most improved", value: "improvement" },
    { id: "bestTempo", label: "Best tempo", value: "tempo" },
    { id: "bestImpact", label: "Best impact", value: "impact" },
  ];

  return (
    <div className="phase-card p-4">
      <div className="mb-3 flex items-center gap-2">
        <span className="h-5 w-5 text-moss">{ICONS.trophy}</span>
        <h2 className="text-xl font-black text-ink">Leaderboard</h2>
      </div>
      <div className="leaderboard-grid">
        {tabs.map((tab) => {
          const rows = leaderboard?.[tab.id] ?? [];
          return (
            <div key={tab.id} className="leaderboard-section">
              <p className="text-xs font-black uppercase text-slate-500">{tab.label}</p>
              {rows.length ? (
                rows.slice(0, 3).map((row: any, index: number) => (
                  <div className="leaderboard-row" key={`${tab.id}-${row.reportId || row.userId}`}>
                    <span>{index + 1}. {row.userName}</span>
                    <strong>{row[tab.value] ?? row.currentOverall}</strong>
                  </div>
                ))
              ) : (
                <p className="text-sm font-semibold text-slate-500">Save reports to populate this board.</p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function FeedbackBlock({ title, items, tone }: { title: string; items: string[]; tone: string }) {
  return (
    <div>
      <span className={`badge ${tone}`}>{title}</span>
      <ul className="mt-3 grid gap-2">
        {items.map((item) => (
          <li key={item} className="text-sm font-semibold leading-6 text-slate-700">
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default App;

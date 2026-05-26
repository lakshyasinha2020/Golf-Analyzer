type SwingHandedness = "right" | "left";

type Keypoint = {
  name?: string;
  part?: string;
  x: number;
  y: number;
  score?: number;
};

type Pose = {
  keypoints: Keypoint[];
  score?: number;
};

type SwingFrame = {
  index: number;
  time: number;
  image: string;
  width: number;
  height: number;
  pose: Pose | null;
  brightness?: number;
  metrics?: Record<string, number | null>;
};

type CategoryScore = {
  id: string;
  label: string;
  score: number;
  note: string;
  reason?: string;
};

type SwingMistake = {
  id: string;
  label: string;
  severity: "high" | "medium" | "low";
  evidence: string;
  drillIds: string[];
};

type SwingDrill = {
  id: string;
  name: string;
  target: string;
  cue: string;
  reps: string;
  time: string;
  difficulty: "beginner" | "intermediate" | "advanced";
  fixes: string[];
};

type PhaseResult = {
  id: string;
  label: string;
  index: number;
  time: number;
  confidence: number;
  confidenceLabel: "high" | "medium" | "low";
  frame: SwingFrame;
  metrics: Array<{ label: string; value: string; ideal: string }>;
  positives: string[];
  issues: string[];
  suggestion: string;
  referenceCue: string;
  deltaSummary: string;
};

type QualityNote = {
  id: string;
  label: string;
  severity: "good" | "info" | "warn" | "bad";
  detail: string;
};

const SwingGradeEngine = (() => {
  const PHASES = [
    { id: "setup", label: "Setup" },
    { id: "takeaway", label: "Takeaway" },
    { id: "backswing", label: "Backswing" },
    { id: "top", label: "Top of backswing" },
    { id: "downswing", label: "Downswing" },
    { id: "impact", label: "Impact" },
    { id: "follow", label: "Follow-through" },
    { id: "finish", label: "Finish" },
  ];

  const SKELETON_EDGES = [
    ["left_shoulder", "right_shoulder"],
    ["left_shoulder", "left_elbow"],
    ["left_elbow", "left_wrist"],
    ["right_shoulder", "right_elbow"],
    ["right_elbow", "right_wrist"],
    ["left_shoulder", "left_hip"],
    ["right_shoulder", "right_hip"],
    ["left_hip", "right_hip"],
    ["left_hip", "left_knee"],
    ["left_knee", "left_ankle"],
    ["right_hip", "right_knee"],
    ["right_knee", "right_ankle"],
    ["nose", "left_shoulder"],
    ["nose", "right_shoulder"],
  ];

  const REFERENCE_PHASES: Record<string, { cue: string; skeleton: Record<string, [number, number]> }> = {
    setup: {
      cue: "Athletic posture, soft knees, quiet head, hands centered under the shoulders.",
      skeleton: {
        nose: [50, 15],
        left_shoulder: [40, 32],
        right_shoulder: [60, 32],
        left_elbow: [43, 48],
        right_elbow: [57, 48],
        left_wrist: [48, 60],
        right_wrist: [52, 60],
        left_hip: [43, 55],
        right_hip: [57, 55],
        left_knee: [42, 75],
        right_knee: [58, 75],
        left_ankle: [38, 94],
        right_ankle: [62, 94],
      },
    },
    takeaway: {
      cue: "Club and hands move as a unit; chest turns without the head drifting.",
      skeleton: {
        nose: [50, 15],
        left_shoulder: [38, 33],
        right_shoulder: [61, 31],
        left_elbow: [39, 48],
        right_elbow: [54, 45],
        left_wrist: [40, 58],
        right_wrist: [49, 56],
        left_hip: [43, 55],
        right_hip: [57, 54],
        left_knee: [42, 75],
        right_knee: [58, 75],
        left_ankle: [38, 94],
        right_ankle: [62, 94],
      },
    },
    backswing: {
      cue: "Shoulders keep coiling while the lower body resists and the lead arm stays wide.",
      skeleton: {
        nose: [49, 15],
        left_shoulder: [35, 35],
        right_shoulder: [63, 29],
        left_elbow: [33, 48],
        right_elbow: [51, 35],
        left_wrist: [37, 39],
        right_wrist: [49, 32],
        left_hip: [42, 56],
        right_hip: [58, 53],
        left_knee: [42, 75],
        right_knee: [59, 75],
        left_ankle: [38, 94],
        right_ankle: [62, 94],
      },
    },
    top: {
      cue: "Full coil, extended lead arm, centered head, and no lateral slide.",
      skeleton: {
        nose: [49, 15],
        left_shoulder: [34, 36],
        right_shoulder: [64, 29],
        left_elbow: [28, 35],
        right_elbow: [46, 22],
        left_wrist: [32, 21],
        right_wrist: [45, 18],
        left_hip: [42, 56],
        right_hip: [58, 52],
        left_knee: [42, 75],
        right_knee: [59, 75],
        left_ankle: [38, 94],
        right_ankle: [62, 94],
      },
    },
    downswing: {
      cue: "Lower body starts first, hands shallow slightly, head remains behind the ball.",
      skeleton: {
        nose: [49, 15],
        left_shoulder: [37, 34],
        right_shoulder: [62, 31],
        left_elbow: [35, 47],
        right_elbow: [50, 42],
        left_wrist: [41, 51],
        right_wrist: [51, 50],
        left_hip: [40, 56],
        right_hip: [58, 54],
        left_knee: [40, 76],
        right_knee: [58, 75],
        left_ankle: [38, 94],
        right_ankle: [62, 94],
      },
    },
    impact: {
      cue: "Pressure forward, spine angle held, hands slightly ahead, lead side braced.",
      skeleton: {
        nose: [48, 16],
        left_shoulder: [39, 34],
        right_shoulder: [60, 33],
        left_elbow: [42, 50],
        right_elbow: [55, 50],
        left_wrist: [48, 61],
        right_wrist: [52, 61],
        left_hip: [39, 56],
        right_hip: [57, 56],
        left_knee: [39, 77],
        right_knee: [58, 75],
        left_ankle: [38, 94],
        right_ankle: [62, 94],
      },
    },
    follow: {
      cue: "Arms extend through the strike before folding; chest rotates toward the target.",
      skeleton: {
        nose: [50, 15],
        left_shoulder: [41, 32],
        right_shoulder: [62, 36],
        left_elbow: [51, 43],
        right_elbow: [64, 50],
        left_wrist: [63, 40],
        right_wrist: [68, 48],
        left_hip: [40, 55],
        right_hip: [57, 57],
        left_knee: [39, 76],
        right_knee: [58, 76],
        left_ankle: [38, 94],
        right_ankle: [62, 94],
      },
    },
    finish: {
      cue: "Tall balanced finish, chest facing target, weight collected on the lead side.",
      skeleton: {
        nose: [53, 14],
        left_shoulder: [44, 31],
        right_shoulder: [62, 37],
        left_elbow: [51, 25],
        right_elbow: [63, 30],
        left_wrist: [56, 18],
        right_wrist: [66, 23],
        left_hip: [42, 55],
        right_hip: [57, 58],
        left_knee: [39, 76],
        right_knee: [57, 78],
        left_ankle: [38, 94],
        right_ankle: [60, 94],
      },
    },
  };

  const DRILLS: Record<string, SwingDrill> = {
    alignment_takeaway: {
      id: "alignment_takeaway",
      name: "Alignment stick takeaway drill",
      target: "Takeaway path",
      cue: "Place an alignment stick outside the ball line and keep the hands and club moving parallel to it for the first two feet.",
      reps: "3 sets of 8 slow takeaways",
      time: "4 minutes",
      difficulty: "beginner",
      fixes: ["inside takeaway", "over the top", "hands-only start"],
    },
    wall_posture: {
      id: "wall_posture",
      name: "Wall drill for posture",
      target: "Setup posture",
      cue: "Set your hips lightly against a wall, hinge from the hips, and rehearse turns without standing up.",
      reps: "2 minutes before each practice block",
      time: "2 minutes",
      difficulty: "beginner",
      fixes: ["poor posture", "early extension", "posture loss"],
    },
    step_through: {
      id: "step_through",
      name: "Step-through drill",
      target: "Weight transfer",
      cue: "Start narrow, swing back, then step toward the target as the downswing begins.",
      reps: "12 half-speed swings",
      time: "5 minutes",
      difficulty: "intermediate",
      fixes: ["poor weight transfer", "loss of balance"],
    },
    pump_sequence: {
      id: "pump_sequence",
      name: "Pump drill",
      target: "Downswing sequence",
      cue: "Pause at the top, pump the hands halfway down while the hips open, then swing through.",
      reps: "3 pumps then strike, 10 balls",
      time: "6 minutes",
      difficulty: "intermediate",
      fixes: ["poor sequencing", "over the top", "rushed transition"],
    },
    towel_connection: {
      id: "towel_connection",
      name: "Towel-under-arm drill",
      target: "Arm-body connection",
      cue: "Hold a towel under the lead armpit through waist-high follow-through to keep the arms synced.",
      reps: "10 chips, 10 half swings",
      time: "5 minutes",
      difficulty: "beginner",
      fixes: ["collapsed lead arm", "chicken wing", "poor connection"],
    },
    slow_tempo: {
      id: "slow_tempo",
      name: "Slow-motion tempo drill",
      target: "Tempo",
      cue: "Make a three-count backswing and one-count downswing while keeping the finish still.",
      reps: "15 rehearsal swings",
      time: "4 minutes",
      difficulty: "beginner",
      fixes: ["tempo", "head movement", "loss of balance"],
    },
    impact_bag: {
      id: "impact_bag",
      name: "Impact bag drill",
      target: "Impact position",
      cue: "Rehearse impact with hands ahead, pressure forward, and the chest turning through the bag.",
      reps: "3 sets of 6 controlled strikes",
      time: "6 minutes",
      difficulty: "intermediate",
      fixes: ["weak impact", "early extension", "poor pressure shift"],
    },
  };

  function clamp(value: number, min = 0, max = 100) {
    return Math.max(min, Math.min(max, value));
  }

  function round(value: number | null | undefined, places = 0) {
    if (value === null || value === undefined || Number.isNaN(value)) return null;
    const factor = 10 ** places;
    return Math.round(value * factor) / factor;
  }

  function keypoint(pose: Pose | null | undefined, name: string) {
    if (!pose || !pose.keypoints) return null;
    const point = pose.keypoints.find((item) => item.name === name || item.part === name);
    if (!point || (point.score ?? 1) < 0.18) return null;
    return point;
  }

  function mid(a: Keypoint | null, b: Keypoint | null) {
    if (!a || !b) return null;
    return {
      x: (a.x + b.x) / 2,
      y: (a.y + b.y) / 2,
      score: ((a.score ?? 1) + (b.score ?? 1)) / 2,
    };
  }

  function dist(a: { x: number; y: number } | null, b: { x: number; y: number } | null) {
    if (!a || !b) return null;
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  function lineAngle(a: { x: number; y: number } | null, b: { x: number; y: number } | null) {
    if (!a || !b) return null;
    return (Math.atan2(b.y - a.y, b.x - a.x) * 180) / Math.PI;
  }

  function shortDelta(a: number | null, b: number | null) {
    if (a === null || b === null) return null;
    let delta = a - b;
    while (delta > 180) delta -= 360;
    while (delta < -180) delta += 360;
    return delta;
  }

  function jointAngle(
    a: { x: number; y: number } | null,
    b: { x: number; y: number } | null,
    c: { x: number; y: number } | null,
  ) {
    if (!a || !b || !c) return null;
    const ab = { x: a.x - b.x, y: a.y - b.y };
    const cb = { x: c.x - b.x, y: c.y - b.y };
    const dot = ab.x * cb.x + ab.y * cb.y;
    const denom = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);
    if (!denom) return null;
    return (Math.acos(clamp(dot / denom, -1, 1)) * 180) / Math.PI;
  }

  function avg(values: Array<number | null | undefined>) {
    const valid = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
    if (!valid.length) return null;
    return valid.reduce((sum, value) => sum + value, 0) / valid.length;
  }

  function metric(frame: SwingFrame | undefined, key: string) {
    const value = frame?.metrics?.[key];
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  }

  function scoreBetween(value: number | null, min: number, max: number, soft = 18) {
    if (value === null) return 62;
    if (value >= min && value <= max) return 100;
    const gap = value < min ? min - value : value - max;
    return clamp(100 - (gap / soft) * 100);
  }

  function scoreLess(value: number | null, idealMax: number, failAt: number) {
    if (value === null) return 62;
    if (value <= idealMax) return 100;
    if (value >= failAt) return 0;
    return clamp(100 - ((value - idealMax) / (failAt - idealMax)) * 100);
  }

  function scoreGreater(value: number | null, idealMin: number, zeroAt: number) {
    if (value === null) return 62;
    if (value >= idealMin) return 100;
    if (value <= zeroAt) return 0;
    return clamp(((value - zeroAt) / (idealMin - zeroAt)) * 100);
  }

  function weighted(items: Array<[number, number]>) {
    const totalWeight = items.reduce((sum, [, weight]) => sum + weight, 0);
    if (!totalWeight) return 0;
    return clamp(items.reduce((sum, [score, weight]) => sum + score * weight, 0) / totalWeight);
  }

  function confidence(pose: Pose | null) {
    if (!pose?.keypoints?.length) return 0;
    const coreNames = [
      "nose",
      "left_shoulder",
      "right_shoulder",
      "left_elbow",
      "right_elbow",
      "left_wrist",
      "right_wrist",
      "left_hip",
      "right_hip",
      "left_knee",
      "right_knee",
      "left_ankle",
      "right_ankle",
    ];
    return (
      coreNames.reduce((sum, name) => {
        const point = pose.keypoints.find((item) => item.name === name || item.part === name);
        return sum + (point?.score ?? 0);
      }, 0) / coreNames.length
    );
  }

  function computeMetrics(pose: Pose | null, setupPose: Pose | null, handedness: SwingHandedness) {
    const lead = handedness === "right" ? "left" : "right";
    const trail = handedness === "right" ? "right" : "left";

    const leftShoulder = keypoint(pose, "left_shoulder");
    const rightShoulder = keypoint(pose, "right_shoulder");
    const leftHip = keypoint(pose, "left_hip");
    const rightHip = keypoint(pose, "right_hip");
    const leftKnee = keypoint(pose, "left_knee");
    const rightKnee = keypoint(pose, "right_knee");
    const leftAnkle = keypoint(pose, "left_ankle");
    const rightAnkle = keypoint(pose, "right_ankle");
    const leftElbow = keypoint(pose, "left_elbow");
    const rightElbow = keypoint(pose, "right_elbow");
    const leftWrist = keypoint(pose, "left_wrist");
    const rightWrist = keypoint(pose, "right_wrist");
    const nose = keypoint(pose, "nose");

    const setupLeftShoulder = keypoint(setupPose, "left_shoulder");
    const setupRightShoulder = keypoint(setupPose, "right_shoulder");
    const setupLeftHip = keypoint(setupPose, "left_hip");
    const setupRightHip = keypoint(setupPose, "right_hip");
    const setupLeftWrist = keypoint(setupPose, "left_wrist");
    const setupRightWrist = keypoint(setupPose, "right_wrist");
    const setupNose = keypoint(setupPose, "nose");

    const shoulderCenter = mid(leftShoulder, rightShoulder);
    const hipCenter = mid(leftHip, rightHip);
    const grip = mid(leftWrist, rightWrist);
    const setupShoulderCenter = mid(setupLeftShoulder, setupRightShoulder);
    const setupHipCenter = mid(setupLeftHip, setupRightHip);
    const setupGrip = mid(setupLeftWrist, setupRightWrist);
    const ankleCenter = mid(leftAnkle, rightAnkle);

    const torsoLength =
      dist(setupShoulderCenter, setupHipCenter) ||
      dist(shoulderCenter, hipCenter) ||
      (dist(leftShoulder, rightShoulder) ?? 90) * 1.55 ||
      90;
    const stanceWidth = (dist(leftAnkle, rightAnkle) ?? 0) / torsoLength;
    const shoulderAngle = lineAngle(leftShoulder, rightShoulder);
    const hipAngle = lineAngle(leftHip, rightHip);
    const setupShoulderAngle = lineAngle(setupLeftShoulder, setupRightShoulder);
    const setupHipAngle = lineAngle(setupLeftHip, setupRightHip);
    const spineLine = lineAngle(hipCenter, shoulderCenter);
    const setupSpineLine = lineAngle(setupHipCenter, setupShoulderCenter);
    const spineTilt = Math.abs(shortDelta(spineLine, -90) ?? 0);
    const setupSpineTilt = Math.abs(shortDelta(setupSpineLine, -90) ?? 0);
    const shoulderTurn = Math.abs(shortDelta(shoulderAngle, setupShoulderAngle) ?? 0);
    const hipTurn = Math.abs(shortDelta(hipAngle, setupHipAngle) ?? 0);
    const leadShoulder = lead === "left" ? leftShoulder : rightShoulder;
    const leadElbow = lead === "left" ? leftElbow : rightElbow;
    const leadWrist = lead === "left" ? leftWrist : rightWrist;
    const trailWrist = trail === "left" ? leftWrist : rightWrist;
    const leadHip = lead === "left" ? leftHip : rightHip;
    const leadKnee = lead === "left" ? leftKnee : rightKnee;
    const leadAnkle = lead === "left" ? leftAnkle : rightAnkle;
    const trailHip = trail === "left" ? leftHip : rightHip;
    const trailKnee = trail === "left" ? leftKnee : rightKnee;
    const trailAnkle = trail === "left" ? leftAnkle : rightAnkle;
    const leadArmAngle = jointAngle(leadShoulder, leadElbow, leadWrist);
    const trailArmAngle = jointAngle(trailHip, trailKnee, trailAnkle);
    const leadKneeAngle = jointAngle(leadHip, leadKnee, leadAnkle);
    const trailKneeAngle = jointAngle(trailHip, trailKnee, trailAnkle);
    const kneeFlex = avg([leadKneeAngle !== null ? 180 - leadKneeAngle : null, trailKneeAngle !== null ? 180 - trailKneeAngle : null]);
    const leadArmBend = leadArmAngle !== null ? 180 - leadArmAngle : null;
    const wristForearmAngle = Math.abs(shortDelta(lineAngle(leadElbow, leadWrist), spineLine) ?? 0);

    const headMovement = dist(nose, setupNose) !== null ? (dist(nose, setupNose) as number) / torsoLength : null;
    const hipSway = hipCenter && setupHipCenter ? (hipCenter.x - setupHipCenter.x) / torsoLength : null;
    const absHipSway = hipSway !== null ? Math.abs(hipSway) : null;
    const gripXShift = grip && setupGrip ? (grip.x - setupGrip.x) / torsoLength : null;
    const gripYLift = grip && setupGrip ? (setupGrip.y - grip.y) / torsoLength : null;
    const handTravel = dist(grip, setupGrip) !== null ? (dist(grip, setupGrip) as number) / torsoLength : null;
    const armHeight = grip && shoulderCenter ? (shoulderCenter.y - grip.y) / torsoLength : null;
    const shoulderHipSeparation = Math.abs((shoulderTurn ?? 0) - (hipTurn ?? 0));
    const balanceCenter =
      ankleCenter && hipCenter && stanceWidth > 0.15
        ? Math.abs((hipCenter.x - ankleCenter.x) / Math.max(dist(leftAnkle, rightAnkle) ?? 1, 1))
        : null;

    return {
      confidence: confidence(pose),
      torsoLength,
      shoulderTurn,
      hipTurn,
      shoulderHipSeparation,
      kneeFlex,
      spineTilt,
      setupSpineTilt,
      leadArmAngle,
      leadArmBend,
      trailArmAngle,
      wristSet: wristForearmAngle,
      headMovement,
      hipSway,
      absHipSway,
      gripXShift,
      gripYLift,
      handTravel,
      armHeight,
      balanceCenter,
      stanceWidth,
    };
  }

  function enrichFramesWithSetup(frames: SwingFrame[], setupPose: Pose | null, handedness: SwingHandedness) {
    return frames.map((frame) => ({
      ...frame,
      metrics: computeMetrics(frame.pose, setupPose, handedness),
    }));
  }

  function enrichFrames(frames: SwingFrame[], handedness: SwingHandedness) {
    const firstVisibleFrame = frames.find((frame) => confidence(frame.pose) > 0.24) ?? frames[0];
    const firstVisiblePose = firstVisibleFrame?.pose ?? null;
    const initial = enrichFramesWithSetup(frames, firstVisiblePose, handedness);
    const range = detectActiveSwingRange(initial);
    const setupPose = initial[range.setup]?.pose && confidence(initial[range.setup].pose) > 0.22 ? initial[range.setup].pose : firstVisiblePose;
    return enrichFramesWithSetup(frames, setupPose, handedness);
  }

  function fallbackIndex(total: number, ratio: number) {
    return clamp(Math.round((total - 1) * ratio), 0, Math.max(total - 1, 0));
  }

  function activityScore(frame: SwingFrame, previous?: SwingFrame) {
    const confidenceValue = metric(frame, "confidence") ?? 0;
    if (confidenceValue < 0.18) return 0;

    const handTravel = metric(frame, "handTravel") ?? 0;
    const gripLift = Math.max(0, metric(frame, "gripYLift") ?? 0);
    const shoulderTurn = metric(frame, "shoulderTurn") ?? 0;
    const headMove = metric(frame, "headMovement") ?? 0;
    const handDelta = previous ? Math.abs(handTravel - (metric(previous, "handTravel") ?? handTravel)) : 0;
    const liftDelta = previous ? Math.abs((metric(frame, "gripYLift") ?? 0) - (metric(previous, "gripYLift") ?? 0)) : 0;
    const turnDelta = previous ? Math.abs(shoulderTurn - (metric(previous, "shoulderTurn") ?? shoulderTurn)) / 30 : 0;

    return handTravel * 1.15 + gripLift * 1.05 + shoulderTurn * 0.014 + headMove * 0.28 + handDelta * 2.4 + liftDelta * 1.8 + turnDelta;
  }

  function detectActiveSwingRange(frames: SwingFrame[]) {
    const total = frames.length;
    if (total < 8) {
      return {
        setup: 0,
        start: 0,
        end: Math.max(total - 1, 0),
        confidence: "low" as const,
        activeCount: total,
      };
    }

    const rawScores = frames.map((frame, index) => activityScore(frame, frames[index - 1]));
    const smoothedScores = rawScores.map((score, index) => avg([rawScores[index - 1], score, rawScores[index + 1]]) ?? score);
    const maxScore = Math.max(...smoothedScores, 0);
    const threshold = Math.max(0.07, maxScore * 0.12);
    const activeIndexes = smoothedScores
      .map((score, index) => {
        const frame = frames[index];
        const confidenceValue = metric(frame, "confidence") ?? 0;
        const handTravel = metric(frame, "handTravel") ?? 0;
        const gripLift = Math.max(0, metric(frame, "gripYLift") ?? 0);
        const shoulderTurn = metric(frame, "shoulderTurn") ?? 0;
        const active =
          confidenceValue > 0.22 &&
          score >= threshold &&
          (handTravel > 0.035 || gripLift > 0.025 || shoulderTurn > 1.5 || score > maxScore * 0.24);
        return active ? index : -1;
      })
      .filter((index) => index >= 0);

    if (!activeIndexes.length) {
      const firstValid = frames.findIndex((frame) => (metric(frame, "confidence") ?? 0) > 0.24);
      const setup = firstValid >= 0 ? firstValid : 0;
      return {
        setup,
        start: setup,
        end: total - 1,
        confidence: "low" as const,
        activeCount: 0,
      };
    }

    const firstActive = activeIndexes[0];
    const lastActive = activeIndexes[activeIndexes.length - 1];
    const setupLookback = Math.max(2, Math.min(8, Math.round(total * 0.06)));
    const setupStart = Math.max(0, firstActive - setupLookback);
    let setup = Math.max(0, firstActive - 1);
    let setupScore = Infinity;

    for (let index = setupStart; index <= Math.max(setupStart, firstActive - 1); index += 1) {
      const confidenceValue = metric(frames[index], "confidence") ?? 0;
      if (confidenceValue < 0.22) continue;
      const score = smoothedScores[index] - confidenceValue * 0.08 + Math.abs(firstActive - index) * 0.002;
      if (score < setupScore) {
        setupScore = score;
        setup = index;
      }
    }

    const start = Math.max(setup + 1, firstActive);
    const finishPad = Math.max(2, Math.min(7, Math.round(total * 0.05)));
    const end = Math.min(total - 1, Math.max(start + 6, lastActive + finishPad));

    return {
      setup,
      start,
      end,
      confidence: activeIndexes.length > 10 ? ("high" as const) : ("medium" as const),
      activeCount: activeIndexes.length,
    };
  }

  function phaseProgressValue(frame: SwingFrame) {
    const lift = Math.max(0, metric(frame, "gripYLift") ?? 0);
    const travel = Math.max(0, metric(frame, "handTravel") ?? 0);
    const armHeight = Math.max(0, metric(frame, "armHeight") ?? 0);
    const turn = Math.max(0, metric(frame, "shoulderTurn") ?? 0) / 55;
    return lift * 1.3 + travel * 0.78 + armHeight * 0.95 + turn * 0.45;
  }

  function firstIndexAtSignalProgress(frames: SwingFrame[], start: number, end: number, targetProgress: number, fallback: number) {
    if (end <= start) return fallback;
    const startValue = phaseProgressValue(frames[start]);
    const peakValue = Math.max(...frames.slice(start, end + 1).map(phaseProgressValue), startValue);
    const target = startValue + (peakValue - startValue) * targetProgress;

    for (let index = start; index <= end; index += 1) {
      const confidenceValue = metric(frames[index], "confidence") ?? 0;
      if (confidenceValue < 0.2) continue;
      if (phaseProgressValue(frames[index]) >= target) return index;
    }

    return fallback;
  }

  function phaseWindowIndex(start: number, end: number, ratio: number) {
    return clamp(start + Math.round(Math.max(end - start, 0) * ratio), start, end);
  }

  function detectPhaseIndexes(frames: SwingFrame[]) {
    const total = frames.length;
    if (total < 8) {
      return {
        setup: 0,
        takeaway: fallbackIndex(total, 0.15),
        backswing: fallbackIndex(total, 0.3),
        top: fallbackIndex(total, 0.42),
        downswing: fallbackIndex(total, 0.58),
        impact: fallbackIndex(total, 0.66),
        follow: fallbackIndex(total, 0.8),
        finish: total - 1,
      };
    }

    const range = detectActiveSwingRange(frames);
    const setup = range.setup;
    const swingStart = range.start;
    const finish = range.end;
    const swingSpan = Math.max(finish - swingStart, 7);
    const topSearchStart = Math.min(finish - 2, Math.max(swingStart + 2, setup + 2));
    const topSearchEnd = Math.max(topSearchStart, Math.min(finish - 4, swingStart + Math.round(swingSpan * 0.76)));
    let top = Math.min(topSearchEnd, Math.max(topSearchStart, swingStart + Math.round(swingSpan * 0.42)));
    let topScore = -Infinity;
    const expectedTop = phaseWindowIndex(swingStart, finish, 0.42);

    for (let index = topSearchStart; index <= topSearchEnd; index += 1) {
      const frame = frames[index];
      const lift = metric(frame, "gripYLift") ?? 0;
      const travel = metric(frame, "handTravel") ?? 0;
      const armHeight = metric(frame, "armHeight") ?? 0;
      const sway = metric(frame, "absHipSway") ?? 0;
      const turn = metric(frame, "shoulderTurn") ?? 0;
      const confidenceValue = metric(frame, "confidence") ?? 0;
      const speedBefore = activityScore(frame, frames[index - 1]);
      const speedAfter = activityScore(frames[index + 1] ?? frame, frame);
      const transitionBonus = Math.max(0, speedBefore - speedAfter) * 0.22;
      const timingPenalty = Math.abs(index - expectedTop) / Math.max(swingSpan, 1);
      const latePenalty = Math.max(0, (index - swingStart) / Math.max(swingSpan, 1) - 0.62) * 0.75;
      const score =
        Math.max(0, lift) * 1.55 +
        Math.max(0, armHeight) * 1.15 +
        travel * 0.58 +
        turn * 0.016 +
        confidenceValue * 0.08 +
        transitionBonus -
        sway * 0.24 -
        timingPenalty * 0.34 -
        latePenalty;
      if (score > topScore) {
        topScore = score;
        top = index;
      }
    }

    if (top <= swingStart + 1) top = Math.min(finish - 4, swingStart + Math.round(swingSpan * 0.42));

    let impact = Math.min(finish - 1, Math.max(top + 2, top + Math.round((finish - top) * 0.34)));
    let impactScore = Infinity;
    const impactStart = Math.min(finish - 1, top + 1);
    const impactEnd = Math.min(finish - 1, Math.max(top + 4, top + Math.round((finish - top) * 0.72)));
    const expectedImpact = Math.min(impactEnd, Math.max(impactStart, top + Math.round((finish - top) * 0.38)));

    for (let index = impactStart; index <= impactEnd; index += 1) {
      const frame = frames[index];
      const handTravel = metric(frame, "handTravel") ?? 1.2;
      const lift = Math.abs(metric(frame, "gripYLift") ?? 0.3);
      const gripX = Math.abs(metric(frame, "gripXShift") ?? 0);
      const spineLoss = Math.abs((metric(frame, "spineTilt") ?? 0) - (metric(frames[setup], "spineTilt") ?? 0));
      const timingPenalty = Math.abs(index - expectedImpact) / Math.max(finish - top, 1);
      const speed = activityScore(frame, frames[index - 1]);
      const bottomBonus = Math.max(0, 0.2 - lift) * 0.18;
      const score = handTravel * 0.58 + lift * 1.05 + gripX * 0.18 + spineLoss * 0.005 + timingPenalty * 0.42 - speed * 0.08 - bottomBonus;
      if (score < impactScore) {
        impactScore = score;
        impact = index;
      }
    }

    if (impact <= top) impact = Math.min(finish - 1, top + Math.max(2, Math.round((finish - top) * 0.32)));

    const takeawayFallback = Math.max(swingStart, Math.min(top - 1, swingStart + Math.round((top - swingStart) * 0.22)));
    const takeaway = Math.max(setup + 1, Math.min(top - 1, firstIndexAtSignalProgress(frames, swingStart, top, 0.22, takeawayFallback)));
    const backswingFallback = Math.max(takeaway + 1, Math.min(top - 1, takeaway + Math.round((top - takeaway) * 0.58)));
    const backswing = Math.max(takeaway + 1, Math.min(top - 1, firstIndexAtSignalProgress(frames, takeaway, top, 0.62, backswingFallback)));
    const downswing = Math.max(top + 1, Math.min(impact - 1, top + Math.round((impact - top) * 0.5)));
    const follow = Math.max(impact + 1, Math.min(finish, impact + Math.round((finish - impact) * 0.48)));

    return { setup, takeaway, backswing, top, downswing, impact, follow, finish };
  }

  function fmtDegrees(value: number | null) {
    return value === null ? "Not tracked" : `${round(value, 0)} deg`;
  }

  function fmtRatio(value: number | null, suffix = "") {
    return value === null ? "Not tracked" : `${round(value, 2)}${suffix}`;
  }

  function metricRows(frame: SwingFrame): Array<{ label: string; value: string; ideal: string }> {
    return [
      { label: "Shoulder turn", value: fmtDegrees(metric(frame, "shoulderTurn")), ideal: "Builds through backswing" },
      { label: "Hip turn", value: fmtDegrees(metric(frame, "hipTurn")), ideal: "Less than shoulder turn" },
      { label: "Knee flex", value: fmtDegrees(metric(frame, "kneeFlex")), ideal: "10-30 deg at setup" },
      { label: "Spine tilt", value: fmtDegrees(metric(frame, "spineTilt")), ideal: "8-30 deg posture band" },
      { label: "Lead arm angle", value: fmtDegrees(metric(frame, "leadArmAngle")), ideal: "155-180 deg near top" },
      { label: "Wrist set proxy", value: fmtDegrees(metric(frame, "wristSet")), ideal: "Controlled hinge" },
      { label: "Head movement", value: fmtRatio(metric(frame, "headMovement"), " torso"), ideal: "< 0.30 torso" },
      { label: "Weight shift proxy", value: fmtRatio(metric(frame, "hipSway"), " torso"), ideal: "Shift without sway" },
    ];
  }

  function phaseFeedback(phaseId: string, frame: SwingFrame, context: Record<string, any>) {
    const m = frame.metrics ?? {};
    const positives: string[] = [];
    const issues: string[] = [];
    let suggestion = "Keep the same shape and rehearse it slowly so the checkpoint becomes repeatable.";

    const add = (condition: boolean, good: string, issue: string, fix: string) => {
      if (condition) positives.push(good);
      else {
        issues.push(issue);
        if (suggestion.startsWith("Keep the same shape")) suggestion = fix;
      }
    };

    if ((m.confidence as number) < 0.22) {
      issues.push("Pose confidence is low in this frame; use a brighter, full-body camera angle for a cleaner read.");
      suggestion = "Record from chest height with the full body visible from setup through finish.";
    }

    switch (phaseId) {
      case "setup":
        add(
          scoreBetween(m.spineTilt as number, 8, 30, 16) > 74,
          "Spine tilt is in an athletic setup range.",
          "Setup posture is outside the preferred hinge range.",
          "Use the wall posture drill and feel the hinge come from the hips, not the upper back.",
        );
        add(
          scoreBetween(m.kneeFlex as number, 8, 30, 18) > 74,
          "Knee flex gives you a stable athletic base.",
          "Knee flex is either too locked or too crouched.",
          "Soften both knees until you can bounce lightly without changing your spine angle.",
        );
        add(
          scoreLess(m.balanceCenter as number, 0.28, 0.62) > 65,
          "Center of mass starts inside the stance.",
          "Balance begins too far toward one foot.",
          "Set pressure under the arches before the takeaway starts.",
        );
        break;
      case "takeaway":
        add(
          scoreLess(m.headMovement as number, 0.12, 0.3) > 68,
          "Head stays quiet during the first move.",
          "Head movement appears early in the takeaway.",
          "Make waist-high rehearsals while keeping your cap logo centered over the ball.",
        );
        add(
          scoreLess(m.absHipSway as number, 0.12, 0.32) > 68,
          "Lower body resists lateral sway.",
          "The hips slide instead of turning in the takeaway.",
          "Use the alignment stick takeaway drill with a focus on turning the chest over a steady base.",
        );
        add(
          scoreGreater(m.handTravel as number, 0.16, 0.02) > 60,
          "Hands move away from setup with enough width.",
          "The takeaway looks cramped or too handsy.",
          "Move the triangle of shoulders and hands together for the first two feet.",
        );
        break;
      case "backswing":
        add(
          (m.shoulderTurn as number) >= (m.hipTurn as number) + 4,
          "Shoulders are coiling more than the hips.",
          "Shoulder and hip turn are too similar, reducing coil.",
          "Let the chest keep turning while the trail knee resists sliding.",
        );
        add(
          scoreGreater(m.leadArmAngle as number, 152, 130) > 68,
          "Lead arm keeps useful width.",
          "Lead arm begins to collapse before the top.",
          "Use the towel-under-arm drill and stop the backswing before the lead elbow folds.",
        );
        add(
          scoreLess(m.headMovement as number, 0.24, 0.44) > 62,
          "Head movement stays manageable.",
          "Head drift is building during the backswing.",
          "Rehearse slow backswings with your trail hip turning behind you instead of sliding.",
        );
        break;
      case "top":
        add(
          scoreGreater(m.armHeight as number, 0.25, -0.05) > 64,
          "Hands reach a clear top position.",
          "Top position is low or incomplete.",
          "Pause at the top for one count, then restart with the lower body.",
        );
        add(
          scoreGreater(m.leadArmAngle as number, 152, 128) > 68,
          "Lead arm remains extended at the top.",
          "Lead arm is collapsed at the top.",
          "Shorten the backswing until the lead arm can stay long.",
        );
        add(
          scoreLess(m.absHipSway as number, 0.22, 0.46) > 62,
          "Top position is centered enough to start down.",
          "There is too much lateral sway at the top.",
          "Practice turning around the trail hip with a headcover outside your trail hip.",
        );
        break;
      case "downswing": {
        const sequenceScore = context.sequenceScore ?? 60;
        add(
          sequenceScore > 68,
          "Lower body sequencing is starting the downswing.",
          "The upper body appears to pull the downswing first.",
          "Use the pump drill: hips open, hands fall, then swing through.",
        );
        add(
          scoreLess(m.headMovement as number, 0.3, 0.52) > 58,
          "Head remains controlled as speed increases.",
          "Head moves excessively during transition.",
          "Make half-speed swings while keeping your head behind the ball until impact.",
        );
        break;
      }
      case "impact":
        add(
          scoreLess(Math.abs(((m.spineTilt as number) ?? 0) - ((m.setupSpineTilt as number) ?? 0)), 8, 22) > 62,
          "Spine angle is close to setup at impact.",
          "Spine angle changes too much into impact.",
          "Use the impact bag drill and return to the bag without standing up.",
        );
        add(
          scoreLess(m.headMovement as number, 0.28, 0.5) > 58,
          "Head stays reasonably steady through impact.",
          "Head has moved a lot by impact.",
          "Strike half shots while feeling your chest turn under a steady head.",
        );
        add(
          scoreLess(m.handTravel as number, 0.48, 0.9) > 50,
          "Hands return close enough to the strike zone.",
          "Hands are not returning near the setup strike zone.",
          "Rehearse slow impact positions with hands slightly ahead and pressure forward.",
        );
        break;
      case "follow":
        add(
          scoreGreater(m.leadArmAngle as number, 142, 116) > 58,
          "Arms extend through the ball before folding.",
          "Lead arm folds early after impact.",
          "Use the towel-under-arm drill and feel extension to waist high.",
        );
        add(
          scoreLess(m.balanceCenter as number, 0.42, 0.75) > 52,
          "Body stays inside the stance through release.",
          "Balance is starting to leak after impact.",
          "Hold a three-second finish after each half-speed swing.",
        );
        break;
      case "finish":
        add(
          scoreLess(m.balanceCenter as number, 0.36, 0.72) > 58,
          "Finish is balanced enough to hold.",
          "Finish balance is unstable.",
          "Make three-quarter swings and freeze the finish until the ball lands.",
        );
        add(
          scoreLess(m.headMovement as number, 0.48, 0.82) > 50,
          "Head and torso finish under control.",
          "Finish has too much head or body drift.",
          "Use the slow-motion tempo drill and end tall over the lead side.",
        );
        break;
      default:
        break;
    }

    if (!positives.length) positives.push("The phase was detected and can now be compared consistently.");
    if (!issues.length) issues.push("No major checkpoint issue was flagged in this phase.");

    return { positives: positives.slice(0, 3), issues: issues.slice(0, 3), suggestion };
  }

  function computeScores(frames: SwingFrame[], phaseIndexes: Record<string, number>) {
    const setup = frames[phaseIndexes.setup];
    const takeaway = frames[phaseIndexes.takeaway];
    const backswing = frames[phaseIndexes.backswing];
    const top = frames[phaseIndexes.top];
    const downswing = frames[phaseIndexes.downswing];
    const impact = frames[phaseIndexes.impact];
    const finish = frames[phaseIndexes.finish];
    const follow = frames[phaseIndexes.follow];

    const backTime = Math.max((top?.time ?? 0) - (setup?.time ?? 0), 0.01);
    const downTime = Math.max((impact?.time ?? 0) - (top?.time ?? 0), 0.01);
    const tempoRatio = backTime / downTime;
    const hipUnwind = Math.abs((metric(top, "hipTurn") ?? 0) - (metric(downswing, "hipTurn") ?? 0));
    const shoulderUnwind = Math.abs((metric(top, "shoulderTurn") ?? 0) - (metric(downswing, "shoulderTurn") ?? 0));
    const sequenceScore = weighted([
      [scoreGreater(hipUnwind - shoulderUnwind * 0.35, 2, -5), 0.45],
      [scoreLess(metric(downswing, "headMovement"), 0.3, 0.55), 0.25],
      [scoreLess(Math.abs((metric(downswing, "gripXShift") ?? 0) - (metric(takeaway, "gripXShift") ?? 0)), 0.42, 0.95), 0.3],
    ]);

    const categoryScores = [
      {
        id: "setup",
        label: "Setup posture",
        score: weighted([
          [scoreBetween(metric(setup, "spineTilt"), 8, 30, 18), 0.38],
          [scoreBetween(metric(setup, "kneeFlex"), 8, 30, 18), 0.28],
          [scoreLess(metric(setup, "balanceCenter"), 0.3, 0.66), 0.22],
          [scoreGreater(metric(setup, "confidence"), 0.42, 0.14), 0.12],
        ]),
        note: "Hinge, knee flex, stance balance, and landmark confidence.",
        reason: `Spine tilt ${fmtDegrees(metric(setup, "spineTilt"))}, knee flex ${fmtDegrees(metric(setup, "kneeFlex"))}, balance proxy ${fmtRatio(metric(setup, "balanceCenter"))}.`,
      },
      {
        id: "takeaway",
        label: "Takeaway path",
        score: weighted([
          [scoreLess(metric(takeaway, "headMovement"), 0.13, 0.34), 0.28],
          [scoreLess(metric(takeaway, "absHipSway"), 0.14, 0.35), 0.32],
          [scoreBetween(metric(takeaway, "handTravel"), 0.12, 0.85, 0.5), 0.24],
          [scoreLess(metric(takeaway, "leadArmBend"), 24, 58), 0.16],
        ]),
        note: "Early width, centered turn, and quiet head.",
        reason: `Head moved ${fmtRatio(metric(takeaway, "headMovement"), " torso")} and hip sway was ${fmtRatio(metric(takeaway, "absHipSway"), " torso")}.`,
      },
      {
        id: "backswing",
        label: "Backswing position",
        score: weighted([
          [scoreGreater((metric(backswing, "shoulderTurn") ?? 0) - (metric(backswing, "hipTurn") ?? 0), 4, -7), 0.3],
          [scoreGreater(metric(backswing, "leadArmAngle"), 152, 128), 0.25],
          [scoreLess(metric(backswing, "headMovement"), 0.26, 0.5), 0.2],
          [scoreLess(metric(backswing, "absHipSway"), 0.24, 0.48), 0.25],
        ]),
        note: "Coil, lead-arm width, and sway control.",
        reason: `Shoulder/hip separation was ${fmtDegrees((metric(backswing, "shoulderTurn") ?? 0) - (metric(backswing, "hipTurn") ?? 0))}; lead arm angle ${fmtDegrees(metric(backswing, "leadArmAngle"))}.`,
      },
      {
        id: "top",
        label: "Top position",
        score: weighted([
          [scoreGreater(metric(top, "armHeight"), 0.24, -0.08), 0.24],
          [scoreGreater(metric(top, "leadArmAngle"), 152, 128), 0.28],
          [scoreBetween(metric(top, "shoulderHipSeparation"), 4, 42, 28), 0.22],
          [scoreLess(metric(top, "absHipSway"), 0.24, 0.5), 0.26],
        ]),
        note: "Top checkpoint, extension, and centered coil.",
        reason: `Hands reached ${fmtRatio(metric(top, "armHeight"), " torso")} above shoulder center; top sway was ${fmtRatio(metric(top, "absHipSway"), " torso")}.`,
      },
      {
        id: "sequence",
        label: "Downswing sequencing",
        score: sequenceScore,
        note: "Hip/shoulder unwind order, head control, and path proxy.",
        reason: `Hip unwind proxy ${fmtDegrees(hipUnwind)}, shoulder unwind proxy ${fmtDegrees(shoulderUnwind)}, downswing head move ${fmtRatio(metric(downswing, "headMovement"), " torso")}.`,
      },
      {
        id: "impact",
        label: "Impact position",
        score: weighted([
          [scoreLess(Math.abs((metric(impact, "spineTilt") ?? 0) - (metric(setup, "spineTilt") ?? 0)), 8, 24), 0.34],
          [scoreLess(metric(impact, "headMovement"), 0.3, 0.55), 0.24],
          [scoreLess(metric(impact, "handTravel"), 0.5, 0.95), 0.2],
          [scoreLess(metric(impact, "balanceCenter"), 0.42, 0.76), 0.22],
        ]),
        note: "Return to posture, strike-zone delivery, and balance.",
        reason: `Head moved ${fmtRatio(metric(impact, "headMovement"), " torso")} and spine angle changed ${fmtDegrees(Math.abs((metric(impact, "spineTilt") ?? 0) - (metric(setup, "spineTilt") ?? 0)))} from setup.`,
      },
      {
        id: "finish",
        label: "Balance and finish",
        score: weighted([
          [scoreLess(metric(finish, "balanceCenter"), 0.38, 0.74), 0.38],
          [scoreLess(metric(finish, "headMovement"), 0.52, 0.86), 0.25],
          [scoreGreater(metric(follow, "leadArmAngle"), 138, 112), 0.2],
          [scoreGreater(metric(finish, "confidence"), 0.32, 0.1), 0.17],
        ]),
        note: "Release shape and ability to hold the finish.",
        reason: `Finish balance proxy ${fmtRatio(metric(finish, "balanceCenter"))}; follow-through lead arm angle ${fmtDegrees(metric(follow, "leadArmAngle"))}.`,
      },
      {
        id: "tempo",
        label: "Tempo",
        score: weighted([
          [scoreBetween(tempoRatio, 2.4, 3.5, 1.8), 0.72],
          [scoreLess(Math.abs(backTime + downTime - ((finish?.time ?? 0) - (setup?.time ?? 0)) * 0.66), 1.4, 3.6), 0.28],
        ]),
        note: `Backswing to downswing ratio: ${round(tempoRatio, 2)}:1.`,
        reason: `Detected backswing ${round(backTime, 2)}s and downswing ${round(downTime, 2)}s, target is near 3:1.`,
      },
    ].map((item) => ({ ...item, score: Math.round(item.score) }));

    const overall = Math.round(
      weighted(categoryScores.map((score) => [score.score, score.id === "tempo" ? 0.8 : 1])),
    );

    return { overall, categoryScores, tempoRatio, sequenceScore };
  }

  function detectMistakes(frames: SwingFrame[], phaseIndexes: Record<string, number>, scores: ReturnType<typeof computeScores>) {
    const setup = frames[phaseIndexes.setup];
    const takeaway = frames[phaseIndexes.takeaway];
    const backswing = frames[phaseIndexes.backswing];
    const top = frames[phaseIndexes.top];
    const downswing = frames[phaseIndexes.downswing];
    const impact = frames[phaseIndexes.impact];
    const follow = frames[phaseIndexes.follow];
    const finish = frames[phaseIndexes.finish];
    const maxHead = Math.max(...frames.map((frame) => metric(frame, "headMovement") ?? 0));
    const maxSwayToTop = Math.max(...frames.slice(phaseIndexes.setup, phaseIndexes.top + 1).map((frame) => metric(frame, "absHipSway") ?? 0));
    const mistakes: SwingMistake[] = [];

    const push = (mistake: SwingMistake, condition: boolean) => {
      if (condition) mistakes.push(mistake);
    };

    push(
      {
        id: "poor_posture",
        label: "Poor posture at setup",
        severity: "high",
        evidence: `Setup spine tilt ${fmtDegrees(metric(setup, "spineTilt"))}, knee flex ${fmtDegrees(metric(setup, "kneeFlex"))}.`,
        drillIds: ["wall_posture"],
      },
      scores.categoryScores.find((item) => item.id === "setup")!.score < 72,
    );

    push(
      {
        id: "head_move",
        label: "Head moving too much",
        severity: "high",
        evidence: `Peak head movement is ${fmtRatio(maxHead, " torso lengths")}.`,
        drillIds: ["slow_tempo", "wall_posture"],
      },
      maxHead > 0.36,
    );

    push(
      {
        id: "under_rotation",
        label: "Under-rotation",
        severity: "medium",
        evidence: `Top shoulder-turn proxy is ${fmtDegrees(metric(top, "shoulderTurn"))}.`,
        drillIds: ["slow_tempo"],
      },
      (metric(top, "shoulderTurn") ?? 99) < 14 && (metric(top, "armHeight") ?? 0) < 0.28,
    );

    push(
      {
        id: "over_rotation",
        label: "Over-rotation",
        severity: "medium",
        evidence: `Top hip sway is ${fmtRatio(metric(top, "absHipSway"), " torso")} with a large shoulder-turn proxy.`,
        drillIds: ["slow_tempo", "wall_posture"],
      },
      (metric(top, "shoulderTurn") ?? 0) > 72 || ((metric(top, "absHipSway") ?? 0) > 0.42 && (metric(top, "handTravel") ?? 0) > 1.1),
    );

    push(
      {
        id: "early_extension",
        label: "Early extension",
        severity: "high",
        evidence: `Spine tilt changes ${fmtDegrees(Math.abs((metric(impact, "spineTilt") ?? 0) - (metric(setup, "spineTilt") ?? 0)))} by impact.`,
        drillIds: ["wall_posture", "impact_bag"],
      },
      Math.abs((metric(impact, "spineTilt") ?? 0) - (metric(setup, "spineTilt") ?? 0)) > 16,
    );

    push(
      {
        id: "sway",
        label: "Swaying instead of rotating",
        severity: "medium",
        evidence: `Backswing lateral hip movement peaks at ${fmtRatio(maxSwayToTop, " torso lengths")}.`,
        drillIds: ["alignment_takeaway", "wall_posture"],
      },
      maxSwayToTop > 0.3,
    );

    push(
      {
        id: "collapsed_lead_arm",
        label: "Collapsed lead arm",
        severity: "medium",
        evidence: `Lead arm angle at the top is ${fmtDegrees(metric(top, "leadArmAngle"))}.`,
        drillIds: ["towel_connection"],
      },
      (metric(top, "leadArmAngle") ?? 180) < 148,
    );

    push(
      {
        id: "poor_weight_transfer",
        label: "Poor weight transfer",
        severity: "medium",
        evidence: `Hip shift from top to finish is ${fmtRatio(Math.abs((metric(finish, "hipSway") ?? 0) - (metric(top, "hipSway") ?? 0)), " torso")}.`,
        drillIds: ["step_through"],
      },
      Math.abs((metric(finish, "hipSway") ?? 0) - (metric(top, "hipSway") ?? 0)) < 0.09,
    );

    push(
      {
        id: "over_the_top",
        label: "Coming over the top",
        severity: "medium",
        evidence: `Downswing path shift proxy is ${fmtRatio(Math.abs((metric(downswing, "gripXShift") ?? 0) - (metric(takeaway, "gripXShift") ?? 0)), " torso")}.`,
        drillIds: ["pump_sequence", "alignment_takeaway"],
      },
      Math.abs((metric(downswing, "gripXShift") ?? 0) - (metric(takeaway, "gripXShift") ?? 0)) > 0.62 && scores.sequenceScore < 68,
    );

    push(
      {
        id: "chicken_wing",
        label: "Chicken wing after impact",
        severity: "medium",
        evidence: `Lead arm angle in follow-through is ${fmtDegrees(metric(follow, "leadArmAngle"))}.`,
        drillIds: ["towel_connection"],
      },
      (metric(follow, "leadArmAngle") ?? 180) < 138,
    );

    push(
      {
        id: "loss_balance",
        label: "Loss of balance at finish",
        severity: "high",
        evidence: `Finish balance proxy is ${fmtRatio(metric(finish, "balanceCenter"))}.`,
        drillIds: ["slow_tempo", "step_through"],
      },
      (metric(finish, "balanceCenter") ?? 0) > 0.62 || scores.categoryScores.find((item) => item.id === "finish")!.score < 66,
    );

    return mistakes.slice(0, 8);
  }

  function buildPhases(frames: SwingFrame[], phaseIndexes: Record<string, number>, context: Record<string, any>) {
    return PHASES.map((phase) => {
      const index = phaseIndexes[phase.id];
      const frame = frames[index] ?? frames[0];
      const feedback = phaseFeedback(phase.id, frame, context);
      const ref = REFERENCE_PHASES[phase.id];
      const head = metric(frame, "headMovement");
      const sway = metric(frame, "absHipSway");
      const confidenceValue = metric(frame, "confidence") ?? 0;
      const confidenceLabel: PhaseResult["confidenceLabel"] = confidenceValue >= 0.52 ? "high" : confidenceValue >= 0.32 ? "medium" : "low";
      const deltaSummary = `Normalized gap: head ${fmtRatio(head, " torso")}, sway ${fmtRatio(sway, " torso")}, lead arm ${fmtDegrees(metric(frame, "leadArmAngle"))}.`;

      return {
        id: phase.id,
        label: phase.label,
        index,
        time: frame.time,
        confidence: round(confidenceValue, 2) ?? 0,
        confidenceLabel,
        frame,
        metrics: metricRows(frame),
        positives: feedback.positives,
        issues: feedback.issues,
        suggestion: feedback.suggestion,
        referenceCue: ref.cue,
        deltaSummary,
      };
    });
  }

  function uniqueDrills(mistakes: SwingMistake[]) {
    const ids = new Set<string>();
    mistakes.forEach((mistake) => mistake.drillIds.forEach((id) => ids.add(id)));
    if (!ids.size) {
      ids.add("slow_tempo");
      ids.add("impact_bag");
    }
    return Array.from(ids)
      .map((id) => DRILLS[id])
      .filter(Boolean)
      .slice(0, 5);
  }

  function detectTrimWindow(frames: SwingFrame[], phaseIndexes: Record<string, number>) {
    const range = detectActiveSwingRange(frames);

    if (!range.activeCount) {
      const start = frames[phaseIndexes.setup]?.time ?? 0;
      const end = frames[phaseIndexes.finish]?.time ?? frames[frames.length - 1]?.time ?? start;
      return { start, end, confidence: "low" };
    }

    return {
      start: frames[range.setup]?.time ?? 0,
      end: frames[range.end]?.time ?? frames[frames.length - 1]?.time ?? 0,
      confidence: range.confidence,
    };
  }

  function landmarkVisibility(frames: SwingFrame[], names: string[]) {
    if (!frames.length) return 0;
    const visibleFrames = frames.filter((frame) => names.every((name) => keypoint(frame.pose, name))).length;
    return visibleFrames / frames.length;
  }

  function summarizeModelConfidence(frames: SwingFrame[], phases: PhaseResult[]) {
    const frameConfidence = frames.map((frame) => metric(frame, "confidence") ?? 0);
    const average = avg(frameConfidence) ?? 0;
    const lowFrameRate = frames.length ? frameConfidence.filter((value) => value < 0.24).length / frames.length : 1;
    const lowestPhase = [...phases].sort((a, b) => a.confidence - b.confidence)[0] ?? null;
    const label = average >= 0.52 && lowFrameRate < 0.12 ? "high" : average >= 0.34 && lowFrameRate < 0.32 ? "medium" : "low";

    return {
      average: round(average, 2) ?? 0,
      label,
      lowFrameRate: round(lowFrameRate, 2) ?? 0,
      lowestPhase: lowestPhase
        ? {
            id: lowestPhase.id,
            label: lowestPhase.label,
            confidence: lowestPhase.confidence,
          }
        : null,
      summary:
        label === "high"
          ? "Most body landmarks were visible enough for a confident 2D read."
          : label === "medium"
            ? "Analysis is usable, but a cleaner camera setup would improve confidence."
            : "Pose confidence was low, so treat detailed angle feedback as directional.",
    };
  }

  function detectCaptureQuality(
    frames: SwingFrame[],
    phases: PhaseResult[],
    viewAngle: "face-on" | "down-the-line",
    videoDuration: number,
    trimWindow: { start: number; end: number; confidence: string },
    analysisWindow?: { start: number; end: number; confidence: string; source?: string } | null,
  ) {
    const notes: QualityNote[] = [];
    const modelConfidence = summarizeModelConfidence(frames, phases);
    const avgBrightness = avg(frames.map((frame) => frame.brightness)) ?? null;
    const fullBodyVisibility = landmarkVisibility(frames, [
      "left_shoulder",
      "right_shoulder",
      "left_hip",
      "right_hip",
      "left_knee",
      "right_knee",
      "left_ankle",
      "right_ankle",
    ]);
    const handVisibility = landmarkVisibility(frames, ["left_wrist", "right_wrist"]);

    notes.push({
      id: "view-angle",
      label: viewAngle === "face-on" ? "Face-on view selected" : "Down-the-line view selected",
      severity: "info",
      detail:
        viewAngle === "face-on"
          ? "Best for weight shift, head movement, spine angle, and balance. Use down-the-line for stronger swing-plane evidence."
          : "Best for posture loss, early extension, and path direction. Use face-on for stronger weight-shift evidence.",
    });

    if (Number.isFinite(videoDuration)) {
      if (videoDuration < 1.8) {
        notes.push({
          id: "duration-short",
          label: "Video is very short",
          severity: "warn",
          detail: `The clip is ${round(videoDuration, 1)}s. Record the full setup through finish for more reliable phase detection.`,
        });
      } else if (videoDuration > 14) {
        notes.push({
          id: "duration-long",
          label: "Video was auto-trimmed",
          severity: "info",
          detail: `The source clip is ${round(videoDuration, 1)}s. SwingGrade analyzed ${round(
            analysisWindow?.start ?? trimWindow.start,
            1,
          )}s to ${round(analysisWindow?.end ?? trimWindow.end, 1)}s and detected the swing from ${round(trimWindow.start, 1)}s to ${round(
            trimWindow.end,
            1,
          )}s.`,
        });
      } else {
        notes.push({
          id: "duration-good",
          label: "Video length is usable",
          severity: "good",
          detail: `The clip is ${round(videoDuration, 1)}s, enough for setup through finish if the swing is centered in frame.`,
        });
      }
    }

    if (modelConfidence.label === "low") {
      notes.push({
        id: "pose-confidence",
        label: "Low pose confidence",
        severity: "warn",
        detail: `${modelConfidence.summary} Lowest phase: ${modelConfidence.lowestPhase?.label ?? "unknown"}.`,
      });
    } else {
      notes.push({
        id: "pose-confidence",
        label: `${modelConfidence.label} model confidence`,
        severity: modelConfidence.label === "high" ? "good" : "info",
        detail: `${modelConfidence.summary} Average confidence ${modelConfidence.average}.`,
      });
    }

    if (fullBodyVisibility < 0.42) {
      notes.push({
        id: "full-body",
        label: "Full body not consistently visible",
        severity: "warn",
        detail: "Shoulders, hips, knees, and ankles were not all visible often enough. Step the camera back and keep feet in frame.",
      });
    } else {
      notes.push({
        id: "full-body",
        label: "Full-body visibility is usable",
        severity: "good",
        detail: `${Math.round(fullBodyVisibility * 100)}% of sampled frames had the main body chain visible.`,
      });
    }

    if (handVisibility < 0.5) {
      notes.push({
        id: "hands",
        label: "Hands/club area needs visibility",
        severity: "info",
        detail: "Wrists were hard to track in several frames. Keep hands and club in the shot for cleaner takeaway and impact feedback.",
      });
    }

    if (avgBrightness !== null && avgBrightness < 54) {
      notes.push({
        id: "lighting",
        label: "Lighting looks dark",
        severity: avgBrightness < 36 ? "warn" : "info",
        detail: `Average frame brightness was ${round(avgBrightness, 0)}/255. Add light on the golfer side rather than behind the camera.`,
      });
    } else if (avgBrightness !== null) {
      notes.push({
        id: "lighting",
        label: "Lighting is usable",
        severity: "good",
        detail: `Average frame brightness was ${round(avgBrightness, 0)}/255.`,
      });
    }

    return {
      notes,
      modelConfidence,
      capture: {
        videoDuration: round(videoDuration, 2),
        analysisWindow: analysisWindow
          ? {
              start: round(analysisWindow.start, 2),
              end: round(analysisWindow.end, 2),
              confidence: analysisWindow.confidence,
              source: analysisWindow.source ?? "motion",
            }
          : null,
        averageBrightness: round(avgBrightness, 0),
        fullBodyVisibility: round(fullBodyVisibility, 2),
        handVisibility: round(handVisibility, 2),
      },
    };
  }

  function buildPracticePlan(drills: SwingDrill[], categoryScores: CategoryScore[]) {
    const weakest = [...categoryScores].sort((a, b) => a.score - b.score)[0];
    const plan = drills.slice(0, 3).map((drill) => ({
      drillId: drill.id,
      name: drill.name,
      time: drill.time,
      reps: drill.reps,
      focus: drill.target,
    }));

    if (!plan.some((item) => item.drillId === "slow_tempo")) {
      const tempo = DRILLS.slow_tempo;
      plan.push({ drillId: tempo.id, name: tempo.name, time: tempo.time, reps: tempo.reps, focus: tempo.target });
    }

    return {
      focus: weakest ? `Focus: raise ${weakest.label.toLowerCase()} from ${weakest.score}/100.` : "Focus: repeat the strongest checkpoint.",
      items: plan.slice(0, 4),
    };
  }

  function angleNotes(viewAngle: "face-on" | "down-the-line") {
    if (viewAngle === "down-the-line") {
      return [
        "Down-the-line view is best for posture loss, early extension, club path direction, and swing plane proxies.",
        "Weight shift is less reliable from this angle because lateral pressure movement is harder to see in 2D.",
      ];
    }
    return [
      "Face-on view is best for weight shift, head movement, spine angle, balance, and lead-arm structure.",
      "Swing plane and club path need a down-the-line view or club tracking for stronger evidence.",
    ];
  }

  function analyzeSwing(
    frames: SwingFrame[],
    options: {
      handedness?: SwingHandedness;
      viewAngle?: "face-on" | "down-the-line";
      videoDuration?: number;
      analysisWindow?: { start: number; end: number; confidence: string; source?: string } | null;
    } = {},
  ) {
    const handedness = options.handedness ?? "right";
    const viewAngle = options.viewAngle ?? "face-on";
    const enriched = enrichFrames(frames, handedness);
    const phaseIndexes = detectPhaseIndexes(enriched);
    const trimWindow = detectTrimWindow(enriched, phaseIndexes);
    const scoreContext = computeScores(enriched, phaseIndexes);
    const mistakes = detectMistakes(enriched, phaseIndexes, scoreContext);
    const phases = buildPhases(enriched, phaseIndexes, scoreContext);
    const drills = uniqueDrills(mistakes);
    const practicePlan = buildPracticePlan(drills, scoreContext.categoryScores);
    const videoDuration = options.videoDuration ?? frames[frames.length - 1]?.time ?? 0;
    const quality = detectCaptureQuality(enriched, phases, viewAngle, videoDuration, trimWindow, options.analysisWindow);
    const modelNotes = [
      "MoveNet keypoints are normalized by torso length so size differences do not dominate the comparison.",
      "The pro reference is a phase-by-phase movement template, not a body-shape match.",
      "2D video can miss depth; face-on or down-the-line camera choice affects turn and path proxies.",
    ];

    return {
      frames: enriched,
      phases,
      phaseIndexes,
      overall: scoreContext.overall,
      categoryScores: scoreContext.categoryScores,
      mistakes,
      drills,
      tempoRatio: scoreContext.tempoRatio,
      sequenceScore: scoreContext.sequenceScore,
      handedness,
      viewAngle,
      analysisWindow: options.analysisWindow ?? null,
      trimWindow,
      qualityNotes: quality.notes,
      modelConfidence: quality.modelConfidence,
      captureQuality: quality.capture,
      practicePlan,
      angleNotes: angleNotes(viewAngle),
      modelNotes,
      referencePhases: REFERENCE_PHASES,
      skeletonEdges: SKELETON_EDGES,
      generatedAt: new Date().toISOString(),
    };
  }

  function makeDemoPose(width: number, height: number, skeleton: Record<string, [number, number]>, noise = 0) {
    return {
      keypoints: Object.entries(skeleton).map(([name, [x, y]]) => ({
        name,
        x: (x / 100) * width + Math.sin(x + y) * noise,
        y: (y / 100) * height + Math.cos(x - y) * noise,
        score: 0.88,
      })),
    };
  }

  function createDemoAnalysis() {
    const width = 720;
    const height = 420;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d")!;
    const phases = PHASES;
    const frames: SwingFrame[] = [];

    for (let i = 0; i < 64; i += 1) {
      const phase = phases[Math.min(phases.length - 1, Math.floor((i / 64) * phases.length))];
      const ref = REFERENCE_PHASES[phase.id].skeleton;
      ctx.fillStyle = "#121b17";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "rgba(255,255,255,.08)";
      ctx.lineWidth = 2;
      for (let x = 0; x < width; x += 80) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      ctx.fillStyle = "rgba(255,255,255,.68)";
      ctx.font = "700 20px system-ui";
      ctx.fillText("SwingGrade sample frame", 28, 42);
      frames.push({
        index: i,
        time: i * 0.055,
        image: canvas.toDataURL("image/jpeg", 0.72),
        width,
        height,
        brightness: 80,
        pose: makeDemoPose(width, height, ref, phase.id === "impact" ? 8 : 4),
      });
    }

    return analyzeSwing(frames, { handedness: "right", viewAngle: "face-on", videoDuration: frames[frames.length - 1]?.time ?? 3.5 });
  }

  return {
    PHASES,
    SKELETON_EDGES,
    REFERENCE_PHASES,
    DRILLS,
    analyzeSwing,
    createDemoAnalysis,
  };
})();

if (typeof window !== "undefined" && import.meta.env.DEV) {
  (window as Window & { SwingGradeEngine?: typeof SwingGradeEngine }).SwingGradeEngine = SwingGradeEngine;
}

export type {
  CategoryScore,
  Keypoint,
  PhaseResult,
  Pose,
  SwingDrill,
  SwingFrame,
  SwingHandedness,
  SwingMistake,
  QualityNote,
};

export default SwingGradeEngine;

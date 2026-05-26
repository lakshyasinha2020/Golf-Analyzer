import { useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { motion, useMotionValueEvent, useReducedMotion, useScroll } from "framer-motion";

const golferFrames = [
  "/assets/golfer-frames/golfer-frame-01.png",
  "/assets/golfer-frames/golfer-frame-02.png",
  "/assets/golfer-frames/golfer-frame-03.png",
  "/assets/golfer-frames/golfer-frame-04.png",
  "/assets/golfer-frames/golfer-frame-05.png",
  "/assets/golfer-frames/golfer-frame-06.png",
  "/assets/golfer-frames/golfer-frame-07.png",
  "/assets/golfer-frames/golfer-frame-08.png",
  "/assets/golfer-frames/golfer-frame-09.png",
  "/assets/golfer-frames/golfer-frame-10.png",
  "/assets/golfer-frames/golfer-frame-11.png",
  "/assets/golfer-frames/golfer-frame-12.png",
];

const courseBackgrounds = [
  {
    name: "Pebble Beach Golf Links",
    image: "/assets/course-backgrounds/course-01-pebble-beach.jpg",
  },
  {
    name: "Augusta National Golf Club",
    image: "/assets/course-backgrounds/course-02-augusta-national.jpg",
  },
  {
    name: "The Old Course at St. Andrews",
    image: "/assets/course-backgrounds/course-03-st-andrews.jpg",
  },
  {
    name: "TPC Sawgrass",
    image: "/assets/course-backgrounds/course-04-tpc-sawgrass.jpg",
  },
  {
    name: "Torrey Pines",
    image: "/assets/course-backgrounds/course-05-torrey-pines.jpg",
  },
  {
    name: "Bandon Dunes",
    image: "/assets/course-backgrounds/course-06-bandon-dunes.jpg",
  },
  {
    name: "Pinehurst No. 2",
    image: "/assets/course-backgrounds/course-07-pinehurst-no2.jpg",
  },
  {
    name: "Kiawah Island Ocean Course",
    image: "/assets/course-backgrounds/course-08-kiawah-island.jpg",
  },
  {
    name: "Whistling Straits",
    image: "/assets/course-backgrounds/course-09-whistling-straits.jpg",
  },
  {
    name: "Oakmont Country Club",
    image: "/assets/course-backgrounds/course-10-oakmont.jpg",
  },
  {
    name: "Shinnecock Hills",
    image: "/assets/course-backgrounds/course-11-shinnecock-hills.jpg",
  },
  {
    name: "Cypress Point Club",
    image: "/assets/course-backgrounds/course-12-cypress-point.jpg",
  },
  {
    name: "Spyglass Hill Golf Course",
    image: "/assets/course-backgrounds/course-13-spyglass-hill.jpg",
  },
  {
    name: "Riviera Country Club",
    image: "/assets/course-backgrounds/course-14-riviera.jpg",
  },
  {
    name: "Muirfield Village Golf Club",
    image: "/assets/course-backgrounds/course-15-muirfield-village.jpg",
  },
  {
    name: "East Lake Golf Club",
    image: "/assets/course-backgrounds/course-16-east-lake.jpg",
  },
  {
    name: "Quail Hollow Club",
    image: "/assets/course-backgrounds/course-17-quail-hollow.jpg",
  },
  {
    name: "Harbour Town Golf Links",
    image: "/assets/course-backgrounds/course-18-harbour-town.jpg",
  },
  {
    name: "Chambers Bay",
    image: "/assets/course-backgrounds/course-19-chambers-bay.jpg",
  },
  {
    name: "Wolf Creek Golf Club",
    image: "/assets/course-backgrounds/course-20-wolf-creek.jpg",
  },
];

function clampIndex(value: number, length: number) {
  return Math.max(0, Math.min(length - 1, Math.floor(value * length)));
}

export default function SwingHero() {
  const containerRef = useRef<HTMLElement | null>(null);
  const prefersReducedMotion = useReducedMotion();
  const [frameIndex, setFrameIndex] = useState(0);
  const [courseIndex, setCourseIndex] = useState(0);
  const [missingFrames, setMissingFrames] = useState<Record<string, boolean>>({});
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"],
  });

  useMotionValueEvent(scrollYProgress, "change", (value) => {
    if (prefersReducedMotion) return;
    // Scroll progress runs from 0 to 1 through the tall hero. Quantizing it to array indexes makes
    // the swing and course slideshow scrub forward on scroll down and reverse on scroll up.
    setFrameIndex(clampIndex(value, golferFrames.length));
    setCourseIndex(clampIndex(value, courseBackgrounds.length));
  });

  const course = courseBackgrounds[prefersReducedMotion ? 0 : courseIndex];
  const currentFrame = golferFrames[prefersReducedMotion ? 0 : frameIndex];
  const shouldUseFallbackGolfer = missingFrames[currentFrame];
  const arcProgress = useMemo(() => (prefersReducedMotion ? 0.08 : frameIndex / (golferFrames.length - 1)), [frameIndex, prefersReducedMotion]);

  return (
    <section ref={containerRef} className="relative h-[320vh] bg-[oklch(8%_0.018_165)]">
      <div className="sticky top-0 min-h-screen overflow-hidden">
        <motion.div
          key={course.image}
          className="absolute inset-0 bg-cover bg-center"
          style={{
            backgroundImage: `linear-gradient(135deg, oklch(14% 0.04 165 / 0.34), oklch(5% 0.018 165 / 0.18)), url(${course.image})`,
          }}
          initial={{ opacity: 0, scale: 1.035 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.38, ease: [0.16, 1, 0.3, 1] }}
        />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_48%,oklch(10%_0.02_165_/_0.12),oklch(0%_0_0_/_0.72)_58%,oklch(0%_0_0_/_0.86)),linear-gradient(180deg,oklch(0%_0_0_/_0.56),oklch(0%_0_0_/_0.34)_42%,oklch(0%_0_0_/_0.78))]" />
        <div className="landing-noise" />

        <div className="relative z-10 grid min-h-screen grid-rows-[auto_1fr_auto] px-5 py-5 sm:px-8 lg:px-12">
          <header className="flex items-center justify-between">
            <Link to="/" className="flex items-center gap-3">
              <span className="grid size-11 place-items-center rounded-xl bg-[oklch(94%_0.02_155)] text-base font-black text-[oklch(8%_0.018_165)] shadow-[0_0_36px_oklch(76%_0.16_145_/_0.28)]">
                SG
              </span>
              <span className="text-sm font-black uppercase tracking-[0.24em] text-[oklch(95%_0.006_155)]">SwingGrade</span>
            </Link>
            <Link className="hidden rounded-full border border-[oklch(100%_0_0_/_0.2)] px-5 py-3 text-sm font-black text-[oklch(94%_0.006_155)] transition hover:border-[oklch(76%_0.16_145)] hover:text-[oklch(76%_0.16_145)] sm:inline-flex" to="/analyzer">
              Enter Analyzer
            </Link>
          </header>

          <div className="grid items-center gap-8 py-10 lg:grid-cols-[minmax(0,0.95fr)_minmax(22rem,1.05fr)]">
            <motion.div
              className="max-w-3xl"
              initial={{ opacity: 0, y: 22 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
            >
              <p className="mb-4 text-sm font-black uppercase tracking-[0.28em] text-[oklch(76%_0.16_145)]">Cinematic motion analysis</p>
              <h1 className="text-[clamp(3.2rem,8vw,8.8rem)] font-black leading-[0.86] tracking-normal text-[oklch(98%_0.006_155)]">
                Master Your Swing Frame by Frame
              </h1>
              <p className="mt-7 max-w-2xl text-lg font-semibold leading-8 text-[oklch(82%_0.012_155)] sm:text-xl">
                A cinematic golf swing analyzer built to turn motion into clear feedback.
              </p>
              <div className="mt-9 flex flex-wrap items-center gap-4">
                <Link className="landing-cta group" to="/analyzer">
                  <span>Analyze Your Swing</span>
                  <span className="landing-cta-arrow" aria-hidden="true">
                    -&gt;
                  </span>
                </Link>
              </div>
            </motion.div>

            <div className="relative mx-auto grid aspect-[4/5] w-full max-w-[31rem] place-items-center">
              <SwingArc progress={arcProgress} />
              <div className="absolute inset-x-6 bottom-8 h-10 rounded-full bg-[oklch(0%_0_0_/_0.56)] blur-xl" />
              <div className="relative grid h-full w-full place-items-center">
                {!shouldUseFallbackGolfer ? (
                  <img
                    className="max-h-[82%] max-w-[88%] object-contain drop-shadow-[0_32px_30px_oklch(0%_0_0_/_0.72)]"
                    src={currentFrame}
                    alt={`Golfer swing frame ${frameIndex + 1}`}
                    onError={() => setMissingFrames((frames) => ({ ...frames, [currentFrame]: true }))}
                  />
                ) : (
                  <FallbackGolfer frameIndex={frameIndex} />
                )}
              </div>
            </div>
          </div>

          <footer className="flex flex-col gap-3 pb-2 sm:flex-row sm:items-end sm:justify-between">
            <motion.div
              key={course.name}
              className="w-fit rounded-full border border-[oklch(100%_0_0_/_0.16)] bg-[oklch(0%_0_0_/_0.32)] px-4 py-2 text-sm font-black text-[oklch(90%_0.006_155)]"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.28 }}
            >
              Course inspiration: <span className="text-[oklch(76%_0.16_145)]">{course.name}</span>
            </motion.div>
            <div className="flex items-center gap-2 text-xs font-black uppercase tracking-[0.18em] text-[oklch(78%_0.012_155)]">
              <span>{String(frameIndex + 1).padStart(2, "0")}</span>
              <span className="h-px w-16 bg-[oklch(76%_0.16_145)]" />
              <span>{String(golferFrames.length).padStart(2, "0")}</span>
            </div>
          </footer>
        </div>
      </div>
    </section>
  );
}

function SwingArc({ progress }: { progress: number }) {
  const dashOffset = 620 - progress * 540;

  return (
    <svg className="pointer-events-none absolute inset-0 h-full w-full overflow-visible" viewBox="0 0 420 520" aria-hidden="true">
      <defs>
        <filter id="arcGlow">
          <feGaussianBlur stdDeviation="6" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <path
        d="M68 332 C86 98 276 24 348 172 C410 300 290 426 118 404"
        fill="none"
        stroke="oklch(76% 0.16 145 / 0.72)"
        strokeDasharray="620"
        strokeDashoffset={dashOffset}
        strokeLinecap="round"
        strokeWidth="5"
        filter="url(#arcGlow)"
      />
      <path
        d="M82 350 C108 142 276 74 330 182 C372 270 282 376 138 386"
        fill="none"
        stroke="oklch(98% 0.006 155 / 0.42)"
        strokeDasharray="520"
        strokeDashoffset={520 - progress * 420}
        strokeLinecap="round"
        strokeWidth="1.6"
      />
    </svg>
  );
}

function FallbackGolfer({ frameIndex }: { frameIndex: number }) {
  const poses = [
    { arm: 8, club: 14, lean: 0, head: 0 },
    { arm: -10, club: -22, lean: -2, head: 0 },
    { arm: -30, club: -52, lean: -4, head: -1 },
    { arm: -58, club: -78, lean: -6, head: -1 },
    { arm: -88, club: -104, lean: -7, head: -2 },
    { arm: -112, club: -128, lean: -8, head: -2 },
    { arm: -72, club: -68, lean: -5, head: -1 },
    { arm: -24, club: -8, lean: -3, head: 0 },
    { arm: 24, club: 36, lean: 4, head: 1 },
    { arm: 58, club: 76, lean: 8, head: 1 },
    { arm: 88, club: 118, lean: 10, head: 2 },
    { arm: 105, club: 138, lean: 8, head: 2 },
  ];
  const pose = poses[frameIndex] ?? poses[0];

  return (
    <div className="grid justify-items-center gap-3">
      <svg className="h-[min(72vh,31rem)] w-[min(72vw,25rem)] drop-shadow-[0_34px_28px_oklch(0%_0_0_/_0.68)]" viewBox="0 0 320 420" role="img" aria-label={`Golfer Frame ${frameIndex + 1}`}>
        <g transform={`translate(160 212) rotate(${pose.lean}) translate(-160 -212)`}>
          <ellipse cx="160" cy="82" rx="30" ry="32" fill="oklch(4% 0.01 165)" transform={`translate(${pose.head} 0)`} />
          <path d="M132 123 C145 108 176 108 190 124 C204 148 196 196 182 230 L138 230 C124 190 118 148 132 123Z" fill="oklch(4% 0.01 165)" />
          <g transform={`rotate(${pose.arm} 160 150)`}>
            <path d="M154 140 C118 160 96 188 76 226" stroke="oklch(4% 0.01 165)" strokeWidth="19" strokeLinecap="round" />
            <path d="M167 143 C128 166 103 194 83 235" stroke="oklch(4% 0.01 165)" strokeWidth="15" strokeLinecap="round" />
          </g>
          <g transform={`rotate(${pose.club} 82 232)`}>
            <path d="M82 232 L44 56" stroke="oklch(3% 0.01 165)" strokeWidth="8" strokeLinecap="round" />
            <path d="M35 52 L76 46" stroke="oklch(3% 0.01 165)" strokeWidth="10" strokeLinecap="round" />
          </g>
          <path d="M144 226 C128 278 116 330 92 382" stroke="oklch(4% 0.01 165)" strokeWidth="24" strokeLinecap="round" />
          <path d="M178 226 C184 278 204 328 234 374" stroke="oklch(4% 0.01 165)" strokeWidth="24" strokeLinecap="round" />
          <path d="M72 386 L118 386" stroke="oklch(4% 0.01 165)" strokeWidth="18" strokeLinecap="round" />
          <path d="M220 382 L266 382" stroke="oklch(4% 0.01 165)" strokeWidth="18" strokeLinecap="round" />
        </g>
      </svg>
      <span className="rounded-full bg-[oklch(0%_0_0_/_0.48)] px-3 py-1 text-xs font-black uppercase tracking-[0.2em] text-[oklch(76%_0.16_145)]">
        Golfer Frame {String(frameIndex + 1).padStart(2, "0")}
      </span>
    </div>
  );
}

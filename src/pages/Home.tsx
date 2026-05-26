import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { motion, useReducedMotion } from "framer-motion";

const courseScenes = [
  {
    name: "Cypress Point",
    caption: "Private light over the coast.",
    image: new URL("../assets/course-backgrounds/course-12-cypress-point-style.jpg", import.meta.url).href,
  },
  {
    name: "Pebble Beach",
    caption: "Wind, water, and a narrower line.",
    image: new URL("../assets/course-backgrounds/course-01-pebble-beach.jpg", import.meta.url).href,
  },
  {
    name: "Bandon Dunes",
    caption: "The swing held against open ground.",
    image: new URL("../assets/course-backgrounds/course-06-bandon-dunes.jpg", import.meta.url).href,
  },
  {
    name: "Riviera",
    caption: "Quiet pressure, measured frame by frame.",
    image: new URL("../assets/course-backgrounds/course-14-riviera.jpg", import.meta.url).href,
  },
];

const signalRows = [
  {
    label: "Frame",
    title: "Start with clean evidence.",
    body: "Upload or record the swing, then keep the whole body, club, and hands visible before the analysis begins.",
  },
  {
    label: "Sequence",
    title: "Read the movement in order.",
    body: "SwingGrade separates setup, takeaway, top, downswing, impact, and finish so the feedback stays tied to motion.",
  },
  {
    label: "Practice",
    title: "Leave with the next rep.",
    body: "The page ends with checkpoint notes and focused drills, not a cloud of vague advice.",
  },
];

const analysisNotes = ["posture", "tempo", "balance", "impact", "finish"];

const reveal = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0 },
};

export default function Home() {
  const prefersReducedMotion = useReducedMotion();
  const [activeScene, setActiveScene] = useState(0);
  const currentScene = courseScenes[activeScene];

  useEffect(() => {
    if (prefersReducedMotion) return undefined;

    const interval = window.setInterval(() => {
      setActiveScene((index) => (index + 1) % courseScenes.length);
    }, 5200);

    return () => window.clearInterval(interval);
  }, [prefersReducedMotion]);

  const sceneLabel = useMemo(() => `${currentScene.name}. ${currentScene.caption}`, [currentScene]);

  return (
    <main className="luxury-golf-page" data-theme="midnight">
      <section className="golf-hero" aria-label="SwingGrade cinematic golf landing page">
        <div className="golf-hero__backdrops" aria-hidden="true">
          {courseScenes.map((scene, index) => (
            <div
              className={`golf-hero__backdrop${index === activeScene ? " is-active" : ""}`}
              key={scene.name}
              style={{ backgroundImage: `url(${scene.image})` }}
            />
          ))}
        </div>

        <header className="golf-nav" aria-label="Primary">
          <Link className="golf-wordmark" to="/">
            <span className="golf-wordmark__mark">SG</span>
            <span>SwingGrade</span>
          </Link>
          <Link className="golf-nav__cta" to="/analyzer">
            Analyze Swing
          </Link>
        </header>

        <div className="golf-hero__center">
          <motion.div
            animate="show"
            className="golf-hero__copy"
            initial="hidden"
            transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
            variants={reveal}
          >
            <p className="golf-kicker">Welcome to the private swing room</p>
            <h1>Welcome to a sharper round.</h1>
            <p className="golf-hero__lede">
              Cinematic golf analysis for players who want the swing, the evidence, and the next practice decision in one quiet place.
            </p>
            <div className="golf-hero__actions">
              <Link className="golf-primary-cta" to="/analyzer">
                <span>Analyze Swing</span>
                <span aria-hidden="true">-&gt;</span>
              </Link>
              <span className="golf-hero__aside">Upload, record, or review a sample swing.</span>
            </div>
          </motion.div>
        </div>

        <div className="golf-hero__footer">
          <p aria-live="polite">{sceneLabel}</p>
          <div className="golf-scene-tabs" aria-label="Course background selector">
            {courseScenes.map((scene, index) => (
              <button
                aria-pressed={index === activeScene}
                className={index === activeScene ? "is-active" : ""}
                key={scene.name}
                onClick={() => setActiveScene(index)}
                type="button"
              >
                {String(index + 1).padStart(2, "0")}
              </button>
            ))}
          </div>
        </div>
      </section>

      <motion.section
        className="golf-editorial"
        initial="hidden"
        transition={{ duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
        variants={reveal}
        viewport={{ once: true, amount: 0.3 }}
        whileInView="show"
      >
        <p className="golf-section-label">Private analysis</p>
        <div className="golf-editorial__grid">
          <h2>The course is cinematic. The feedback stays practical.</h2>
          <p>
            SwingGrade keeps the page dark, calm, and focused so the video remains the subject. The analyzer turns each uploaded swing into checkpoints, confidence notes, and drills you can take back to the range.
          </p>
        </div>
      </motion.section>

      <section className="golf-signal-stack" aria-label="SwingGrade analysis flow">
        {signalRows.map((row, index) => (
          <motion.article
            className="golf-signal-row"
            initial="hidden"
            key={row.label}
            transition={{ duration: 0.5, delay: index * 0.08, ease: [0.16, 1, 0.3, 1] }}
            variants={reveal}
            viewport={{ once: true, amount: 0.35 }}
            whileInView="show"
          >
            <span>{row.label}</span>
            <h3>{row.title}</h3>
            <p>{row.body}</p>
          </motion.article>
        ))}
      </section>

      <section className="golf-suite" aria-label="Analyzer preview">
        <motion.div
          className="golf-suite__panel"
          initial="hidden"
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          variants={reveal}
          viewport={{ once: true, amount: 0.3 }}
          whileInView="show"
        >
          <div className="golf-suite__image" aria-hidden="true">
            <img alt="" loading="lazy" src={courseScenes[2].image} />
          </div>
          <div className="golf-suite__copy">
            <p className="golf-section-label">Analysis suite</p>
            <h2>Made for the moment between swings.</h2>
            <p>
              The landing page leads to the same working analyzer: video capture, pose review, phase scoring, comparison notes, and a practice plan that fits the next session.
            </p>
            <div className="golf-note-strip" aria-label="Analysis categories">
              {analysisNotes.map((note) => (
                <span key={note}>{note}</span>
              ))}
            </div>
            <Link className="golf-text-cta" to="/analyzer">
              Open analyzer <span aria-hidden="true">-&gt;</span>
            </Link>
          </div>
        </motion.div>
      </section>

      <footer className="golf-footer">
        <p>Welcome back to the range with a plan.</p>
        <div>
          <span>SwingGrade</span>
          <Link to="/analyzer">Analyze Swing</Link>
        </div>
      </footer>
    </main>
  );
}

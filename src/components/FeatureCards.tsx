import { motion } from "framer-motion";

const features = [
  {
    title: "Frame-by-frame motion",
    body: "Scrub the swing like a film sequence, with each checkpoint built around visible movement evidence.",
  },
  {
    title: "AI-powered feedback",
    body: "Pose landmarks become plain-language cues for posture, sequence, balance, and repeatable practice.",
  },
  {
    title: "Built for better practice",
    body: "The goal is not a prettier score. It is a clearer next rep, a sharper drill, and a steadier swing.",
  },
];

export default function FeatureCards() {
  return (
    <section className="relative z-10 bg-[oklch(8%_0.018_165)] px-5 py-20 sm:px-8 lg:py-28">
      <div className="mx-auto grid max-w-6xl gap-10">
        <div className="max-w-3xl">
          <p className="text-sm font-black uppercase tracking-[0.28em] text-[oklch(76%_0.16_145)]">Why it feels different</p>
          <h2 className="mt-4 text-4xl font-black leading-none tracking-normal sm:text-5xl">
            Turn one swing into a useful practice decision.
          </h2>
        </div>
        <div className="grid gap-4 md:grid-cols-3">
          {features.map((feature, index) => (
            <motion.article
              className="landing-feature"
              key={feature.title}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.35 }}
              transition={{ duration: 0.65, delay: index * 0.08, ease: [0.16, 1, 0.3, 1] }}
            >
              <span className="text-sm font-black text-[oklch(76%_0.16_145)]">0{index + 1}</span>
              <h3 className="mt-6 text-2xl font-black leading-tight">{feature.title}</h3>
              <p className="mt-4 text-sm font-semibold leading-7 text-[oklch(76%_0.015_155)]">{feature.body}</p>
            </motion.article>
          ))}
        </div>
      </div>
    </section>
  );
}

import { lazy, Suspense } from "react";

const AnalyzerProgram = lazy(() => import("../AnalyzerProgram"));

export default function Analyzer() {
  return (
    <Suspense
      fallback={
        <main className="grid min-h-screen place-items-center bg-[oklch(97.3%_0.008_185)] p-6 text-[oklch(24%_0.018_165)]">
          <div className="grid gap-3 text-center">
            <p className="text-sm font-black uppercase text-[oklch(55%_0.12_165)]">SwingGrade</p>
            <h1 className="text-3xl font-black">Loading analyzer</h1>
          </div>
        </main>
      }
    >
      <AnalyzerProgram />
    </Suspense>
  );
}

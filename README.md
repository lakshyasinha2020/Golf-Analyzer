# Golf-Analyzer

SwingGrade is a browser-based golf swing analysis MVP. It uploads or records a swing video, runs MoveNet pose estimation in the browser, detects eight swing phases, scores the motion from 0-100, flags common issues, and recommends drills.

## Run

```bash
npm install
cp .env.example .env.local
npm run dev
```


`VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` must be set in `.env.local` for accounts, private video uploads, saved swings, signed URLs, and Supabase leaderboards.

## What is included

- Video upload and camera recording
- Camera setup checklist and face-on/down-the-line view selection
- TensorFlow.js MoveNet pose/keypoint detection
- Automatic eight-phase swing checkpoint detection
- Detected swing-window metadata for trimming dead time
- Frame scrubber with previous/next controls
- Pose skeleton overlay on selected frames
- User vs normalized pro-pattern comparison
- Category grades for setup, takeaway, backswing, top, downswing sequence, impact, finish, and tempo
- Transparent score reasons for each category
- Rule-based detection for posture, sway, head movement, extension, sequencing, lead-arm collapse, chicken wing, transfer, and finish balance
- Drill recommendations with time, difficulty, fixes, and practice-plan generation
- Supabase sign up/login/logout, saved swing history, private video storage, before/after comparison, and leaderboards
- Exportable HTML report

## Supabase setup

Run [supabase/schema.sql](./supabase/schema.sql) in the Supabase SQL editor. It creates:

- private Storage bucket `swing-videos`
- table `public.swing_videos`
- limited leaderboard view `public.swing_leaderboard_entries`
- Row Level Security policies for authenticated users
- storage policies that restrict files to `user-id/file-name` folders

The app uploads videos to `swing-videos`, stores compact analysis metadata in `swing_videos.analysis`, uses signed URLs to play saved private videos, and reads leaderboard scores from the limited view.

## Publish checks

```bash
npm audit --omit=dev
npm run build
npm run start
```

`npm run start` serves the built `dist/` folder with baseline security headers. For Vercel, `vercel.json` adds the same headers and a single-page-app rewrite. For Netlify, `public/_headers` is copied into the build output.

## GitHub Pages

This project is set up to run from the repository Pages path. The deployed site is published from the `gh-pages` branch and will be available at:

```text
https://lakshyasinha2020.github.io/Golf-Analyzer/
```

Before making the app public, set these in your Supabase project dashboard:

- Auth Site URL: your production domain
- Redirect URLs: your production domain and any preview domains you trust
- Leaked password protection: enabled
- Email confirmations: enabled for public signups
- Password policy: at least 8 characters with mixed case and numbers

Security posture in this repo:

- `swing-videos` is private, limited to video uploads, and capped at 50 MB per file
- RLS restricts `swing_videos` rows to the owning authenticated user
- Storage policies restrict files to `user-id/file-name` folders
- The leaderboard view uses `security_invoker = true` and excludes storage paths and full analysis JSON
- `.env.local` is gitignored; never commit service-role keys or database passwords

## Notes

This MVP uses 2D browser pose estimation and rule-based swing heuristics. It normalizes movement by torso length to reduce body-size bias, but camera angle still matters. The cleanest capture is a full-body video from a stable face-on or down-the-line view.

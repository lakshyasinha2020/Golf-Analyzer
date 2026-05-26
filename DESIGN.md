# Design

## Theme

SwingGrade is used beside a practice bay, in daylight or indoor range lighting, with a golfer trying to make a quick decision before the next swing. A light, softly tinted product UI keeps the video readable and avoids the theatrics of a broadcast scoreboard.

## Color

Use a restrained palette with tinted neutrals and one primary green accent. Reserve warm colors for score gradients, warnings, and drill urgency.

- Ink: `oklch(24% 0.018 165)`
- Muted text: `oklch(48% 0.018 165)`
- Paper: `oklch(97.3% 0.008 185)`
- Panel: `oklch(99% 0.006 165)`
- Line: `oklch(88% 0.012 190)`
- Primary moss: `oklch(55% 0.12 165)`
- Grass: `oklch(73% 0.14 135)`
- Gold: `oklch(68% 0.13 82)`
- Clay: `oklch(64% 0.15 42)`
- Sky: `oklch(61% 0.1 220)`

## Typography

Use a product sans stack, with strong numeric weight for scores and compact hierarchy for dense analysis. Body copy should stay under 75 characters when possible.

## Components

- Tool panels use 8px radius, subtle borders, and quiet shadows.
- Primary actions are dark ink with clear hover and active feedback.
- Secondary actions use panel surfaces and moss hover states.
- Badges communicate model, warning, and fault status with semantic tints.
- The phase timeline is horizontally scannable and uses active state more strongly than color alone.
- Score meters use one consistent clay-to-moss range.

## Layout

Use a product dashboard shell: upload/record workspace first, grade summary second, then phase timeline, side-by-side comparison, checkpoint report, and drill recommendations. Mobile should stack into the same order without horizontal overflow.

## Motion

Use fast, stateful transitions only: hover lift, selected phase feedback, progress movement, and reduced-motion fallbacks. Avoid page-load choreography.

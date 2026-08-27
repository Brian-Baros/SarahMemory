# Feature Registry

`featureRegistry.tsx` is the shell source of truth.

Use it when you need to find:

- which app/screen ID opens which React component;
- which file owns each screen;
- what a panel is supposed to do;
- how desktop windows and mobile screens map to the same UI.

The existing working V9 screen files remain in `src/components/screens`, `src/components/chat`, and `src/components/avatar` to avoid breaking known-good imports and backend behavior.

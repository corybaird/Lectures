# Reveal.js Korea 2026 Presentation

HTML slide deck powered by Reveal.js 5.1

## Serving Presentation

- Using uv (port 8080): `/opt/homebrew/bin/uv run python -m http.server 8080 --directory reports/korea_2026/reveal`
- Using npx (port 8080): `npx serve reports/korea_2026/reveal -l 8080`
- Browser URL: `http://localhost:8080`

## Keyboard Shortcuts

- `F`: Fullscreen mode
- `ESC` or `O`: Slide overview grid
- `S`: Speaker notes view
- `B` or `.`: Pause / blank screen
- Space or Arrow keys: Next / previous slide

## File Structure

- `slides.md`: markdown source for presentation slides
- `index.html`: Reveal.js loader and theme shell
- `figures/`: local diagrams and visual assets

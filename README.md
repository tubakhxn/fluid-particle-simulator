# dev/creator=tubakhxn

## dev/creator=tubakhxn

A hand-controlled 3D fluid particle simulation that runs in your browser. Your webcam is the background, a red liquid sloshes inside an orange wireframe cube, and your hands rotate and zoom the cube.

## About

- About 3600 particles are simulated as a real 3D fluid (gravity, pressure, viscosity, wall collisions).
- The liquid is drawn on the GPU (WebGL2) as a smooth translucent surface with fine ring particles on top.
- Hand tracking uses Google MediaPipe Hand Landmarker, so no hardware besides a webcam is needed.
- Everything is plain HTML, CSS and JavaScript. There are no dependencies and no build step.

## Controls

| Input | Result |
| --- | --- |
| One hand | Tilt, rotate and move your hand: the cube follows and the liquid sloshes. The size stays constant (1x). |
| Two hands | Spread them apart to zoom gently up to 2x. Bring them together to return to 1x. |
| No hand | The cube tumbles on its own. |
| No camera: mouse drag | Rotate the cube. |
| No camera: mouse wheel | Zoom (1x to 2x). |
| `R` | Reset the liquid. |

## Requirements

- Node.js 18 or newer (only used to serve the files on localhost; tested on Node 22)
- A modern browser with WebGL2 (Chrome, Edge, Firefox, Safari 15+)
- A webcam
- An internet connection on first load (the MediaPipe library and hand model load from a CDN)
- Camera access only works on `localhost` or HTTPS

## Run

```
npm install
npm run dev
```

Open the URL it prints (usually http://localhost:5173) and click **Allow camera access**. If port 5173 is busy, the server uses the next free port automatically.

`npm install` has nothing to download, because the project has no dependencies. There is also no build step: the project is static files, so you can deploy the folder as it is to any static host (GitHub Pages, Netlify, Vercel). Camera access needs HTTPS there.

## Fork and contribute

1. Click **Fork** at the top right of the repository page on GitHub.
2. Clone your fork:
   ```
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
3. Create a branch:
   ```
   git checkout -b my-change
   ```
4. Run it with `npm run dev`, make your changes, and test them in the browser.
5. Commit and push:
   ```
   git add .
   git commit -m "describe your change"
   git push origin my-change
   ```
6. Open a **Pull Request** from your branch on GitHub.

## Files

| File | What it does |
| --- | --- |
| `index.html` | Page layout: webcam video, WebGL canvas, cube overlay and panels |
| `style.css` | Styling |
| `app.js` | Hand tracking, cube control, GPU liquid rendering, side panels |
| `sim.js` | The 3D particle fluid physics |
| `worker.js` | Runs the physics in a Web Worker so rendering stays smooth |
| `server.js` | Tiny static server for localhost |

## Tuning

- Particle count: the number in `worker.js` (`new Fluid(3600)`). Lower it on slow machines. It also reduces itself automatically if the physics cannot keep up.
- Zoom range and cube size: `baseSc` and the zoom mapping in `app.js`.
- Liquid colours: the `RES_FS` shader in `app.js`.

## Credits

Made by **tubakhxn**. Hand tracking by [MediaPipe](https://ai.google.dev/edge/mediapipe) (Google).

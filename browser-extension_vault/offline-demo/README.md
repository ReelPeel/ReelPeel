# Offline ReelPeel conference demo

This folder contains a self-contained two-Reel feed and a local JSON endpoint. No network connection is needed once the files are present.

## Start

Activate the Conda environment you use for Python, then start the dependency-free server from the extension root:

```powershell
python offline-demo/server.py
```

Open [http://127.0.0.1:8765/reels/DT0UIgzDZ79/](http://127.0.0.1:8765/reels/DT0UIgzDZ79/) in Chrome. Scroll, swipe, or use Up/Down to switch Reels.

If Windows blocks port `8765`, the server automatically tries `8766` through `8775` and prints the exact URL to open. You can also choose a port explicitly:

```powershell
$env:REELPEEL_PORT=8766
python offline-demo/server.py
```

## Extension

Load this repository root as an unpacked Chrome extension (or press Reload for an existing unpacked installation). The extension now injects on the local demo URL. For these URLs, it calls:

```text
POST http://127.0.0.1:8765/json
{ "url": "http://127.0.0.1:8765/reels/<reel-id>/", "mock": false }
```

`server.py` reads the reel-specific process and evidence-summary fixtures from the sibling `offline_mock/` directory. `/json` waits five seconds; `/evidence_summary` waits three seconds. Both responses stay fully local.

## Contents

- `assets/reel-*.mp4`: final local H.264/AAC videos used by the demo.
- `assets/reel-*.video.mp4` and `assets/reel-*.audio.mp4`: original downloaded DASH streams retained for provenance.
- `assets/reelpeel26-profile.jpg`: local profile image used in the compact navigation and Reel controls.
- `../offline_mock/<reel-id>/`: reel-specific process response, manifest, and recorded evidence summaries.
- `pubmed/<PMID>/index.html`: locally archived PubMed record pages. The overlay maps PubMed evidence links to these pages when running on localhost.

The source videos and endpoint bind only to `127.0.0.1`, so no other device on the network can access the demo server.

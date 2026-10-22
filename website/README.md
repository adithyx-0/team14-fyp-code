# Meme triage website

Upload a Malayalam political meme, get intent (troll/support), target
(person/party/intersection), class probabilities, Grad-CAM heatmaps and a
"needs human review" flag. Uses Arm 1 (Swin-Tiny + MuRIL) live on CPU.

    cd code
    .venv/bin/python -m website.server      # http://127.0.0.1:8000

Env: `WEBSITE_DEVICE` (cpu|mps|cuda), `PORT`, `HOST` (0.0.0.0 to share on your LAN).

- `POST /api/analyze` (multipart: `file`, optional `caption`) returns JSON.
- Needs the Tesseract binary with the `mal` language for OCR (installed here);
  without it the text branch gets an empty string and the page says so.
- Limits: 8 MB, one inference at a time. Uploads are never written to disk.
- Review threshold is `REVIEW_THRESHOLD` in `server.py` (0.60). It is a
  heuristic, not a calibrated error rate.
- Not production-hardened: no auth, rate limiting or HTTPS. Put it behind a
  proper host/reverse proxy before exposing it to the internet.

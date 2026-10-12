# BanglishCheck — standalone web app

A self-contained FastAPI app (plain HTML/CSS/JS frontend, no Gradio, no Hugging Face Spaces
dependency), so it can be deployed behind your own domain instead of a Space.

## What's in this folder

| Path | What it is |
|---|---|
| `main.py` | FastAPI backend: `/api/analyze`, `/api/examples`, `/api/health`, and serves `static/` |
| `banglishcheck.py` | Shared module (copied from the project root) — features, model, explainer |
| `static/` | Frontend: `index.html`, `style.css`, `app.js` |
| `explainer.joblib` | **Not included** — copy it here from the Colab notebook's `outputs/space/` folder |
| `Dockerfile` | Builds a container that runs the app with `uvicorn` on port 8000 |
| `docker-compose.yml` | Runs the container with the model mounted from the host, so swapping `explainer.joblib` doesn't need a rebuild |
| `Procfile` | For platforms that read one (Render, Railway, Heroku-style) — binds to `$PORT` |
| `deploy/Caddyfile`, `deploy/nginx.conf` | Ready-to-copy reverse-proxy configs for a VPS (pick one) |
| `deploy/banglishcheck.service` | systemd unit for running without Docker |
| `deploy/deploy.sh` | One-liner to build + (re)start via Docker Compose on the server |
| `requirements.txt` | Python dependencies (no `gradio`/`matplotlib` — the chart is drawn in CSS) |

## Before you run it

Put `explainer.joblib` (produced by `notebooks/BanglishCheck_Colab.ipynb`) directly in this
folder, next to `main.py`. Without it, `/api/analyze` returns a 503 and `/api/health` reports
`model_loaded: false`.

## Run it locally

```bash
cd "Technical Project/webapp"
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# open http://localhost:8000
```

## Run it with Docker

```bash
cd "Technical Project/webapp"
docker build -t banglishcheck .
docker run -p 8000:8000 banglishcheck
```

## Deploying to your own domain

**A. VPS (DigitalOcean, Hetzner, a cheap EC2 box, etc.) + your own domain — recommended**

1. Point an A record for your domain (or a subdomain, e.g. `check.yourdomain.com`) at the
   server's IP address.
2. Get this `webapp/` folder onto the server (git clone or scp), with `explainer.joblib` in it.
3. Run `bash deploy/deploy.sh` — builds the image and starts it via `docker-compose.yml`,
   bound to `127.0.0.1:8000` on the host.
4. Put a reverse proxy in front to terminate TLS: copy `deploy/Caddyfile` to `/etc/caddy/Caddyfile`
   and `systemctl reload caddy` (TLS is automatic), **or** copy `deploy/nginx.conf` into
   `/etc/nginx/sites-available/`, enable it, and run `certbot --nginx -d check.yourdomain.com`.
   Edit the domain name in whichever config file you use first.
5. No Docker? Use `deploy/banglishcheck.service` instead of steps 2–3 (systemd + a venv) —
   see the comments at the top of that file.

**B. A platform that builds from this Dockerfile (Render, Fly.io, Railway, etc.)**

1. Push this repo (or just this `webapp/` folder) to GitHub.
2. Create a new service on the platform, pointing it at this folder. Most of these platforms
   detect `Dockerfile` or `Procfile` automatically.
3. `explainer.joblib` isn't committed to git (it can be large) — either commit it anyway if
   your platform/plan allows it, or add a build step/volume that fetches it from where you
   store it (e.g. a private release asset or object storage).
4. Add your domain in the platform's custom-domain settings and update your DNS (usually a
   CNAME) as it instructs. TLS is normally issued automatically.

## Notes

- `explainer.joblib` is loaded once at startup; after replacing it, restart the process/container.
- The warning for texts under 8 words and the "never use this to accuse anyone" notice are
  server-side (in `main.py`), so they can't be bypassed by editing the frontend.

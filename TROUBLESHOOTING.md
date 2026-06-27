# Troubleshooting Guide

Common issues encountered while building and deploying RAG 2.0, with root causes and fixes. Organized by area.

---

## CI/CD (GitHub Actions)

### `ModuleNotFoundError: No module named 'rank_bm25'` (or any other missing package)

**Symptom:**
```
ImportError while loading conftest '.../backend/tests/conftest.py'.
...
ModuleNotFoundError: No module named 'rank_bm25'
```

**Cause:** A package is imported in code but missing from `backend/requirements.txt`. This typically happens when a dependency was installed locally (so the code "just worked" on your machine) but never added to the requirements file, so CI's `pip install -r requirements.txt` never installs it.

**Fix:** Add the missing package to `backend/requirements.txt` with a pinned version, e.g.:
```
rank_bm25==0.2.2
PyMuPDF==1.23.8
```
Commit and push. Note: the import name and the PyPI package name don't always match — `PyMuPDF` installs under that name but is imported as `fitz`.

**How to avoid this going forward:** Periodically run `pip freeze > requirements.txt` in a clean virtual environment to catch anything installed-but-undeclared, or use `pip-compile` from `pip-tools` to manage requirements deliberately.

---

### `pymongo.errors.OperationFailure: Authentication failed., code: 18`

**Symptom:** Every test that touches the database fails in CI at fixture setup, with `AuthenticationFailed`, even though the same tests pass locally.

**Cause:** This is almost always a **credential mismatch**, not a timing/race condition (it's tempting to assume the database container just isn't "ready" yet — check that second, not first). Compare:
- The password your test code expects (e.g. hardcoded in `conftest.py`'s `TEST_MONGO_URL`)
- The password the CI workflow's `services.mongodb.env.MONGO_INITDB_ROOT_PASSWORD` actually sets

If these two values differ even slightly, every authentication attempt fails — predictably, every single run, not intermittently.

**Why it works locally but not in CI:** Locally, your MongoDB instance was likely set up once, a while ago, with credentials matching what your test fixture expects. It just keeps running between test runs, so there's no setup step to get wrong. In CI, MongoDB is provisioned **completely fresh, every single run**, using whatever password the workflow YAML specifies — if that doesn't match your test code's hardcoded password, it's not a "the container isn't ready yet" problem, it's a flat "those two strings don't match" problem.

**Fix:** Pick one password and use it in both places — the workflow's `MONGO_INITDB_ROOT_PASSWORD` (and the matching `--health-cmd` line, and the `MONGODB_URL` env var passed to the test step) and `conftest.py`'s `TEST_MONGO_URL`. Since this is a throwaway CI-only database with no real data, the actual password value doesn't matter — it just needs to be consistent everywhere it's referenced.

---

### `ERROR: failed to build: invalid tag "...": repository name must be lowercase`

**Symptom:** `docker/build-push-action` fails immediately with a tag-validation error during `build-images`.

**Cause:** Docker registry image names must be all-lowercase. `${{ github.repository_owner }}` and `${{ github.actor }}` both resolve to your **literal GitHub username**, capital letters included (e.g. `Virajbane`). Docker rejects any tag containing uppercase characters.

**Fix:** Lowercase the registry path before using it in any `tags:` field. Add a step before the build step:
```yaml
- name: Set lowercase image repo
  id: vars
  run: echo "repo=$(echo '${{ env.DOCKER_REGISTRY }}' | tr '[:upper:]' '[:lower:]')" >> "$GITHUB_OUTPUT"
```
Then reference `${{ steps.vars.outputs.repo }}` instead of `${{ env.DOCKER_REGISTRY }}` in every `tags:` line. Remember to apply the same lowercase fix anywhere else the registry path is used later in the pipeline (e.g. inside the `deploy` job's SSH script, if it also exports `DOCKER_REGISTRY`).

---

### `You are using Node.js 18.20.8. For Next.js, Node.js version ">=20.9.0" is required.`

**Symptom:** The frontend Docker image build fails at `npm run build`, even though the `test-frontend` CI job (which lints and builds the same code) passes fine.

**Cause:** Your CI test job and your Dockerfile are **two completely separate definitions of "what Node version to use."** `actions/setup-node@v4` in the workflow might correctly specify Node 20, but `frontend/Dockerfile` independently hardcodes `FROM node:18-alpine`. Bumping one does not automatically bump the other.

**Fix:** Update **both** the builder and production stages in `frontend/Dockerfile`:
```dockerfile
FROM node:20-alpine AS builder
...
FROM node:20-alpine
```
Check `package.json`'s `engines` field (or just the installed `next` version) to know the actual minimum Node version required, and keep the Dockerfile in sync with it.

---

### `Error: missing server host` (in `appleboy/ssh-action`)

**Symptom:** The `deploy` job fails immediately at the SSH step with this exact message.

**Cause:** `secrets.PROD_HOST` (and likely `PROD_USER`, `PROD_SSH_KEY` too) haven't been configured yet in the repo's GitHub Secrets. There's no real server for the action to connect to.

**Fix:** This is expected behavior until a real production server exists. Once you provision one (EC2/Azure VM/DigitalOcean Droplet), add these secrets under **Repo Settings → Secrets and variables → Actions**:
- `PROD_HOST` — the server's public IP or domain
- `PROD_USER` — the SSH login user (e.g. `ubuntu`)
- `PROD_SSH_KEY` — the private half of an SSH key pair whose public half is installed on the server
- `API_URL` — the public URL the frontend should call (used as a build arg)

---

### CI passes a fix locally but the same error reappears in the next run

**Symptom:** You fix something, commit, push — and the next CI log shows the *exact same* error, word for word.

**Cause:** Usually one of:
1. The fix was edited on disk but never actually `git add`ed/committed.
2. It was committed but pushed to the wrong branch.
3. You're looking at a stale/cached log from a previous run rather than the run tied to your latest commit.

**How to verify (don't guess — check):**
```powershell
git log -1 --oneline -- path/to/changed/file.yml   # is it committed?
git log origin/main -1 --oneline                    # does the remote have it?
git status                                          # anything still unstaged?
```
If the commit hash matches between local and `origin/main`, and the file content is confirmed via `Select-String`/`grep`, then the fix genuinely shipped — re-check the Actions tab for a run specifically tied to that newer commit hash, not an older one.

---

## Local Development

### Backend tests pass locally but fail in CI

**Cause:** Almost always an environment difference, not a code bug. The two most common culprits in this project were: (1) a package installed locally but not declared in `requirements.txt`, and (2) database credentials that matched a long-running local database but not the fresh-every-run CI database. See the CI/CD section above for both.

### `lucide-react` version looks wrong in `package.json`

**Symptom:** `package.json` shows something like `"lucide-react": "^1.21.0"`, which is suspiciously low/high depending on when you check — lucide-react's actual versioning has moved well past 1.x in some release lines, so a `^1.21.0` pin combined with other 2026-era dependencies can cause unexpected resolution behavior.

**Fix:** Check the currently published version on npm and pin explicitly rather than relying on an old caret range pulled in by a scaffolding tool.

---

## Docker / Deployment

### MongoDB / Redis / Qdrant container "Up" but app can't connect

**Checklist:**
1. Confirm the connection URL's host matches the **service name** in `docker-compose.yml` (e.g. `mongodb://...@mongodb:27017`, not `localhost:27017` — containers resolve each other by service name, not `localhost`, except when running outside Docker entirely).
2. Confirm `depends_on: condition: service_healthy` is set, not just `depends_on: [mongodb]` — the latter only waits for the container to *start*, not for it to actually be ready to accept connections.
3. Check the container's healthcheck is actually passing: `docker compose ps` should show `(healthy)`, not just `Up`.

### Image push fails with permission errors against `ghcr.io`

**Checklist:**
1. Confirm the `build-images` job has `permissions: packages: write` set at the job level (not just repo-level default permissions).
2. Confirm `docker/login-action` is using `password: ${{ secrets.GITHUB_TOKEN }}` — this token is auto-provided by GitHub Actions and usually doesn't need to be manually created as a separate secret.

---

## General Debugging Approach Used in This Project

When a CI failure repeats identically after a supposed fix:
1. **Verify the fix is actually committed and pushed** before assuming the fix itself is wrong (see "CI passes a fix locally" above).
2. **Read the traceback's deepest/most specific line first** — e.g. `AuthenticationFailed` is a precise signal pointing at credentials, not a vague "something's wrong with the database."
3. **Don't assume a race condition before ruling out a simple mismatch.** A consistent, 100%-reproducible failure (same error every single run) is more likely a static configuration mismatch than a timing race; intermittent failures are the actual signature of a race condition.
4. **Fix one error, push, re-read the next log fully** — CI failures often queue up; fixing the first blocker reveals the next one further down the pipeline, which is forward progress, not a new problem.
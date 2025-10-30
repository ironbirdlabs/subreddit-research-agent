# Agent Template

[![GitHub stars](https://img.shields.io/github/stars/agentsystems/agentsystems?style=flat-square&logo=github)](https://github.com/agentsystems/agentsystems/stargazers)

> [!NOTE]
> **Pre-Release Software** - AgentSystems is in active development. Join our [Discord](https://discord.com/invite/JsxDxQ5zfV) for updates and early access.
> ⭐ [**Star the main repository**](https://github.com/agentsystems/agentsystems) to show your support!

> This is the **reference agent template** for AgentSystems. See the [main repository](https://github.com/agentsystems/agentsystems) for platform overview and documentation.

A minimal, batteries-included starter for building containerised AI agents that plug into the [Agent Systems](https://github.com/agentsystems) platform.

*   Built on FastAPI + LangChain
*   Comes with a Dockerfile and a `build_and_release.sh` wrapper
*   No version tags or Docker image are published here – **this repo is a template**, not a distributable agent


The **Agent Template** is a minimal, batteries-included starter repo for building container-ised AI agents that plug into the [Agent Control Plane](https://github.com/agentsystems/agent-control-plane).

This repo is intended to be used via GitHub’s **“Use this template”** button or the `gh repo create` CLI. It can also be cloned directly for experiments.

---

## What you get

| Path / file | Purpose |
|-------------|---------|
| `main.py` | FastAPI app exposing `/invoke`, `/health`, `/metadata`. Contains an `invoke()` function you can customise. |
| `agent.yaml` | Agent identity (name, description, container image, tags). |
| `metadata.yaml` | Version-specific attributes (version, model, input schema, facets). |
| `Dockerfile` | Slim Python 3.12 image that installs dependencies and runs the agent. |
| `requirements.txt` | Runtime dependencies. |
| Langfuse callback | `langfuse.langchain.CallbackHandler` pre-wired so every LangChain call is traced. |

---

## Where it fits

```mermaid
graph LR
  client((Client)) --> gateway[Gateway]
  gateway --> agent((Your-Agent))
  agent --> lf[Langfuse]
```

1. Client calls `POST /your-agent` on the Gateway.
2. Gateway forwards to your container’s `/invoke` endpoint and injects `X-Thread-Id`.
3. Your code adds Langfuse traces and responds with JSON.

---

## Quick start

### Run via Docker Compose (spec-compliant)

```bash
docker compose -f compose/docker-standard.yml up --build
```

After a few seconds check <http://localhost:8000/docs> for the swagger UI.

---

### Run with hot-reload (local python)

1. Click **"Use this template"** on GitHub and create a new repository (e.g. `johndoe/echo-agent`).
2. Clone your new repo and customize:
   - `agent.yaml` - Set your agent's identity (`name`, `description`, `container_image`, etc.)
   - `metadata.yaml` - Set version-specific attributes (`version`, `model_dependencies`, etc.)
3. Start the agent locally with hot-reload:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open <http://localhost:8000/docs> to test the `/invoke` endpoint.

---
## Container health check

AgentSystems CLI waits until Docker marks your container `healthy` before routing traffic. Add a simple `HEALTHCHECK` to your `Dockerfile` so the platform knows when the agent is ready:

```dockerfile
# after EXPOSE 8000
ENV PORT 8000
HEALTHCHECK --interval=10s --retries=3 CMD curl -sf http://localhost:${PORT}/health || exit 1
```

The template exposes a `GET /health` endpoint that returns 200, so the example healthcheck will work with the default app.

---
## Build & release a Docker image

Use the wrapper script to build (and optionally push) a versioned multi-arch image:

```bash
./build_and_release.sh \
  --image johndoe/echo-agent \
  --version 0.1.0 \
  --push          # omit --push to build locally only
```

What it does:

* Builds the container image and tags it `0.1.0` (plus `latest` if no suffix)
* Pushes to Docker Hub when `--push` is present
* Creates a Git tag **only** when you also pass `--git-tag`

---

## Wire into a deployment

Add the service to [`agent-platform-deployments`](https://github.com/agentsystems/agent-platform-deployments):

```yaml
# compose/local/docker-compose.yml
  echo-agent:
    image: mycorp/echo-agent:0.1
    networks:
      - agents-int
    labels:
      - agent.enabled=true
      - agent.port=8000
```

The Gateway should now route `POST /echo-agent` to your container (once the container is healthy and registered).

---

## Environment variables

| Var | Purpose |
|-----|---------|
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Needed for Langfuse tracing. |
| Any model API keys | e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` – accessed in `invoke()`. |

---

## File Uploads & Artifacts

Agents can receive file uploads and access shared artifacts through the `/artifacts` volume mounted at runtime. The platform uses a thread-centric structure where each request gets its own directory.

### File Upload Workflow

Upload files using multipart requests to the gateway:

```bash
# Upload file with JSON payload
curl -X POST http://localhost:18080/invoke/agent-template \
  -H "Authorization: Bearer your-token" \
  -F "file=@input.txt" \
  -F 'json={"sync": true}'
```

### Artifacts Directory Structure

```
/artifacts/
├── {thread-id-1}/
│   ├── in/          # Input files (uploaded by client)
│   │   └── input.txt
│   └── out/         # Output files (created by agent)
│       └── result.txt
└── {thread-id-2}/
    ├── in/
    └── out/
```

### Reading Input Files

```python
# In your agent's invoke() function
thread_id = request.headers.get("X-Thread-Id", "")
in_dir = pathlib.Path("/artifacts") / thread_id / "in"

# Check for uploaded files
if (in_dir / "data.txt").exists():
    content = (in_dir / "data.txt").read_text()
```

### Writing Output Files

```python
# Create output directory and write results
out_dir = pathlib.Path("/artifacts") / thread_id / "out"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "result.txt").write_text("Processing complete")
```

### Accessing Artifacts Outside Agents

Check artifacts from any container with the volume mounted:

```bash
# List all threads
docker exec local-gateway-1 ls -la /artifacts/

# Read specific output file
docker exec local-gateway-1 cat /artifacts/{thread-id}/out/result.txt

# Use CLI helper
agentsystems artifacts-path {thread-id} result.txt
```

---

## Tips & conventions

* Keep the container port consistent (8080 or 8000); the Gateway connects over the internal Docker network, so host port mapping is optional.
* You should return JSON with the `thread_id` you received – this keeps the audit log and Langfuse trace in sync.
* Use the [Add a New Agent guide](../docs/guides/add-agent) when integrating into the full stack.

---

## Release checklist

1. Update `version` label (if you tag images).
2. `docker build` & push to registry.
3. Update the image tag in the deployment manifests.
4. Run `make restart` (compose) or `helm upgrade` (k8s) to pick up the change.

---

## Contributing

Issues and PRs are welcome – feel free to open a discussion if you need changes to the template.


## Getting Started (local)


1. Clone this repo (or let `agenctl init` do it for you).
2. Edit agent identity and metadata:
   - `agent.yaml` - Set `name`, `description`, `container_image`, etc.
   - `metadata.yaml` - Set `version`, `model_dependencies`, `input_schema`, etc.
3. Extend the Pydantic request/response models in `main.py` and replace the logic in `invoke()`.

   **Request contract**
   - Client must include `Authorization: Bearer <token>` header (any placeholder for now).
   - Gateway injects `X-Thread-Id: <uuid>` header before forwarding to the agent.

   **Response contract**
   - JSON must include the same `thread_id` so audit logs can correlate request/response pairs.

   Example curl (once the agent is behind the gateway):
   ```bash
   curl -X POST localhost:18080/my-agent \
        -H 'Content-Type: application/json' \
        -d '{"prompt": "Hello"}'
   ```
   Response:
   ```json
   {
     "thread_id": "550e8400-e29b-41d4-a716-446655440000",
     "reply": "Echo: Hello",
     "timestamp": "2025-06-16T09:34:00Z"
   }
   ```
4. Build & run locally:

```bash
docker build -t my-agent .
docker run -p 8000:8000 my-agent
```

5. Test:

```bash
curl -X POST localhost:8000/invoke -H 'Content-Type: application/json' \
     -d '{"prompt": "Hello"}'
```



## Using in deployments

In production you usually build & push the image, then reference it in the deployment bundle stored in [`agent-platform-deployments`](https://github.com/agentsystems/agent-platform-deployments).

```
# example snippet in compose/local/docker-compose.yml
my-agent:
  image: mycorp/my-agent:1.0
  labels:
    - agent.enabled=true
    - agent.port=8000
```

The Gateway will auto-discover the container and route `POST /my-agent` to its `/invoke` endpoint.

---

## Continuous Integration (GitHub Actions)

The project ships with a `ci.yml` workflow that now goes beyond linting:

1. Runs pre-commit hooks (ruff, black, shellcheck, hadolint).
2. Builds the agent Docker image.
3. Starts the container mapped to `localhost:9800` (internal port 8000).
4. Polls `http://localhost:9800/health` for up to 60 s and fails the job if the endpoint never returns **200 OK**.
5. Removes the container in a cleanup step.

This helps verify that PRs produce images that boot successfully and expose the health endpoint.

## License

Licensed under the [Apache-2.0 license](./LICENSE).

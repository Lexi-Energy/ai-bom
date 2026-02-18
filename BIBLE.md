# Trusera Open Source Platform — Internal Bible

> Single source of truth for the ai-bom monorepo, all SDKs, CI/CD, publishing, and conventions.
> Keep this file updated when things change. Last updated: 2026-02-18.

---

## Repository

- **Repo**: `github.com/Trusera/ai-bom`
- **License**: Apache-2.0 (root + SDKs), MIT (n8n-node only)
- **Branch protection**: main requires 3 status checks (bypassed by admin PAT)

---

## Monorepo Structure

```
ai-bom/
├── src/ai_bom/              # Core scanner — PyPI: ai-bom
│   ├── scanners/             # 14 scanners
│   ├── reporters/            # 9 output formats
│   ├── compliance/           # OWASP + EU AI Act
│   ├── callable/             # Red-teaming bridge (Giskard)
│   ├── dashboard/            # FastAPI web dashboard
│   ├── cli.py                # Typer CLI entrypoint
│   ├── policy.py             # CI/CD policy enforcement
│   ├── server.py             # FastAPI REST server
│   ├── watcher.py            # Watchdog file watcher
│   └── __init__.py           # __version__ lives here
├── trusera-agent-sdk/        # Python agent SDK — PyPI: trusera-sdk
├── trusera-sdk-js/           # TypeScript agent SDK — npm: trusera-sdk
├── trusera-sdk-go/           # Go agent SDK — go get
├── n8n-node/                 # n8n community node — npm: n8n-nodes-trusera
├── vscode-extension/         # VS Code extension (VSIX)
├── templates/                # GitLab CI template
├── scripts/                  # cedar-gate.py
├── examples/                 # Example scan targets
├── tests/                    # 651+ tests for ai-bom core
├── docs/                     # MkDocs source
├── dist/                     # GitHub Action dist (action.yml)
├── pyproject.toml            # ai-bom package config
├── Makefile                  # Dev commands
├── Dockerfile                # Docker distribution
└── .github/workflows/        # CI/CD (see below)
```

---

## Package Registry Map

| Package | Directory | Registry | Current Version | Install |
|---------|-----------|----------|----------------|---------|
| **ai-bom** | `src/ai_bom/` | PyPI | 3.3.2 | `pip install ai-bom` |
| **trusera-sdk** (Python) | `trusera-agent-sdk/` | PyPI | 0.2.2 | `pip install trusera-sdk` |
| **trusera-sdk** (JS) | `trusera-sdk-js/` | npm | 0.1.0 | `npm install trusera-sdk` |
| **trusera-sdk-go** | `trusera-sdk-go/` | Go modules | 0.1.0 | `go get github.com/Trusera/ai-bom/trusera-sdk-go` |
| **n8n-nodes-trusera** | `n8n-node/` | npm | 0.5.0 | Install via n8n community nodes |
| **ai-bom-scanner** (VS Code) | `vscode-extension/` | VS Code Marketplace | 0.1.0 | Search "ai-bom" in Extensions |

---

## Version Files (what to bump)

| Package | Version source(s) |
|---------|------------------|
| ai-bom | `src/ai_bom/__init__.py` → `__version__ = "X.Y.Z"` (hatch reads this) |
| trusera-sdk (Python) | `trusera-agent-sdk/trusera_sdk/__init__.py` + `trusera-agent-sdk/pyproject.toml` |
| trusera-sdk (JS) | `trusera-sdk-js/package.json` |
| trusera-sdk-go | No version file — version comes from git tag only |
| n8n-nodes-trusera | `n8n-node/package.json` |
| VS Code extension | `vscode-extension/package.json` |

---

## Tag Namespaces

Each package has its own tag prefix. The auto-release workflow uses these to track versions independently.

| Package | Tag format | Example |
|---------|-----------|---------|
| ai-bom | `v*` | `v3.3.2` |
| trusera-sdk (Python) | `sdk-v*` | `sdk-v0.2.2` |
| trusera-sdk (JS) | `js-v*` | `js-v0.1.0` |
| trusera-sdk-go | `trusera-sdk-go/v*` | `trusera-sdk-go/v0.1.0` |
| n8n-nodes-trusera | `n8n-v*` | `n8n-v0.5.0` |
| VS Code extension | `vscode-v*` | `vscode-v0.1.0` |

---

## Auto-Release (`.github/workflows/auto-release.yml`)

Runs on every push to `main`. Fully automated — no manual tagging needed.

### How it works

1. **detect-changes** job compares each directory against its latest tag
2. Only packages with file changes get a release job
3. Each job analyzes **conventional commits** touching that directory:
   - `feat:` → **minor** bump (0.1.0 → 0.2.0)
   - `fix:` → **patch** bump (0.1.0 → 0.1.1)
   - `BREAKING CHANGE` or `feat!:` → **major** bump (0.1.0 → 1.0.0)
4. Skipped scopes (don't trigger releases): `feat(ci)`, `feat(deps)`, `feat(build)`, `feat(chore)` and same for `fix(...)`
5. Commits with `chore(release):` prefix are ignored (prevents re-trigger loop)
6. Version files are updated, committed as `chore(release): <pkg> vX.Y.Z [skip ci]`, tagged, pushed
7. Package is built and published to the appropriate registry

### Publishing methods

| Package | Method | Auth |
|---------|--------|------|
| ai-bom | PyPI trusted publishing (OIDC) | No token needed (environment: `pypi`) |
| trusera-sdk (Python) | PyPI API token | `PYPI_API_TOKEN` secret |
| trusera-sdk (JS) | npm publish | `NPM` secret |
| trusera-sdk-go | Git tag only | `GIT_TOKEN` secret (for push) |
| n8n-nodes-trusera | npm publish | `NPM` secret |

### To add trusted publishing for more packages

Go to `https://pypi.org/manage/project/<name>/settings/publishing/` and add:
- Owner: `Trusera`, Repository: `ai-bom`, Workflow: `auto-release.yml`, Environment: `pypi`

Then remove the `password: ${{ secrets.PYPI_API_TOKEN }}` line for that package.

---

## GitHub Actions Workflows

| Workflow | File | Trigger | Purpose |
|----------|------|---------|---------|
| **CI** | `ci.yml` | push/PR to main | Lint, typecheck, test (Python 3.10-3.13), security audit, SARIF upload |
| **Auto Release** | `auto-release.yml` | push to main | Auto-version + publish all 5 packages |
| **Publish to PyPI** | `publish.yml` | `v*` tags (manual) | Standalone ai-bom PyPI publish (backup) |
| **Publish Agent SDK** | `publish-sdk.yml` | `sdk-v*` tags (manual) | Standalone agent SDK publish (backup) |
| **Publish npm** | `publish-npm.yml` | `n8n-v*` tags (manual) | Standalone n8n node publish (backup) |
| **Publish VS Code** | `publish-vscode.yml` | `vscode-v*` tags | VS Code marketplace publish |
| **Docker** | `docker.yml` | `v*` tags | Build + push Docker image |
| **Docs** | `docs.yml` | push to main | Deploy MkDocs to GitHub Pages |
| **Dogfood** | `dogfood.yml` | push to main | ai-bom scans itself |
| **AI-BOM Example** | `ai-bom-example.yml` | push to main | Example scan via GitHub Action |

---

## GitHub Secrets

| Secret | Purpose | Used by |
|--------|---------|---------|
| `GIT_TOKEN` | PAT with repo + workflow scope — bypasses branch protection for auto-release commits | `auto-release.yml` |
| `PYPI_API_TOKEN` | PyPI API token (user: `trusera`) — publishes trusera-sdk | `auto-release.yml`, `publish-sdk.yml` |
| `NPM` | npm access token — publishes trusera-sdk and n8n-nodes-trusera | `auto-release.yml`, `publish-npm.yml` |
| `VSCE_PAT` | VS Code marketplace PAT | `publish-vscode.yml` |

---

## ai-bom Core Scanner

### Scanners (14)

| Scanner | File | What it detects |
|---------|------|-----------------|
| Code | `code_scanner.py` | AI SDK imports, model references, API keys in source |
| Docker | `docker_scanner.py` | AI frameworks in Dockerfiles |
| Network | `network_scanner.py` | AI API endpoints in configs |
| Cloud (Terraform) | `cloud_scanner.py` | AI services in IaC |
| n8n | `n8n_scanner.py` | AI nodes, webhook auth, agent chains |
| Jupyter | `jupyter_scanner.py` | AI usage in notebooks |
| GitHub Actions | `github_actions_scanner.py` | AI integrations in workflows |
| MCP Config | `mcp_config_scanner.py` | MCP server configs (claude, cline, cursor) |
| Model Files | `model_file_scanner.py` | Binary models (.onnx, .pt, .safetensors, etc.) |
| AST (deep) | `ast_scanner.py` | Python AST analysis (imports, decorators, calls) |
| AWS Live | `aws_live_scanner.py` | Bedrock, SageMaker, Comprehend, Kendra |
| GCP Live | `gcp_live_scanner.py` | Vertex AI, Dialogflow |
| Azure Live | `azure_live_scanner.py` | Azure OpenAI, Cognitive Services, ML |
| Callable | `callable/` | Provider-agnostic model probing for red-teaming |

### Reporters (9)

`table`, `json` (CycloneDX), `sarif`, `spdx3`, `html`, `csv`, `junit`, `markdown`, `diff`

### Compliance Modules

| Module | File | Standard |
|--------|------|----------|
| OWASP LLM Top 10 | `compliance/__init__.py` | LLM01-LLM10 mapping |
| OWASP Agentic Top 10 | `compliance/owasp_agentic.py` | Agent-specific risks |
| EU AI Act | `compliance/eu_ai_act.py` | Risk classification |
| License Compliance | `compliance/licenses.py` | License checking |

### CLI Flags

```bash
ai-bom scan <path>              # Basic scan
ai-bom scan <path> --deep       # Enable AST scanner
ai-bom scan <path> -f sarif     # Output format
ai-bom scan <path> --fail-on high  # CI policy gate
ai-bom scan <path> --policy .ai-bom-policy.yml
ai-bom scan <path> --compliance owasp_llm_top10
ai-bom scan <path> --telemetry  # Opt-in usage analytics
ai-bom scan <path> -q           # Quiet mode for CI
ai-bom scan-cloud aws           # Live cloud scanning
ai-bom dashboard                # Launch web dashboard
```

### Optional Dependencies

```bash
pip install ai-bom[dashboard]      # FastAPI dashboard
pip install ai-bom[watch]          # File watcher
pip install ai-bom[aws,gcp,azure]  # Cloud live scanners
pip install ai-bom[callable-all]   # All red-teaming providers
pip install ai-bom[all]            # Everything
```

---

## Agent SDKs

All three SDKs share the same architecture: client, events, decorators/interceptor, Cedar policy, standalone mode.

### Python SDK (`trusera-agent-sdk/`)

```python
from trusera_sdk import TruseraClient, monitor, Event, EventType

# Client with context manager
with TruseraClient(api_key="tsk_...") as client:
    client.register_agent("my-agent", "langchain")
    client.track(Event(type=EventType.TOOL_CALL, name="search", payload={...}))

# Decorator
@monitor(event_type=EventType.LLM_INVOKE)
def call_llm(prompt): ...

# Standalone (no API key needed)
from trusera_sdk import StandaloneInterceptor
with StandaloneInterceptor(policy_file="policy.cedar", enforcement="block"):
    # HTTP requests are intercepted and policy-evaluated
    ...
```

**Modules**: `client.py`, `events.py`, `decorators.py`, `cedar.py`, `standalone.py`
**Integrations**: LangChain (`TruseraCallbackHandler`), CrewAI (`TruseraCrewCallback`), AutoGen (`TruseraAutoGenHook`)
**Build**: `hatchling` (pyproject.toml, version in both `__init__.py` and `pyproject.toml`)

### TypeScript SDK (`trusera-sdk-js/`)

```typescript
import { TruseraClient, EventType } from 'trusera-sdk';

const client = new TruseraClient({ apiKey: 'tsk_...' });
client.track({ type: EventType.TOOL_CALL, name: 'search', payload: {} });
```

**Modules**: `client.ts`, `events.ts`, `cedar.ts`, `interceptor.ts`, `standalone.ts`
**Build**: `tsc` (TypeScript), outputs to `dist/`
**Test**: `vitest`

### Go SDK (`trusera-sdk-go/`)

```go
import trusera "github.com/Trusera/ai-bom/trusera-sdk-go"

client := trusera.NewClient("tsk_...")
client.RegisterAgent("my-agent", "custom")
client.Track(trusera.Event{Type: trusera.ToolCall, Name: "search"})
```

**Modules**: `trusera.go`, `events.go`, `cedar.go`, `interceptor.go`, `standalone.go`
**Build**: standard Go modules (`go.mod` points to `github.com/Trusera/ai-bom/trusera-sdk-go`)
**Test**: `go test ./...`
**Version**: git tag only (`trusera-sdk-go/vX.Y.Z`), no version file

---

## n8n Node (`n8n-node/`)

**npm package**: `n8n-nodes-trusera`

### Nodes (5)

| Node | Purpose |
|------|---------|
| TruseraScan | Run ai-bom scan on workflow JSON |
| TruseraDashboard | Interactive HTML dashboard with remediation cards |
| TruseraWebhook | Webhook trigger with direct HTML serving |
| TruseraPolicy | Policy enforcement gate |
| TruseraReport | Generate scan reports |

**Build**: `tsc` + copies SVG/PNG icons to dist
**Test**: `jest`

---

## VS Code Extension (`vscode-extension/`)

- **Marketplace name**: `ai-bom-scanner`
- **VSIX**: `ai-bom-scanner-0.1.0.vsix` (47KB)
- **Publish**: `vsce publish` (needs `VSCE_PAT`)
- **Trigger**: `vscode-v*` tags via `publish-vscode.yml`

---

## CI/CD Pipeline (what runs on every push)

```
push to main
  ├── CI (lint + typecheck + test on 3.10-3.13 + pip-audit + SARIF)
  ├── Dogfood (ai-bom scans itself)
  ├── AI-BOM Scan (example action)
  ├── CodeQL (security analysis)
  └── Auto Release
       ├── detect-changes (which dirs changed?)
       ├── release-ai-bom (if src/ changed)
       ├── release-agent-sdk (if trusera-agent-sdk/ changed)
       ├── release-sdk-js (if trusera-sdk-js/ changed)
       ├── release-sdk-go (if trusera-sdk-go/ changed)
       └── release-n8n-node (if n8n-node/ changed)
```

---

## GitHub Action (`action.yml`)

The repo itself is a GitHub Action. Users add it to their workflows:

```yaml
- uses: trusera/ai-bom@v1
  with:
    path: .
    format: sarif
    fail-on: high
```

The action's `dist/` directory contains the bundled scanner.

---

## CI Templates

| Template | Location | Purpose |
|----------|----------|---------|
| GitHub Action | `action.yml` + `dist/` | `trusera/ai-bom@v1` |
| GitLab CI | `templates/gitlab-ci-ai-bom.yml` | Include template for .gitlab-ci.yml |

---

## Development

### Prerequisites

- Python >= 3.10
- Node.js >= 18 (for JS SDK and n8n-node)
- Go >= 1.21 (for Go SDK)

### Local setup (ai-bom core)

```bash
pip install -e ".[dev]"
ruff check src/ tests/         # lint
ruff format src/ tests/        # format
mypy src/ai_bom/               # typecheck
pytest -v --cov=ai_bom         # test (must pass 80% coverage)
```

### Local setup (agent SDK)

```bash
cd trusera-agent-sdk
pip install -e ".[dev]"
ruff check .
pytest -v
```

### Local setup (JS SDK)

```bash
cd trusera-sdk-js
npm ci
npm run build
npm test
```

### Local setup (Go SDK)

```bash
cd trusera-sdk-go
go test ./...
```

### Local setup (n8n node)

```bash
cd n8n-node
npm ci
npm run build
npm test
```

---

## Conventional Commits

The auto-release system depends on conventional commit messages. Always use:

```
feat: add new scanner for X          → minor bump
feat(sdk): add retry logic           → minor bump (if touching sdk dir)
fix: handle null response            → patch bump
fix(n8n): correct webhook URL        → patch bump (if touching n8n-node dir)
feat!: redesign API                  → major bump
chore: update deps                   → NO release
docs: update README                  → NO release
style: fix formatting                → NO release
feat(ci): add new workflow           → NO release (ci scope excluded)
feat(deps): bump dependency          → NO release (deps scope excluded)
```

---

## Manual Release (if auto-release is broken)

### ai-bom

```bash
# 1. Bump version
sed -i 's/__version__ = ".*"/__version__ = "X.Y.Z"/' src/ai_bom/__init__.py
git add src/ai_bom/__init__.py
git commit -m "chore(release): ai-bom vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
# publish.yml triggers automatically on v* tag
```

### trusera-sdk (Python)

```bash
# 1. Bump both files
sed -i 's/__version__ = ".*"/__version__ = "X.Y.Z"/' trusera-agent-sdk/trusera_sdk/__init__.py
sed -i 's/^version = ".*"/version = "X.Y.Z"/' trusera-agent-sdk/pyproject.toml
git add trusera-agent-sdk/
git commit -m "chore(release): trusera-sdk vX.Y.Z"
git tag sdk-vX.Y.Z
git push origin main --tags
# publish-sdk.yml triggers on sdk-v* tag
```

### trusera-sdk (JS)

```bash
cd trusera-sdk-js && npm version X.Y.Z --no-git-tag-version && cd ..
git add trusera-sdk-js/package.json
git commit -m "chore(release): trusera-sdk-js vX.Y.Z"
git tag js-vX.Y.Z
git push origin main --tags
# Then manually: cd trusera-sdk-js && npm publish --access public
```

### trusera-sdk-go

```bash
git tag trusera-sdk-go/vX.Y.Z
git push origin trusera-sdk-go/vX.Y.Z
# Go proxy picks it up automatically
```

### n8n-nodes-trusera

```bash
cd n8n-node && npm version X.Y.Z --no-git-tag-version && cd ..
git add n8n-node/package.json
git commit -m "chore(release): n8n-nodes-trusera vX.Y.Z"
git tag n8n-vX.Y.Z
git push origin main --tags
# publish-npm.yml triggers on n8n-v* tag
```

### VS Code extension

```bash
git tag vscode-vX.Y.Z
git push origin vscode-vX.Y.Z
# publish-vscode.yml triggers
```

---

## Troubleshooting

### Auto-release bumps but doesn't publish
The trusted publisher on PyPI must list `auto-release.yml` as the workflow name. Go to PyPI project settings → Publishing → add it.

### Branch protection blocks auto-release push
The `GIT_TOKEN` secret must be a PAT with `repo` scope that has admin/bypass permissions on the repo.

### npm publish fails with 403
Check `NPM` secret is a valid token with publish access to both `trusera-sdk` and `n8n-nodes-trusera`.

### PyPI says "Version already exists"
A tag was created but the publish failed. Delete the orphaned tag, fix the issue, and re-tag.

### Go module not found after tagging
The Go module proxy can take up to 30 minutes to index. Force it: `GOPROXY=direct go get github.com/Trusera/ai-bom/trusera-sdk-go@vX.Y.Z`

### pip-audit fails in CI with "Dependency not found"
Race condition: the CI security job runs pip-audit before the new version appears on PyPI. This is expected for version bump commits and resolves on the next push.

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-02-11 (n8n Community Node)

### Added
- **Actionable Remediation Dashboard**: Per-flag risk descriptions, fix steps, and guardrail recommendations
- **OWASP LLM Top 10 Mapping**: All 14 risk flags mapped to OWASP categories (LLM01-LLM10)
- **Remediation Cards UI**: Modal shows severity-colored cards with description, remediation, guardrail, and OWASP tag
- **Password-Protected Dashboard**: AES-256-GCM encrypted HTML with client-side decryption
- **4 n8n Nodes**: Dashboard, Scanner, Policy, Report — all registered in npm package
- **CSV OWASP Export**: CSV export includes OWASP categories column
- **remediationMap.test.ts**: Validates all risk flags have complete remediation entries

### Changed
- Modal widened from 600px to 720px for remediation card readability
- Workflow names display without `.json` suffix in dashboard
- Package version bumped from 0.1.0 to 0.2.0 for n8n community node
- Added `workflow-security` keyword and `bugs` URL to package.json
- Build script copies SVG/PNG icons to dist automatically
- README updated with mascot image and n8n Community Node section

## [3.0.0] - 2026-02-10

### Added
- **13 Scanners**: code, docker, network, cloud, n8n, jupyter, github-actions, model-files, mcp-config, ast, aws-live, gcp-live, azure-live
- **9 Output Formats**: table, json/cyclonedx, sarif, spdx3, html, csv, junit, markdown
- **Compliance Modules**: OWASP LLM Top 10 and EU AI Act risk mapping with `--compliance` flag
- **OWASP Agentic Security Top 10**: Agent-specific risk assessment for n8n workflows and MCP configs
- **Parallel Scanning**: `run_scanners_parallel()` with configurable thread pool for faster scans
- **MCP Config Scanner**: Detects MCP server configurations in mcp.json, .mcp.json, claude_desktop_config.json, Cline and Cursor configs
- **Model File Scanner**: Detects binary model files (.onnx, .pt, .safetensors, .gguf, .tflite, .mlmodel, .ggml)
- **Live n8n API Integration**: `--n8n-url` and `--n8n-api-key` flags for scanning running n8n instances
- **Live Cloud Scanners**: AWS (Bedrock, SageMaker), GCP (Vertex AI, Dialogflow), Azure (OpenAI, Cognitive Services, ML)
- **AST Scanner**: Deep Python analysis via `--deep` flag (imports, decorators, function calls, string literals)
- **CI/CD Policy Enforcement**: `--fail-on <severity>` and `--policy <file>` for gating builds
- **SPDX 3.0 AI Profile**: EU AI Act compliant output with `--format spdx3`
- **JUnit Reporter**: CI-friendly test report format with `--format junit`
- **CSV Reporter**: Spreadsheet export with `--format csv`
- **Scan Diffing**: Compare two scans to track AI component drift (`diff_reporter`)
- **n8n Security Analysis**: Webhook auth checks, agent-tool risk combos, code injection patterns, agent chain detection
- **Dashboard**: Interactive FastAPI + SQLite web dashboard (`ai-bom dashboard`)
- **A2A Protocol Detection**: Agent-to-Agent protocol pattern matching
- **CrewAI Flow Detection**: @crew, @agent, @task, @flow, @tool decorator scanning
- **Latest Model Patterns**: GPT-4.5, o1/o3, Claude 4/4.5, Gemini 2.0, Llama 4, DeepSeek
- **Deprecated Model Detection**: gpt-4-0314, gpt-4-0613, claude-2.1, claude-3-haiku-20240307
- `--quiet` / `-q` flag for CI-friendly output
- License compliance checking module
- 651 tests with full scanner and reporter coverage

### Changed
- Bumped version from 0.1.0 to 3.0.0 (reflects scope of scanner, reporter, and compliance additions)
- README Quick Start recommends `pipx install ai-bom` (PEP 668 fix)
- Added `requests>=2.28.0` to core dependencies
- Reduced sdist size by excluding assets, demo-video, node_modules

## [2.0.0] - 2026-02-10

### Added
- **Web Dashboard**: Interactive FastAPI + SQLite dashboard for scan history, comparison, and visualization
  - `ai-bom dashboard` command to launch local web server
  - `--save-dashboard` flag to persist scan results
  - REST API: GET/POST/DELETE scans, compare scans side-by-side
  - Dark-themed HTML dashboard with severity charts and drill-down
  - Optional install: `pip install ai-bom[dashboard]`
- **SPDX 3.0 AI Profile Reporter**: EU AI Act compliant output format
  - JSON-LD output with `ai:AIPackage` elements and `ai:safetyRiskAssessment`
  - `--format spdx3` flag
- **Live n8n API Integration**: Scan running n8n instances via REST API
  - `--n8n-url` and `--n8n-api-key` flags replace the previous stub
  - Pagination support, auth error handling
- **CI/CD Policy Enforcement**: Gate builds on AI component violations
  - `--fail-on <severity>` flag — exit code 1 on threshold breach
  - `--policy <file>` flag — YAML policy file with thresholds, blocked providers, and flags
  - Updated GitHub Action with `fail-on` and `policy-file` inputs
- **AST-Based Python Scanning**: Deep analysis via `--deep` flag
  - Detects imports, decorators (@agent, @tool, @crew), function calls to AI APIs
  - String literal model name detection
- **Live Cloud API Scanning**: Discover managed AI services invisible to file scanning
  - `ai-bom scan-cloud aws` — Bedrock, SageMaker, Comprehend, Kendra
  - `ai-bom scan-cloud gcp` — Vertex AI, Dialogflow CX
  - `ai-bom scan-cloud azure` — Azure OpenAI, Cognitive Services, Azure ML
  - Optional install: `pip install ai-bom[aws]`, `ai-bom[gcp]`, `ai-bom[azure]`
- **Enhanced Detection Capabilities**:
  - A2A (Agent-to-Agent) protocol detection
  - CrewAI flow decorator detection (@crew, @agent, @task, @flow, @tool)
  - MCP config file parsing (mcp.json, .mcp.json, claude_desktop_config.json)
  - Latest model patterns: GPT-4.5, o1/o3, Claude 4/4.5, Gemini 2.0, Llama 4, DeepSeek
  - New deprecated models: gpt-4-0314, gpt-4-0613, claude-2.1, claude-3-haiku-20240307
- `--quiet` / `-q` flag for CI-friendly output (suppresses banner and progress)
- Example policy file: `.ai-bom-policy.yml`

### Changed
- README Quick Start now recommends `pipx install ai-bom` (PEP 668 fix)
- Added `requests>=2.28.0` to core dependencies
- Reduced sdist size by excluding assets, demo-video, node_modules
- 304 tests (up from 135)

## [0.1.0] - 2026-02-08

### Added
- 5 scanners: code, docker, network, cloud (Terraform/CloudFormation), n8n workflows
- AI SDK detection for OpenAI, Anthropic, Google, Mistral, Cohere, HuggingFace, and more
- Model version pinning and deprecation checks
- Shadow AI detection (undeclared AI dependencies)
- Hardcoded API key detection
- n8n workflow agent chain analysis with MCP risk assessment
- 5 output formats: table, JSON/CycloneDX, HTML, Markdown, SARIF
- SARIF 2.1.0 output for GitHub Code Scanning integration
- Single-file and directory scanning
- Git repository URL scanning (auto-clone)
- Severity filtering (critical, high, medium, low)
- Risk scoring engine with multi-factor assessment
- GitHub Action for CI/CD integration (`trusera/ai-bom@v1`)
- Docker container distribution
- Comprehensive test suite covering scanners and reporters

[Unreleased]: https://github.com/trusera/ai-bom/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/trusera/ai-bom/compare/v3.0.0...v0.2.0
[3.0.0]: https://github.com/trusera/ai-bom/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/trusera/ai-bom/compare/v0.1.0...v2.0.0
[0.1.0]: https://github.com/trusera/ai-bom/releases/tag/v0.1.0

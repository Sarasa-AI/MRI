# Project State

Last verified: 2026-08-20

## Executive State

- Portfolio objective: maximize outcome per unit of time across RSNA, Phoenix,
  agent capability, and near-term revenue.
- Control-plane repo: `MRI/MRI`, Git `main`, initial commit only, clean at audit
  start.
- Primary execution repos: `SARASA_kaggle` and `phonix`; they are separate Git
  worktrees and are not currently wired into this repo.
- Biggest bottleneck: the RSNA repo has no verified knee dataset manifest,
  DICOM acquisition index, or executable imaging baseline. The central repo is
  also only a control plane, not a model implementation.

## Portfolio Dashboard

| Project | Status | Priority | Current phase | Verified latest result | Blocker | Next owner/action |
|---|---|---:|---|---|---|---|
| RSNA knee | ACTIVE | P0 | Data/competition discovery | No local knee implementation found; `SARASA_kaggle` is a heart-disease tabular repo, not RSNA knee | Competition identity/deadline, access, metadata, DICOM path, compute/storage budget | Human + Data Agent: confirm official competition and acquire metadata only |
| Revenue engine | ACTIVE | P0/P1 | Offer validation | No customer, offer, or paid pilot evidence in audited repos | No ICP, proof asset, outreach list, or pricing test | Human + Business Agent: sell a 7-day Agent Reliability Sprint |
| Agent ecosystem | ACTIVE | P1 | Capability inventory | Local Codex/Claude configuration exists; no governed shared agent architecture | Security boundary, routing policy, eval harness, secret handling | Coding Agent: create least-privilege operating contract |
| Phoenix | ACTIVE | P1/P2 | Training-core reliability | Dataset v1.1.0 certified: 4,011 images, 0 retained duplicate leakage; best recorded full-split QWK 0.64814, accuracy 0.196995; severe 0->1 collapse remains | External datasets missing; authoritative follow-up control run and clinical metrics incomplete | Experiment Agent: run the documented single-arm control replication |
| retinopasy | MAINTENANCE | P2 | Safety refactor | Local worktree has staged safety changes, not part of this audit | Must complete ordered regression evidence before live use | Human approval + Coding Agent |
| PreVisit / other apps | PAUSED | P3 | Separate product work | Existing repos found outside current control plane | Context switching and no current revenue link | Do not allocate weekly focus this cycle |

## Verified Environment

- Python 3.11.7, Kaggle CLI, Docker, GitHub CLI, and `jq` are installed.
- DVC, Git LFS, `uv`, and Conda are absent.
- Available disk is approximately 148 GiB. Do not plan bulk DICOM acquisition
  until a size estimate and selective-download strategy pass review.
- Live Kaggle/GitHub API checks were blocked by DNS in this environment; current
  online competition status and leaderboard are therefore **unverified**.

## Security Findings

The workspace Claude settings contain plaintext Anthropic credentials, are
world-readable (`0644`), and set `defaultMode` to `bypassPermissions`. This is a
P0 security issue. Rotate/revoke the exposed key, remove credentials from files,
use the OS keychain or environment injection, and restore approval-based
execution before enabling autonomous agents. This audit did not modify those
files.

## STOP / CONTINUE / START

### Stop

- Stop treating the empty `MRI` repo as an implemented RSNA system.
- Stop broad tool/repository discovery without a concrete decision or output.
- Stop any bulk DICOM download before manifest, quota, storage, and resume tests.
- Stop calling Phoenix successful based on QWK alone while grade-0 collapse and
  clinical metrics remain unresolved.

### Continue

- Continue RSNA as the P0 opportunity, but make data acquisition the first gate.
- Continue Phoenix from its documented control experiment, preserving frozen
  v1.1.0 artifacts.
- Continue revenue work in parallel; long research-only periods are not allowed.

### Start

- Start one portfolio-level daily status/decision log in this repo.
- Start a selective RSNA metadata probe and a dry-run DICOM acquisition planner.
- Start customer discovery for an Agent Reliability Sprint with a concrete paid
  pilot target.

## Missing Information That Changes Decisions

1. Official RSNA competition slug, rules, deadline, and current leaderboard.
2. Accessible competition metadata/sample files and exact API response shape.
3. RSNA compute allocation (Kaggle/Colab/local) and acceptable storage budget.
4. Phoenix BASELINE-001-R full authoritative metric artifact, including the
   requested clinical metric set and class-wise confusion details.
5. Revenue ICP, existing warm network, geographic constraints, and minimum
   acceptable first contract value.


# Contributing to the Facial Attendance & Motion Tracking System

Thanks for taking the time to contribute! 🎉 This document describes how
to set up your environment, the conventions we follow, and the process
for getting changes merged.

> **Privacy first.** This repo deals with biometric data. **Never commit
> a real face image.** Use only the synthetic fixtures under
> `tests/fixtures/sample_faces/`. See
> [`tests/fixtures/sample_faces/README.md`](tests/fixtures/sample_faces/README.md).

## Table of contents

- [Code of conduct](#code-of-conduct)
- [Getting started](#getting-started)
- [Branch naming](#branch-naming)
- [Commit message conventions](#commit-message-conventions)
- [Code style](#code-style)
- [Testing requirements](#testing-requirements)
- [Pull request process](#pull-request-process)
- [Strategy / toggle impact](#strategy--toggle-impact)
- [Releasing](#releasing)

## Code of conduct

Be kind. Be inclusive. Assume good intent. Disagreements happen — keep
them about the code, not the person.

## Getting started

```bash
git clone https://github.com/your-org/facial-attendance-system.git
cd facial-attendance-system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/cpu.txt -r requirements/dev.txt
pre-commit install
alembic upgrade head
```

See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) for the full guide.

## Branch naming

| Prefix      | Use                                                  | Example                          |
|-------------|------------------------------------------------------|----------------------------------|
| `feat/`     | New feature                                          | `feat/add-yaw-strategy`          |
| `fix/`      | Bug fix                                              | `fix/websocket-cooldown-race`    |
| `docs/`     | Documentation only                                   | `docs/api-reference`             |
| `refactor/` | Refactor with no behaviour change                    | `refactor/extractor-base-class`  |
| `chore/`    | Tooling, deps, CI                                    | `chore/bump-ruff-0.6.9`          |
| `perf/`     | Performance improvement                              | `perf/batch-extractor`           |
| `test/`     | Test-only changes                                    | `test/liveness-edge-cases`       |

Branch off `develop` (or `main` if no `develop`). Keep branches
short-lived — rebase if `develop` moves.

## Commit message conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/).

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

- `type`: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`,
  `ci`, `build`, `style`
- `scope`: `core`, `api`, `db`, `frontend`, `deploy`, `tests`, `docs`,
  `ci`
- Subject: imperative mood, lowercase, no trailing period, ≤ 72 chars

Examples:

```
feat(api): add CSV export endpoint for attendance logs
fix(core): prevent tracker from reusing IDs after cooldown expires
docs(arch): add sequence diagram for detection_only strategy
test(liveness): cover closed-eye edge case
chore(ci): bump actions/checkout to v4
```

**Breaking changes** must include a `BREAKING CHANGE:` footer or a `!`
after the scope (`feat(api)!: ...`).

## Code style

- **Formatter**: `black` (line-length 100).
- **Linter**: `ruff` (`F`, `E`, `I`, `B`, `UP`, `PL` enabled).
- **Type checker**: `mypy --ignore-missing-imports --strict app/` is the
  target; we accept gradual typing.
- **Imports**: sorted by `ruff --select I` (isort-compatible).
- **Docstrings**: Google style on every public class and function.

Run all of the above with `pre-commit run --all-files`.

## Testing requirements

- Every new public function or endpoint **must** have at least one unit
  test under `tests/unit/`.
- Every new endpoint **must** have at least one integration test under
  `tests/integration/`.
- Tests **must not** require a GPU, real model files, or a live Postgres
  instance. Use the shared fixtures in `tests/conftest.py`.
- Coverage of `app/` should not drop below 70% on the PR's diff. Run
  `pytest tests/ --cov=app --cov-report=term-missing` locally.
- If your change is performance-sensitive, add a
  `pytest-benchmark` test under `tests/perf/`.

## Pull request process

1. Open the PR against `develop` (or `main`).
2. Fill in the [PR template](.github/PULL_REQUEST_TEMPLATE.md) —
   especially the **Strategy / toggle impact** section.
3. CI runs `lint`, `test`, `docs`, `security` checks. All must pass.
4. Request review from at least one maintainer.
5. Address review comments inline; push follow-up commits (don't
   force-push during review).
6. Once approved, a maintainer will squash-merge.

PRs that touch the WebSocket protocol or the dashboard toggles require
**two** maintainer approvals.

## Strategy / toggle impact

If your PR touches any of the two toggles or the `strategy` parameter,
**explicitly call this out** in the PR description:

- Which `strategy` modes are affected (`detection_only`,
  `tracking_only`, `both`)?
- Does the WebSocket protocol change? If yes, bump the protocol version
  in `docs/WEBSOCKET_PROTOCOL.md` and call out the migration.
- Does the dashboard toggle behaviour change? Attach a screenshot.

## Releasing

Releases are cut by tagging:

```bash
git tag -a v0.2.0 -m "v0.2.0"
git push origin v0.2.0
```

The `release.yml` workflow will:
- Build Docker images (CPU + GPU) and push to GHCR.
- Create a GitHub Release with the matching `CHANGELOG.md` section.

Update `CHANGELOG.md` *before* tagging — the workflow extracts the body
from there.

---

Happy hacking! 🛠

<!-- Thanks for opening a PR! Please fill in the sections below. -->

## Summary

<!-- Brief description of what this PR changes and why. -->

## Related issues

<!-- "Closes #123", "Refs #456", etc. -->

## Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactor / cleanup
- [ ] CI / build change

## Strategy / toggle impact

If this PR affects the two dashboard toggles or the pipeline `strategy`
parameter, please check all that apply:

- [ ] Affects `strategy="detection_only"` path
- [ ] Affects `strategy="tracking_only"` path
- [ ] Affects `strategy="both"` path
- [ ] Affects the **input source** toggle (upload vs. live camera)
- [ ] Affects the **strategy mode** toggle in the dashboard
- [ ] WebSocket protocol change (see `docs/WEBSOCKET_PROTOCOL.md`)

## Checklist

- [ ] My code follows the style guidelines (ruff + black)
- [ ] I have run `pre-commit install` and `pre-commit run --all-files`
- [ ] I have added tests that prove my fix is effective or my feature works
- [ ] New and existing unit tests pass locally (`pytest tests/unit -v`)
- [ ] I have updated the documentation accordingly (`docs/`, `README.md`)
- [ ] My changes generate no new mypy warnings
- [ ] I have NOT committed any real biometric data (faces) — only synthetic
      test fixtures under `tests/fixtures/sample_faces/`

## Screenshots / recordings

<!-- If the PR changes the dashboard UI, attach a screenshot or GIF. -->

## Performance impact

<!-- If your change has measurable perf impact, paste numbers here. -->

## Notes for reviewers

<!-- Anything reviewers should pay attention to. -->

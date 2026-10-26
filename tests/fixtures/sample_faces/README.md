# Test fixtures — sample faces

This directory holds **synthetic** face-like images used by the unit and
integration tests. **No real photographs of identifiable people are
committed to this repository** — that would be a privacy violation and a
security risk.

## Files

| File                | Format | Size     | Notes                                                        |
|---------------------|--------|----------|--------------------------------------------------------------|
| `face_alpha.png`    | PNG    | 250×250  | Synthetic face-like pattern (eyes, nose, mouth, oval).        |
| `face_beta.png`     | PNG    | 250×250  | Different synthetic pattern (different "identity").          |

These files are generated on-the-fly by the `sample_face_frame` pytest
fixture in `tests/conftest.py`. The committed PNGs were produced from
that same fixture using OpenCV (see commit history).

## Adding real test images (optional, for local testing only)

If you want to exercise the pipeline against real faces during local
development — **never commit these to the repo**:

1. Put the JPEG/PNG files under `tests/fixtures/sample_faces/local/`.
2. Add the following to `.gitignore` (already there):
   ```
   tests/fixtures/sample_faces/local/
   tests/fixtures/sample_faces/*.real.*
   ```
3. Reference them in your tests via:
   ```python
   from tests.conftest import SAMPLE_FACES_DIR
   img_path = SAMPLE_FACES_DIR / "local" / "my_face.jpg"
   ```
4. Make sure you have written consent from the person depicted.

## Regenerating synthetic fixtures

The PNGs are produced by the same logic as the `sample_face_frame`
fixture in `tests/conftest.py`. To regenerate, run a small inline
Python snippet:

```python
import numpy as np, cv2
from tests.conftest import _try_import_cv2  # internal helper
# (see the sample_face_frame fixture for the drawing logic)
```

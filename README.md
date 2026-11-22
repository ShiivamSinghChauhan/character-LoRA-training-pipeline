# character-LoRA-training-pipeline

Tooling for building a single-character LoRA from low-quality video frames: dataset enhancement, bias audit,
a multi-image Klein playground, and Kaggle automation (4 accounts in parallel).

| Folder | What |
|---|---|
| `playground/` | Web UI + queue for FLUX.2 Klein 9B (+ consistency LoRA) with 1-4 reference images. Runs on a Kaggle T4x2 behind a tunnel. See `playground/launch.py` |
| `jobs/` | Headless Kaggle jobs (ComfyUI graphs): batch enhancement, hair/skin tests, base-model bias audit |
| `eval/` | ArcFace identity / bias evaluation |
| `kaggle_tools/` | Playwright helpers that drive Kaggle through Chrome debug ports (`start_chrome_profiles.ps1` keeps one logged-in profile per account) |
| `prompts/` | Locked editing prompts |
| `docs/` | Research notes |

No credentials live in this repo. Kaggle logins stay in local Chrome profiles, the HF token in Kaggle secrets.

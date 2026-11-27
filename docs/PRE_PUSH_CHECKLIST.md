# Atlas pre-push checklist

**Ready for manual GitHub staging, review, and push as an intermediate checkpoint.** The source and offline checks passed. The research and portability tasks remain in [Pending work](PENDING_WORK.md).

| Git item | Verified state |
| --- | --- |
| Workspace | Atlas repository root |
| Branch | `atlas` |
| HEAD | `7c277168c81ffbf18601bf00b013c0250bbc53bb` |
| Working tree | This checklist and `docs/REPRODUCIBILITY.md` contain the final cleanup updates. Ignored working data and the local archive remain on disk. |
| Staging area at the final check | 82 checkpoint paths were staged. The two updated documents still had working-tree changes to include before committing. |
| Current HEAD history | One initial commit with an empty tree. No tracked files require untracking. |
| Remote | `origin` uses GitHub for fetch and push. The URL has no embedded HTTP user information. |

## Proposed Git contents

Include `.env.example`, `.gitignore`, `.python-version`, `README.md`, `Makefile`, `pyproject.toml`, `uv.lock`, `.github/workflows/quality.yml`, and `configs/`, `docs/`, `infrastructure/`, `scripts/`, `src/` and `tests/`. The JSONL test inputs are synthetic fixtures. The three evidence JSON files under `docs/evidence/` are metadata and verification summaries.

The read-only preview includes staged paths and eligible untracked paths. Review any working-tree edits to staged files before committing.

```sh
GIT_OPTIONAL_LOCKS=0 git ls-files --cached --others --exclude-standard | LC_ALL=C sort -u
GIT_OPTIONAL_LOCKS=0 git diff --name-only
```

Keep `.venv/`, `artifacts/`, `data/`, `results/`, model and database storage, runtime logs, temporary build files, application and editor caches, and real environment or credential files excluded. `.env.example` and the configuration templates remain eligible. The local preview is at `artifacts/checkpoint/proposed-files.txt`.

## Security and large files

The candidate scan checked provider and Git tokens, AWS key formats, private-key markers, credential-bearing URLs, personal account paths, email patterns and binary payloads. The candidate set contains configuration variables and synthetic fixture values. Downloaded corpus text, real question or answer rows, API usage logs and personal machine paths are absent from the Git proposal. No likely secret was found in the candidates. The empty commit tree also contains no credential-bearing file. This check covers the Atlas candidate set and its current HEAD history.

The proposed inventory contains 82 files. The largest is `uv.lock` at about 380 KiB, and none exceeds 1 MiB. Most remaining disk use is in ignored directories, including the primary virtual environment and dataset material.

## Disk and runtime inventory

The following inventory records the first Phase 2.5 cleanup. Values show allocated space when each workspace entry was measured separately. Nested rows are included in their parent. Shared package files and APFS allocation make these per-entry values approximate.

| Workspace entry | Before | After |
| --- | ---: | ---: |
| Main `.venv/` | 887 MiB | 887 MiB |
| `data/` including about 271 MiB of downloads | 381.7 MiB | 381.7 MiB |
| `artifacts/` including caches, indexes, graph snapshots, logs and local evidence | 1,413.3 MiB | 1,405.3 MiB |
| `results/` with synthetic outputs | 7.2 MiB | 7.2 MiB |
| `src/` | 844 KiB | 336 KiB |
| `tests/` including the synthetic fixtures | 416 KiB | 108 KiB |
| `configs/`, `infrastructure/`, and `scripts/` | 56 KiB | 56 KiB |
| `docs/` | 112 KiB | About 192 KiB |
| Top-level pytest and Ruff caches | 64 KiB | Removed |

The cleanup removed 9.91 MiB of pre-existing scratch. This included the pytest and Ruff caches, Python bytecode under source and tests, two previous pytest scratch trees, and one empty profiling output. The post-cleanup test scratch was also discarded after retaining its log, JUnit and coverage files. The workspace measurements net about 8.7 MiB less disk after writing the new evidence and reports. Whole-volume free space fluctuated during the check. These figures apply to the measured workspace entries.

The default generated graph and vector roots occupy about 268 KiB and 28 KiB respectively. Other synthetic graph, vector and run outputs are under `artifacts/delivery/`, `artifacts/completion/` and `artifacts/phase2/`. The Phase 2 fixture tree is about 900 KiB. The public tokenizer cache is about 5 MiB. No standalone model-weight directory was present at the workspace root.

The Atlas-root process inventory showed editor and coding sessions, with no separately confirmed idle Atlas service to stop. Docker's local daemon timed out during the scoped query. The project labels checked were `elevate-atlas`, `elevate-atlas-storage-check`, `elevate-atlas-postgres-age-check`, `elevate-atlas-native-check`, and `elevate-atlas-native-apoc-check`. The earlier verification records identify Qdrant, Milvus, Neo4j, Memgraph and PostgreSQL/AGE images and persistent service data used by the tests. Current container states, volume sizes, image ownership and RAM usage could not be confirmed. No service was stopped or restarted.

## Evidence preservation

- The local archive is `artifacts/checkpoint/atlas-phase1-phase2-evidence.tar.gz`. It contains seven payload files and a member checksum manifest. Its size is 17,175 bytes.
- Archive SHA-256 is `153f806a41d0ccf80d54d92f833241c976eb4f855397a503aa0bb5b04592752d`. The archive was reopened and all seven member hashes matched. A separate whole-archive SHA-256 check passed.
- The Git checkpoint includes [the evidence summary](evidence/checkpoint-summary.json), [dataset manifests and checksums](evidence/dataset-manifests.json) and [post-cleanup test outcomes](evidence/checkpoint-verification.json). It includes the Phase 2 findings on labels, leakage and retrieval scoring, the measured optimization, the shared index and earlier database checks, the blocked Qdrant repeat, all four native framework results, router aggregate CV and the dry-run and model-cost estimates.
- Original research evidence remains in `artifacts/delivery/`, `artifacts/completion/`, `artifacts/phase2/`, and `results/`. An independent, approved destination is still needed for these originals, the archive and its SHA file, dataset versions and permitted inputs, old split files under `artifacts/phase2/prior-data/`, and any dedicated database volume snapshots needed for replay. A same-disk archive is a recoverable copy within Atlas.

Verify the local archive with:

```sh
(cd artifacts/checkpoint && shasum -a 256 -c atlas-phase1-phase2-evidence.tar.gz.sha256)
tar -tzf artifacts/checkpoint/atlas-phase1-phase2-evidence.tar.gz
```

## Authorized additional cleanup

All four approved paths were removed. Whole-workspace allocated disk usage was measured with `du -sk` immediately before and after removal. The raw measurements are in `artifacts/checkpoint/additional-cleanup/cleanup-result.json`.

| Workspace measurement | Allocated space |
| --- | ---: |
| Before | 2,748,468 KiB (2.62 GiB) |
| After | 1,364,472 KiB (1.30 GiB) |
| Reclaimed | 1,383,996 KiB (1.32 GiB) |

The approved directories contained generated dependency or compiler files. The original clean-run records, proposal extract, HippoRAG results, package inventory, and archive checksum were stored outside those paths. After removal, all 1,469 files in the retained data and evidence inventory matched their pre-cleanup sizes and SHA-256 hashes. The inventory includes data, results, Phase 1 and Phase 2 evidence outside the approved paths, the compact evidence archive and its sidecar, and `docs/PENDING_WORK.md`. Docker and its persistent volumes were left untouched during this cleanup.

| Removed resource | Restoration verified before removal |
| --- | --- |
| `artifacts/phase2/clean-venv/` and `artifacts/phase2/uv-clean/` | A locked install into a fresh environment and cache completed successfully. The rebuilt environment ran all 20 offline V4 fixture questions with zero errors. |
| `artifacts/phase2/swift-cache/` | A fresh Swift module cache produced an extract that matched the retained `artifacts/phase2/proposal.txt` byte for byte. |
| `artifacts/phase2/hipporag-2.0.0a5.tar.gz` | The pinned source download matched the original archive and the retained `artifacts/phase2/hippo-archive.sha256`. |

Run the restoration commands from the Atlas root. Set `PROPOSAL_PDF` to the original project-design PDF before running Swift.

```sh
UV_PROJECT_ENVIRONMENT="$PWD/artifacts/phase2/clean-venv" \
  UV_CACHE_DIR="$PWD/artifacts/phase2/uv-clean" \
  uv sync --locked --extra graph --extra router --extra faiss

swift -module-cache-path artifacts/phase2/swift-cache artifacts/phase2/extract_proposal.swift \
  "$PROPOSAL_PDF" > artifacts/checkpoint/proposal-reextract.txt
cmp artifacts/phase2/proposal.txt artifacts/checkpoint/proposal-reextract.txt

curl -fsSL 'https://codeload.github.com/OSU-NLP-Group/HippoRAG/tar.gz/d5c8329422e0a0b834a15874545cb6a74b4f9b26' \
  -o artifacts/phase2/hipporag-2.0.0a5.tar.gz &&
  shasum -a 256 -c artifacts/phase2/hippo-archive.sha256
```

The pinned HippoRAG archive SHA-256 is `0aa7bad1b5753ce6991826ac91a6550aae66d2a5c0901679b5272c4742b82b2a`. See [Reproducibility](REPRODUCIBILITY.md#native-fixture-runs) for the native fixture workflow.

## Post-cleanup verification

After the additional cleanup, the primary `.venv` imported the FAISS and LightRAG dependencies on Python 3.12.13 and completed all 20 offline V4 fixture questions with zero errors. The compact archive passed its whole-file SHA-256 check again, and all seven payloads matched their member hashes. The follow-up verification is in `artifacts/checkpoint/additional-cleanup/`.

The first Phase 2.5 cleanup recorded the following results.

| Check | Result |
| --- | --- |
| Pytest | 55 passed and three opt-in database cases skipped, with zero failures |
| Coverage | 87.13% core line coverage and 75.21% package line coverage |
| Ruff | Lint and formatting passed |
| Mypy | Passed for 25 source files |
| Dependency lock | `uv lock --offline --check` passed |
| Offline V4 smoke | 20 of 20 fixture questions completed, zero errors |

Raw verification is in `artifacts/checkpoint/tests.txt`, `artifacts/checkpoint/tests-junit.xml`, `artifacts/checkpoint/coverage.json`, and `artifacts/checkpoint/smoke-run.json`. See also [exact reproduction commands](REPRODUCIBILITY.md).

## Manual Git commands

Review the preview from the Atlas root. Include the final versions of this checklist and `docs/REPRODUCIBILITY.md` when you run the staging command.

```sh
git branch --show-current
git status --short --untracked-files=all
git ls-files --others --exclude-standard
git diff --cached --name-only

git add -- .env.example .gitignore .python-version README.md Makefile pyproject.toml uv.lock \
  .github/workflows/quality.yml configs docs infrastructure scripts src tests
git diff --cached --stat
git diff --cached --check
git diff --cached
git commit -m "Checkpoint Atlas implementation and Phase 2 audit"
git push -u origin atlas
```

Confirm the resulting checkpoint before resuming the [pending work](PENDING_WORK.md).

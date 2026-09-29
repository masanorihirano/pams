# Release procedure

This document describes how to release a new version of `pams` to
[PyPI](https://pypi.org/project/pams/). It is written for maintainers who have
admin access to `masanorihirano/pams`.

Most of the work is automated by two workflows:

- [`.github/workflows/bump.yml`](.github/workflows/bump.yml) ("Version Bump")
- [`.github/workflows/release.yml`](.github/workflows/release.yml) ("Release")

You create one branch, merge two pull requests and check the result.

## Overview

```text
main ──(you) create branch──▶ release/X.Y.Z
                                   │  `create` event
                                   ▼
             bump.yml opens "Bumping version from A to X.Y.Z"
             create-pull-request/patch-<sha> ──(you) merge──▶ release/X.Y.Z
                                   │
       (you) open PR release/X.Y.Z ──▶ main; its body is the release notes
       (you) merge it with a merge commit
                                   │  `pull_request` closed + merged
                                   ▼
release.yml
  tagging              check pyproject.toml and pams/version.py == X.Y.Z,
                       push the annotated tag X.Y.Z on the merge commit
  final test           black, isort, pflake8, mypy, pytest on
                       {ubuntu, macos, windows}-latest x Python 3.10-3.14
  release test         uv build, uv publish to TestPyPI
  release test check   install pams==X.Y.Z from TestPyPI, run pytest tests/
  release              uv build again (same uv), uv publish to PyPI,
                       checksums, GitHub release X.Y.Z, delete release/X.Y.Z
```

In past releases, the time from merging the release PR to the published GitHub
release was about 20 to 30 minutes.

**Rules that matter most**

- Name the branch exactly `release/X.Y.Z`, for example `release/0.3.0`. Do not
  use a `v` prefix. The tag is `X.Y.Z`, taken from the branch name.
- Create the branch yourself, from an up-to-date `main`.
- Merge the version bump PR into `release/X.Y.Z` **before** you merge the
  release PR.
- The release PR body is copied verbatim into the GitHub release notes.
- Do **not** click "Delete branch" on the merged release PR. The workflow
  deletes `release/X.Y.Z` as its last step.
- If a job of the Release workflow fails, use **Re-run failed jobs**. Do not use
  "Re-run all jobs". See [Troubleshooting](#troubleshooting-and-recovery).
- A version can never be uploaded to PyPI or TestPyPI again with different
  content. Never delete files on PyPI; yank the release instead.

## Before you start

The commands below assume a POSIX shell (Git Bash works on Windows) with `git`,
`gh` (logged in to GitHub), `curl`, `python` and `uv`. Set the version once:

```bash
VERSION=0.3.0
```

1. **`main` is green.** Check the latest CI runs on `main`:

   ```bash
   gh run list --workflow=ci-python.yml --branch main --limit 5
   ```

   The `final test` job of the release runs the same OS and Python matrix. A
   failure on `main` is likely to fail the release too.

2. **Open PRs are merged or deferred.** The release branch contains only what is
   on `main` when you create it.

   ```bash
   gh pr list --state open --base main
   ```

3. **Choose the version.** While `pams` is `0.x`, a minor bump (`0.3.0`) is for
   breaking changes, such as dropping a Python version or changing simulation
   results. A patch bump (`0.3.1`) is for fixes only.

4. **The version is unused.** All of these must be empty or `404`:

   ```bash
   curl -s -o /dev/null -w '%{http_code}\n' "https://pypi.org/pypi/pams/${VERSION}/json"       # 404
   curl -s -o /dev/null -w '%{http_code}\n' "https://test.pypi.org/pypi/pams/${VERSION}/json"  # 404
   git ls-remote --tags origin "$VERSION"                  # empty
   git ls-remote --heads origin 'release/*' 'create-pull-request/*'   # empty
   ```

   These checks do not show files that were uploaded and then deleted, and such
   file names can never be used again. If files for this version were ever
   deleted on PyPI or TestPyPI, choose the next version.

5. **The upload tokens are valid.** The workflows use the repository secrets
   `TEST_PYPI_TOKEN` (TestPyPI) and `PYPI_TOKEN` (PyPI).

   ```bash
   gh secret list -R masanorihirano/pams
   ```

   GitHub shows only the date on which each secret was set. There is no way to
   test a token without uploading: `uv publish --dry-run` does not check
   credentials. Instead:

   - Log in to <https://pypi.org/manage/account/> and
     <https://test.pypi.org/manage/account/>.
   - Under "API tokens", check that the token for `pams` still exists, and look
     at when it was last used. It should be around the last release.
   - If in doubt, create a new token scoped to the project `pams` and store it:

     ```bash
     gh secret set PYPI_TOKEN -R masanorihirano/pams        # prompts for the value
     gh secret set TEST_PYPI_TOKEN -R masanorihirano/pams
     ```

   A bad token can be fixed during the release (see
   [Troubleshooting](#troubleshooting-and-recovery)), but checking it first
   avoids a failed run.

6. **Draft the release notes.** They go into the release PR body. See
   [Step 3](#step-3-open-the-release-pr) for the format.

## Step 1: Create the release branch

Create `release/X.Y.Z` from the current `main` and push it:

```bash
git fetch origin
git push origin "origin/main:refs/heads/release/${VERSION}"
```

Alternatively, in the web UI, open the branch selector on the repository page,
type `release/X.Y.Z` and choose "Create branch: release/X.Y.Z from main".

Why it has to be done this way:

- The `create` event runs `bump.yml` **as it exists in the new branch**. A branch
  created from an old commit, such as an old tag, runs an old `bump.yml`.
- Only a branch created with your own credentials (git push, the web UI or
  `gh api`) triggers the workflow. A branch created by another workflow with
  `GITHUB_TOKEN` does not.
- The `make pr` job runs only for branches whose name starts with `release/`.
  For a name that uv would normalize, such as `release/v0.3.0`, the "Bump
  version" step fails with `release/... is not of the form release/X.Y.Z`.
  Other valid versions are accepted as they are: `release/0.3` would release
  version `0.3`. Check the name before pushing.

Pushing the branch also starts CI-min, CI-notebooks and codecov CI on it.

## Step 2: Merge the version bump PR

1. Wait for the Version Bump run to finish and open the PR.

   ```bash
   gh run list --workflow=bump.yml --limit 3
   gh pr list --base "release/${VERSION}"
   ```

2. Review the PR. It should look like this:

   - Title: `Bumping version from 0.2.2 to 0.3.0`.
   - Head branch: `create-pull-request/patch-<short sha>`, opened by
     `github-actions[bot]`.
   - Changes: the `version` line in `pyproject.toml` and `pams/version.py`.
     `uv.lock` is not committed, so it does not appear.

   The PR is created with `GITHUB_TOKEN`, so its checks may not start, or may
   wait behind an "Approve workflows to run" button. `release/*` branches have no
   required checks, so this does not block the merge. Merging the PR pushes to
   `release/X.Y.Z`, which runs CI-min, CI-notebooks and codecov CI on the bumped
   branch anyway.

3. Merge it with a merge commit, and delete its head branch. Deleting the
   `create-pull-request/...` branch is safe.

   ```bash
   gh pr merge <bump-PR-number> --merge --delete-branch
   ```

4. Check that the release branch now has the new version:

   ```bash
   git fetch origin
   git show "origin/release/${VERSION}:pams/version.py"
   git show "origin/release/${VERSION}:pyproject.toml" | grep '^version'
   ```

   (In Git Bash on Windows, prefix `git show <rev>:<path>` with
   `MSYS_NO_PATHCONV=1`.)

**Last-minute fixes** can land on the release branch before Step 4. Open a PR
with base `release/X.Y.Z`, or push to it directly. They reach `main` through the
release PR. Do not change the version in these fixes.

## Step 3: Open the release PR

Open a PR from `release/X.Y.Z` into `main`. Past releases used the title
`Release/X.Y.Z`, which is GitHub's default for this branch name.

```bash
gh pr create --base main --head "release/${VERSION}" \
  --title "Release/${VERSION}" --body-file release-notes.md
```

**The PR body is the release note.** When the PR is merged, the `release` job
creates the GitHub release with this body:

```text
<release PR body at merge time>

This release is automatically generated.
Please see the pull request.
[<release PR URL>](<release PR URL>)
```

- Everything in the body is copied, including checklists, HTML comments and
  footers added by bots or tools. Remove what should not be in the release
  notes.
- You can edit the body at any time before merging. Use the web UI, or:

  ```bash
  gh api -X PATCH repos/masanorihirano/pams/pulls/<release-PR-number> -F body=@release-notes.md
  ```

- Editing the body **after** the merge does not change the release, and
  re-running the workflow also uses the body as it was at merge time. Edit the
  GitHub release instead:

  ```bash
  gh release edit "$VERSION" --notes-file release-notes.md
  ```

Past release notes are short. They have lower-case category headers and ` - `
bullets, and mention contributors with `@`. For example, 0.2.2 (#108):

```text
maintenance:
 - drop python 3.8 supports
 - support python 3.12, 3.13
 - drop poetry legacy installation
```

and 0.2.1 (#101):

```text
feat:
 - add order expiration logs @ryuji-hashimoto0110

maintenance:
 - CI update (yaml formatting, change default python version to 3.11 from 3.9)
```

For a larger release, use categories such as `breaking changes:`, `features:`,
`bug fixes:`, `documentation:` and `maintenance:`, and add the PR number to each
bullet.

## Step 4: Wait for the checks and merge

1. The release PR runs CI (3 OS x Python 3.10 to 3.14), CI-notebooks, codecov CI
   and doc. Wait until they pass:

   ```bash
   gh pr checks <release-PR-number> --watch
   ```

   If a leg fails because of a flaky test (see
   [Notes for 0.3.0](#notes-for-030-the-first-release-with-uv)), use "Re-run
   failed jobs" on that run.

2. Branch protection on `main` requires one approving review. Either get an
   approval from another maintainer, or merge as an administrator. In the web UI,
   tick the option to bypass the branch protection rules. On the command line,
   use `--admin`.

3. Merge with **a merge commit**, as for every past release. Do not delete the
   branch.

   ```bash
   gh pr merge <release-PR-number> --merge --admin
   ```

   Squash or rebase merges would also work, because every job uses the PR's
   `merge_commit_sha`. The merge commit is the convention.

Closing the release PR without merging does nothing: no tag and no upload. You
can reopen it and merge it later.

## Step 5: Watch the Release workflow

```bash
gh run list --workflow=release.yml --limit 3
gh run watch <run-id>
```

The workflow runs these jobs in order:

| Job in the UI | What it does | State if it fails |
|---|---|---|
| `tagging` | "Check that the version was bumped": `pyproject.toml` and `pams/version.py` at the merge commit must both equal `X.Y.Z`. "Tag the merge commit" pushes the annotated tag `X.Y.Z`. If the tag already points to this commit, it succeeds without doing anything. | No tag, nothing uploaded. |
| `final test (<os>, <python>)` | 15 legs, at most 5 at a time, with fail-fast: lint, mypy and `pytest tests/`. | Tag pushed, nothing uploaded. |
| `release test (3.11)` | `uv build`, then "test release" runs `uv publish` to TestPyPI with `TEST_PYPI_TOKEN`. It records the uv version it used. | Tag pushed, maybe some files on TestPyPI. |
| `release test check (3.11)` | Polls TestPyPI for up to 30 x 20 s, installs `pams==X.Y.Z` from there and the latest `pytest`, removes `pams/` and `pyproject.toml` from the checkout and runs `pytest tests/` against the installed package. | Files on TestPyPI, nothing on PyPI. |
| `release (3.11)` | Builds again with the **same uv version** as `release test`, so the files are byte-identical. "release" runs `uv publish` to PyPI with `PYPI_TOKEN`. "Generate checksum" writes `pams-X.Y.Z-checksums.txt`. "Create release" runs `gh release create`. "remove branch" deletes `release/X.Y.Z`. | Depends on the step, see below. |

If a job fails, find its row in
[Troubleshooting](#troubleshooting-and-recovery) before doing anything else.

## Step 6: Verify the release

```bash
git fetch origin --tags
git cat-file -t "$VERSION"                  # tag (annotated)
git rev-parse "${VERSION}^{commit}"         # must equal the next line
gh pr view <release-PR-number> --json mergeCommit -q .mergeCommit.oid
git ls-remote --heads origin "release/${VERSION}"   # empty: the branch was deleted
```

**GitHub release.** It should be named `X.Y.Z`, be marked Latest, and have three
assets: `pams-X.Y.Z-py3-none-any.whl`, `pams-X.Y.Z.tar.gz` and
`pams-X.Y.Z-checksums.txt`. Check the checksums and compare them with PyPI:

```bash
gh release view "$VERSION"
gh release download "$VERSION" -D "release-${VERSION}"
(cd "release-${VERSION}" && sha256sum -c "pams-${VERSION}-checksums.txt")
curl -s "https://pypi.org/pypi/pams/${VERSION}/json" \
  | python -c "import json,sys; [print(u['digests']['sha256'], u['filename']) for u in json.load(sys.stdin)['urls']]"
```

**PyPI page.** Open `https://pypi.org/project/pams/X.Y.Z/`. Check the README,
"Requires: Python", the license and the classifiers.

**Clean install.** Install on the oldest and the newest supported Python. For
0.3.0 these are 3.10 and 3.14; expect numpy 1.x on 3.10 and numpy 2.x on 3.14.

```bash
for py in 3.10 3.14; do
  uv run --no-project --isolated --refresh-package pams --python "$py" \
    --with "pams==${VERSION}" \
    python -c "import pams, numpy; print(pams.__version__, numpy.__version__)"
done
```

Without uv, create a fresh venv with that Python and run
`python -m pip install --no-cache-dir pams==X.Y.Z`.

**Documentation.**

- The docs are on Read the Docs at <https://pams.hirano.dev/>, with the projects
  `pams` (English) and `pams-ja` (Japanese).
- Pushing the tag starts a build of the `stable` version. Check that the build
  succeeded on the Read the Docs build pages of both projects, or check that the
  date changed:

  ```bash
  curl -sI https://pams.hirano.dev/en/stable/ | grep -i last-modified
  curl -sI https://pams.hirano.dev/ja/stable/ | grep -i last-modified
  ```

- Versions for individual tags (such as `/en/0.3.0/`) are not active on Read the
  Docs. Only `latest` and `stable` are built.
- The `doc` workflow also deploys to GitHub Pages on every push to `main`,
  including the release merge.

## Troubleshooting and recovery

### General rules

- **Use "Re-run failed jobs"**: the button on the run page, or
  `gh run rerun <run-id> --failed`. It re-runs the failed jobs and the jobs that
  depend on them, including matrix legs that fail-fast cancelled. Successful
  jobs are not run again, and their outputs (the tag and the uv version) are
  reused. To re-run one job, get its ID with
  `gh run view <run-id> --json jobs --jq '.jobs[] | {name, databaseId}'` and run
  `gh run rerun --job <databaseId>`.
- **Avoid "Re-run all jobs".** `release test` installs the newest uv.
  - If uv had a new release since the first run, the rebuilt wheel differs (its
    `WHEEL` file names the uv version). TestPyPI then rejects it with
    `400 File already exists`, and this version can no longer pass.
  - If the GitHub release already exists, `gh release create` fails with
    `a release with the same tag name already exists`.
  - It is only safe while nothing has been uploaded to TestPyPI.
- **A re-run cannot pick up a fix.** It uses the original event: the same merge
  commit, PR body and branch name. To change code or the version, you need a new
  release PR.
- A run can be re-run for up to 30 days after it started, and at most 50 times.
- **Upload rules on PyPI and TestPyPI:**
  - Uploading a file that is identical to an existing one is skipped silently,
    so it is safe.
  - Uploading a different file under an existing name is rejected
    (`400 File already exists`).
  - A deleted file name can never be used again.
  - New files can be added to a release only within 14 days of its first upload.

### No Version Bump run, or no bump PR

- Check `gh run list --workflow=bump.yml --limit 5`.
  - If the `make pr` job was skipped, the branch name does not start with
    `release/`.
  - If there is no run at all, check how the branch was created (see
    [Step 1](#step-1-create-the-release-branch)). A branch created by a
    workflow with `GITHUB_TOKEN` does not start a run.
- Start it by hand. This is harmless: if the version is already bumped, the
  workflow finds no changes and opens no PR.

  ```bash
  gh workflow run bump.yml --ref "release/${VERSION}"
  ```

- If "Bump version" failed with `is not of the form release/X.Y.Z`, delete the
  wrongly named branch and create `release/X.Y.Z` again:

  ```bash
  git push origin --delete release/<wrong-name>
  ```

- If the release branch got new commits while the bump PR was open and you ran
  the bump again, a second bump PR appears. Merge one, then close the other and
  delete its branch.

### The release PR was merged before the bump PR

`tagging` fails with `The version in pyproject.toml / pams/version.py does not
match X.Y.Z. Merge the version bump pull request into the release branch
first.`

- Nothing was tagged or uploaded, and `release/X.Y.Z` still exists.
- Re-running cannot help, because it checks the same merge commit.
- To recover:
  1. Merge the bump PR into `release/X.Y.Z`.
  2. Open a **new** PR `release/X.Y.Z` into `main` with the release notes.
  3. Merge it. This starts a new Release run.

### `tagging`: `Tag X.Y.Z already exists on another commit`

1. Make sure that the tag is a stray one: no GitHub release and no files on PyPI
   or TestPyPI for this version.
2. Delete the tag and re-run:

   ```bash
   git push origin ":refs/tags/${VERSION}"
   gh run rerun <run-id> --failed
   ```

### `final test` fails

- **Flaky test** (for example a timing assertion): `gh run rerun <run-id>
  --failed`. The failed leg and every leg cancelled by fail-fast run again, and
  the tag is reused.
- **Real failure:** the tag exists but nothing was uploaded, so the version can
  still be used.
  1. Delete the tag: `git push origin ":refs/tags/${VERSION}"` (and
     `git tag -d "$VERSION"` locally if you fetched it).
  2. Fix the problem on `release/X.Y.Z`, which still exists.
  3. Open a new release PR and merge it.

### `release test` fails

- **Authentication error (403) from TestPyPI:** replace `TEST_PYPI_TOKEN`, then
  `gh run rerun <run-id> --failed`. Nothing has been uploaded, so a newer uv does
  no harm here. `release` uses whatever version this re-run records.
- **Only some files were uploaded:** `gh run rerun <run-id> --failed`. This works
  if uv has not had a new release in the meantime; otherwise see the next item.
- **`400 File already exists`:** TestPyPI already has a different file with this
  name, and this version can no longer be used.
  - Release the next version (for example `X.Y.(Z+1)`) with the full procedure.
  - Delete the leftover `release/X.Y.Z` branch by hand.
  - The tag `X.Y.Z` can stay without a PyPI release. This happened with 0.0.4.

### `release test check` fails

- **`pams==X.Y.Z did not become available on TestPyPI`:** TestPyPI was slow.
  Run `gh run rerun <run-id> --failed`.
- **A flaky test failed:** `gh run rerun <run-id> --failed`.
- **The tests fail against the installed package** (a packaging bug). The files
  are on TestPyPI and cannot be replaced, and nothing is on PyPI.
  - Fix the bug and release the next version.
  - Delete the leftover `release/X.Y.Z` branch by hand. The tag `X.Y.Z` can
    stay, as described above.

### `release` fails in "release" (`uv publish` to PyPI)

- **Authentication error (403):** replace `PYPI_TOKEN`, then
  `gh run rerun <run-id> --failed`. uv is pinned to the version `release test`
  used, so the rebuilt files are identical.
- **Network error or partial upload:** `gh run rerun <run-id> --failed`.
  Identical files are skipped and missing ones are uploaded. Do it within 14 days
  of the first PyPI upload.

### `release` fails in "Create release"

PyPI has the version, but there is no GitHub release: `gh` deletes its draft
when an upload fails.

- Run `gh run rerun <run-id> --failed`. It rebuilds identical files, `uv publish`
  skips them, then it creates the release and deletes the branch.
- If a re-run is no longer possible, create the release by hand from the files
  on PyPI:

  ```bash
  mkdir "release-${VERSION}" && cd "release-${VERSION}"
  curl -s "https://pypi.org/pypi/pams/${VERSION}/json" \
    | python -c "import json,sys; print('\n'.join(u['url'] for u in json.load(sys.stdin)['urls']))" \
    | xargs -n1 curl -sSLO
  sha256sum -- * > "pams-${VERSION}-checksums.txt"
  PR=<release-PR-number>
  PR_URL=$(gh pr view "$PR" --json url -q .url)
  gh pr view "$PR" --json body -q .body > notes.md
  printf '\nThis release is automatically generated.\nPlease see the pull request.\n[%s](%s)\n' \
    "$PR_URL" "$PR_URL" >> notes.md
  gh release create "$VERSION" "pams-${VERSION}-py3-none-any.whl" "pams-${VERSION}.tar.gz" \
    "pams-${VERSION}-checksums.txt" --verify-tag --title "$VERSION" --notes-file notes.md
  git push origin --delete "release/${VERSION}"
  ```

### `release` fails in "remove branch"

Everything is published. Usually someone deleted `release/X.Y.Z` already, for
example with the "Delete branch" button.

- **Do not re-run.** A re-run would stop at `gh release create` because the
  release exists.
- If the branch still exists, delete it by hand:
  `git push origin --delete "release/${VERSION}"`.

### A broken release is on PyPI

- Do **not** delete files or the release. The file names could never be used
  again.
- Yank the release instead:
  1. Open <https://pypi.org/manage/project/pams/releases/>.
  2. Select the version, choose "Options", then "Yank", and give a reason.
  3. Installers then skip it unless it is pinned with `==`.
- Fix forward with the next patch version, using the same procedure. Leave the
  tag in place. Optionally add a note to the GitHub release.

## Notes for 0.3.0 (the first release with uv)

- **First end-to-end run.** The workflows were rewritten for uv in #131 and
  #140, and 0.3.0 is the first release that runs them.
  - The last release, 0.2.2 (October 2024), used the old Poetry-based workflow.
  - The logs of those runs are no longer available.
  - Watch every job, especially `release test`, which is the first upload.
- **Tokens.** The secrets `PYPI_TOKEN` and `TEST_PYPI_TOKEN` were set on
  2022-10-20. They were last used successfully for 0.2.2 on 2024-10-23. Check
  them as described in [Before you start](#before-you-start).
- **Flaky test.**
  `tests/pams/runners/test_agent_parallel.py::TestMultiProcessAgentParallelRunner::test_parallel_efficiency`
  has a timing assertion.
  - It has failed several times in CI on `main`, on macOS and Windows legs.
  - In the release, `final test` runs after the tag is pushed. A flaky failure
    there stops the release with the tag already in place.
  - "Re-run failed jobs" recovers from this.
- **uv is not pinned** in `release test`. It uses the newest uv, and `release`
  pins itself to that version. This is why "Re-run all jobs" should be avoided.
- **`release test check` installs the newest `pytest`**, while the `dev` group
  pins `pytest<8`. A local simulation with pytest 9.1.1 passed; it only printed
  `PytestReturnNotNoneWarning` warnings.
- **Read the Docs.**
  - `stable` has shown 0.2.0 since September 2023. The builds for the 0.2.1 and
    0.2.2 tags failed because the `.readthedocs.yml` at those tags had no
    `build.os`.
  - The current `.readthedocs.yml` builds, so the 0.3.0 tag should update
    `stable`. Check it after the release.
- **Open PRs.** The open PRs are based on a commit before the switch to uv. Merge
  `main` into them and let CI pass before merging them into the release.

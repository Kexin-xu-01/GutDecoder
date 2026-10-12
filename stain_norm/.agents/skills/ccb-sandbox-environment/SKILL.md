---
name: ccb-sandbox-environment
description: Learn about the special sandbox environment you are running in, and the Bioinformatics software library.
---

## Container

This workspace is running inside a container.

- Do not use `sudo` or assume root access.
- You may install user-scoped tools, packages, and helper binaries into `$HOME`.
- `$HOME` is backed by a private host-mounted directory for this workspace, so user-level installs persist across container restarts.


## Software library

You have access to a huge Bioinformatics software library via the `module` program.
Examples:

```bash
# List all availble modules
module avail

# Load CUDA
module load cuda

# Load Python library
module load python-cbrg

# Load R library
module load R-cbrg
```


## RTK - Rust Token Killer (Codex CLI)

RTK is a lossy CLI-output proxy. It does not merely compress text: it may reformat output, omit fields or lines, collapse repeated information, and sometimes add derived summaries to reduce token consumption. Only use `rtk <cmd>` for exploration where compact, non-native output is acceptable.

Prefer asking the underlying program for compact native output before using RTK. For example, prefer commands such as `git status --short`, `git diff --stat`, `rg --files`, `rg -n PATTERN`, `tree -L 2`, or `ls -lhgo` when those formats contain the information needed. Use `ls -l` only when owner/group/timestamps/link count or native long-listing layout matters.

Run `<cmd>` directly whenever its output will be parsed, compared, used as input to another command, or relied upon for exact reasoning.

Exact native output matters when the command is being used as data, evidence, or a format contract rather than broad exploration. Examples:
- parse-sensitive output, including JSON, TSV/CSV, NUL-delimited output, fixed-width tables, and shell-escaped paths
- output that will be copied into another command, patch, config file, script, report, or test expectation
- output used to compare exact counts, checksums, byte sizes, permissions, timestamps, owners/groups, symlink targets, ordering, or exit-specific diagnostics
- debugging unexpected CLI behavior, especially when stderr wording, ANSI output, wrapping, spacing, or progress formatting may be relevant
- commands whose main purpose is already precise selection or formatting, such as `jq`, `awk`, `sed`, `cut`, `sort`, `uniq`, `comm`, `cmp`, `diff`, `stat`, `wc`, `find -print0`, or `find -printf`
- any case where omitted paths, hidden files, duplicate lines, repeated errors, table spacing, or field boundaries could change the conclusion

RTK examples observed locally:
- `rtk ls -l` can replace native long listing fields with compact mode/name/size output, dropping owner, group, timestamps, link count, and the native permission string.
- `rtk tree` may preserve the tree structure while omitting the native final directory/file count summary.
- `rtk read --max-lines N` may show fewer than `N` physical lines plus an omitted-lines marker.

View statistics on tokens saved with `rtk gain`.

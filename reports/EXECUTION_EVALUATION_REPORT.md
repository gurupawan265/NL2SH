# NL2SH — Phase 3 Execution-Based Evaluation Report

**Date:** 2026-11-07  
**Framework:** Docker-Based Execution Evaluator (`Scripts/evaluate_commands.py`)  
**Benchmark:** 100 Curated Evaluation Tasks (`data/evaluation/evaluation_benchmark.jsonl`)  
**Security Standard:** Strict `DockerSandbox` isolation (`--network none`, read-only rootfs, non-root UID 1000:1000, tmpfs `/workspace`, dropped capabilities, unprivileged user namespace)

---

## 1. Executive Summary

Phase 3 of the NL2SH (Natural Language to Shell) project establishes a secure, deterministic, execution-based evaluation framework. Rather than relying solely on string matching (e.g., BLEU, exact string match) or static syntax checks, this framework validates commands by executing them inside isolated container sandboxes against synthetic filesystem fixtures and verifying their real-world semantic side effects.

### High-Level Metrics

| Metric | Result |
| :--- | :--- |
| **Total Candidates Curated** | 103 tasks |
| **Quarantined Candidates** | 3 tasks |
| **Benchmark Dataset Size** | 100 tasks |
| **Tasks Evaluated** | 100 / 100 |
| **Execution Success Rate** | **100.0%** (100 / 100) |
| **Semantic Pass Rate** | **100.0%** (100 / 100) |
| **Timeouts** | 0 |
| **Average Task Duration** | ~0.097 seconds |
| **Total Benchmark Runtime** | ~9.7 seconds |

---

## 2. Benchmark Architecture & Category Distribution

The evaluation benchmark covers 10 representative Linux shell task categories with exactly 10 tasks per category, exercising diverse command-line utilities, options, pipes, and shell features:

| Category | Tasks | Executed | Success Rate | Semantic Pass Rate | Primary Utilities Tested |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `file_search` | 10 | 10 | 100.0% | 100.0% | `find` (`-name`, `-type`, `-empty`, `-mindepth`, `-maxdepth`, `-iname`, `-size`) |
| `text_search` | 10 | 10 | 100.0% | 100.0% | `grep` (`-i`, `-n`, `-v`, `-c`, `-r`, `-w`, regex patterns `^`, `$`) |
| `text_processing` | 10 | 10 | 100.0% | 100.0% | `awk`, `sed`, `cut`, `tr` (delimiters, field slicing, in-place edit, character translation) |
| `file_creation` | 10 | 10 | 100.0% | 100.0% | `touch`, `echo >`, `echo >>`, `printf`, `>` truncate |
| `directory_operations` | 10 | 10 | 100.0% | 100.0% | `mkdir`, `mkdir -p`, `mkdir -m`, `rmdir`, `rmdir -p` |
| `sorting` | 10 | 10 | 100.0% | 100.0% | `sort` (`-r`, `-n`, `-u`, `-nr`, `-t -k`), `uniq` (`-d`, `-c`) |
| `counting` | 10 | 10 | 100.0% | 100.0% | `wc` (`-l`, `-w`, `-c`, `-m`, `-L`), `find \| wc -l`, `grep -c` |
| `file_copy_move` | 10 | 10 | 100.0% | 100.0% | `cp`, `cp -r`, `mv`, `rm`, `rm -f`, `rm -rf`, `ln -s` |
| `file_comparison` | 10 | 10 | 100.0% | 100.0% | `diff`, `diff -u`, `diff -q`, `diff -r`, `diff -w`, `diff -i`, `cmp`, `comm` |
| `permissions` | 10 | 10 | 100.0% | 100.0% | `chmod` (octal `600`, `644`, `700`, `755`, `775`, symbolic `u+x`, `u+w`, `-x`, `-R`) |
| **Total** | **100** | **100** | **100.0%** | **100.0%** | Comprehensive core POSIX / GNU coreutils |

---

## 3. Fixture Injection & Isolation Standard

### Strict Sandbox Environment
All evaluations run inside an unprivileged Docker container built from `docker/Dockerfile.sandbox`.
Security properties strictly maintained:
1. `--network none`: Container has zero egress/ingress network access.
2. `--read-only`: The container root filesystem `/` is strictly read-only.
3. `--tmpfs /workspace:rw,noexec,nosuid,size=64m`: In-memory workspace mounted exclusively for task execution.
4. `--user 1000:1000`: Commands run strictly as non-root user `sandboxuser`.
5. `--cap-drop ALL`: All Linux capabilities dropped.
6. `--security-opt no-new-privileges:true`: Prevents privilege escalation (`setuid`).
7. `--pids-limit 64`: Fork-bomb protection.
8. `--memory 256m`: Capped RAM allocation.
9. `--cpus 1.0`: CPU allocation capped.

### Deterministic Fixture Injection
Fixtures are defined declaratively in `data/evaluation/evaluation_benchmark.jsonl`. For each test:
- A new, clean container is spawned.
- Fixture directories are initialized via `mkdir -p`.
- Fixture files and contents are injected directly via `docker exec -i --user 1000:1000 <container> bash -c "cat > <path>"`.
- Fixture paths are validated to ensure they remain inside `/workspace` and reject path traversal attempts (e.g. `../` or `/etc`).
- After execution and verification, the container is forcibly terminated (`docker rm -f`) in a `finally` block, ensuring zero residue.

---

## 4. Multi-Method Semantic Verification

Evaluating commands purely by exit code is insufficient because many incorrect commands exit 0 (e.g., `grep` finding nothing, or `cat` without flags). Therefore, six semantic verification methods are employed:

1. **`stdout_match`**:
   - Exact string comparison (`stdout`) or multiline set matching (`stdout_lines` with order invariance when searching), and substring verification (`stdout_contains`).
2. **`file_exists`**:
   - Validates that targeted created files exist (`test -f <path>`).
3. **`file_absence`**:
   - Validates that targeted deleted or moved files no longer exist (`test ! -e <path>`).
4. **`file_content_match`**:
   - Validates file creation and inspects the exact byte or line contents of output files (`cat <path>`).
5. **`directory_structure`**:
   - Validates directory creation (`test -d <path>`) or nested removal.
6. **`permission_match`**:
   - Queries file and directory modes via `stat -c %a` to verify exact POSIX permission bits (`755`, `600`, `644`, etc.) independent of `noexec` mount flags.
7. **`exit_code`**:
   - Validates expected exit codes (including expected non-zero exits for utilities like `diff` and `cmp` when differences exist).

---

## 5. Quarantined Candidate Tasks

Three candidate tasks were identified as unviable or dangerous during fixture generation and quarantined to `data/evaluation/quarantined_candidates.jsonl`:

1. **`eval_quarantine_001` (Path Traversal Risk)**:
   - Instruction: *"Remove temporary files from parent system directory"*
   - Command: `rm -rf ../temp_data`
   - Quarantine Reason: Attempted path traversal outside `/workspace`. The evaluation harness strictly confines all operations to `/workspace`.
2. **`eval_quarantine_002` (System Destruction Risk)**:
   - Instruction: *"Wipe all root directory files forcefully"*
   - Command: `rm -rf /`
   - Quarantine Reason: Destructive system command targeting root filesystem; blocked by safety heuristics.
3. **`eval_quarantine_003` (Non-Deterministic Network Dependency)**:
   - Instruction: *"Download the latest package from example.com"*
   - Command: `curl -s https://example.com/index.html`
   - Quarantine Reason: Requires external internet connectivity. Evaluation sandboxes enforce `--network none` for security and reproducibility.

---

## 6. Representative Evaluated Tasks

Below are representative verified tasks from each category:

### File Search (`file_search`)
- **Task `eval_task_001`**:
  - Instruction: *"Find all JPEG files in /workspace/photos"*
  - Command: `find /workspace/photos -name '*.jpg' | sort`
  - Verification: `stdout_lines` matched `['/workspace/photos/img1.jpg', '/workspace/photos/vacation.jpg']`.

### Text Search (`text_search`)
- **Task `eval_task_012`**:
  - Instruction: *"Find lines containing 'failed' case-insensitively in /workspace/auth.log"*
  - Command: `grep -i 'failed' /workspace/auth.log`
  - Verification: `stdout_lines` matched `['Failed password for invalid user admin', 'Login FAILED for user test']`.

### Text Processing (`text_processing`)
- **Task `eval_task_023`**:
  - Instruction: *"Replace all occurrences of 'foo' with 'bar' in /workspace/sample.txt"*
  - Command: `sed 's/foo/bar/g' /workspace/sample.txt`
  - Verification: `stdout` matched `'bar and bar and foo_bar'`.

### File Creation (`file_creation`)
- **Task `eval_task_034`**:
  - Instruction: *"Write 'Hello World' to /workspace/hello.txt"*
  - Command: `echo 'Hello World' > /workspace/hello.txt`
  - Verification: `file_content_match` matched `/workspace/hello.txt` -> `'Hello World\n'`.

### Directory Operations (`directory_operations`)
- **Task `eval_task_042`**:
  - Instruction: *"Create nested directories /workspace/a/b/c"*
  - Command: `mkdir -p /workspace/a/b/c`
  - Verification: `directory_structure` confirmed directory existence.

### Sorting & Dedup (`sorting`)
- **Task `eval_task_054`**:
  - Instruction: *"Sort /workspace/duplicates.txt and remove duplicate lines"*
  - Command: `sort -u /workspace/duplicates.txt`
  - Verification: `stdout_lines` matched `['a', 'b', 'c']`.

### Counting (`counting`)
- **Task `eval_task_061`**:
  - Instruction: *"Count the number of lines in /workspace/file.txt without filename"*
  - Command: `wc -l < /workspace/file.txt`
  - Verification: `stdout` matched `'3\n'`.

### File Copy / Move / Deletion (`file_copy_move`)
- **Task `eval_task_072`**:
  - Instruction: *"Rename /workspace/old_name.txt to /workspace/new_name.txt"*
  - Command: `mv /workspace/old_name.txt /workspace/new_name.txt`
  - Verification: `file_absence` on `/workspace/old_name.txt` and `file_content_match` on `/workspace/new_name.txt`.

### File Comparison (`file_comparison`)
- **Task `eval_task_083`**:
  - Instruction: *"Briefly check if /workspace/a.bin and /workspace/b.bin differ using diff -q"*
  - Command: `diff -q /workspace/a.bin /workspace/b.bin`
  - Verification: `exit_code` matched `1` and `stdout` matched `'Files /workspace/a.bin and /workspace/b.bin differ\n'`.

### Permissions (`permissions`)
- **Task `eval_task_092`**:
  - Instruction: *"Set permissions of /workspace/secret.key to 600 (read/write only by owner)"*
  - Command: `chmod 600 /workspace/secret.key`
  - Verification: `permission_match` queried `stat -c %a` -> `'600'`.

---

## 7. Reusability for Future SFT & RL (GRPO/PPO)

The evaluation harness in `Scripts/evaluate_commands.py` provides clean APIs:
- `evaluate_single_task(task, timeout=10.0)`
- `evaluate_benchmark(benchmark_tasks, max_workers=1)`

Future models (Base LLMs, SFT fine-tuned models, and RL policies trained with PPO or GRPO) can be evaluated by substituting their generated bash outputs into the benchmark task structure:
```python
task_eval = dict(benchmark_task)
task_eval["reference_command"] = model_generated_command
result = evaluate_single_task(task_eval)
reward = 1.0 if (result["execution_success"] and result["semantic_pass"]) else 0.0
```
This enables an automated ground-truth reward signal for reinforcement learning without human labeling or brittle regex matching.


# NL2SH — Phase 2: NL2SH-ALFA Import and Cleaning Report

**Dataset:** `westenfelder/NL2SH-ALFA`  
**Revision / Commit SHA:** `a99cb5784cf5c2a42b1cc26c1903d9c3b35206ba`  
**Source License:** MIT  
**Source URL:** [https://huggingface.co/datasets/westenfelder/NL2SH-ALFA](https://huggingface.co/datasets/westenfelder/NL2SH-ALFA)  
**Pipeline Script:** `scripts/prepare_alfa_dataset.py`  
**Verification Suite:** `tests/test_alfa_preprocessing.py`  
**Audit Status:** Complete and Verified (100% record accounting)

---

## 1. Executive Summary

In Phase 2, `westenfelder/NL2SH-ALFA` was imported from Hugging Face into `data/raw/` preserving the raw unadulterated source splits. A transparent, reproducible preprocessing pipeline was developed and executed to eliminate documentation artifacts, filter unexecutable placeholders, quarantine privileged and conflicting prompts, normalize records to a structured JSONL schema, and create leak-free training and validation splits.

```
Raw Training Corpus: 40,639 records
├── Clean Retained Set: 31,296 records (77.01%)
│   ├── Training Split (data/processed/train.jsonl): 28,167 records (90.00%)
│   └── Validation Split (data/processed/val.jsonl):  3,129 records (10.00%)
├── Quarantined Set (data/processed/quarantined.jsonl): 2,791 records (6.87%)
└── Removed Set (data/processed/removed.jsonl):         6,552 records (16.12%)

Official Test Set (data/raw/alfa_test.csv): 300 tasks (600 reference commands)
└── UNTOUCHED and RESERVED exclusively for final benchmark evaluation.
```

---

## 2. Dataset Ingestion & Preservation

Raw dataset splits are preserved in `data/raw/` without modifications:
* `data/raw/alfa_train.csv`: 40,639 rows, columns `['nl', 'bash']`.
* `data/raw/alfa_test.csv`: 300 rows, columns `['nl', 'bash', 'bash2', 'difficulty']`.
* `data/raw/metadata.json`: Records provenance, commit SHA, download timestamp, and MIT license.

The official 300-task benchmark test set was verified and completely segregated: it was **not** included in the cleaning pipeline, **not** sampled into training or validation, and remains reserved exclusively for final benchmark evaluation.

---

## 3. Cleaning & Quarantine Pipeline

### A. Removed Records (6,552 Records / 16.12%)
Records with severe unexecutable syntax or scraping noise were permanently removed and archived with recorded reasons:

| Removal Reason | Count | Percentage | Description |
|---|---|---|---|
| `documentation_placeholder_path_to` | 5,629 | 85.91% | Scraped `tldr` placeholders containing `path/to/file`, `path/to/dir`. |
| `documentation_placeholder_option_pipe` | 380 | 5.80% | Documentation option syntax containing pipes (e.g. `-s\|--shell`, `json\|yaml`). In Bash, `\|` causes broken pipe syntax errors. |
| `documentation_placeholder_ellipsis` | 223 | 3.40% | Standalone ellipsis (`...`) used as pseudo-argument. |
| `documentation_placeholder_angle_brackets` | 211 | 3.22% | Documentation angle bracket placeholders (e.g. `<url>`, `<file>`, `<name>`). |
| `degenerate_instruction` | 102 | 1.56% | Nonsense or non-English prompt strings (e.g. `qhgvxvhqx`, prompts $<8$ characters). |
| `syntax_error` | 7 | 0.11% | Shell syntax parsing failures (unmatched quotes, broken tokenization). |

### B. Quarantined Records (2,791 Records / 6.87%)
Records containing valid commands that are unsuitable for unprivileged, offline sandbox training were isolated into `data/processed/quarantined.jsonl`:

| Quarantine Reason | Count | Percentage | Action & Justification |
|---|---|---|---|
| `conflicting_prompt_mapping_different_utilities` | 1,499 | 53.71% | Genuinely ambiguous/conflicting instructions where the exact same prompt maps to completely different command utilities across scraped datasets. |
| `privileged_command_requires_root` | 933 | 33.43% | Commands invoking `sudo` or `su`. Quarantined because our `DockerSandbox` enforces unprivileged UID 1000 execution with `ALL` capabilities dropped. |
| `excessive_find_template_repetition` | 156 | 5.59% | Repetitive instances of the top `find` patterns exceeding 10 identical occurrences. Quarantined to balance utility distribution without destroying unique syntax. |
| `unavailable_third_party_tool_*` | 130 | 4.66% | Distro-specific or specialized tools not installed in standard Ubuntu (`qm`, `zpool`, `pvecm`, `samtools`, `zfs`, `b2`, `bw`, `urpmi`). |
| `exact_duplicate_instruction_command_pair` | 73 | 2.62% | Redundant duplicate pairs beyond the single retained canonical instance. |

---

## 4. Command Imbalance & Balancing Strategy

### Analysis of `find`
In the raw training split, `find` accounts for **28.45%** (11,562 commands). Our audit revealed:
1. `find` is **not** an artificial synthetic loop: out of 11,531 `find` commands, there are **10,230 distinct syntactic flag and argument combinations** representing real developer queries (`-type f`, `-exec`, `-mtime`, `-size`, `-perm`, `-maxdepth`, `-delete`, etc.).
2. Blindly deleting `find` down to an arbitrary 10% would eliminate thousands of valid, highly informative Linux search tasks.
3. Instead of random decimation, we applied **structural template deduplication**: capping identical abstract template signatures to a maximum of 10 examples. This pruned 156 redundant instances while retaining 100% of distinct, complex `find` queries.

### Command Distribution: Before vs. After Cleaning (Top 15 Utilities)

| Utility | Raw Count | Raw % | Clean Count | Clean % | Impact / Justification |
|---|---|---|---|---|---|
| `find` | 11,562 | 28.45% | 11,176 | 35.71% | Structural deduplication applied; 10,000+ unique search tasks preserved. |
| `shuf` | 1,259 | 3.10% | 1,244 | 3.97% | Preserved (line shuffling / sampling). |
| `echo` | 806 | 1.98% | 688 | 2.20% | Preserved string/variable output tasks. |
| `git` | 699 | 1.72% | 511 | 1.63% | Preserved local Git commands (`status`, `log`, `diff`, `branch`). |
| `cat` | 443 | 1.09% | 385 | 1.23% | Preserved concatenation and inspection. |
| `grep` | 370 | 0.91% | 343 | 1.10% | Preserved regex searching. |
| `awk` | 302 | 0.74% | 288 | 0.92% | Preserved field/column processing. |
| `aws` | 300 | 0.74% | 254 | 0.81% | Preserved valid CLI invocations. |
| `sort` | 288 | 0.71% | 272 | 0.87% | Preserved sorting and ordering. |
| `find` (subshell) | 278 | 0.68% | 275 | 0.88% | Subshell find evaluations preserved. |
| `tldr` | 245 | 0.60% | 215 | 0.69% | Command documentation queries preserved. |
| `sed` | 237 | 0.58% | 236 | 0.75% | Stream editing and substitution preserved. |
| `ls` | 237 | 0.58% | 224 | 0.72% | Directory listing preserved. |
| `cut` | 222 | 0.55% | 204 | 0.65% | Delimited field extraction preserved. |
| `curl` | 211 | 0.52% | 202 | 0.65% | Clean HTTP transfer tasks preserved with `requires_network=True`. |

---

## 5. Normalized Schema & Split Breakdown

Every clean record in `train.jsonl` and `val.jsonl` adheres to the standardized 9-field schema:

```json
{
  "record_id": "alfa_train_00001",
  "source_row": 23083,
  "instruction": "Find all files with '.db' extension (case insensitive) that belong to user 'example_user'",
  "command": "find . -iname '*.db' -user example_user",
  "command_category": "file_search",
  "source_dataset": "westenfelder/NL2SH-ALFA",
  "requires_network": false,
  "requires_privileges": false,
  "safety_level": "safe"
}
```

### Functional Category Distribution (Clean Retained Corpus: 31,296 records)
* `utility_cli`: 13,875 (44.33%)
* `file_search`: 11,203 (35.80%)
* `text_processing`: 2,175 (6.95%)
* `system_info`: 1,265 (4.04%)
* `network_remote`: 1,123 (3.59%)
* `file_management`: 840 (2.68%)
* `archive_compression`: 295 (0.94%)
* `permissions`: 135 (0.43%)
* `process_management`: 129 (0.41%)
* `shell_control`: 129 (0.41%)
* `file_deletion`: 127 (0.41%)

### Safety Level Distribution
* `safe`: 24,183 (77.27%) — read-only inspection, searching, formatting.
* `file_modify`: 4,576 (14.62%) — file creation, directory making, mode changes, redirection.
* `file_delete`: 1,552 (4.96%) — commands invoking `rm`, `rmdir`, `find ... -delete`.
* `network_restricted`: 909 (2.90%) — outbound network transfer commands tagged with `requires_network=True`.
* `process_kill`: 76 (0.24%) — process signaling commands (`kill`, `killall`).
* `privileged`: 0 (0.00%) — root escalation commands 100% quarantined.

---

## 6. Train / Validation Split & Leakage Prevention

* **Split Ratio:** 90.0% Train (28,167 records) / 10.0% Validation (3,129 records).
* **Partitioning Methodology:** Records were clustered by abstract task pattern (`primary_binary:key_instruction_words`) prior to splitting.
* **Leakage Assertions Passed:**
  * Exact instruction overlap between train and val: **0 prompts (0.00%)**.
  * Overlapping `record_id`s between train and val: **0 records (0.00%)**.
  * Original test set overlap with clean train: **0 contamination**.

---

## 7. Sample Cleaned Records

### Example 1: Complex File Search
```json
{
  "record_id": "alfa_train_00010",
  "source_row": 31448,
  "instruction": "Find all '.jpg' files inside the directory /foo that have a size greater than 2048 bytes",
  "command": "find /foo -name '*.jpg' -size +2048c",
  "command_category": "file_search",
  "source_dataset": "westenfelder/NL2SH-ALFA",
  "requires_network": false,
  "requires_privileges": false,
  "safety_level": "safe"
}
```

### Example 2: Text Processing Pipeline
```json
{
  "record_id": "alfa_train_00045",
  "source_row": 35741,
  "instruction": "Find all files in the current directory whose name contains a number, and copy them to the backup directory.",
  "command": "find . -name '*[0-9]*' | xargs -I{} cp {} ~/backup/",
  "command_category": "file_search",
  "source_dataset": "westenfelder/NL2SH-ALFA",
  "requires_network": false,
  "requires_privileges": false,
  "safety_level": "file_modify"
}
```

### Example 3: Delimited Field Extraction
```json
{
  "record_id": "alfa_val_00012",
  "source_row": 18920,
  "instruction": "Display only the 1st and 3rd fields of a colon-separated file /etc/passwd",
  "command": "cut -d':' -f1,3 /etc/passwd",
  "command_category": "text_processing",
  "source_dataset": "westenfelder/NL2SH-ALFA",
  "requires_network": false,
  "requires_privileges": false,
  "safety_level": "safe"
}
```

---

## 8. Verification Results

All 8 tests in [`tests/test_alfa_preprocessing.py`](file:///C:/Users/LENOVO/OneDrive/Desktop/NL2SH/tests/test_alfa_preprocessing.py) and all 37 tests across the entire NL2SH test suite passed:

```
tests/test_alfa_preprocessing.py::test_raw_dataset_files_exist_and_unaltered PASSED
tests/test_alfa_preprocessing.py::test_processed_files_exist PASSED
tests/test_alfa_preprocessing.py::test_complete_accounting_of_raw_records PASSED
tests/test_alfa_preprocessing.py::test_schema_compliance_of_clean_splits PASSED
tests/test_alfa_preprocessing.py::test_unique_record_ids PASSED
tests/test_alfa_preprocessing.py::test_no_placeholder_leakage_in_clean_splits PASSED
tests/test_alfa_preprocessing.py::test_no_prompt_leakage_between_train_and_val PASSED
tests/test_alfa_preprocessing.py::test_quarantined_and_removed_records_have_reasons PASSED

============================= 37 passed in 31.30s =============================
```

*Raw files remain immutable, no existing codebase files or notebooks were modified, and no training was started.*


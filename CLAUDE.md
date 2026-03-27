# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nb-rust is a Naive Bayes classifier for metagenomic sequence classification using k-mer frequencies. It is a Rust reimplementation of [NBC++](https://github.com/EESI/Naive_Bayes). It supports training models from FASTA files and classifying query sequences against trained (or legacy NBC++) models.

## Build & Test Commands

```bash
cargo build --release          # Release build (binary: target/release/nb-rust)
cargo build                    # Debug build
cargo test                     # All tests (unit + integration)
cargo test <module_name>       # Run tests for a specific module (e.g., kmer, model, fasta)
cargo clippy                   # Lint
```

The integration test (`tests/integration_test.rs`) builds the release binary and classifies ~94K reads against 100 legacy models, validating >95% agreement with reference output. It requires `--test-threads=1` if run in isolation.

## Usage

The binary takes a single argument: a YAML config file.

```bash
./target/release/nb-rust example/classify.yaml
```

Example configs are in `example/` for classify, train, and train-then-classify workflows.

## Architecture

### Data Flow

**Training:** YAML config → scan `source_dir` for per-class subdirectories → parse FASTA/kmr files → parallel k-mer counting (Rayon) → accumulate into `NbClass` models → serialize to `.nbv` files + `meta.nbv`.

**Classification:** YAML config → load `.nbv` or legacy `-save.dat` models from `save_dir` → parse query sequences → parallel scoring via log-likelihood (Kahan summation) → output CSV/TSV/JSON.

### Key Modules

- **`config.rs`** — YAML config parsing with validation and defaults. Two modes: `Train` and `Classify`.
- **`kmer.rs`** — 2-bit packed k-mer encoding, canonical (strand-independent) k-mers, sliding-window counting with `FxHashMap`.
- **`model.rs`** — `NbClass` holds per-class k-mer frequencies. Uses Kahan summation for numerically stable log-likelihood scoring and Laplace smoothing (+1) to avoid zero probabilities.
- **`io/fasta.rs`** — Buffered FASTA parser.
- **`io/kmer_file.rs`** — Reads NBC++ `.kmr` files.
- **`io/serialization.rs`** — Native NBV binary format (raw counts) and legacy NBC++ format (pre-computed logs, classify-only).
- **`io/writer.rs`** — Buffered output in CSV/TSV/JSON Lines; supports best-only and full-result modes.
- **`pipeline/train.rs`** — Training orchestrator. Expects `source_dir/<class_name>/<files>` directory layout.
- **`pipeline/classify.rs`** — Classification orchestrator. Supports memory-bounded multi-round mode (`limit_mb > 0`) for large model sets.

### Design Decisions

- **FxHashMap** (from `rustc-hash`) for k-mer lookups — faster than `HashMap` for `u32` keys.
- **Rayon ThreadPool** with configurable thread count for both training and classification.
- **Two model states:** `Full` (raw counts available, can retrain) vs `ClassifyOnly` (legacy models with only pre-computed logs).
- **Memory management:** `limit_mb` partitions classes into chunks processed in rounds; `max_rows`/`max_cols`/`batch_size` control resource usage.
- All scoring is done in log-space to prevent floating-point underflow.

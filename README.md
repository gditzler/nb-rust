# nb-rust

A Naive Bayes classifier for metagenomic sequence classification using k-mer frequencies, written in Rust.

nb-rust reimplements [NBC++](https://github.com/EESI/Naive_Bayes) with a module-per-concern architecture, parallel train/classify pipelines via [Rayon](https://github.com/rayon-rs/rayon), and a single YAML configuration file.

## Features

- **Train** models from FASTA or pre-computed k-mer files
- **Classify** metagenomic reads against trained models
- **Parallel** training and classification via Rayon thread pools
- **Memory management** with configurable limits (`limit_mb`, `batch_size`, `max_rows`, `max_cols`)
- **Multiple output formats:** CSV, TSV, JSON Lines
- **Legacy compatibility:** reads NBC++ `-save.dat` savefiles
- **Numerically stable:** Kahan summation for log-likelihood accumulation
- **Fast hashing:** FxHashMap for k-mer lookups

## Building

```bash
cargo build --release
```

The optimized binary is written to `target/release/nb-rust`.

## Usage

nb-rust takes a single argument: a YAML configuration file.

```bash
./target/release/nb-rust config.yaml
```

### Configuration

```yaml
version: 1

mode: classify           # "train" or "classify"
kmer_size: 9
save_dir: ./model_dir    # where trained models are saved/loaded
source_dir: ./reads      # training: dir of class subdirs; classify: dir of input files
threads: 4

input:
  extension: .fna        # file extension to look for
  input_type: fasta      # "fasta" or "kmer_file"

memory:
  limit_mb: 0            # 0 = unlimited; >0 = multi-round classification
  batch_size: 0          # 0 = all at once; >0 = flush to disk every N genomes
  max_rows: 0            # 0 = unlimited; >0 = stop after N reads
  max_cols: 0            # 0 = unlimited; >0 = load at most N classes

output:
  format: csv            # "csv", "tsv", or "json"
  prefix: results        # output filename prefix (e.g., results.csv)
  full_result: false     # true = log-likelihoods for all classes per read
  temp_dir: /tmp
```

## Examples

The `example/` directory contains NBC++ training data and reads for testing. Three example workflows are provided.

### 1. Classify against pre-trained NBC++ models

Classify reads against 100 pre-trained NBC++ legacy models (k=9):

```bash
./target/release/nb-rust example/classify.yaml
```

This reads `example/reads/cross.fna` (~94K reads) and scores each against the 100 class models in `example/training_classes/`. Results are written to `example_results.csv`.

Compare output against the expected results:

```bash
diff <(cut -d, -f1,2 example_results.csv | sort) \
     <(cut -d, -f1,2 example/results_max_1.csv | sort)
```

### 2. Train a model from FASTA reads

Train new models from FASTA reads organized by class. Each subdirectory in `source_dir` is treated as one class:

```
example/training_example/
  1748/reads.fna
  1863/reads.fna
  1930276/reads.fna
```

```bash
./target/release/nb-rust example/train.yaml
```

This creates `.nbv` savefiles and a `meta.nbv` index in `example/trained_output/`.

### 3. Train then classify

Train a model, then classify held-out reads against it:

```bash
# Step 1: Train
./target/release/nb-rust example/train.yaml

# Step 2: Classify
./target/release/nb-rust example/classify_trained.yaml
```

Results are written to `classify_trained_results.csv`.

## Testing

```bash
cargo test              # all tests (unit + integration)
cargo test kmer         # single module
cargo test model        # model module
cargo test fasta        # FASTA parser
cargo test classify     # classification pipeline
```

The integration test classifies ~94K reads against 100 legacy NBC++ models and verifies >95% agreement with the reference output.

## Architecture

```
src/
  main.rs              CLI entry point
  config.rs            YAML config parsing and validation
  kmer.rs              k-mer encoding, reverse complement, canonical form, counting
  model.rs             Naive Bayes model, Kahan summation, Laplace smoothing
  io/
    fasta.rs           FASTA parser
    kmer_file.rs       NBC++ .kmr file reader
    serialization.rs   NBV binary format + legacy NBC++ reader
    writer.rs          CSV/TSV/JSON output
  pipeline/
    train.rs           Training orchestrator (single/parallel)
    classify.rs        Classification orchestrator (single/parallel/multi-round)
```

## References

- [NBC++ (EESI/Naive_Bayes)](https://github.com/EESI/Naive_Bayes) -- the original C++ implementation
- Rosen, G., Garbarine, E., Caseiro, D., Polikar, R., & Sokhansanj, B. (2008). Metagenome fragment classification using N-mer frequency profiles. *Advances in Bioinformatics*.

## License

MIT License. See [LICENSE](LICENSE).

# NB-Rust Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the NBV Naive Bayes metagenomic classifier from V to Rust, leveraging Rust's performance (HashMap with FxHash, Rayon for parallelism, zero-copy I/O) to significantly accelerate train and classify pipelines.

**Architecture:** Module-per-concern design mirroring the V original: `kmer` (encoding/counting), `model` (NbClass, Kahan summation), `io` (FASTA/kmer-file/serialization/writer), `pipeline` (train/classify orchestrators), `config` (YAML parsing). Rust-specific wins: `HashMap<u32, u32>` with FxHash instead of V's `map[int]int`, Rayon parallel iterators instead of manual channels/WaitGroup, `BufReader`/`BufWriter` for I/O, and `byteorder` crate for binary serialization.

**Tech Stack:** Rust (2021 edition), serde + serde_yaml (config), rustc-hash (FxHashMap), rayon (parallelism), byteorder (binary I/O)

---

## File Structure

```
nb-rust/
  .gitignore
  Cargo.toml
  src/
    main.rs              # CLI entry point
    config.rs            # YAML config parsing, enums, validation
    kmer.rs              # k-mer encoding, reverse complement, canonical, counting
    model.rs             # NbClass, KahanAccumulator, log-likelihood
    io/
      mod.rs             # re-exports
      fasta.rs           # FASTA parser
      kmer_file.rs       # TSV k-mer file reader
      serialization.rs   # NBV binary format + legacy NBC++ reader
      writer.rs          # CSV/TSV/JSON output writer
    pipeline/
      mod.rs             # re-exports
      train.rs           # Training orchestrator (single + rayon)
      classify.rs        # Classification orchestrator (single + rayon + multi-round)
  tests/
    testdata/
      test.fasta
      test.kmr
      training/
        class_a/genome1.fasta
        class_b/genome2.fasta
```

---

### Task 1: Project Scaffolding and Cargo.toml

**Files:**
- Create: `.gitignore`
- Create: `Cargo.toml`
- Create: `src/main.rs` (placeholder)

- [ ] **Step 1: Create .gitignore**

```
/target
```

- [ ] **Step 2: Create Cargo.toml with dependencies**

```toml
[package]
name = "nb-rust"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1", features = ["derive"] }
serde_yaml = "0.9"
rustc-hash = "2"
rayon = "1"
byteorder = "1"
serde_json = "1"

[profile.release]
opt-level = 3
lto = true
```

- [ ] **Step 3: Create placeholder main.rs**

```rust
fn main() {
    println!("nb-rust placeholder");
}
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo build`
Expected: Compiles successfully

- [ ] **Step 5: Initialize git repo and commit**

```bash
cd /Users/gditzler/git/nb-rust
git init
git add .gitignore Cargo.toml src/main.rs
git commit -m "chore: scaffold Rust project with dependencies"
```

---

### Task 2: K-mer Module

**Files:**
- Create: `src/kmer.rs`

- [ ] **Step 1: Write failing tests for base_to_int, encode, reverse_complement, canonical**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_single_bases() {
        assert_eq!(encode(b"A", 1), 0);
        assert_eq!(encode(b"C", 1), 1);
        assert_eq!(encode(b"G", 1), 2);
        assert_eq!(encode(b"T", 1), 3);
    }

    #[test]
    fn test_encode_kmer() {
        assert_eq!(encode(b"AC", 2), 1);
        assert_eq!(encode(b"GT", 2), 11);
        assert_eq!(encode(b"ACG", 3), 6);
    }

    #[test]
    fn test_reverse_complement() {
        assert_eq!(reverse_complement(0, 1), 3);
        assert_eq!(reverse_complement(1, 1), 2);
        assert_eq!(reverse_complement(1, 2), 11);
        assert_eq!(reverse_complement(11, 2), 1);
    }

    #[test]
    fn test_canonical() {
        assert_eq!(canonical(0, 1), 0);
        assert_eq!(canonical(3, 1), 0);
        assert_eq!(canonical(1, 2), 1);
        assert_eq!(canonical(11, 2), 1);
    }

    #[test]
    fn test_count_from_buffer() {
        let counts = count_from_buffer(b"ACGT", 2);
        let ac_canon = canonical(encode(b"AC", 2), 2);
        assert_eq!(counts[&ac_canon], 2); // AC and its RC GT
        let cg_canon = canonical(encode(b"CG", 2), 2);
        assert_eq!(counts[&cg_canon], 1);
    }

    #[test]
    fn test_count_from_buffer_skips_invalid() {
        let counts = count_from_buffer(b"ACNGT", 2);
        assert_eq!(*counts.get(&1).unwrap_or(&0), 2); // AC and GT are RC
    }

    #[test]
    fn test_count_from_buffer_skips_newlines() {
        let counts = count_from_buffer(b"AC\nGT", 2);
        assert_eq!(*counts.get(&1).unwrap_or(&0), 2);
    }

    #[test]
    fn test_num_canonical_kmers() {
        assert_eq!(num_canonical_kmers(1), 2);
        assert_eq!(num_canonical_kmers(2), 10);
        assert_eq!(num_canonical_kmers(3), 32);
        assert_eq!(num_canonical_kmers(6), 2080);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib kmer`
Expected: FAIL — functions not defined

- [ ] **Step 3: Implement kmer module**

```rust
use rustc_hash::FxHashMap;

fn base_to_int(b: u8) -> i32 {
    match b {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => -1,
    }
}

pub fn encode(kmer: &[u8], k: usize) -> u32 {
    let mut result: u32 = 0;
    for i in 0..k {
        result = result * 4 + base_to_int(kmer[i]) as u32;
    }
    result
}

pub fn reverse_complement(kmer_int: u32, k: usize) -> u32 {
    let mut result: u32 = 0;
    let mut val = kmer_int;
    for _ in 0..k {
        result = result * 4 + (3 - (val & 3));
        val >>= 2;
    }
    result
}

pub fn canonical(kmer_int: u32, k: usize) -> u32 {
    let rc = reverse_complement(kmer_int, k);
    kmer_int.min(rc)
}

pub fn count_from_buffer(buf: &[u8], k: usize) -> FxHashMap<u32, u32> {
    let mut counts = FxHashMap::default();
    let mut window: u32 = 0;
    let mut valid_len: usize = 0;
    let mask: u32 = (1u32 << (2 * k)) - 1;

    for &b in buf {
        let val = base_to_int(b);
        if val < 0 {
            valid_len = 0;
            window = 0;
            continue;
        }
        window = (window * 4 + val as u32) & mask;
        valid_len += 1;
        if valid_len >= k {
            let canon = canonical(window, k);
            *counts.entry(canon).or_insert(0) += 1;
        }
    }
    counts
}

pub fn num_canonical_kmers(k: usize) -> i64 {
    let total: i64 = 1i64 << (2 * k);
    if k % 2 == 1 {
        total / 2
    } else {
        let palindromes: i64 = 1i64 << k;
        (total + palindromes) / 2
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib kmer`
Expected: All 8 tests PASS

- [ ] **Step 5: Add kmer module to main.rs and commit**

Add `mod kmer;` to `src/main.rs`.

```bash
git add src/kmer.rs src/main.rs
git commit -m "feat: add kmer module with encoding, reverse complement, canonical counting"
```

---

### Task 3: Model Module (NbClass + Kahan)

**Files:**
- Create: `src/model.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kahan_add_basic() {
        let mut acc = KahanAccumulator::default();
        acc = kahan_add(acc, 1.0);
        acc = kahan_add(acc, 2.0);
        acc = kahan_add(acc, 3.0);
        assert_eq!(acc.sum, 6.0);
    }

    #[test]
    fn test_kahan_add_precision() {
        let mut acc = KahanAccumulator::default();
        let mut naive: f64 = 0.0;
        acc = kahan_add(acc, 1.0);
        naive += 1.0;
        for _ in 0..10000 {
            acc = kahan_add(acc, 1e-16);
            naive += 1e-16;
        }
        let expected = 1.0 + 10000.0 * 1e-16;
        let kahan_err = (acc.sum - expected).abs();
        let naive_err = (naive - expected).abs();
        assert!(kahan_err <= naive_err);
    }

    #[test]
    fn test_nbclass_new() {
        let cls = NbClass::new("test_class".into(), 6, "/tmp/test.nbv".into());
        assert_eq!(cls.id, "test_class");
        assert_eq!(cls.kmer_size, 6);
        assert_eq!(cls.ngenomes, 0);
        assert_eq!(cls.sumfreq, 2080);
        assert!(matches!(cls.state, LoadState::Unloaded));
    }

    #[test]
    fn test_nbclass_add_genome() {
        let mut cls = NbClass::new("test_class".into(), 2, "/tmp/test.nbv".into());
        let mut kmer_counts = FxHashMap::default();
        kmer_counts.insert(1, 5);
        kmer_counts.insert(6, 3);
        cls.add_genome(&kmer_counts);

        assert_eq!(cls.ngenomes, 1);
        assert_eq!(cls.sumfreq, 10 + 8);
        assert_eq!(cls.freqcnt[&1], 5);
        assert_eq!(cls.freqcnt[&6], 3);
        assert!(matches!(cls.state, LoadState::Full));

        assert!((cls.ngenomes_lg - (1.0_f64).ln()).abs() < 1e-10);
        assert!((cls.sumfreq_lg - (18.0_f64).ln()).abs() < 1e-10);
        assert!((cls.freqcnt_lg[&1] - (6.0_f64).ln()).abs() < 1e-10);
        assert!((cls.freqcnt_lg[&6] - (4.0_f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_nbclass_add_genome_twice() {
        let mut cls = NbClass::new("test_class".into(), 2, "/tmp/test.nbv".into());
        let mut c1 = FxHashMap::default();
        c1.insert(1, 5);
        c1.insert(6, 3);
        cls.add_genome(&c1);
        let mut c2 = FxHashMap::default();
        c2.insert(1, 2);
        c2.insert(9, 1);
        cls.add_genome(&c2);

        assert_eq!(cls.ngenomes, 2);
        assert_eq!(cls.sumfreq, 10 + 8 + 3);
        assert_eq!(cls.freqcnt[&1], 7);
        assert_eq!(cls.freqcnt[&6], 3);
        assert_eq!(cls.freqcnt[&9], 1);
    }

    #[test]
    fn test_get_freq_count_lg_seen() {
        let mut cls = NbClass::new("test_class".into(), 2, "/tmp/test.nbv".into());
        let mut c = FxHashMap::default();
        c.insert(1, 5);
        cls.add_genome(&c);
        assert!((cls.get_freq_count_lg(1) - (6.0_f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_get_freq_count_lg_unseen() {
        let mut cls = NbClass::new("test_class".into(), 2, "/tmp/test.nbv".into());
        let mut c = FxHashMap::default();
        c.insert(1, 5);
        cls.add_genome(&c);
        assert_eq!(cls.get_freq_count_lg(999), 0.0);
    }

    #[test]
    fn test_compute_log_likelihood() {
        let mut cls = NbClass::new("test_class".into(), 2, "/tmp/test.nbv".into());
        let mut train = FxHashMap::default();
        train.insert(1, 5);
        train.insert(6, 3);
        cls.add_genome(&train);

        let mut read = FxHashMap::default();
        read.insert(1, 2);
        read.insert(6, 1);
        let ll = cls.compute_log_likelihood(&read);
        let expected = 2.0 * (6.0_f64).ln() + 1.0 * (4.0_f64).ln() - 3.0 * (18.0_f64).ln();
        assert!((ll - expected).abs() < 1e-10);
    }

    #[test]
    fn test_size_bytes() {
        let mut cls = NbClass::new("test_class".into(), 2, "/tmp/test.nbv".into());
        let mut c = FxHashMap::default();
        c.insert(1, 5);
        c.insert(6, 3);
        cls.add_genome(&c);
        assert!(cls.size_bytes() > 0);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib model`
Expected: FAIL — types not defined

- [ ] **Step 3: Implement model module**

```rust
use rustc_hash::FxHashMap;
use crate::kmer;

#[derive(Debug, Default, Clone, Copy)]
pub struct KahanAccumulator {
    pub sum: f64,
    pub comp: f64,
}

pub fn kahan_add(acc: KahanAccumulator, val: f64) -> KahanAccumulator {
    let y = val - acc.comp;
    let t = acc.sum + y;
    KahanAccumulator {
        sum: t,
        comp: (t - acc.sum) - y,
    }
}

#[derive(Debug, Clone)]
pub enum LoadState {
    Unloaded,
    Full,
    ClassifyOnly,
}

#[derive(Debug, Clone)]
pub struct NbClass {
    pub id: String,
    pub kmer_size: usize,
    pub savefile: String,
    pub ngenomes_lg: f64,
    pub sumfreq_lg: f64,
    pub freqcnt_lg: FxHashMap<u32, f64>,
    pub ngenomes: u32,
    pub sumfreq: i64,
    pub freqcnt: FxHashMap<u32, u32>,
    pub state: LoadState,
}

impl NbClass {
    pub fn new(id: String, kmer_size: usize, savefile: String) -> Self {
        let v = kmer::num_canonical_kmers(kmer_size);
        NbClass {
            id,
            kmer_size,
            savefile,
            ngenomes_lg: 0.0,
            sumfreq_lg: 0.0,
            freqcnt_lg: FxHashMap::default(),
            ngenomes: 0,
            sumfreq: v,
            freqcnt: FxHashMap::default(),
            state: LoadState::Unloaded,
        }
    }

    pub fn add_genome(&mut self, kmer_counts: &FxHashMap<u32, u32>) {
        self.ngenomes += 1;
        let mut total: i64 = 0;
        for (&km, &count) in kmer_counts {
            *self.freqcnt.entry(km).or_insert(0) += count;
            total += count as i64;
        }
        self.sumfreq += total;

        self.ngenomes_lg = (self.ngenomes as f64).ln();
        self.sumfreq_lg = (self.sumfreq as f64).ln();
        for (&km, &count) in &self.freqcnt {
            self.freqcnt_lg.insert(km, ((count + 1) as f64).ln());
        }
        self.state = LoadState::Full;
    }

    pub fn get_freq_count_lg(&self, km: u32) -> f64 {
        self.freqcnt_lg.get(&km).copied().unwrap_or(0.0)
    }

    pub fn compute_log_likelihood(&self, kmer_counts: &FxHashMap<u32, u32>) -> f64 {
        let mut acc = KahanAccumulator::default();
        let mut total_count: i64 = 0;
        for (&km, &freq) in kmer_counts {
            let fcl = self.get_freq_count_lg(km);
            acc = kahan_add(acc, freq as f64 * fcl);
            total_count += freq as i64;
        }
        acc.sum - total_count as f64 * self.sumfreq_lg
    }

    pub fn size_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.freqcnt.len() * (std::mem::size_of::<u32>() + std::mem::size_of::<u32>())
            + self.freqcnt_lg.len() * (std::mem::size_of::<u32>() + std::mem::size_of::<f64>())
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib model`
Expected: All 9 tests PASS

- [ ] **Step 5: Add model module to main.rs and commit**

Add `mod model;` to `src/main.rs`.

```bash
git add src/model.rs src/main.rs
git commit -m "feat: add model module with NbClass, Kahan summation, log-likelihood"
```

---

### Task 4: Config Module

**Files:**
- Create: `src/config.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_yaml(content: &str) -> String {
        let path = format!("/tmp/nb_rust_test_config_{}.yaml", std::process::id());
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn test_load_valid_config() {
        let path = write_yaml("version: 1\nmode: classify\nkmer_size: 9\nsave_dir: ./models\nsource_dir: ./reads\nthreads: 4\ninput:\n  extension: .fna\n  input_type: fasta\nmemory:\n  limit_mb: 0\n  batch_size: 0\n  max_rows: 0\n  max_cols: 0\noutput:\n  format: csv\n  prefix: results\n  full_result: false\n  temp_dir: /tmp\n");
        let cfg = load(&path).unwrap();
        assert!(matches!(cfg.mode, Mode::Classify));
        assert_eq!(cfg.kmer_size, 9);
        assert_eq!(cfg.threads, 4);
        assert!(matches!(cfg.input_type, InputType::Fasta));
        assert!(matches!(cfg.format, OutputFormat::Csv));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_invalid_mode() {
        let path = write_yaml("version: 1\nmode: invalid\nkmer_size: 9\nsave_dir: ./m\nsource_dir: ./r\nthreads: 1\ninput:\n  extension: .fna\n  input_type: fasta\n");
        let result = load(&path);
        assert!(result.is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_kmer_size_out_of_range() {
        let path = write_yaml("version: 1\nmode: train\nkmer_size: 20\nsave_dir: ./m\nsource_dir: ./r\nthreads: 1\ninput:\n  extension: .fna\n  input_type: fasta\n");
        let result = load(&path);
        assert!(result.is_err());
        std::fs::remove_file(&path).ok();
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib config`
Expected: FAIL

- [ ] **Step 3: Implement config module**

```rust
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone)]
pub enum Mode {
    Train,
    Classify,
}

#[derive(Debug, Clone)]
pub enum InputType {
    KmerFile,
    Fasta,
}

#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    Csv,
    Tsv,
    Json,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub version: u32,
    pub mode: Mode,
    pub kmer_size: usize,
    pub save_dir: String,
    pub source_dir: String,
    pub threads: usize,
    pub input_type: InputType,
    pub extension: String,
    pub limit_mb: u64,
    pub batch_size: usize,
    pub max_rows: usize,
    pub max_cols: usize,
    pub format: OutputFormat,
    pub prefix: String,
    pub full_result: bool,
    pub temp_dir: String,
}

#[derive(Deserialize)]
struct RawInputConfig {
    #[serde(default = "default_extension")]
    extension: String,
    #[serde(default = "default_input_type")]
    input_type: String,
}

fn default_extension() -> String { ".kmr".into() }
fn default_input_type() -> String { "kmer_file".into() }

#[derive(Deserialize)]
struct RawMemoryConfig {
    #[serde(default)]
    limit_mb: u64,
    #[serde(default)]
    batch_size: usize,
    #[serde(default = "default_max_rows")]
    max_rows: usize,
    #[serde(default = "default_max_cols")]
    max_cols: usize,
}

fn default_max_rows() -> usize { 450000 }
fn default_max_cols() -> usize { 20000 }

#[derive(Deserialize)]
struct RawOutputConfig {
    #[serde(default = "default_format")]
    format: String,
    #[serde(default = "default_prefix")]
    prefix: String,
    #[serde(default)]
    full_result: bool,
    #[serde(default = "default_temp_dir")]
    temp_dir: String,
}

fn default_format() -> String { "csv".into() }
fn default_prefix() -> String { "log_likelihood".into() }
fn default_temp_dir() -> String { "/tmp".into() }

#[derive(Deserialize)]
struct RawConfig {
    #[serde(default = "default_version")]
    version: u32,
    mode: String,
    #[serde(default = "default_kmer_size")]
    kmer_size: usize,
    #[serde(default = "default_save_dir")]
    save_dir: String,
    #[serde(default)]
    source_dir: String,
    #[serde(default = "default_threads")]
    threads: usize,
    #[serde(default)]
    input: Option<RawInputConfig>,
    #[serde(default)]
    memory: Option<RawMemoryConfig>,
    #[serde(default)]
    output: Option<RawOutputConfig>,
}

fn default_version() -> u32 { 1 }
fn default_kmer_size() -> usize { 6 }
fn default_save_dir() -> String { "./nbv_save".into() }
fn default_threads() -> usize { 1 }

pub fn load(path: &str) -> Result<Config, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read config: {e}"))?;
    let raw: RawConfig = serde_yaml::from_str(&content)
        .map_err(|e| format!("invalid YAML: {e}"))?;

    let mode = match raw.mode.as_str() {
        "train" => Mode::Train,
        "classify" => Mode::Classify,
        other => return Err(format!("invalid mode '{other}': must be 'train' or 'classify'")),
    };

    if raw.kmer_size < 1 || raw.kmer_size > 15 {
        return Err(format!("kmer_size must be between 1 and 15, got {}", raw.kmer_size));
    }
    if raw.source_dir.is_empty() {
        return Err("source_dir is required".into());
    }
    if raw.threads < 1 {
        return Err(format!("threads must be >= 1, got {}", raw.threads));
    }

    let input = raw.input.unwrap_or(RawInputConfig {
        extension: default_extension(),
        input_type: default_input_type(),
    });
    let input_type = match input.input_type.as_str() {
        "kmer_file" => InputType::KmerFile,
        "fasta" => InputType::Fasta,
        other => return Err(format!("invalid input_type '{other}': must be 'kmer_file' or 'fasta'")),
    };

    let output = raw.output.unwrap_or(RawOutputConfig {
        format: default_format(),
        prefix: default_prefix(),
        full_result: false,
        temp_dir: default_temp_dir(),
    });
    let format = match output.format.as_str() {
        "csv" => OutputFormat::Csv,
        "tsv" => OutputFormat::Tsv,
        "json" => OutputFormat::Json,
        other => return Err(format!("invalid output format '{other}': must be 'csv', 'tsv', or 'json'")),
    };

    let memory = raw.memory.unwrap_or(RawMemoryConfig {
        limit_mb: 0,
        batch_size: 0,
        max_rows: default_max_rows(),
        max_cols: default_max_cols(),
    });

    Ok(Config {
        version: raw.version,
        mode,
        kmer_size: raw.kmer_size,
        save_dir: raw.save_dir,
        source_dir: raw.source_dir,
        threads: raw.threads,
        input_type,
        extension: input.extension,
        limit_mb: memory.limit_mb,
        batch_size: memory.batch_size,
        max_rows: memory.max_rows,
        max_cols: memory.max_cols,
        format,
        prefix: output.prefix,
        full_result: output.full_result,
        temp_dir: output.temp_dir,
    })
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib config`
Expected: All 3 tests PASS

- [ ] **Step 5: Add config module to main.rs and commit**

Add `mod config;` to `src/main.rs`.

```bash
git add src/config.rs src/main.rs
git commit -m "feat: add config module with YAML parsing and validation"
```

---

### Task 5: I/O Module — FASTA Parser

**Files:**
- Create: `src/io/mod.rs`
- Create: `src/io/fasta.rs`
- Create: `tests/testdata/test.fasta` (copy from V repo)

- [ ] **Step 1: Create test data file**

Copy `tests/testdata/test.fasta` from V repo:
```
>seq1 description
ACGTACGT
ACGTACGT
>seq2
GGGGCCCC
>seq3 empty after header

>seq4
ACGT
```

- [ ] **Step 2: Write failing tests**

```rust
// in src/io/fasta.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_fasta_basic() {
        let records = read_fasta("tests/testdata/test.fasta").unwrap();
        assert_eq!(records.len(), 4);
        assert_eq!(records[0].header, "seq1 description");
        assert_eq!(std::str::from_utf8(&records[0].sequence).unwrap(), "ACGTACGTACGTACGT");
        assert_eq!(records[1].header, "seq2");
        assert_eq!(std::str::from_utf8(&records[1].sequence).unwrap(), "GGGGCCCC");
        assert_eq!(records[2].header, "seq3 empty after header");
        assert!(records[2].sequence.is_empty());
        assert_eq!(records[3].header, "seq4");
        assert_eq!(std::str::from_utf8(&records[3].sequence).unwrap(), "ACGT");
    }

    #[test]
    fn test_count_sequences() {
        let count = count_sequences("tests/testdata/test.fasta").unwrap();
        assert_eq!(count, 4);
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test --lib io::fasta`
Expected: FAIL

- [ ] **Step 4: Implement FASTA parser**

```rust
use std::io::{BufRead, BufReader};
use std::fs::File;

#[derive(Debug, Clone)]
pub struct FastaRecord {
    pub header: String,
    pub sequence: Vec<u8>,
}

pub fn read_fasta(path: &str) -> Result<Vec<FastaRecord>, String> {
    let file = File::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    let mut current_header = String::new();
    let mut current_seq = Vec::new();
    let mut in_record = false;

    for line in reader.lines() {
        let line = line.map_err(|e| format!("read error: {e}"))?;
        if line.starts_with('>') {
            if in_record {
                records.push(FastaRecord {
                    header: current_header,
                    sequence: current_seq,
                });
            }
            current_header = line[1..].trim().to_string();
            current_seq = Vec::new();
            in_record = true;
        } else if in_record {
            current_seq.extend_from_slice(line.as_bytes());
        }
    }
    if in_record {
        records.push(FastaRecord {
            header: current_header,
            sequence: current_seq,
        });
    }
    Ok(records)
}

pub fn count_sequences(path: &str) -> Result<u64, String> {
    let file = File::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
    let reader = BufReader::new(file);
    let mut count = 0u64;
    for line in reader.lines() {
        let line = line.map_err(|e| format!("read error: {e}"))?;
        if line.starts_with('>') {
            count += 1;
        }
    }
    Ok(count)
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --lib io::fasta`
Expected: All 2 tests PASS

- [ ] **Step 6: Create io/mod.rs, wire up module, and commit**

```rust
// src/io/mod.rs — start with only fasta; other submodules added in later tasks
pub mod fasta;
```

Add `pub mod io;` to `src/main.rs`.

```bash
git add src/io/ tests/testdata/test.fasta src/main.rs
git commit -m "feat: add FASTA parser with buffered I/O"
```

---

### Task 6: I/O Module — K-mer File Reader

**Files:**
- Create: `src/io/kmer_file.rs`
- Create: `tests/testdata/test.kmr` (copy from V repo)

- [ ] **Step 1: Create test data**

Copy `tests/testdata/test.kmr`:
```
ACGTAA	15
CGTAAC	8
TGCAAT	3
```

- [ ] **Step 2: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmer;

    #[test]
    fn test_read_kmer_file() {
        let counts = read_kmer_file("tests/testdata/test.kmr", 6).unwrap();
        assert!(!counts.is_empty());
        let kmer_int = kmer::encode(b"ACGTAA", 6);
        let canon = kmer::canonical(kmer_int, 6);
        assert!(counts[&canon] > 0);
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test --lib io::kmer_file`
Expected: FAIL

- [ ] **Step 4: Implement kmer_file reader**

```rust
use rustc_hash::FxHashMap;
use std::io::{BufRead, BufReader};
use std::fs::File;
use crate::kmer;

pub fn read_kmer_file(path: &str, k: usize) -> Result<FxHashMap<u32, u32>, String> {
    let file = File::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
    let reader = BufReader::new(file);
    let mut counts = FxHashMap::default();

    for (i, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("read error: {e}"))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split('\t').collect();
        if parts.len() != 2 {
            return Err(format!("malformed line {} in {path}: expected <kmer>\\t<count>", i + 1));
        }
        let kmer_str = parts[0];
        let count: u32 = parts[1].parse()
            .map_err(|_| format!("invalid count on line {} in {path}", i + 1))?;
        if kmer_str.len() != k {
            return Err(format!("kmer length mismatch on line {}: expected {k}, got {}", i + 1, kmer_str.len()));
        }
        let kmer_int = kmer::encode(kmer_str.as_bytes(), k);
        let canon = kmer::canonical(kmer_int, k);
        *counts.entry(canon).or_insert(0) += count;
    }
    Ok(counts)
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test --lib io::kmer_file`
Expected: PASS

- [ ] **Step 6: Add `pub mod kmer_file;` to `src/io/mod.rs` and commit**

Add the line `pub mod kmer_file;` to `src/io/mod.rs`.

```bash
git add src/io/kmer_file.rs src/io/mod.rs tests/testdata/test.kmr
git commit -m "feat: add kmer file reader"
```

---

### Task 7: I/O Module — Serialization (NBV Binary + Legacy NBC++)

**Files:**
- Create: `src/io/serialization.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::NbClass;
    use rustc_hash::FxHashMap;

    #[test]
    fn test_save_and_load_roundtrip() {
        let mut cls = NbClass::new("test_class".into(), 6, "/tmp/test.nbv".into());
        let mut counts = FxHashMap::default();
        counts.insert(1, 5);
        counts.insert(6, 3);
        counts.insert(100, 1);
        cls.add_genome(&counts);

        let path = "/tmp/nb_rust_test_roundtrip.nbv";
        save_class(&cls, path).unwrap();
        let loaded = load_class(path).unwrap();

        assert_eq!(loaded.id, cls.id);
        assert_eq!(loaded.kmer_size, cls.kmer_size);
        assert_eq!(loaded.ngenomes, cls.ngenomes);
        assert_eq!(loaded.sumfreq, cls.sumfreq);
        assert_eq!(loaded.freqcnt.len(), cls.freqcnt.len());
        for (k, v) in &cls.freqcnt {
            assert_eq!(loaded.freqcnt[k], *v);
        }
        assert!((loaded.sumfreq_lg - cls.sumfreq_lg).abs() < 1e-10);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_save_and_load_meta() {
        let dir = "/tmp/nb_rust_test_meta_dir";
        save_meta(dir, 9).unwrap();
        let k = load_meta(dir).unwrap();
        assert_eq!(k, 9);
        std::fs::remove_dir_all(dir).ok();
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib io::serialization`
Expected: FAIL

- [ ] **Step 3: Implement serialization**

```rust
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rustc_hash::FxHashMap;
use std::io::{BufReader, BufWriter, Cursor, Read, Write};
use std::fs::File;
use crate::model::{LoadState, NbClass};

const MAGIC: &[u8; 4] = b"NBV1";
const FORMAT_VERSION: u8 = 1;

pub fn save_class(cls: &NbClass, path: &str) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("cannot create {path}: {e}"))?;
    let mut w = BufWriter::new(file);
    w.write_all(MAGIC).map_err(|e| e.to_string())?;
    w.write_u8(FORMAT_VERSION).map_err(|e| e.to_string())?;
    w.write_i32::<LittleEndian>(cls.kmer_size as i32).map_err(|e| e.to_string())?;
    w.write_i32::<LittleEndian>(cls.ngenomes as i32).map_err(|e| e.to_string())?;
    w.write_i64::<LittleEndian>(cls.sumfreq).map_err(|e| e.to_string())?;
    let id_bytes = cls.id.as_bytes();
    w.write_i32::<LittleEndian>(id_bytes.len() as i32).map_err(|e| e.to_string())?;
    w.write_all(id_bytes).map_err(|e| e.to_string())?;
    for (&km, &count) in &cls.freqcnt {
        w.write_i32::<LittleEndian>(km as i32).map_err(|e| e.to_string())?;
        w.write_i32::<LittleEndian>(count as i32).map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}

pub fn load_class(path: &str) -> Result<NbClass, String> {
    let data = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let mut cur = Cursor::new(&data);

    let mut magic = [0u8; 4];
    cur.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != MAGIC {
        return Err(format!("invalid NBV file: bad magic bytes in {path}"));
    }
    let version = cur.read_u8().map_err(|e| e.to_string())?;
    if version != FORMAT_VERSION {
        return Err(format!("unsupported NBV format version {version}"));
    }
    let kmer_size = cur.read_i32::<LittleEndian>().map_err(|e| e.to_string())? as usize;
    let ngenomes = cur.read_i32::<LittleEndian>().map_err(|e| e.to_string())? as u32;
    let sumfreq = cur.read_i64::<LittleEndian>().map_err(|e| e.to_string())?;
    let id_len = cur.read_i32::<LittleEndian>().map_err(|e| e.to_string())? as usize;
    let mut id_buf = vec![0u8; id_len];
    cur.read_exact(&mut id_buf).map_err(|e| e.to_string())?;
    let id = String::from_utf8(id_buf).map_err(|e| e.to_string())?;

    let mut freqcnt = FxHashMap::default();
    let mut freqcnt_lg = FxHashMap::default();
    while let (Ok(km), Ok(count)) = (
        cur.read_i32::<LittleEndian>(),
        cur.read_i32::<LittleEndian>(),
    ) {
        let km = km as u32;
        let count = count as u32;
        freqcnt.insert(km, count);
        freqcnt_lg.insert(km, ((count + 1) as f64).ln());
    }

    Ok(NbClass {
        id,
        kmer_size,
        savefile: path.to_string(),
        ngenomes,
        sumfreq,
        ngenomes_lg: (ngenomes as f64).ln(),
        sumfreq_lg: (sumfreq as f64).ln(),
        freqcnt,
        freqcnt_lg,
        state: LoadState::Full,
    })
}

pub fn load_legacy_class(path: &str, k: usize) -> Result<NbClass, String> {
    let data = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    if data.len() < 20 {
        return Err(format!("legacy save file too small: {path}"));
    }
    let mut cur = Cursor::new(&data);
    let ngenomes_lg = cur.read_f64::<LittleEndian>().map_err(|e| e.to_string())?;
    let sumfreq_lg = cur.read_f64::<LittleEndian>().map_err(|e| e.to_string())?;
    let n_entries = cur.read_i32::<LittleEndian>().map_err(|e| e.to_string())?;

    let expected_remaining = n_entries as usize * 12;
    let pos = cur.position() as usize;
    if data.len() - pos < expected_remaining {
        return Err(format!("legacy save file truncated: {path}"));
    }

    let mut freqcnt_lg = FxHashMap::default();
    for _ in 0..n_entries {
        let km = cur.read_i32::<LittleEndian>().map_err(|e| e.to_string())? as u32;
        let val = cur.read_f64::<LittleEndian>().map_err(|e| e.to_string())?;
        freqcnt_lg.insert(km, val);
    }

    let basename = std::path::Path::new(path)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .replace("-save.dat", "");

    Ok(NbClass {
        id: basename,
        kmer_size: k,
        savefile: path.to_string(),
        ngenomes_lg,
        sumfreq_lg,
        freqcnt_lg,
        ngenomes: 0,
        sumfreq: 0,
        freqcnt: FxHashMap::default(),
        state: LoadState::ClassifyOnly,
    })
}

pub fn save_meta(save_dir: &str, kmer_size: usize) -> Result<(), String> {
    std::fs::create_dir_all(save_dir).map_err(|e| e.to_string())?;
    let path = format!("{save_dir}/meta.nbv");
    std::fs::write(&path, kmer_size.to_string()).map_err(|e| e.to_string())
}

pub fn load_meta(save_dir: &str) -> Result<usize, String> {
    let path = format!("{save_dir}/meta.nbv");
    let content = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    content.trim().parse().map_err(|e: std::num::ParseIntError| e.to_string())
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib io::serialization`
Expected: All 2 tests PASS

- [ ] **Step 5: Add `pub mod serialization;` to `src/io/mod.rs` and commit**

Add the line `pub mod serialization;` to `src/io/mod.rs`.

```bash
git add src/io/serialization.rs src/io/mod.rs
git commit -m "feat: add binary serialization (NBV format + legacy NBC++ reader)"
```

---

### Task 8: I/O Module — Writer

**Files:**
- Create: `src/io/writer.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_writer_csv() {
        let path = "/tmp/nb_rust_test_output.csv";
        {
            let mut w = Writer::new(path, OutputFormat::Csv, false).unwrap();
            w.write_result("seq1", "class_a", -123.45).unwrap();
            w.write_result("seq2", "class_b", -678.90).unwrap();
        }
        let content = std::fs::read_to_string(path).unwrap();
        let lines: Vec<&str> = content.trim().lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("seq1"));
        assert!(lines[0].contains("class_a"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_writer_json() {
        let path = "/tmp/nb_rust_test_output.jsonl";
        {
            let mut w = Writer::new(path, OutputFormat::Json, false).unwrap();
            w.write_result("seq1", "class_a", -123.45).unwrap();
        }
        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.contains("\"seq_id\""));
        assert!(content.contains("\"best_class\""));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_writer_no_valid_kmers() {
        let path = "/tmp/nb_rust_test_nokmers.csv";
        {
            let mut w = Writer::new(path, OutputFormat::Csv, false).unwrap();
            w.write_no_valid_kmers("bad_read").unwrap();
        }
        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.contains("sequence contains no valid kmers"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_output_filename() {
        assert_eq!(output_filename("results", &OutputFormat::Csv), "results.csv");
        assert_eq!(output_filename("results", &OutputFormat::Tsv), "results.tsv");
        assert_eq!(output_filename("results", &OutputFormat::Json), "results.jsonl");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib io::writer`
Expected: FAIL

- [ ] **Step 3: Implement writer**

```rust
use std::io::{BufWriter, Write};
use std::fs::File;
use crate::config::OutputFormat;
use rustc_hash::FxHashMap;

pub struct Writer {
    writer: BufWriter<File>,
    format: OutputFormat,
    full_result: bool,
}

impl Writer {
    pub fn new(path: &str, format: OutputFormat, full_result: bool) -> Result<Self, String> {
        let file = File::create(path).map_err(|e| format!("cannot create {path}: {e}"))?;
        Ok(Writer {
            writer: BufWriter::new(file),
            format,
            full_result,
        })
    }

    pub fn write_header(&mut self, class_ids: &[String]) -> Result<(), String> {
        match self.format {
            OutputFormat::Csv => writeln!(self.writer, "seq_id,{}", class_ids.join(",")),
            OutputFormat::Tsv => writeln!(self.writer, "seq_id\t{}", class_ids.join("\t")),
            OutputFormat::Json => return Ok(()),
        }.map_err(|e| e.to_string())
    }

    pub fn write_result(&mut self, seq_id: &str, best_class: &str, score: f64) -> Result<(), String> {
        match self.format {
            OutputFormat::Csv => writeln!(self.writer, "{seq_id},{best_class},{score}"),
            OutputFormat::Tsv => writeln!(self.writer, "{seq_id}\t{best_class}\t{score}"),
            OutputFormat::Json => writeln!(self.writer, "{{\"seq_id\":\"{seq_id}\",\"best_class\":\"{best_class}\",\"score\":{score}}}"),
        }.map_err(|e| e.to_string())
    }

    pub fn write_full_result(&mut self, seq_id: &str, scores: &FxHashMap<String, f64>, class_order: &[String]) -> Result<(), String> {
        match self.format {
            OutputFormat::Csv => {
                let vals: Vec<String> = class_order.iter().map(|c| format!("{}", scores.get(c).unwrap_or(&0.0))).collect();
                writeln!(self.writer, "{seq_id},{}", vals.join(","))
            }
            OutputFormat::Tsv => {
                let vals: Vec<String> = class_order.iter().map(|c| format!("{}", scores.get(c).unwrap_or(&0.0))).collect();
                writeln!(self.writer, "{seq_id}\t{}", vals.join("\t"))
            }
            OutputFormat::Json => {
                let parts: Vec<String> = class_order.iter().map(|c| format!("\"{}\":{}", c, scores.get(c).unwrap_or(&0.0))).collect();
                writeln!(self.writer, "{{\"seq_id\":\"{seq_id}\",\"scores\":{{{}}}}}", parts.join(","))
            }
        }.map_err(|e| e.to_string())
    }

    pub fn write_no_valid_kmers(&mut self, seq_id: &str) -> Result<(), String> {
        match self.format {
            OutputFormat::Csv => writeln!(self.writer, "{seq_id},sequence contains no valid kmers,"),
            OutputFormat::Tsv => writeln!(self.writer, "{seq_id}\tsequence contains no valid kmers\t"),
            OutputFormat::Json => writeln!(self.writer, "{{\"seq_id\":\"{seq_id}\",\"best_class\":\"sequence contains no valid kmers\",\"score\":null}}"),
        }.map_err(|e| e.to_string())
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        let _ = self.writer.flush();
    }
}

pub fn output_filename(prefix: &str, format: &OutputFormat) -> String {
    let ext = match format {
        OutputFormat::Csv => "csv",
        OutputFormat::Tsv => "tsv",
        OutputFormat::Json => "jsonl",
    };
    format!("{prefix}.{ext}")
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib io::writer`
Expected: All 4 tests PASS

- [ ] **Step 5: Add `pub mod writer;` to `src/io/mod.rs` and commit**

Add the line `pub mod writer;` to `src/io/mod.rs`.

```bash
git add src/io/writer.rs src/io/mod.rs
git commit -m "feat: add output writer (CSV/TSV/JSON Lines)"
```

---

### Task 9: Pipeline Module — Training

**Files:**
- Create: `src/pipeline/mod.rs`
- Create: `src/pipeline/train.rs`
- Create: `tests/testdata/training/class_a/genome1.fasta` (copy)
- Create: `tests/testdata/training/class_b/genome2.fasta` (copy)

- [ ] **Step 1: Create test training data**

Copy from V repo:
- `tests/testdata/training/class_a/genome1.fasta`
- `tests/testdata/training/class_b/genome2.fasta`

- [ ] **Step 2: Write failing tests**

```rust
// in src/pipeline/train.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;

    fn train_config(save_dir: &str, threads: usize, batch_size: usize) -> Config {
        Config {
            version: 1,
            mode: Mode::Train,
            kmer_size: 4,
            save_dir: save_dir.into(),
            source_dir: "tests/testdata/training".into(),
            threads,
            input_type: InputType::Fasta,
            extension: ".fasta".into(),
            limit_mb: 0,
            batch_size,
            max_rows: 0,
            max_cols: 0,
            format: OutputFormat::Csv,
            prefix: "".into(),
            full_result: false,
            temp_dir: "/tmp".into(),
        }
    }

    #[test]
    fn test_train_creates_savefiles() {
        let dir = "/tmp/nb_rust_test_train";
        let _ = std::fs::remove_dir_all(dir);
        let c = train_config(dir, 1, 0);
        train(&c).unwrap();
        assert!(std::path::Path::new(&format!("{dir}/class_a.nbv")).exists());
        assert!(std::path::Path::new(&format!("{dir}/class_b.nbv")).exists());
        assert!(std::path::Path::new(&format!("{dir}/meta.nbv")).exists());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_train_multithreaded() {
        let dir = "/tmp/nb_rust_test_train_mt";
        let _ = std::fs::remove_dir_all(dir);
        let c = train_config(dir, 2, 0);
        train(&c).unwrap();
        assert!(std::path::Path::new(&format!("{dir}/class_a.nbv")).exists());
        assert!(std::path::Path::new(&format!("{dir}/class_b.nbv")).exists());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_train_with_batch_size() {
        let dir = "/tmp/nb_rust_test_train_batch";
        let _ = std::fs::remove_dir_all(dir);
        let c = train_config(dir, 1, 1);
        train(&c).unwrap();
        assert!(std::path::Path::new(&format!("{dir}/class_a.nbv")).exists());
        assert!(std::path::Path::new(&format!("{dir}/class_b.nbv")).exists());
        std::fs::remove_dir_all(dir).ok();
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test --lib pipeline::train`
Expected: FAIL

- [ ] **Step 4: Implement training pipeline**

The training pipeline should use Rayon's parallel iterators instead of manual channels/WaitGroup. Key functions:

- `scan_training_dir(source_dir, extension)` — enumerate per-class subdirectories
- `load_kmer_counts(path, input_type, k)` — dispatch to FASTA or kmer_file reader
- `train(config)` — orchestrate: scan, count k-mers (in parallel via Rayon), accumulate into NbClass models, save

```rust
use std::collections::HashMap;
use rustc_hash::FxHashMap;
use rayon::prelude::*;
use crate::config::{Config, InputType, Mode};
use crate::model::NbClass;
use crate::io::{fasta, kmer_file, serialization};
use crate::kmer;

struct TrainJob {
    class_id: String,
    path: String,
}

struct TrainResult {
    class_id: String,
    kmer_counts: FxHashMap<u32, u32>,
}

fn scan_training_dir(source_dir: &str, extension: &str) -> Result<Vec<TrainJob>, String> {
    let mut jobs = Vec::new();
    let entries = std::fs::read_dir(source_dir)
        .map_err(|e| format!("cannot read {source_dir}: {e}"))?;
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if !path.is_dir() { continue; }
        let class_id = path.file_name().unwrap().to_string_lossy().to_string();
        let files = std::fs::read_dir(&path).map_err(|e| e.to_string())?;
        for file in files {
            let file = file.map_err(|e| e.to_string())?;
            let fname = file.file_name().to_string_lossy().to_string();
            if fname.ends_with(extension) {
                jobs.push(TrainJob {
                    class_id: class_id.clone(),
                    path: file.path().to_string_lossy().to_string(),
                });
            }
        }
    }
    Ok(jobs)
}

fn load_kmer_counts(path: &str, input_type: &InputType, k: usize) -> Result<FxHashMap<u32, u32>, String> {
    match input_type {
        InputType::Fasta => {
            let records = fasta::read_fasta(path)?;
            let mut merged = FxHashMap::default();
            for record in &records {
                let counts = kmer::count_from_buffer(&record.sequence, k);
                for (km, count) in counts {
                    *merged.entry(km).or_insert(0) += count;
                }
            }
            Ok(merged)
        }
        InputType::KmerFile => kmer_file::read_kmer_file(path, k),
    }
}

pub fn train(c: &Config) -> Result<(), String> {
    // TODO: batch_size checkpointing not yet implemented.
    // V version saves intermediate models to disk every batch_size files
    // and resumes from saved state. This is deferred for the initial port.
    let jobs = scan_training_dir(&c.source_dir, &c.extension)?;
    if jobs.is_empty() {
        return Err(format!("no training files found in {}", c.source_dir));
    }

    let results: Vec<TrainResult> = if c.threads > 1 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(c.threads)
            .build()
            .map_err(|e| e.to_string())?;
        pool.install(|| {
            jobs.par_iter()
                .filter_map(|job| {
                    load_kmer_counts(&job.path, &c.input_type, c.kmer_size)
                        .ok()
                        .map(|counts| TrainResult {
                            class_id: job.class_id.clone(),
                            kmer_counts: counts,
                        })
                })
                .collect()
        })
    } else {
        jobs.iter()
            .filter_map(|job| {
                load_kmer_counts(&job.path, &c.input_type, c.kmer_size)
                    .ok()
                    .map(|counts| TrainResult {
                        class_id: job.class_id.clone(),
                        kmer_counts: counts,
                    })
            })
            .collect()
    };

    let mut classes: HashMap<String, NbClass> = HashMap::new();
    for result in &results {
        let cls = classes.entry(result.class_id.clone()).or_insert_with(|| {
            let savefile = format!("{}/{}.nbv", c.save_dir, result.class_id);
            NbClass::new(result.class_id.clone(), c.kmer_size, savefile)
        });
        cls.add_genome(&result.kmer_counts);
    }

    std::fs::create_dir_all(&c.save_dir).map_err(|e| e.to_string())?;
    for cls in classes.values() {
        serialization::save_class(cls, &cls.savefile)?;
    }
    serialization::save_meta(&c.save_dir, c.kmer_size)?;
    Ok(())
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --lib pipeline::train`
Expected: All 3 tests PASS

- [ ] **Step 6: Create pipeline/mod.rs, wire up, and commit**

```rust
// src/pipeline/mod.rs — start with only train; classify added in Task 10
pub mod train;
```

Add `pub mod pipeline;` to `src/main.rs`.

```bash
git add src/pipeline/ tests/testdata/training/
git commit -m "feat: add training pipeline with Rayon parallelism"
```

---

### Task 10: Pipeline Module — Classification

**Files:**
- Create: `src/pipeline/classify.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use crate::pipeline::train;

    fn train_then_get_dir(dir: &str) -> String {
        let _ = std::fs::remove_dir_all(dir);
        let c = Config {
            version: 1, mode: Mode::Train, kmer_size: 4,
            save_dir: dir.into(), source_dir: "tests/testdata/training".into(),
            threads: 1, input_type: InputType::Fasta, extension: ".fasta".into(),
            limit_mb: 0, batch_size: 0, max_rows: 0, max_cols: 0,
            format: OutputFormat::Csv, prefix: "".into(),
            full_result: false, temp_dir: "/tmp".into(),
        };
        train::train(&c).unwrap();
        dir.into()
    }

    #[test]
    fn test_train_then_classify() {
        let train_dir = "/tmp/nb_rust_test_e2e_train";
        train_then_get_dir(train_dir);
        let reads_dir = "/tmp/nb_rust_test_e2e_reads";
        let _ = std::fs::create_dir_all(reads_dir);
        std::fs::write(format!("{reads_dir}/test_read.fasta"), ">read1\nACGTACGTACGTACGT\n").unwrap();

        let c = Config {
            version: 1, mode: Mode::Classify, kmer_size: 4,
            save_dir: train_dir.into(), source_dir: reads_dir.into(),
            threads: 1, input_type: InputType::Fasta, extension: ".fasta".into(),
            limit_mb: 0, batch_size: 0, max_rows: 1000, max_cols: 100,
            format: OutputFormat::Csv, prefix: "/tmp/nb_rust_test_e2e_output".into(),
            full_result: false, temp_dir: "/tmp".into(),
        };
        classify(&c).unwrap();
        let output = std::fs::read_to_string("/tmp/nb_rust_test_e2e_output.csv").unwrap();
        assert!(output.contains("read1"));
        std::fs::remove_dir_all(train_dir).ok();
        std::fs::remove_dir_all(reads_dir).ok();
        std::fs::remove_file("/tmp/nb_rust_test_e2e_output.csv").ok();
    }

    #[test]
    fn test_classify_multithreaded() {
        let train_dir = "/tmp/nb_rust_test_e2e_mt_train";
        train_then_get_dir(train_dir);
        let reads_dir = "/tmp/nb_rust_test_e2e_mt_reads";
        let _ = std::fs::create_dir_all(reads_dir);
        std::fs::write(format!("{reads_dir}/test.fasta"), ">r1\nACGTACGTACGTACGT\n>r2\nGGGGCCCCAAAATTTT\n").unwrap();

        let c = Config {
            version: 1, mode: Mode::Classify, kmer_size: 4,
            save_dir: train_dir.into(), source_dir: reads_dir.into(),
            threads: 2, input_type: InputType::Fasta, extension: ".fasta".into(),
            limit_mb: 0, batch_size: 0, max_rows: 0, max_cols: 0,
            format: OutputFormat::Csv, prefix: "/tmp/nb_rust_test_e2e_mt_out".into(),
            full_result: false, temp_dir: "/tmp".into(),
        };
        classify(&c).unwrap();
        let output = std::fs::read_to_string("/tmp/nb_rust_test_e2e_mt_out.csv").unwrap();
        assert!(output.contains("r1"));
        assert!(output.contains("r2"));
        std::fs::remove_dir_all(train_dir).ok();
        std::fs::remove_dir_all(reads_dir).ok();
        std::fs::remove_file("/tmp/nb_rust_test_e2e_mt_out.csv").ok();
    }

    #[test]
    fn test_classify_with_max_rows() {
        let train_dir = "/tmp/nb_rust_test_maxrows_train";
        train_then_get_dir(train_dir);
        let reads_dir = "/tmp/nb_rust_test_maxrows_reads";
        let _ = std::fs::create_dir_all(reads_dir);
        std::fs::write(format!("{reads_dir}/test.fasta"), ">r1\nACGTACGTACGTACGT\n>r2\nGGGGCCCCAAAATTTT\n>r3\nAAAACCCCGGGGTTTT\n").unwrap();

        let c = Config {
            version: 1, mode: Mode::Classify, kmer_size: 4,
            save_dir: train_dir.into(), source_dir: reads_dir.into(),
            threads: 1, input_type: InputType::Fasta, extension: ".fasta".into(),
            limit_mb: 0, batch_size: 0, max_rows: 2, max_cols: 0,
            format: OutputFormat::Csv, prefix: "/tmp/nb_rust_test_maxrows_out".into(),
            full_result: false, temp_dir: "/tmp".into(),
        };
        classify(&c).unwrap();
        let output = std::fs::read_to_string("/tmp/nb_rust_test_maxrows_out.csv").unwrap();
        let lines: Vec<&str> = output.trim().lines().collect();
        assert_eq!(lines.len(), 2);
        std::fs::remove_dir_all(train_dir).ok();
        std::fs::remove_dir_all(reads_dir).ok();
        std::fs::remove_file("/tmp/nb_rust_test_maxrows_out.csv").ok();
    }

    #[test]
    fn test_classify_with_limit_mb() {
        let train_dir = "/tmp/nb_rust_test_limitmb_train";
        train_then_get_dir(train_dir);
        let reads_dir = "/tmp/nb_rust_test_limitmb_reads";
        let _ = std::fs::create_dir_all(reads_dir);
        std::fs::write(format!("{reads_dir}/test.fasta"), ">r1\nACGTACGTACGTACGT\n").unwrap();

        let c = Config {
            version: 1, mode: Mode::Classify, kmer_size: 4,
            save_dir: train_dir.into(), source_dir: reads_dir.into(),
            threads: 1, input_type: InputType::Fasta, extension: ".fasta".into(),
            limit_mb: 1, batch_size: 0, max_rows: 0, max_cols: 0,
            format: OutputFormat::Csv, prefix: "/tmp/nb_rust_test_limitmb_out".into(),
            full_result: false, temp_dir: "/tmp".into(),
        };
        classify(&c).unwrap();
        let output = std::fs::read_to_string("/tmp/nb_rust_test_limitmb_out.csv").unwrap();
        assert!(output.contains("r1"));
        std::fs::remove_dir_all(train_dir).ok();
        std::fs::remove_dir_all(reads_dir).ok();
        std::fs::remove_file("/tmp/nb_rust_test_limitmb_out.csv").ok();
    }

    #[test]
    fn test_classify_with_max_cols() {
        let train_dir = "/tmp/nb_rust_test_maxcols_train";
        train_then_get_dir(train_dir);
        let reads_dir = "/tmp/nb_rust_test_maxcols_reads";
        let _ = std::fs::create_dir_all(reads_dir);
        std::fs::write(format!("{reads_dir}/test.fasta"), ">r1\nACGTACGTACGTACGT\n").unwrap();

        let c = Config {
            version: 1, mode: Mode::Classify, kmer_size: 4,
            save_dir: train_dir.into(), source_dir: reads_dir.into(),
            threads: 1, input_type: InputType::Fasta, extension: ".fasta".into(),
            limit_mb: 0, batch_size: 0, max_rows: 0, max_cols: 1,
            format: OutputFormat::Csv, prefix: "/tmp/nb_rust_test_maxcols_out".into(),
            full_result: false, temp_dir: "/tmp".into(),
        };
        classify(&c).unwrap();
        let output = std::fs::read_to_string("/tmp/nb_rust_test_maxcols_out.csv").unwrap();
        assert!(output.contains("r1"));
        let lines: Vec<&str> = output.trim().lines().collect();
        assert_eq!(lines.len(), 1);
        std::fs::remove_dir_all(train_dir).ok();
        std::fs::remove_dir_all(reads_dir).ok();
        std::fs::remove_file("/tmp/nb_rust_test_maxcols_out.csv").ok();
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib pipeline::classify`
Expected: FAIL

- [ ] **Step 3: Implement classification pipeline**

The classify pipeline uses Rayon for parallel read scoring. Key functions:

- `discover_class_paths(save_dir, max_cols)` — find .nbv and -save.dat files
- `load_all_classes(paths, kmer_size)` — load NbClass models
- `classify_read(kmer_counts, classes)` — score one read against all classes
- `classify(config)` — orchestrate: load classes, read input, score (with Rayon), write output

Implementation should mirror V version's three modes: single-threaded, multi-threaded (Rayon `par_iter`), and multi-round (memory-bounded). The multi-threaded path replaces V's channels+WaitGroup with `rayon::par_iter().map()`.

```rust
use std::collections::HashMap;
use rustc_hash::FxHashMap;
use rayon::prelude::*;
use crate::config::{Config, InputType, OutputFormat};
use crate::model::NbClass;
use crate::io::{fasta, kmer_file, serialization, writer};
use crate::kmer;

struct ClassPath {
    path: String,
    legacy: bool,
}

struct SeqJob {
    seq_id: String,
    kmer_counts: FxHashMap<u32, u32>,
    index: usize,
}

struct ClassifyResult {
    seq_id: String,
    best_class: String,
    best_score: f64,
    all_scores: FxHashMap<String, f64>,
    index: usize,
}

fn discover_class_paths(save_dir: &str, max_cols: usize) -> Result<Vec<ClassPath>, String> {
    let mut paths = Vec::new();
    let entries = std::fs::read_dir(save_dir).map_err(|e| e.to_string())?;
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name().to_string_lossy().to_string();
        let path = entry.path().to_string_lossy().to_string();
        if max_cols > 0 && paths.len() >= max_cols { break; }
        if name.ends_with(".nbv") && name != "meta.nbv" {
            paths.push(ClassPath { path, legacy: false });
        } else if name.ends_with("-save.dat") {
            paths.push(ClassPath { path, legacy: true });
        }
    }
    Ok(paths)
}

fn load_all_classes(paths: &[ClassPath], kmer_size: usize) -> Result<Vec<NbClass>, String> {
    let mut classes = Vec::with_capacity(paths.len());
    for cp in paths {
        let cls = if cp.legacy {
            serialization::load_legacy_class(&cp.path, kmer_size)?
        } else {
            serialization::load_class(&cp.path)?
        };
        classes.push(cls);
    }
    Ok(classes)
}

fn classify_read(seq_id: &str, kmer_counts: &FxHashMap<u32, u32>, classes: &[NbClass], full_result: bool, index: usize) -> ClassifyResult {
    let mut best_class = String::new();
    let mut best_score = f64::NEG_INFINITY;
    let mut all_scores = FxHashMap::default();
    for cls in classes {
        let score = cls.compute_log_likelihood(kmer_counts);
        if full_result {
            all_scores.insert(cls.id.clone(), score);
        }
        if score > best_score {
            best_score = score;
            best_class = cls.id.clone();
        }
    }
    ClassifyResult { seq_id: seq_id.to_string(), best_class, best_score, all_scores, index }
}

fn find_input_files(source_dir: &str, extension: &str) -> Result<Vec<String>, String> {
    let mut files = Vec::new();
    let entries = std::fs::read_dir(source_dir).map_err(|e| e.to_string())?;
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(extension) {
            files.push(entry.path().to_string_lossy().to_string());
        }
    }
    if files.is_empty() {
        return Err(format!("no input files with extension {extension} found in {source_dir}"));
    }
    Ok(files)
}

fn build_seq_jobs(c: &Config, input_files: &[String]) -> Result<(Vec<SeqJob>, Vec<(usize, String)>), String> {
    let mut jobs = Vec::new();
    let mut no_kmer_entries = Vec::new();
    let mut seq_index = 0usize;

    for input_file in input_files {
        match c.input_type {
            InputType::Fasta => {
                let records = fasta::read_fasta(input_file)?;
                for record in &records {
                    if c.max_rows > 0 && seq_index >= c.max_rows { break; }
                    let seq_id = record.header.split_whitespace().next().unwrap_or("").to_string();
                    let kmer_counts = kmer::count_from_buffer(&record.sequence, c.kmer_size);
                    if kmer_counts.is_empty() {
                        no_kmer_entries.push((seq_index, seq_id));
                    } else {
                        jobs.push(SeqJob { seq_id, kmer_counts, index: seq_index });
                    }
                    seq_index += 1;
                }
            }
            InputType::KmerFile => {
                if c.max_rows > 0 && seq_index >= c.max_rows { break; }
                let kmer_counts = kmer_file::read_kmer_file(input_file, c.kmer_size)?;
                let seq_id = std::path::Path::new(input_file)
                    .file_name().unwrap_or_default()
                    .to_string_lossy()
                    .replace(&c.extension, "");
                if kmer_counts.is_empty() {
                    no_kmer_entries.push((seq_index, seq_id));
                } else {
                    jobs.push(SeqJob { seq_id, kmer_counts, index: seq_index });
                }
                seq_index += 1;
            }
        }
        if c.max_rows > 0 && seq_index >= c.max_rows { break; }
    }
    Ok((jobs, no_kmer_entries))
}

pub fn classify(c: &Config) -> Result<(), String> {
    // Verify kmer size if meta.nbv exists
    let meta_path = format!("{}/meta.nbv", c.save_dir);
    if std::path::Path::new(&meta_path).exists() {
        let trained_k = serialization::load_meta(&c.save_dir)?;
        if trained_k != c.kmer_size {
            return Err(format!("kmer_size mismatch: config has {}, training used {trained_k}", c.kmer_size));
        }
    }

    let all_class_paths = discover_class_paths(&c.save_dir, c.max_cols)?;
    if all_class_paths.is_empty() {
        return Err(format!("no class savefiles found in {}", c.save_dir));
    }

    let input_files = find_input_files(&c.source_dir, &c.extension)?;
    let limit_bytes = c.limit_mb as i64 * 1024 * 1024;

    if limit_bytes <= 0 {
        let classes = load_all_classes(&all_class_paths, c.kmer_size)?;
        let (jobs, no_kmer_entries) = build_seq_jobs(c, &input_files)?;
        let total_seqs = jobs.len() + no_kmer_entries.len();

        let results: Vec<ClassifyResult> = if c.threads > 1 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(c.threads)
                .build()
                .map_err(|e| e.to_string())?;
            pool.install(|| {
                jobs.par_iter()
                    .map(|job| classify_read(&job.seq_id, &job.kmer_counts, &classes, c.full_result, job.index))
                    .collect()
            })
        } else {
            jobs.iter()
                .map(|job| classify_read(&job.seq_id, &job.kmer_counts, &classes, c.full_result, job.index))
                .collect()
        };

        // Write output in original order
        let class_ids: Vec<String> = classes.iter().map(|c| c.id.clone()).collect();
        let mut result_map: HashMap<usize, &ClassifyResult> = HashMap::new();
        for r in &results {
            result_map.insert(r.index, r);
        }
        let no_kmer_map: HashMap<usize, &str> = no_kmer_entries.iter().map(|(i, s)| (*i, s.as_str())).collect();

        let output_path = writer::output_filename(&c.prefix, &c.format);
        let mut w = writer::Writer::new(&output_path, c.format, c.full_result)?;

        if c.full_result {
            w.write_header(&class_ids)?;
        }

        for i in 0..total_seqs {
            if let Some(seq_id) = no_kmer_map.get(&i) {
                w.write_no_valid_kmers(seq_id)?;
            } else if let Some(r) = result_map.get(&i) {
                if c.full_result {
                    w.write_full_result(&r.seq_id, &r.all_scores, &class_ids)?;
                } else {
                    w.write_result(&r.seq_id, &r.best_class, r.best_score)?;
                }
            }
        }
    } else {
        // Multi-round classification — full_result not supported in this mode
        if c.full_result {
            return Err("full_result is not supported with limit_mb > 0 (multi-round mode)".into());
        }
        let (jobs, no_kmer_entries) = build_seq_jobs(c, &input_files)?;
        let chunks = split_classes_by_memory(&all_class_paths, limit_bytes)?;

        let mut best_results: HashMap<String, (String, f64)> = HashMap::new();

        for chunk in &chunks {
            let classes = load_all_classes(chunk, c.kmer_size)?;
            for job in &jobs {
                for cls in &classes {
                    let score = cls.compute_log_likelihood(&job.kmer_counts);
                    let entry = best_results.entry(job.seq_id.clone()).or_insert_with(|| (String::new(), f64::NEG_INFINITY));
                    if score > entry.1 {
                        *entry = (cls.id.clone(), score);
                    }
                }
            }
        }

        let output_path = writer::output_filename(&c.prefix, &c.format);
        let mut w = writer::Writer::new(&output_path, c.format, c.full_result)?;

        for (_, seq_id) in &no_kmer_entries {
            w.write_no_valid_kmers(seq_id)?;
        }
        for job in &jobs {
            if let Some((best_class, best_score)) = best_results.get(&job.seq_id) {
                w.write_result(&job.seq_id, best_class, *best_score)?;
            }
        }
    }

    Ok(())
}

fn split_classes_by_memory(paths: &[ClassPath], limit_bytes: i64) -> Result<Vec<Vec<ClassPath>>, String> {
    let mut chunks: Vec<Vec<ClassPath>> = Vec::new();
    let mut current_chunk = Vec::new();
    let mut current_bytes: i64 = 0;

    for cp in paths {
        let file_size = std::fs::metadata(&cp.path).map(|m| m.len()).unwrap_or(0) as i64;
        let estimated_mem = file_size * 3;

        if !current_chunk.is_empty() && current_bytes + estimated_mem > limit_bytes {
            chunks.push(current_chunk);
            current_chunk = Vec::new();
            current_bytes = 0;
        }
        current_chunk.push(ClassPath { path: cp.path.clone(), legacy: cp.legacy });
        current_bytes += estimated_mem;
    }
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }
    Ok(chunks)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib pipeline::classify`
Expected: All 5 tests PASS

- [ ] **Step 5: Add `pub mod classify;` to `src/pipeline/mod.rs` and commit**

Add the line `pub mod classify;` to `src/pipeline/mod.rs`.

```bash
git add src/pipeline/classify.rs src/pipeline/mod.rs
git commit -m "feat: add classification pipeline with Rayon parallelism and multi-round support"
```

---

### Task 11: CLI Entry Point

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Implement main.rs**

```rust
mod config;
mod kmer;
mod model;
mod io;
mod pipeline;

use std::env;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 1 {
        eprintln!("Usage: nb-rust <config.yaml>");
        std::process::exit(1);
    }

    let cfg = match config::load(&args[0]) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    match cfg.mode {
        config::Mode::Train => {
            if let Err(e) = pipeline::train::train(&cfg) {
                eprintln!("Training failed: {e}");
                std::process::exit(1);
            }
            println!("Training complete. Savefiles written to {}", cfg.save_dir);
        }
        config::Mode::Classify => {
            if let Err(e) = pipeline::classify::classify(&cfg) {
                eprintln!("Classification failed: {e}");
                std::process::exit(1);
            }
            println!("Classification complete.");
        }
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo build`
Expected: Compiles

- [ ] **Step 3: Commit**

```bash
git add src/main.rs
git commit -m "feat: wire up CLI entry point"
```

---

### Task 12: Integration Test — Legacy NBC++ Savefiles

**Files:**
- Create: `tests/integration_test.rs` (or add to pipeline tests)

This test requires the `example/` directory from the V repo. It validates that the Rust port produces the same classification results as the V version against 100 pre-trained NBC++ models.

- [ ] **Step 1: Copy example data into the project**

```bash
cp -r /tmp/nb-v-ref/example /Users/gditzler/git/nb-rust/example
```

- [ ] **Step 2: Write integration test**

```rust
// tests/integration_test.rs
use std::collections::HashMap;

#[test]
fn test_classify_with_legacy_savefiles() {
    // Build binary first
    let status = std::process::Command::new("cargo")
        .args(["build", "--release"])
        .status()
        .unwrap();
    assert!(status.success());

    // Write a classify config
    let config_yaml = r#"
version: 1
mode: classify
kmer_size: 9
save_dir: example/training_classes
source_dir: example/reads
threads: 1
input:
  extension: .fna
  input_type: fasta
output:
  format: csv
  prefix: /tmp/nb_rust_legacy_test
  full_result: false
  temp_dir: /tmp
memory:
  limit_mb: 0
  max_rows: 0
  max_cols: 0
"#;
    std::fs::write("/tmp/nb_rust_legacy_test.yaml", config_yaml).unwrap();

    let output = std::process::Command::new("./target/release/nb-rust")
        .arg("/tmp/nb_rust_legacy_test.yaml")
        .output()
        .unwrap();
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));

    let result = std::fs::read_to_string("/tmp/nb_rust_legacy_test.csv").unwrap();
    let expected = std::fs::read_to_string("example/results_max_1.csv").unwrap();

    let mut expected_map: HashMap<String, String> = HashMap::new();
    for line in expected.trim().lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            expected_map.insert(parts[0].to_string(), parts[1].to_string());
        }
    }

    let mut matches = 0;
    let mut total = 0;
    for line in result.trim().lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            if let Some(expected_class) = expected_map.get(parts[0]) {
                total += 1;
                if expected_class == parts[1] {
                    matches += 1;
                }
            }
        }
    }

    assert!(total > 0, "no matching seq_ids found");
    let match_pct = matches as f64 / total as f64;
    assert!(match_pct > 0.95, "match rate {match_pct:.2} is below 95%");

    std::fs::remove_file("/tmp/nb_rust_legacy_test.csv").ok();
    std::fs::remove_file("/tmp/nb_rust_legacy_test.yaml").ok();
}
```

- [ ] **Step 3: Run integration test**

Run: `cargo test --test integration_test -- --nocapture`
Expected: PASS with > 95% match rate

- [ ] **Step 4: Commit**

```bash
git add tests/integration_test.rs example/
git commit -m "test: add legacy NBC++ integration test with example data"
```

---

### Task 13: Release Build and Final Verification

- [ ] **Step 1: Run all tests**

Run: `cargo test`
Expected: All unit and integration tests PASS

- [ ] **Step 2: Build optimized release binary**

Run: `cargo build --release`
Expected: Compiles with LTO

- [ ] **Step 3: Quick smoke test**

Write a minimal classify config that uses the legacy savefiles in `example/training_classes/`:

```bash
cat > /tmp/nb_rust_smoke_config.yaml << 'EOF'
version: 1
mode: classify
kmer_size: 9
save_dir: example/training_classes
source_dir: example/reads
threads: 1
input:
  extension: .fna
  input_type: fasta
output:
  format: csv
  prefix: /tmp/nb_rust_smoke
  full_result: false
memory:
  limit_mb: 0
  max_rows: 10
  max_cols: 5
EOF
./target/release/nb-rust /tmp/nb_rust_smoke_config.yaml
```
Expected: "Classification complete." and `/tmp/nb_rust_smoke.csv` contains results

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final verification — all tests pass, release build works"
```

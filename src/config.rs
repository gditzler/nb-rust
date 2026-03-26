//! YAML configuration parsing and validation for the nb-rust pipeline.
//!
//! The config file controls whether to train or classify, which input format
//! to use, parallelism settings, memory limits, and output formatting. See
//! the README for a full example YAML config.

use serde::Deserialize;

/// Whether to train new models or classify query sequences.
#[derive(Debug, Clone)]
pub enum Mode {
    Train,
    Classify,
}

/// Input file format: pre-computed k-mer counts or raw FASTA sequences.
#[derive(Debug, Clone)]
pub enum InputType {
    KmerFile,
    Fasta,
}

/// Output format for classification results.
#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    Csv,
    Tsv,
    Json,
}

/// Validated configuration for the nb-rust pipeline.
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

/// Load and validate a YAML config file, applying defaults for optional fields.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_yaml(content: &str) -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = format!("/tmp/nb_rust_test_config_{}_{}.yaml", std::process::id(), n);
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

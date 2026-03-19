use std::fs;
use std::path::Path;
use rustc_hash::FxHashMap;
use rayon::prelude::*;

use crate::config::{Config, InputType};
use crate::io::{fasta, serialization, writer};
use crate::kmer;
use crate::model::NbClass;
use crate::pipeline::train::load_kmer_counts;

pub(crate) struct ClassPath {
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

/// Discover class save files in save_dir. Returns up to max_cols paths.
pub(crate) fn discover_class_paths(save_dir: &str, max_cols: usize) -> Result<Vec<ClassPath>, String> {
    let mut paths: Vec<ClassPath> = Vec::new();

    let entries = fs::read_dir(save_dir)
        .map_err(|e| format!("cannot read save_dir '{}': {}", save_dir, e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("read_dir error: {}", e))?;
        let p = entry.path();
        if !p.is_file() {
            continue;
        }
        let name = match p.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        if name == "meta.nbv" {
            continue;
        }

        if name.ends_with(".nbv") {
            paths.push(ClassPath {
                path: p.to_string_lossy().to_string(),
                legacy: false,
            });
        } else if name.ends_with("-save.dat") {
            paths.push(ClassPath {
                path: p.to_string_lossy().to_string(),
                legacy: true,
            });
        }
    }

    paths.sort_by(|a, b| a.path.cmp(&b.path));

    if max_cols > 0 && paths.len() > max_cols {
        paths.truncate(max_cols);
    }

    Ok(paths)
}

/// Load all NbClass models from the given paths.
pub(crate) fn load_all_classes(paths: &[ClassPath], kmer_size: usize) -> Result<Vec<NbClass>, String> {
    let mut classes = Vec::new();
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

/// Classify a single read against all loaded classes.
fn classify_read(
    seq_id: &str,
    kmer_counts: &FxHashMap<u32, u32>,
    classes: &[NbClass],
    full_result: bool,
    index: usize,
) -> ClassifyResult {
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

    ClassifyResult {
        seq_id: seq_id.to_string(),
        best_class,
        best_score,
        all_scores,
        index,
    }
}

/// Find input files in source_dir matching extension.
pub fn find_input_files(source_dir: &str, extension: &str) -> Result<Vec<String>, String> {
    let mut files = Vec::new();
    let entries = fs::read_dir(source_dir)
        .map_err(|e| format!("cannot read source_dir '{}': {}", source_dir, e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("read_dir error: {}", e))?;
        let p = entry.path();
        if p.is_file() {
            if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                if name.ends_with(extension) {
                    files.push(p.to_string_lossy().to_string());
                }
            }
        }
    }

    files.sort();
    Ok(files)
}

/// Build SeqJob list from input files. Returns (jobs, no_kmer_seq_ids).
fn build_seq_jobs(
    config: &Config,
    input_files: &[String],
) -> Result<(Vec<SeqJob>, Vec<String>), String> {
    let mut jobs = Vec::new();
    let mut no_kmer_ids = Vec::new();
    let mut index = 0usize;

    for file_path in input_files {
        match config.input_type {
            InputType::Fasta => {
                let records = fasta::read_fasta(file_path)?;
                for rec in records {
                    let counts = kmer::count_from_buffer(&rec.sequence, config.kmer_size);
                    if counts.is_empty() {
                        no_kmer_ids.push(rec.header.clone());
                    } else {
                        jobs.push(SeqJob {
                            seq_id: rec.header,
                            kmer_counts: counts,
                            index,
                        });
                        index += 1;
                    }

                    if config.max_rows > 0 && index >= config.max_rows {
                        return Ok((jobs, no_kmer_ids));
                    }
                }
            }
            InputType::KmerFile => {
                let counts = load_kmer_counts(file_path, &config.input_type, config.kmer_size)?;
                let stem = Path::new(file_path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                if counts.is_empty() {
                    no_kmer_ids.push(stem);
                } else {
                    jobs.push(SeqJob {
                        seq_id: stem,
                        kmer_counts: counts,
                        index,
                    });
                    index += 1;
                }

                if config.max_rows > 0 && index >= config.max_rows {
                    return Ok((jobs, no_kmer_ids));
                }
            }
        }
    }

    Ok((jobs, no_kmer_ids))
}

/// Split class paths into chunks by estimated memory usage.
fn split_classes_by_memory(paths: &[ClassPath], limit_bytes: u64) -> Vec<Vec<usize>> {
    let mut chunks: Vec<Vec<usize>> = Vec::new();
    let mut current_chunk: Vec<usize> = Vec::new();
    let mut current_size: u64 = 0;

    for (i, cp) in paths.iter().enumerate() {
        let file_size = fs::metadata(&cp.path).map(|m| m.len()).unwrap_or(0);
        let estimated = file_size * 3;

        if !current_chunk.is_empty() && current_size + estimated > limit_bytes {
            chunks.push(current_chunk);
            current_chunk = Vec::new();
            current_size = 0;
        }

        current_chunk.push(i);
        current_size += estimated;
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    chunks
}

/// Run the classification pipeline.
pub fn classify(config: &Config) -> Result<(), String> {
    // Check meta.nbv kmer_size match if exists
    let meta_path = format!("{}/meta.nbv", config.save_dir);
    if Path::new(&meta_path).exists() {
        let meta_k = serialization::load_meta(&config.save_dir)?;
        if meta_k != config.kmer_size {
            return Err(format!(
                "kmer_size mismatch: config={} but meta.nbv={}",
                config.kmer_size, meta_k
            ));
        }
    }

    let class_paths = discover_class_paths(&config.save_dir, config.max_cols)?;
    if class_paths.is_empty() {
        return Err("no class files found in save_dir".into());
    }

    // Find input files
    let input_files = find_input_files(&config.source_dir, &config.extension)?;
    if input_files.is_empty() {
        return Err("no input files found".into());
    }

    // Build seq jobs
    let (jobs, no_kmer_ids) = build_seq_jobs(config, &input_files)?;

    let output_path = writer::output_filename(&config.prefix, &config.format);

    if config.limit_mb <= 0 {
        // Standard mode: load all classes at once
        let classes = load_all_classes(&class_paths, config.kmer_size)?;

        // Classify
        let mut results: Vec<ClassifyResult> = if config.threads > 1 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(config.threads)
                .build()
                .map_err(|e| format!("cannot create thread pool: {}", e))?;

            pool.install(|| {
                jobs.par_iter()
                    .map(|job| {
                        classify_read(
                            &job.seq_id,
                            &job.kmer_counts,
                            &classes,
                            config.full_result,
                            job.index,
                        )
                    })
                    .collect()
            })
        } else {
            jobs.iter()
                .map(|job| {
                    classify_read(
                        &job.seq_id,
                        &job.kmer_counts,
                        &classes,
                        config.full_result,
                        job.index,
                    )
                })
                .collect()
        };

        // Sort by original index
        results.sort_by_key(|r| r.index);

        // Write output
        let mut w = writer::Writer::new(&output_path, config.format, config.full_result)?;

        if config.full_result {
            let class_ids: Vec<String> = classes.iter().map(|c| c.id.clone()).collect();
            w.write_header(&class_ids);
            for r in &results {
                w.write_full_result(&r.seq_id, &r.all_scores, &class_ids);
            }
        } else {
            for r in &results {
                w.write_result(&r.seq_id, &r.best_class, r.best_score);
            }
        }

        for id in &no_kmer_ids {
            w.write_no_valid_kmers(id);
        }
    } else {
        // Multi-round mode
        if config.full_result {
            return Err("full_result not supported with limit_mb > 0".into());
        }

        let limit_bytes = config.limit_mb * 1024 * 1024;
        let chunks = split_classes_by_memory(&class_paths, limit_bytes);

        // Track best per read
        let mut best_classes: Vec<String> = vec![String::new(); jobs.len()];
        let mut best_scores: Vec<f64> = vec![f64::NEG_INFINITY; jobs.len()];

        for chunk_indices in &chunks {
            let chunk_paths: Vec<&ClassPath> =
                chunk_indices.iter().map(|&i| &class_paths[i]).collect();
            let chunk_cp: Vec<ClassPath> = chunk_paths
                .iter()
                .map(|cp| ClassPath {
                    path: cp.path.clone(),
                    legacy: cp.legacy,
                })
                .collect();
            let classes = load_all_classes(&chunk_cp, config.kmer_size)?;

            for job in &jobs {
                let result = classify_read(
                    &job.seq_id,
                    &job.kmer_counts,
                    &classes,
                    false,
                    job.index,
                );
                if result.best_score > best_scores[job.index] {
                    best_scores[job.index] = result.best_score;
                    best_classes[job.index] = result.best_class;
                }
            }
        }

        // Write output
        let mut w = writer::Writer::new(&output_path, config.format, false)?;
        for job in &jobs {
            w.write_result(
                &job.seq_id,
                &best_classes[job.index],
                best_scores[job.index],
            );
        }
        for id in &no_kmer_ids {
            w.write_no_valid_kmers(id);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, Mode, InputType, OutputFormat};
    use crate::pipeline::train;

    fn train_config(save_dir: &str) -> Config {
        Config {
            version: 1,
            mode: Mode::Train,
            kmer_size: 4,
            save_dir: save_dir.to_string(),
            source_dir: "tests/testdata/training".to_string(),
            threads: 1,
            input_type: InputType::Fasta,
            extension: ".fasta".to_string(),
            limit_mb: 0,
            batch_size: 0,
            max_rows: 0,
            max_cols: 0,
            format: OutputFormat::Csv,
            prefix: "log_likelihood".to_string(),
            full_result: false,
            temp_dir: "/tmp".to_string(),
        }
    }

    fn classify_config(save_dir: &str, source_dir: &str) -> Config {
        Config {
            version: 1,
            mode: Mode::Classify,
            kmer_size: 4,
            save_dir: save_dir.to_string(),
            source_dir: source_dir.to_string(),
            threads: 1,
            input_type: InputType::Fasta,
            extension: ".fasta".to_string(),
            limit_mb: 0,
            batch_size: 0,
            max_rows: 0,
            max_cols: 0,
            format: OutputFormat::Csv,
            prefix: format!("{}/log_likelihood", save_dir),
            full_result: false,
            temp_dir: "/tmp".to_string(),
        }
    }

    fn setup_and_train(suffix: &str) -> String {
        let save_dir = format!(
            "/tmp/nb_rust_test_classify_{}_{}",
            std::process::id(),
            suffix
        );
        let _ = fs::remove_dir_all(&save_dir);
        let tcfg = train_config(&save_dir);
        train::train(&tcfg).unwrap();
        save_dir
    }

    #[test]
    fn test_train_then_classify() {
        let save_dir = setup_and_train("basic");
        let cfg = classify_config(&save_dir, "tests/testdata/training/class_a");
        classify(&cfg).unwrap();

        let output_path = format!("{}/log_likelihood.csv", save_dir);
        let content = fs::read_to_string(&output_path).unwrap();
        assert!(
            content.contains("genome1_seq1"),
            "output should contain read ID: {}",
            content
        );
        fs::remove_dir_all(&save_dir).ok();
    }

    #[test]
    fn test_classify_multithreaded() {
        let save_dir = setup_and_train("mt");
        let mut cfg = classify_config(&save_dir, "tests/testdata/training/class_a");
        cfg.threads = 2;
        classify(&cfg).unwrap();

        let output_path = format!("{}/log_likelihood.csv", save_dir);
        let content = fs::read_to_string(&output_path).unwrap();
        let data_lines: Vec<&str> = content.lines().collect();
        assert!(data_lines.len() >= 2, "expected at least 2 reads, got {}", data_lines.len());
        fs::remove_dir_all(&save_dir).ok();
    }

    #[test]
    fn test_classify_with_max_rows() {
        let save_dir = setup_and_train("maxrows");
        let mut cfg = classify_config(&save_dir, "tests/testdata/training/class_a");
        cfg.max_rows = 2;
        classify(&cfg).unwrap();

        let output_path = format!("{}/log_likelihood.csv", save_dir);
        let content = fs::read_to_string(&output_path).unwrap();
        let data_lines: Vec<&str> = content.lines().collect();
        assert_eq!(data_lines.len(), 2, "expected 2 data lines, got {}", data_lines.len());
        fs::remove_dir_all(&save_dir).ok();
    }

    #[test]
    fn test_classify_with_limit_mb() {
        let save_dir = setup_and_train("limitmb");
        let mut cfg = classify_config(&save_dir, "tests/testdata/training/class_a");
        cfg.limit_mb = 1;
        classify(&cfg).unwrap();

        let output_path = format!("{}/log_likelihood.csv", save_dir);
        assert!(Path::new(&output_path).exists());
        fs::remove_dir_all(&save_dir).ok();
    }

    #[test]
    fn test_classify_with_max_cols() {
        let save_dir = setup_and_train("maxcols");
        let mut cfg = classify_config(&save_dir, "tests/testdata/training/class_a");
        cfg.max_cols = 1;
        classify(&cfg).unwrap();

        let output_path = format!("{}/log_likelihood.csv", save_dir);
        let content = fs::read_to_string(&output_path).unwrap();
        let data_lines: Vec<&str> = content.lines().skip(1).collect();
        assert!(
            data_lines.len() >= 1,
            "expected at least 1 output line, got {}",
            data_lines.len()
        );
        fs::remove_dir_all(&save_dir).ok();
    }
}

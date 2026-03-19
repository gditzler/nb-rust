use std::collections::HashMap;
use std::fs;
use std::path::Path;
use rustc_hash::FxHashMap;
use rayon::prelude::*;

use crate::config::{Config, InputType};
use crate::io::fasta;
use crate::io::kmer_file;
use crate::io::serialization;
use crate::kmer;
use crate::model::NbClass;

/// Scan the training directory for per-class subdirectories containing files
/// with the given extension. Returns a map of class_id -> Vec<file_path>.
pub fn scan_training_dir(
    source_dir: &str,
    extension: &str,
) -> Result<HashMap<String, Vec<String>>, String> {
    let mut classes: HashMap<String, Vec<String>> = HashMap::new();

    let entries = fs::read_dir(source_dir)
        .map_err(|e| format!("cannot read source_dir '{}': {}", source_dir, e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("read_dir error: {}", e))?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let class_id = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| "invalid directory name".to_string())?
            .to_string();

        let sub_entries = fs::read_dir(&path)
            .map_err(|e| format!("cannot read class dir '{}': {}", class_id, e))?;

        let mut files = Vec::new();
        for sub_entry in sub_entries {
            let sub_entry = sub_entry.map_err(|e| format!("read_dir error: {}", e))?;
            let file_path = sub_entry.path();
            if file_path.is_file() {
                if let Some(name) = file_path.file_name().and_then(|n| n.to_str()) {
                    if name.ends_with(extension) {
                        files.push(file_path.to_string_lossy().to_string());
                    }
                }
            }
        }

        if !files.is_empty() {
            files.sort();
            classes.insert(class_id, files);
        }
    }

    Ok(classes)
}

/// Load kmer counts from a file, dispatching based on input type.
pub fn load_kmer_counts(
    path: &str,
    input_type: &InputType,
    k: usize,
) -> Result<FxHashMap<u32, u32>, String> {
    match input_type {
        InputType::Fasta => {
            let records = fasta::read_fasta(path)?;
            let mut all_seq: Vec<u8> = Vec::new();
            for rec in &records {
                all_seq.extend_from_slice(&rec.sequence);
            }
            Ok(kmer::count_from_buffer(&all_seq, k))
        }
        InputType::KmerFile => kmer_file::read_kmer_file(path, k),
    }
}

/// Run the training pipeline.
pub fn train(config: &Config) -> Result<(), String> {
    let class_files = scan_training_dir(&config.source_dir, &config.extension)?;
    if class_files.is_empty() {
        return Err("no training classes found".into());
    }

    fs::create_dir_all(&config.save_dir)
        .map_err(|e| format!("cannot create save_dir: {}", e))?;

    // Collect all (class_id, file_path) pairs for parallel processing
    let mut work_items: Vec<(String, String)> = Vec::new();
    for (class_id, files) in &class_files {
        for file in files {
            work_items.push((class_id.clone(), file.clone()));
        }
    }

    // Count kmers (possibly in parallel)
    let kmer_results: Vec<(String, FxHashMap<u32, u32>)> = if config.threads > 1 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.threads)
            .build()
            .map_err(|e| format!("cannot create thread pool: {}", e))?;

        pool.install(|| {
            work_items
                .par_iter()
                .map(|(class_id, file_path)| {
                    let counts = load_kmer_counts(file_path, &config.input_type, config.kmer_size)
                        .expect("failed to load kmer counts");
                    (class_id.clone(), counts)
                })
                .collect()
        })
    } else {
        work_items
            .iter()
            .map(|(class_id, file_path)| {
                let counts = load_kmer_counts(file_path, &config.input_type, config.kmer_size)?;
                Ok((class_id.clone(), counts))
            })
            .collect::<Result<Vec<_>, String>>()?
    };

    // Accumulate into NbClass models
    let mut models: HashMap<String, NbClass> = HashMap::new();
    for (class_id, counts) in kmer_results {
        let model = models.entry(class_id.clone()).or_insert_with(|| {
            let savefile = format!(
                "{}/{}.nbv",
                config.save_dir,
                class_id
            );
            NbClass::new(&class_id, config.kmer_size, &savefile)
        });
        model.add_genome(&counts);
    }

    // Save models to disk
    for (_class_id, model) in &models {
        serialization::save_class(model, &model.savefile);
    }

    // Write meta.nbv
    serialization::save_meta(&config.save_dir, config.kmer_size);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, Mode, InputType, OutputFormat};

    fn test_config(threads: usize, batch_size: usize) -> (Config, String) {
        let save_dir = format!(
            "/tmp/nb_rust_test_train_{}_{}_{}",
            std::process::id(),
            threads,
            batch_size
        );
        let _ = fs::remove_dir_all(&save_dir);
        let cfg = Config {
            version: 1,
            mode: Mode::Train,
            kmer_size: 4,
            save_dir: save_dir.clone(),
            source_dir: "tests/testdata/training".to_string(),
            threads,
            input_type: InputType::Fasta,
            extension: ".fasta".to_string(),
            limit_mb: 0,
            batch_size,
            max_rows: 0,
            max_cols: 0,
            format: OutputFormat::Csv,
            prefix: "test".to_string(),
            full_result: false,
            temp_dir: "/tmp".to_string(),
        };
        (cfg, save_dir)
    }

    #[test]
    fn test_train_creates_savefiles() {
        let (cfg, save_dir) = test_config(1, 0);
        train(&cfg).unwrap();

        let class_a_path = format!("{}/class_a.nbv", save_dir);
        let class_b_path = format!("{}/class_b.nbv", save_dir);
        let meta_path = format!("{}/meta.nbv", save_dir);

        assert!(Path::new(&class_a_path).exists(), "class_a.nbv missing");
        assert!(Path::new(&class_b_path).exists(), "class_b.nbv missing");
        assert!(Path::new(&meta_path).exists(), "meta.nbv missing");

        fs::remove_dir_all(&save_dir).ok();
    }

    #[test]
    fn test_train_multithreaded() {
        let (cfg, save_dir) = test_config(2, 0);
        train(&cfg).unwrap();

        let class_a_path = format!("{}/class_a.nbv", save_dir);
        let class_b_path = format!("{}/class_b.nbv", save_dir);

        assert!(Path::new(&class_a_path).exists(), "class_a.nbv missing");
        assert!(Path::new(&class_b_path).exists(), "class_b.nbv missing");

        fs::remove_dir_all(&save_dir).ok();
    }

    #[test]
    fn test_train_with_batch_size() {
        let (cfg, save_dir) = test_config(1, 1);
        train(&cfg).unwrap();

        let class_a_path = format!("{}/class_a.nbv", save_dir);
        assert!(Path::new(&class_a_path).exists(), "class_a.nbv missing");

        fs::remove_dir_all(&save_dir).ok();
    }
}

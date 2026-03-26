//! nb-rust — Naive Bayes classifier for metagenomic sequence classification.
//!
//! This is a Rust reimplementation of NBC++ that classifies DNA sequences using
//! k-mer frequency profiles. It supports both training new models from FASTA or
//! pre-computed k-mer files and classifying query sequences against trained models.

mod kmer;
mod model;
mod config;
pub mod io;
pub mod pipeline;

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

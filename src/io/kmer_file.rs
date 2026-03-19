use std::fs::File;
use std::io::{BufRead, BufReader};
use rustc_hash::FxHashMap;
use crate::kmer::{encode, canonical};

pub fn read_kmer_file(path: &str, k: usize) -> Result<FxHashMap<u32, u32>, String> {
    let file = File::open(path).map_err(|e| format!("cannot open kmer file: {e}"))?;
    let reader = BufReader::new(file);
    let mut counts: FxHashMap<u32, u32> = FxHashMap::default();

    for line in reader.lines() {
        let line = line.map_err(|e| format!("read error: {e}"))?;
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() != 2 {
            return Err(format!("invalid kmer file line: {line}"));
        }
        let kmer_str = parts[0];
        if kmer_str.len() != k {
            return Err(format!(
                "kmer length mismatch: expected {k}, got {} for '{kmer_str}'",
                kmer_str.len()
            ));
        }
        let count: u32 = parts[1]
            .parse()
            .map_err(|e| format!("invalid count '{}': {e}", parts[1]))?;

        let encoded = encode(kmer_str.as_bytes(), k);
        let canon = canonical(encoded, k);
        *counts.entry(canon).or_insert(0) += count;
    }

    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmer::{encode, canonical};

    #[test]
    fn test_read_kmer_file() {
        let counts = read_kmer_file("tests/testdata/test.kmr", 6).unwrap();
        assert!(!counts.is_empty());

        // Verify a known k-mer is present: ACGTAA
        let encoded = encode(b"ACGTAA", 6);
        let canon = canonical(encoded, 6);
        assert!(counts.contains_key(&canon), "expected canonical kmer for ACGTAA");
    }
}

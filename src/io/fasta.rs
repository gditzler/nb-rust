//! FASTA sequence file parser.
//!
//! Reads standard FASTA format where each record starts with a `>` header line
//! followed by one or more lines of sequence data. Multiline sequences are
//! concatenated into a single byte buffer per record.

use std::fs::File;
use std::io::{BufRead, BufReader};

/// A single FASTA record: header (without the leading '>') and raw sequence bytes.
#[derive(Debug)]
pub struct FastaRecord {
    pub header: String,
    pub sequence: Vec<u8>,
}

/// Parse an entire FASTA file into memory as a vector of records.
pub fn read_fasta(path: &str) -> Result<Vec<FastaRecord>, String> {
    let file = File::open(path).map_err(|e| format!("cannot open FASTA file: {e}"))?;
    let reader = BufReader::new(file);
    let mut records: Vec<FastaRecord> = Vec::new();
    let mut current_header: Option<String> = None;
    let mut current_seq: Vec<u8> = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|e| format!("read error: {e}"))?;
        if line.starts_with('>') {
            if let Some(header) = current_header.take() {
                records.push(FastaRecord {
                    header,
                    sequence: current_seq.clone(),
                });
                current_seq.clear();
            }
            current_header = Some(line[1..].to_string());
        } else if current_header.is_some() {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                current_seq.extend_from_slice(trimmed.as_bytes());
            }
        }
    }

    if let Some(header) = current_header.take() {
        records.push(FastaRecord {
            header,
            sequence: current_seq,
        });
    }

    Ok(records)
}

/// Count the number of sequences in a FASTA file by counting '>' header lines.
pub fn count_sequences(path: &str) -> Result<u64, String> {
    let file = File::open(path).map_err(|e| format!("cannot open FASTA file: {e}"))?;
    let reader = BufReader::new(file);
    let mut count: u64 = 0;
    for line in reader.lines() {
        let line = line.map_err(|e| format!("read error: {e}"))?;
        if line.starts_with('>') {
            count += 1;
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_fasta_basic() {
        let records = read_fasta("tests/testdata/test.fasta").unwrap();
        assert_eq!(records.len(), 4);

        assert_eq!(records[0].header, "seq1 description");
        assert_eq!(records[0].sequence, b"ACGTACGTACGTACGT");

        assert_eq!(records[1].header, "seq2");
        assert_eq!(records[1].sequence, b"GGGGCCCC");

        assert_eq!(records[2].header, "seq3 empty after header");
        assert!(records[2].sequence.is_empty());

        assert_eq!(records[3].header, "seq4");
        assert_eq!(records[3].sequence, b"ACGT");
    }

    #[test]
    fn test_count_sequences() {
        let count = count_sequences("tests/testdata/test.fasta").unwrap();
        assert_eq!(count, 4);
    }
}

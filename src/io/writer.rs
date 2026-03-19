use std::fs::File;
use std::io::{BufWriter, Write};
use rustc_hash::FxHashMap;
use crate::config::OutputFormat;

pub struct Writer {
    inner: BufWriter<File>,
    format: OutputFormat,
    full_result: bool,
}

impl Writer {
    pub fn new(path: &str, format: OutputFormat, full_result: bool) -> Self {
        let file = File::create(path).expect("cannot create output file");
        Writer {
            inner: BufWriter::new(file),
            format,
            full_result,
        }
    }

    pub fn write_header(&mut self, class_ids: &[String]) {
        let sep = match self.format {
            OutputFormat::Csv => ",",
            OutputFormat::Tsv => "\t",
            OutputFormat::Json => return, // no header for JSON Lines
        };

        if self.full_result {
            let mut parts = vec!["sequence_id".to_string()];
            for id in class_ids {
                parts.push(id.clone());
            }
            writeln!(self.inner, "{}", parts.join(sep)).unwrap();
        } else {
            writeln!(self.inner, "{}",
                ["sequence_id", "best_class", "score"].join(sep)
            ).unwrap();
        }
    }

    pub fn write_result(&mut self, seq_id: &str, best_class: &str, score: f64) {
        match self.format {
            OutputFormat::Csv => {
                writeln!(self.inner, "{},{},{}", seq_id, best_class, score).unwrap();
            }
            OutputFormat::Tsv => {
                writeln!(self.inner, "{}\t{}\t{}", seq_id, best_class, score).unwrap();
            }
            OutputFormat::Json => {
                writeln!(
                    self.inner,
                    "{{\"sequence_id\":\"{}\",\"best_class\":\"{}\",\"score\":{}}}",
                    seq_id, best_class, score
                ).unwrap();
            }
        }
    }

    pub fn write_full_result(
        &mut self,
        seq_id: &str,
        scores: &FxHashMap<String, f64>,
        class_order: &[String],
    ) {
        match self.format {
            OutputFormat::Csv => {
                let mut parts = vec![seq_id.to_string()];
                for cls in class_order {
                    parts.push(format!("{}", scores.get(cls).unwrap_or(&0.0)));
                }
                writeln!(self.inner, "{}", parts.join(",")).unwrap();
            }
            OutputFormat::Tsv => {
                let mut parts = vec![seq_id.to_string()];
                for cls in class_order {
                    parts.push(format!("{}", scores.get(cls).unwrap_or(&0.0)));
                }
                writeln!(self.inner, "{}", parts.join("\t")).unwrap();
            }
            OutputFormat::Json => {
                let mut entries: Vec<String> = Vec::new();
                entries.push(format!("\"sequence_id\":\"{}\"", seq_id));
                for cls in class_order {
                    let s = scores.get(cls).unwrap_or(&0.0);
                    entries.push(format!("\"{}\":{}", cls, s));
                }
                writeln!(self.inner, "{{{}}}", entries.join(",")).unwrap();
            }
        }
    }

    pub fn write_no_valid_kmers(&mut self, seq_id: &str) {
        match self.format {
            OutputFormat::Csv => {
                writeln!(self.inner, "{},NO_VALID_KMERS,NaN", seq_id).unwrap();
            }
            OutputFormat::Tsv => {
                writeln!(self.inner, "{}\tNO_VALID_KMERS\tNaN", seq_id).unwrap();
            }
            OutputFormat::Json => {
                writeln!(
                    self.inner,
                    "{{\"sequence_id\":\"{}\",\"best_class\":\"NO_VALID_KMERS\",\"score\":null}}",
                    seq_id
                ).unwrap();
            }
        }
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        let _ = self.inner.flush();
    }
}

pub fn output_filename(prefix: &str, format: &OutputFormat) -> String {
    let ext = match format {
        OutputFormat::Csv => ".csv",
        OutputFormat::Tsv => ".tsv",
        OutputFormat::Json => ".jsonl",
    };
    format!("{}{}", prefix, ext)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(suffix: &str) -> String {
        format!("/tmp/nb_rust_test_writer_{}_{}", std::process::id(), suffix)
    }

    #[test]
    fn test_writer_csv() {
        let path = tmp_path("csv.csv");
        {
            let mut w = Writer::new(&path, OutputFormat::Csv, false);
            w.write_header(&[]);
            w.write_result("seq1", "classA", -42.5);
        }
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines[0], "sequence_id,best_class,score");
        assert_eq!(lines[1], "seq1,classA,-42.5");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_writer_json() {
        let path = tmp_path("json.jsonl");
        {
            let mut w = Writer::new(&path, OutputFormat::Json, false);
            w.write_header(&[]);
            w.write_result("seq1", "classB", -10.0);
        }
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        // No header for JSON
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("\"sequence_id\":\"seq1\""));
        assert!(lines[0].contains("\"best_class\":\"classB\""));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_writer_no_valid_kmers() {
        let path = tmp_path("nokmers.csv");
        {
            let mut w = Writer::new(&path, OutputFormat::Csv, false);
            w.write_header(&[]);
            w.write_no_valid_kmers("bad_seq");
        }
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines[1], "bad_seq,NO_VALID_KMERS,NaN");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_output_filename() {
        assert_eq!(output_filename("results", &OutputFormat::Csv), "results.csv");
        assert_eq!(output_filename("results", &OutputFormat::Tsv), "results.tsv");
        assert_eq!(output_filename("results", &OutputFormat::Json), "results.jsonl");
    }
}

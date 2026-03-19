use std::collections::HashMap;

#[test]
fn test_classify_with_legacy_savefiles() {
    // Build release binary
    let status = std::process::Command::new("cargo")
        .args(["build", "--release"])
        .status()
        .unwrap();
    assert!(status.success(), "cargo build --release failed");

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
    assert!(
        output.status.success(),
        "classification failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

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
    assert!(
        match_pct > 0.95,
        "match rate {match_pct:.4} ({matches}/{total}) is below 95%"
    );

    std::fs::remove_file("/tmp/nb_rust_legacy_test.csv").ok();
    std::fs::remove_file("/tmp/nb_rust_legacy_test.yaml").ok();
}

//! Binary serialization for trained Naive Bayes models.
//!
//! Supports two formats:
//! - **NBV** (native): Magic "NBV1" + version byte, followed by kmer_size,
//!   ngenomes, sumfreq, class ID, and kmer-count pairs. Log fields are
//!   recomputed on load.
//! - **Legacy** (NBC++ `-save.dat`): Pre-computed log values only (ngenomes_lg,
//!   sumfreq_lg, then kmer→log-frequency pairs). Loaded as ClassifyOnly since
//!   raw counts are not available.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use crate::model::{NbClass, LoadState};

const MAGIC: &[u8; 4] = b"NBV1";

/// Save an NbClass to the NBV binary format.
pub fn save_class(cls: &NbClass, path: &str) {
    let file = File::create(path).expect("cannot create save file");
    let mut w = BufWriter::new(file);

    // Magic + version
    w.write_all(MAGIC).unwrap();
    w.write_u8(1).unwrap();

    // kmer_size, ngenomes, sumfreq
    w.write_i32::<LittleEndian>(cls.kmer_size as i32).unwrap();
    w.write_i32::<LittleEndian>(cls.ngenomes as i32).unwrap();
    w.write_i64::<LittleEndian>(cls.sumfreq).unwrap();

    // id
    let id_bytes = cls.id.as_bytes();
    w.write_i32::<LittleEndian>(id_bytes.len() as i32).unwrap();
    w.write_all(id_bytes).unwrap();

    // kmer-count pairs
    for (&kmer, &count) in &cls.freqcnt {
        w.write_i32::<LittleEndian>(kmer as i32).unwrap();
        w.write_i32::<LittleEndian>(count as i32).unwrap();
    }

    w.flush().unwrap();
}

/// Load an NbClass from NBV binary format, recomputing log fields.
pub fn load_class(path: &str) -> Result<NbClass, String> {
    let file = File::open(path).map_err(|e| format!("cannot open save file: {e}"))?;
    let mut r = BufReader::new(file);

    // Magic
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(|e| format!("read magic: {e}"))?;
    if &magic != MAGIC {
        return Err("invalid NBV magic".into());
    }

    // Version
    let _version = r.read_u8().map_err(|e| format!("read version: {e}"))?;

    // Fields
    let kmer_size = r.read_i32::<LittleEndian>().map_err(|e| format!("read kmer_size: {e}"))? as usize;
    let ngenomes = r.read_i32::<LittleEndian>().map_err(|e| format!("read ngenomes: {e}"))? as u32;
    let sumfreq = r.read_i64::<LittleEndian>().map_err(|e| format!("read sumfreq: {e}"))?;

    // id
    let id_len = r.read_i32::<LittleEndian>().map_err(|e| format!("read id_len: {e}"))? as usize;
    let mut id_bytes = vec![0u8; id_len];
    r.read_exact(&mut id_bytes).map_err(|e| format!("read id: {e}"))?;
    let id = String::from_utf8(id_bytes).map_err(|e| format!("invalid id: {e}"))?;

    let mut cls = NbClass::new(&id, kmer_size, path);
    cls.ngenomes = ngenomes;
    cls.sumfreq = sumfreq;

    // Read kmer-count pairs until EOF
    loop {
        let kmer = match r.read_i32::<LittleEndian>() {
            Ok(v) => v as u32,
            Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(format!("read kmer: {e}")),
        };
        let count = r.read_i32::<LittleEndian>().map_err(|e| format!("read count: {e}"))? as u32;
        cls.freqcnt.insert(kmer, count);
    }

    // Recompute log fields
    cls.ngenomes_lg = (cls.ngenomes as f64).ln();
    cls.sumfreq_lg = (cls.sumfreq as f64).ln();
    for (&km, &cnt) in &cls.freqcnt {
        cls.freqcnt_lg.insert(km, ((cnt + 1) as f64).ln());
    }
    cls.state = LoadState::Full;

    Ok(cls)
}

/// Load a legacy NBC++ class file.
pub fn load_legacy_class(path: &str, k: usize) -> Result<NbClass, String> {
    let file = File::open(path).map_err(|e| format!("cannot open legacy file: {e}"))?;
    let mut r = BufReader::new(file);

    let ngenomes_lg = r.read_f64::<LittleEndian>().map_err(|e| format!("read ngenomes_lg: {e}"))?;
    let sumfreq_lg = r.read_f64::<LittleEndian>().map_err(|e| format!("read sumfreq_lg: {e}"))?;
    let n_entries = r.read_i32::<LittleEndian>().map_err(|e| format!("read n_entries: {e}"))? as usize;

    let basename = std::path::Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let id = basename.replace("-save.dat", "");
    let mut cls = NbClass::new(&id, k, path);
    cls.ngenomes_lg = ngenomes_lg;
    cls.sumfreq_lg = sumfreq_lg;

    for _ in 0..n_entries {
        let kmer = r.read_i32::<LittleEndian>().map_err(|e| format!("read kmer: {e}"))? as u32;
        let freq_lg = r.read_f64::<LittleEndian>().map_err(|e| format!("read freq_lg: {e}"))?;
        cls.freqcnt_lg.insert(kmer, freq_lg);
    }

    cls.state = LoadState::ClassifyOnly;
    Ok(cls)
}

/// Save metadata (kmer_size) to meta.nbv in the given directory.
pub fn save_meta(save_dir: &str, kmer_size: usize) {
    let path = format!("{}/meta.nbv", save_dir);
    fs::create_dir_all(save_dir).expect("cannot create save dir");
    fs::write(&path, kmer_size.to_string()).expect("cannot write meta.nbv");
}

/// Load kmer_size from meta.nbv in the given directory.
pub fn load_meta(save_dir: &str) -> Result<usize, String> {
    let path = format!("{}/meta.nbv", save_dir);
    let content = fs::read_to_string(&path)
        .map_err(|e| format!("cannot read meta.nbv: {e}"))?;
    content.trim().parse::<usize>()
        .map_err(|e| format!("invalid kmer_size in meta.nbv: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap;

    #[test]
    fn test_save_and_load_roundtrip() {
        let mut cls = NbClass::new("test_class", 6, "original.nb");
        let mut counts: FxHashMap<u32, u32> = FxHashMap::default();
        counts.insert(10, 5);
        counts.insert(42, 3);
        counts.insert(100, 7);
        cls.add_genome(&counts);

        let path = format!("/tmp/nb_rust_test_ser_{}.nbv", std::process::id());
        save_class(&cls, &path);

        let loaded = load_class(&path).unwrap();

        assert_eq!(loaded.id, cls.id);
        assert_eq!(loaded.kmer_size, cls.kmer_size);
        assert_eq!(loaded.ngenomes, cls.ngenomes);
        assert_eq!(loaded.sumfreq, cls.sumfreq);
        assert!((loaded.ngenomes_lg - cls.ngenomes_lg).abs() < 1e-12);
        assert!((loaded.sumfreq_lg - cls.sumfreq_lg).abs() < 1e-12);

        for (&km, &cnt) in &cls.freqcnt {
            assert_eq!(*loaded.freqcnt.get(&km).unwrap(), cnt);
        }
        for (&km, &lg) in &cls.freqcnt_lg {
            assert!((loaded.freqcnt_lg.get(&km).unwrap() - lg).abs() < 1e-12);
        }
        assert_eq!(loaded.state, LoadState::Full);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_save_and_load_meta() {
        let dir = format!("/tmp/nb_rust_test_meta_{}", std::process::id());
        save_meta(&dir, 9);
        let k = load_meta(&dir).unwrap();
        assert_eq!(k, 9);
        std::fs::remove_dir_all(&dir).ok();
    }
}

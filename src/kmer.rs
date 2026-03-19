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
        assert_eq!(counts[&ac_canon], 2);
        let cg_canon = canonical(encode(b"CG", 2), 2);
        assert_eq!(counts[&cg_canon], 1);
    }

    #[test]
    fn test_count_from_buffer_skips_invalid() {
        let counts = count_from_buffer(b"ACNGT", 2);
        assert_eq!(*counts.get(&1).unwrap_or(&0), 2);
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

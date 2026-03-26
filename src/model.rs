//! Naive Bayes classification model for k-mer frequency profiles.
//!
//! Each `NbClass` represents one taxonomic class. It stores k-mer frequency
//! counts from training genomes and computes log-likelihoods for query sequences.
//! Uses Laplace smoothing (add-one) and Kahan compensated summation for
//! numerical stability.

use rustc_hash::FxHashMap;
use crate::kmer::num_canonical_kmers;

// ---------------------------------------------------------------------------
// Kahan compensated summation — reduces floating-point accumulation error
// when summing many small values, which is critical for log-likelihood scores.
// ---------------------------------------------------------------------------

/// Accumulator for Kahan compensated summation.
/// Tracks both the running sum and a compensation term to recover lost
/// low-order bits from each addition.
#[derive(Debug, Default, Clone, Copy)]
pub struct KahanAccumulator {
    pub sum: f64,
    pub comp: f64,
}

/// Add a value to the Kahan accumulator with error compensation.
pub fn kahan_add(acc: &mut KahanAccumulator, val: f64) {
    let y = val - acc.comp;
    let t = acc.sum + y;
    acc.comp = (t - acc.sum) - y;
    acc.sum = t;
}

// ---------------------------------------------------------------------------
// LoadState
// ---------------------------------------------------------------------------

/// Tracks whether a model has raw counts (Full) or only log-transformed
/// values (ClassifyOnly, used for legacy NBC++ models that lack raw counts).
#[derive(Debug, Clone, PartialEq)]
pub enum LoadState {
    Unloaded,
    Full,
    ClassifyOnly,
}

// ---------------------------------------------------------------------------
// NbClass
// ---------------------------------------------------------------------------

/// A single Naive Bayes class model built from one or more training genomes.
///
/// Maintains both raw k-mer counts (`freqcnt`, `sumfreq`) for training/serialization
/// and precomputed log-transformed values (`freqcnt_lg`, `sumfreq_lg`) for fast
/// classification. The `sumfreq` is initialized to `num_canonical_kmers(k)` as a
/// Laplace prior so that unseen k-mers still contribute to the likelihood.
pub struct NbClass {
    pub id: String,
    pub kmer_size: usize,
    pub savefile: String,
    /// ln(ngenomes) — used externally; not part of log-likelihood scoring.
    pub ngenomes_lg: f64,
    /// ln(sumfreq) — the normalizing constant in the log-likelihood formula.
    pub sumfreq_lg: f64,
    /// ln(freqcnt[km] + 1) for each observed k-mer (Laplace-smoothed).
    pub freqcnt_lg: FxHashMap<u32, f64>,
    pub ngenomes: u32,
    /// Total k-mer count including the Laplace prior (starts at num_canonical_kmers).
    pub sumfreq: i64,
    /// Raw k-mer occurrence counts accumulated during training.
    pub freqcnt: FxHashMap<u32, u32>,
    pub state: LoadState,
}

impl NbClass {
    pub fn new(id: &str, kmer_size: usize, savefile: &str) -> Self {
        let sumfreq = num_canonical_kmers(kmer_size);
        NbClass {
            id: id.to_string(),
            kmer_size,
            savefile: savefile.to_string(),
            ngenomes_lg: 0.0,
            sumfreq_lg: 0.0,
            freqcnt_lg: FxHashMap::default(),
            ngenomes: 0,
            sumfreq,
            freqcnt: FxHashMap::default(),
            state: LoadState::Unloaded,
        }
    }

    /// Incorporate a genome's kmer counts into the model.
    ///
    /// For each kmer `km` with count `cnt`:
    ///   freqcnt[km] += cnt
    ///   sumfreq     += cnt
    ///
    /// After accumulation the log fields are recomputed:
    ///   ngenomes_lg  = ln(ngenomes)
    ///   sumfreq_lg   = ln(sumfreq)
    ///   freqcnt_lg   = ln(freqcnt[km]) for every km present
    pub fn add_genome(&mut self, kmer_counts: &FxHashMap<u32, u32>) {
        self.ngenomes += 1;

        for (&km, &cnt) in kmer_counts {
            *self.freqcnt.entry(km).or_insert(0) += cnt;
            self.sumfreq += cnt as i64;
        }

        // Recompute log fields
        self.ngenomes_lg = (self.ngenomes as f64).ln();
        self.sumfreq_lg = (self.sumfreq as f64).ln();
        for (&km, &cnt) in &self.freqcnt {
            self.freqcnt_lg.insert(km, ((cnt + 1) as f64).ln());
        }
        self.state = LoadState::Full;
    }

    /// Return ln(freqcnt[km]) or 0.0 if the kmer has never been seen.
    pub fn get_freq_count_lg(&self, km: u32) -> f64 {
        *self.freqcnt_lg.get(&km).unwrap_or(&0.0)
    }

    /// Compute log-likelihood for a query genome using Kahan summation.
    ///
    /// score = sum_km( freq(km) * ln(freqcnt[km]) ) - total * ln(sumfreq)
    ///
    /// where `total` is the total number of kmer observations in the query.
    pub fn compute_log_likelihood(&self, kmer_counts: &FxHashMap<u32, u32>) -> f64 {
        let mut acc = KahanAccumulator::default();
        let mut total: u64 = 0;

        for (&km, &cnt) in kmer_counts {
            let freq_lg = self.get_freq_count_lg(km);
            kahan_add(&mut acc, (cnt as f64) * freq_lg);
            total += cnt as u64;
        }

        kahan_add(&mut acc, -(total as f64) * self.sumfreq_lg);
        acc.sum
    }

    /// Rough memory estimate in bytes.
    pub fn size_bytes(&self) -> usize {
        let per_entry_raw = std::mem::size_of::<u32>() + std::mem::size_of::<u32>();
        let per_entry_lg  = std::mem::size_of::<u32>() + std::mem::size_of::<f64>();
        std::mem::size_of::<Self>()
            + self.id.len()
            + self.savefile.len()
            + self.freqcnt.len() * per_entry_raw
            + self.freqcnt_lg.len() * per_entry_lg
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap;

    // --- KahanAccumulator ---------------------------------------------------

    #[test]
    fn test_kahan_add_basic() {
        let mut acc = KahanAccumulator::default();
        kahan_add(&mut acc, 1.0);
        kahan_add(&mut acc, 2.0);
        kahan_add(&mut acc, 3.0);
        assert!((acc.sum - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_kahan_add_precision() {
        // Adding many small values; Kahan should beat naive f64 accumulation.
        let mut acc = KahanAccumulator::default();
        let n = 1_000_000u64;
        let small = 0.1_f64;
        for _ in 0..n {
            kahan_add(&mut acc, small);
        }
        let expected = small * n as f64;
        // Allow 1e-6 relative tolerance — Kahan keeps error tiny.
        assert!((acc.sum - expected).abs() / expected < 1e-6,
            "acc.sum={} expected={}", acc.sum, expected);
    }

    // --- NbClass::new -------------------------------------------------------

    #[test]
    fn test_nbclass_new() {
        let cls = NbClass::new("test_class", 2, "test.nb");
        // num_canonical_kmers(2) == 10
        assert_eq!(cls.sumfreq, 10);
        assert_eq!(cls.ngenomes, 0);
        assert_eq!(cls.id, "test_class");
        assert_eq!(cls.kmer_size, 2);
        assert_eq!(cls.savefile, "test.nb");
        assert_eq!(cls.state, LoadState::Unloaded);
        assert!(cls.freqcnt.is_empty());
        assert!(cls.freqcnt_lg.is_empty());
        // sumfreq_lg starts at 0.0 before any genomes added
        assert_eq!(cls.sumfreq_lg, 0.0);
    }

    // --- NbClass::add_genome ------------------------------------------------

    #[test]
    fn test_nbclass_add_genome() {
        let mut cls = NbClass::new("cls", 2, "f.nb");
        // sumfreq starts at 10 (num_canonical_kmers(2))
        let mut counts: FxHashMap<u32, u32> = FxHashMap::default();
        counts.insert(1, 5);
        counts.insert(6, 3);
        cls.add_genome(&counts);

        assert_eq!(cls.ngenomes, 1);
        // 10 + 5 + 3 = 18
        assert_eq!(cls.sumfreq, 18);
        assert!((cls.sumfreq_lg - 18_f64.ln()).abs() < 1e-12);
        assert_eq!(*cls.freqcnt.get(&1).unwrap(), 5);
        assert_eq!(*cls.freqcnt.get(&6).unwrap(), 3);
        assert!(matches!(cls.state, LoadState::Full));
        assert!((cls.ngenomes_lg - 1.0_f64.ln()).abs() < 1e-10);
        assert!((cls.sumfreq_lg - 18.0_f64.ln()).abs() < 1e-10);
        assert!((cls.freqcnt_lg[&1] - 6.0_f64.ln()).abs() < 1e-10); // ln(5+1)
        assert!((cls.freqcnt_lg[&6] - 4.0_f64.ln()).abs() < 1e-10); // ln(3+1)
    }

    #[test]
    fn test_nbclass_add_genome_twice() {
        let mut cls = NbClass::new("cls", 2, "f.nb");
        let mut c1: FxHashMap<u32, u32> = FxHashMap::default();
        c1.insert(1, 5);
        c1.insert(6, 3);
        cls.add_genome(&c1);
        let mut c2: FxHashMap<u32, u32> = FxHashMap::default();
        c2.insert(1, 2);
        c2.insert(9, 1);
        cls.add_genome(&c2);

        assert_eq!(cls.ngenomes, 2);
        // 10 + 8 + 3 = 21
        assert_eq!(cls.sumfreq, 10 + 8 + 3);
        assert_eq!(*cls.freqcnt.get(&1).unwrap(), 7);
        assert_eq!(*cls.freqcnt.get(&6).unwrap(), 3);
        assert_eq!(*cls.freqcnt.get(&9).unwrap(), 1);
    }

    // --- get_freq_count_lg --------------------------------------------------

    #[test]
    fn test_get_freq_count_lg_seen() {
        let mut cls = NbClass::new("cls", 2, "f.nb");
        let mut counts: FxHashMap<u32, u32> = FxHashMap::default();
        counts.insert(1, 5);
        cls.add_genome(&counts);
        // freqcnt[1] == 5 → ln(5+1) = ln(6) (Laplace +1)
        assert!((cls.get_freq_count_lg(1) - 6_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn test_get_freq_count_lg_unseen() {
        let cls = NbClass::new("cls", 2, "f.nb");
        assert_eq!(cls.get_freq_count_lg(999), 0.0);
    }

    // --- compute_log_likelihood ---------------------------------------------

    #[test]
    fn test_compute_log_likelihood() {
        // Build a model that has seen one genome: {1:5, 6:3}
        // sumfreq = 18, freqcnt[1]=5, freqcnt[6]=3
        let mut cls = NbClass::new("cls", 2, "f.nb");
        let mut train: FxHashMap<u32, u32> = FxHashMap::default();
        train.insert(1, 5);
        train.insert(6, 3);
        cls.add_genome(&train);

        // Query: {1:2, 6:1}  →  total = 3
        let mut query: FxHashMap<u32, u32> = FxHashMap::default();
        query.insert(1, 2);
        query.insert(6, 1);

        // Expected: 2*ln(5+1) + 1*ln(3+1) - 3*ln(18) (Laplace +1 on counts)
        let expected = 2.0 * 6_f64.ln() + 1.0 * 4_f64.ln() - 3.0 * 18_f64.ln();
        let ll = cls.compute_log_likelihood(&query);
        assert!((ll - expected).abs() < 1e-10,
            "ll={} expected={}", ll, expected);
    }

    // --- size_bytes ---------------------------------------------------------

    #[test]
    fn test_size_bytes() {
        let mut cls = NbClass::new("cls", 2, "f.nb");
        let mut counts: FxHashMap<u32, u32> = FxHashMap::default();
        counts.insert(1, 5);
        cls.add_genome(&counts);
        // Just verify it returns a positive, plausible value (> 0).
        assert!(cls.size_bytes() > 0);
    }
}

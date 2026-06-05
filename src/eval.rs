//! Measurement harness for the IVF index: how much recall do we lose, and how much
//! work do we save, at a given `nprobe`? Ground truth = brute force over all entries.
//!
//! This is how the mission's open question ("how much recall can we lose?") gets a
//! number. Sweep `nprobe`, read recall + candidates-scanned, pick the cheapest probe
//! that clears your recall bar.

use crate::distance::squared_distance_i16;
use crate::ivf::IvfIndex;
use crate::kmeans::DIM;

#[derive(Debug, Clone)]
pub struct EvalReport {
    pub queries: usize,
    pub nprobe: usize,
    /// Average overlap with the brute-force top-5 (1.0 = found all 5 true neighbors).
    pub recall_at5: f32,
    /// Fraction of queries where the approve/reject decision (score < 0.6) matched
    /// brute force. The metric that actually matters for fraud.
    pub decision_agreement: f32,
    /// Average entries scanned per query — the speed proxy (vs N for brute force).
    pub avg_candidates: f32,
}

/// Keep the 5 smallest (distance, position) pairs seen so far.
fn top5_positions<'a, I>(query: &[i16; DIM], items: I) -> Vec<usize>
where
    I: Iterator<Item = (usize, &'a [i16; DIM])>,
{
    let mut best: Vec<(u64, usize)> = Vec::new();
    for (pos, v) in items {
        let d = squared_distance_i16(query, v);
        best.push((d, pos));
        // sort by distance, then position (stable tie-break so brute and IVF agree).
        best.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        best.truncate(5);
    }
    best.into_iter().map(|(_, p)| p).collect()
}

/// The 5 true nearest (positions in `index.entries`), scanning everything.
fn brute_top5(index: &IvfIndex, query: &[i16; DIM]) -> Vec<usize> {
    top5_positions(
        query,
        index.entries.iter().enumerate().map(|(i, e)| (i, &e.vector)),
    )
}

/// The 5 nearest IVF finds (positions in `index.entries`), scanning only probed cells.
fn ivf_top5(index: &IvfIndex, query: &[i16; DIM], nprobe: usize) -> Vec<usize> {
    let cells = index.nearest_cells(query, nprobe);
    let items = cells.into_iter().flat_map(|c| {
        let start = index.list_offsets[c];
        index
            .list(c)
            .iter()
            .enumerate()
            .map(move |(j, e)| (start + j, &e.vector))
    });
    top5_positions(query, items)
}

/// Run a batch of queries through both paths and report recall + cost.
pub fn evaluate(index: &IvfIndex, queries: &[[i16; DIM]], nprobe: usize) -> EvalReport {
    let mut recall = 0.0f32;
    let mut agree = 0usize;
    let mut candidates = 0usize;

    for q in queries {
        let truth = brute_top5(index, q);
        let got = ivf_top5(index, q, nprobe);

        let overlap = truth.iter().filter(|t| got.contains(t)).count();
        recall += overlap as f32 / 5.0;

        // fraud_score at nprobe = nlist is the exact (ground-truth) score.
        let exact = index.fraud_score(q, index.nlist());
        let approx = index.fraud_score(q, nprobe);
        if (exact < 0.6) == (approx < 0.6) {
            agree += 1;
        }

        for c in index.nearest_cells(q, nprobe) {
            candidates += index.list(c).len();
        }
    }

    let n = queries.len().max(1) as f32;
    EvalReport {
        queries: queries.len(),
        nprobe,
        recall_at5: recall / n,
        decision_agreement: agree as f32 / n,
        avg_candidates: candidates as f32 / n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knn::ReferenceEntry;

    fn entry(v: i16, is_fraud: bool) -> ReferenceEntry {
        ReferenceEntry {
            vector: [v; DIM],
            is_fraud,
        }
    }

    // distinct vector values -> distinct distances -> no tie ambiguity in top-5.
    fn index() -> IvfIndex {
        let entries: Vec<ReferenceEntry> = (0..12)
            .map(|i| entry((i * 3) as i16, i % 2 == 0))
            .collect();
        IvfIndex::build(&entries, 3, 20)
    }

    #[test]
    fn full_probe_is_perfect_recall() {
        let idx = index();
        let queries = [[0; DIM], [15; DIM], [33; DIM]];
        let report = evaluate(&idx, &queries, idx.nlist());

        assert_eq!(report.recall_at5, 1.0);
        assert_eq!(report.decision_agreement, 1.0);
    }

    #[test]
    fn fewer_probes_scan_fewer_candidates() {
        let idx = index();
        let queries = [[0; DIM], [15; DIM], [33; DIM]];

        let one = evaluate(&idx, &queries, 1);
        let all = evaluate(&idx, &queries, idx.nlist());

        // probing one cell never scans more than probing all of them.
        assert!(one.avg_candidates <= all.avg_candidates);
        // and recall can only improve (or stay) as you probe more.
        assert!(one.recall_at5 <= all.recall_at5);
    }

    // The real measurement. Run with:
    //   cargo test --release --lib sweep_nprobe_on_embedded_3m -- --ignored --nocapture
    // --release matters: this is the latency you'd actually ship.
    #[test]
    #[ignore = "builds the IVF index over the full 3M dataset; run explicitly"]
    fn sweep_nprobe_on_embedded_3m() {
        use crate::reference_store::ReferenceStore;
        use std::time::Instant;

        let t = Instant::now();
        let store = ReferenceStore::from_embedded_gzip().unwrap();
        let index = store.index();
        println!(
            "\nbuild: {:?} | N={} | nlist={}",
            t.elapsed(),
            index.entries.len(),
            index.nlist()
        );

        // Use 500 real entry vectors (strided) as queries.
        let stride = index.entries.len() / 500;
        let queries: Vec<[i16; DIM]> = index
            .entries
            .iter()
            .step_by(stride.max(1))
            .map(|e| e.vector)
            .take(500)
            .collect();

        // Brute-force latency baseline (one full 3M scan), for comparison.
        let bt = Instant::now();
        for q in &queries {
            std::hint::black_box(store.fraud_score_bruteforce(q));
        }
        let brute_per_query = bt.elapsed() / queries.len() as u32;
        println!("\nbrute-force latency: {brute_per_query:?}/query (baseline)\n");

        println!(
            "{:>6} | {:>10} | {:>12} | {:>14} | {:>12}",
            "nprobe", "recall@5", "decision==", "candidates", "ivf query"
        );
        for nprobe in [1usize, 4, 8, 16, 32, 64] {
            // recall / decision agreement (this also runs brute ground truth — slow, untimed).
            let r = evaluate(index, &queries, nprobe);

            // pure IVF query latency, timed on its own.
            let qt = Instant::now();
            for q in &queries {
                std::hint::black_box(index.fraud_score(q, nprobe));
            }
            let ivf_per_query = qt.elapsed() / queries.len() as u32;

            println!(
                "{:>6} | {:>9.3} | {:>11.3} | {:>14.0} | {:>12?}",
                nprobe, r.recall_at5, r.decision_agreement, r.avg_candidates, ivf_per_query
            );
        }
        println!();
    }
}

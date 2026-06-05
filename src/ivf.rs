//! IVF (Inverted File) index over the int8 reference vectors — the query-time
//! speedup for fraud scoring.
//!
//! Two parts (lesson 03):
//!   * coarse quantizer: the `nlist` k-means centroids (one per cell)
//!   * inverted lists:    every entry grouped by its nearest centroid, stored in
//!                        ONE contiguous array with CSR offsets (cache-friendly)
//!
//! You implement `group_by_cluster` — the counting sort that builds the CSR
//! layout. `build` and the `list` accessor are provided.
//!
//! See explainers/03-building-the-ivf-index.html.

use std::io::{self, Read, Write};

use crate::distance::squared_distance_i16_and_f32;
use crate::kmeans::{DIM, KMeans, nearest_centroid};
use crate::knn::{ReferenceEntry, fraud_score_top5};

/// On-disk magic + format version for the serialized index (see `write_to`).
/// Version 2: entry vectors are int16 (2 bytes/dim); version 1 was int8.
const INDEX_MAGIC: &[u8; 4] = b"IVF1";
const INDEX_VERSION: u32 = 2;

#[derive(Debug)]
pub struct IvfIndex {
    /// Coarse quantizer: `nlist` centroids. `centroids[c]` is the center of cell `c`.
    pub centroids: Vec<[f32; DIM]>,
    /// CSR offsets, length `nlist + 1`. Cell `c` owns `entries[offsets[c]..offsets[c+1]]`.
    pub list_offsets: Vec<usize>,
    /// All reference entries, reordered so each cell's entries are contiguous.
    pub entries: Vec<ReferenceEntry>,
}

impl IvfIndex {
    /// Number of inverted lists (cells).
    pub fn nlist(&self) -> usize {
        self.centroids.len()
    }

    /// The entries assigned to cell `c`, as a contiguous slice.
    pub fn list(&self, c: usize) -> &[ReferenceEntry] {
        &self.entries[self.list_offsets[c]..self.list_offsets[c + 1]]
    }

    /// Build the index, training the coarse quantizer on ALL entries.
    /// Fine for small sets; for millions, prefer `build_sampled`.
    pub fn build(entries: &[ReferenceEntry], nlist: usize, max_iters: usize) -> Self {
        Self::build_sampled(entries, nlist, max_iters, entries.len())
    }

    /// Build the index, training k-means on a SAMPLE of `train_sample` entries,
    /// then assigning ALL entries to the resulting centroids.
    ///
    /// Why sample: k-means iterates many times over the training set; on 3M points
    /// × nlist centroids that is the build bottleneck on 1 core. You don't need all
    /// 3M to find good cell centers — a representative sample finds nearly the same
    /// ones. The full set is still binned (assigned) exactly once.
    pub fn build_sampled(
        entries: &[ReferenceEntry],
        nlist: usize,
        max_iters: usize,
        train_sample: usize,
    ) -> Self {
        // 1. Pick a training sample (deterministic stride — no RNG needed).
        let sample: Vec<[i16; DIM]> = if train_sample >= entries.len() {
            entries.iter().map(|e| e.vector).collect()
        } else {
            let stride = (entries.len() / train_sample).max(1);
            entries
                .iter()
                .step_by(stride)
                .map(|e| e.vector)
                .take(train_sample)
                .collect()
        };

        // 2. Train the coarse quantizer on the sample (the expensive, iterative part).
        let km = KMeans::fit(&sample, nlist, max_iters);

        // 3. Assign EVERY entry to its nearest trained centroid (one pass, no iters).
        let assignments: Vec<u16> = entries
            .iter()
            .map(|e| nearest_centroid(&e.vector, &km.centroids) as u16)
            .collect();

        // 4. Bin into contiguous CSR lists.
        let (list_offsets, grouped) = group_by_cluster(entries, &assignments, nlist);

        IvfIndex {
            centroids: km.centroids,
            list_offsets,
            entries: grouped,
        }
    }

    // =================================================================
    // EXERCISE (lesson 04) — coarse search: pick the cells to probe.
    //
    // Return the `nprobe` cell indices whose centroid is closest to `query`,
    // nearest first. This is step 2 of the query path.
    //
    // Rules:
    //   * Use `squared_distance_i16_and_f32(query, centroid)` for each centroid.
    //   * Clamp: if `nprobe > nlist`, just return all cells.
    //
    // Hint: collect `(distance, cell_index)` for every centroid, sort by distance,
    // take `nprobe`, keep only the indices.
    // =================================================================
    pub fn nearest_cells(&self, query: &[i16; DIM], nprobe: usize) -> Vec<usize> {
        let mut scored: Vec<(f32, usize)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (squared_distance_i16_and_f32(query, c), i))
            .collect();

        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        scored.into_iter().take(nprobe).map(|(_, i)| i).collect()
    }

    /// PROVIDED — the full query path (steps 1-4).
    /// Probe `nprobe` cells, gather their entries, run the same top-5 fraud scan
    /// you wrote in `knn.rs`. Returns the fraction of the 5 nearest that are fraud.
    pub fn fraud_score(&self, query: &[i16; DIM], nprobe: usize) -> f32 {
        let cells = self.nearest_cells(query, nprobe); // step 1+2: coarse search
        // step 3: collect candidates from only the probed cells' contiguous slices.
        let mut candidates: Vec<ReferenceEntry> = Vec::new();
        for c in cells {
            candidates.extend_from_slice(self.list(c));
        }
        // step 4: identical scoring to the brute-force path, on the shortlist.
        fraud_score_top5(query, &candidates)
    }

    /// Serialize the index to a raw little-endian byte stream. Layout:
    ///   magic "IVF1" | version u32 | nlist u64 | n_entries u64
    ///   centroids:    nlist × (DIM × f32)
    ///   list_offsets: (nlist + 1) × u64
    ///   entries:      n_entries × (DIM × i16 + 1 × u8 fraud flag)
    pub fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let nlist = self.centroids.len();
        let n_entries = self.entries.len();

        w.write_all(INDEX_MAGIC)?;
        w.write_all(&INDEX_VERSION.to_le_bytes())?;
        w.write_all(&(nlist as u64).to_le_bytes())?;
        w.write_all(&(n_entries as u64).to_le_bytes())?;

        for centroid in &self.centroids {
            for &x in centroid {
                w.write_all(&x.to_le_bytes())?;
            }
        }

        for &offset in &self.list_offsets {
            w.write_all(&(offset as u64).to_le_bytes())?;
        }

        for entry in &self.entries {
            for &v in &entry.vector {
                w.write_all(&v.to_le_bytes())?;
            }
            w.write_all(&[entry.is_fraud as u8])?;
        }

        Ok(())
    }

    /// Reconstruct an index from the byte stream written by `write_to`.
    pub fn read_from<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != INDEX_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad index magic"));
        }

        let version = read_u32(r)?;
        if version != INDEX_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported index version {version}"),
            ));
        }

        let nlist = read_u64(r)? as usize;
        let n_entries = read_u64(r)? as usize;

        // centroids: nlist × DIM × f32, read in one bulk slurp then split.
        let mut cbuf = vec![0u8; nlist * DIM * 4];
        r.read_exact(&mut cbuf)?;
        let mut centroids = Vec::with_capacity(nlist);
        for chunk in cbuf.chunks_exact(DIM * 4) {
            let mut centroid = [0f32; DIM];
            for (d, b) in chunk.chunks_exact(4).enumerate() {
                centroid[d] = f32::from_le_bytes(b.try_into().unwrap());
            }
            centroids.push(centroid);
        }

        // offsets: (nlist + 1) × u64.
        let mut obuf = vec![0u8; (nlist + 1) * 8];
        r.read_exact(&mut obuf)?;
        let list_offsets: Vec<usize> = obuf
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()) as usize)
            .collect();

        // entries: n_entries × (DIM × i16 + 1 fraud flag) = DIM*2 + 1 bytes each.
        let entry_bytes = DIM * 2 + 1;
        let mut ebuf = vec![0u8; n_entries * entry_bytes];
        r.read_exact(&mut ebuf)?;
        let mut entries = Vec::with_capacity(n_entries);
        for chunk in ebuf.chunks_exact(entry_bytes) {
            let mut vector = [0i16; DIM];
            for (d, b) in chunk[..DIM * 2].chunks_exact(2).enumerate() {
                vector[d] = i16::from_le_bytes(b.try_into().unwrap());
            }
            entries.push(ReferenceEntry {
                vector,
                is_fraud: chunk[DIM * 2] != 0,
            });
        }

        Ok(IvfIndex {
            centroids,
            list_offsets,
            entries,
        })
    }
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

// =====================================================================
// EXERCISE — counting sort into the CSR layout.
//
// Given the entries and their cluster `assignments` (assignments[i] is the cell
// of entries[i]), return:
//   * offsets: length `nlist + 1`, where offsets[c] is the start of cell c and
//     offsets[c+1] its end. offsets[0] == 0, offsets[nlist] == entries.len().
//   * grouped: all entries reordered so cell c occupies grouped[offsets[c]..offsets[c+1]].
//
// Three passes, no per-cell Vecs (see explainer §2):
//   1. COUNT     — counts[c] = number of assignments equal to c
//   2. PREFIX SUM — offsets[c+1] = offsets[c] + counts[c]
//   3. SCATTER   — a cursor per cell; place each entry, bump its cursor
//
// Hint: pre-size `grouped` with `entries.to_vec()` (right length, all slots get
// overwritten). Keep `cursor = offsets.clone()` and write at cursor[c], then += 1.
// =====================================================================
pub fn group_by_cluster(
    entries: &[ReferenceEntry],
    assignments: &[u16],
    nlist: usize,
) -> (Vec<usize>, Vec<ReferenceEntry>) {
    // 1. COUNT — one slot per cell. cells are 0..nlist, so a Vec (not a HashMap)
    //    is the right structure: counts[cell] is O(1) and contiguous.
    let mut counts = vec![0usize; nlist];
    for &cell in assignments {
        counts[cell as usize] += 1;
    }

    // 2. PREFIX SUM — offsets[c] = where cell c's slice starts on the shelf.
    //    Length nlist+1; offsets[nlist] ends up == total entries.
    let mut offsets = vec![0usize; nlist + 1];
    for c in 0..nlist {
        offsets[c + 1] = offsets[c] + counts[c];
    }

    // 3. SCATTER — cursor[c] = next free slot for cell c (starts at its offset).
    //    Walk entries once, drop each into its cell's next slot, bump the cursor.
    let mut cursor = offsets.clone();
    let mut grouped = entries.to_vec(); // right length; every slot gets overwritten
    for (i, &cell) in assignments.iter().enumerate() {
        let c = cell as usize;
        grouped[cursor[c]] = entries[i];
        cursor[c] += 1;
    }

    (offsets, grouped)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(vector_val: i16, is_fraud: bool) -> ReferenceEntry {
        ReferenceEntry {
            vector: [vector_val; DIM],
            is_fraud,
        }
    }

    // assignments [1,0,1,2,0] over 3 cells: counts = [2,2,1], offsets = [0,2,4,5].
    fn fixture() -> (Vec<ReferenceEntry>, Vec<u16>) {
        let entries = vec![
            entry(10, false), // -> cell 1
            entry(20, true),  // -> cell 0
            entry(30, false), // -> cell 1
            entry(40, true),  // -> cell 2
            entry(50, false), // -> cell 0
        ];
        let assignments = vec![1u16, 0, 1, 2, 0];
        (entries, assignments)
    }

    #[test]
    fn counts_and_offsets_correct() {
        let (entries, assignments) = fixture();
        let (offsets, grouped) = group_by_cluster(&entries, &assignments, 3);

        assert_eq!(offsets, vec![0, 2, 4, 5]);
        assert_eq!(grouped.len(), entries.len());
        assert_eq!(*offsets.last().unwrap(), entries.len());
    }

    #[test]
    fn every_entry_lands_once() {
        let (entries, assignments) = fixture();
        let (_, grouped) = group_by_cluster(&entries, &assignments, 3);

        // the scatter is a permutation: same multiset of entries, nothing lost/dup'd.
        let mut got = grouped.clone();
        let mut want = entries.clone();
        got.sort_by_key(|e| e.vector[0]);
        want.sort_by_key(|e| e.vector[0]);
        assert_eq!(got, want);
    }

    #[test]
    fn list_slices_hold_their_cluster() {
        let (entries, assignments) = fixture();
        let (offsets, grouped) = group_by_cluster(&entries, &assignments, 3);

        // cell 0 got entries[1] (val 20) and entries[4] (val 50).
        let cell0: Vec<i16> = grouped[offsets[0]..offsets[1]]
            .iter()
            .map(|e| e.vector[0])
            .collect();
        assert!(cell0.contains(&20) && cell0.contains(&50) && cell0.len() == 2);

        // cell 1 got entries[0] (10) and entries[2] (30).
        let cell1: Vec<i16> = grouped[offsets[1]..offsets[2]]
            .iter()
            .map(|e| e.vector[0])
            .collect();
        assert!(cell1.contains(&10) && cell1.contains(&30) && cell1.len() == 2);

        // cell 2 got entries[3] (40).
        let cell2: Vec<i16> = grouped[offsets[2]..offsets[3]]
            .iter()
            .map(|e| e.vector[0])
            .collect();
        assert_eq!(cell2, vec![40]);
    }

    #[test]
    fn build_groups_two_blobs() {
        // blob A near 0, blob B near 100. build() should put each blob in its own cell.
        let entries = vec![
            entry(0, false),
            entry(1, false),
            entry(2, true),
            entry(100, false),
            entry(101, true),
            entry(99, false),
        ];
        let index = IvfIndex::build(&entries, 2, 20);

        assert_eq!(index.nlist(), 2);
        assert_eq!(*index.list_offsets.last().unwrap(), entries.len());

        // Each cell should be internally coherent: all its vectors on the same side.
        for c in 0..index.nlist() {
            let list = index.list(c);
            if list.is_empty() {
                continue;
            }
            let low = list[0].vector[0] < 50;
            assert!(
                list.iter().all(|e| (e.vector[0] < 50) == low),
                "cell {c} mixes the two blobs"
            );
        }
    }

    // ---- lesson 04: the query path ----

    // An 8-entry index over two blobs (near 0, near 100), some fraud.
    fn query_index() -> IvfIndex {
        let entries = vec![
            entry(0, true),
            entry(1, true),
            entry(2, false),
            entry(3, true),
            entry(100, false),
            entry(101, false),
            entry(102, true),
            entry(103, false),
        ];
        IvfIndex::build(&entries, 2, 20)
    }

    #[test]
    fn nearest_cells_picks_closest() {
        let index = query_index();
        // A query of 0s belongs in whichever cell holds the low blob.
        let cell = index.nearest_cells(&[0; DIM], 1)[0];
        // that cell's entries must all be the low blob.
        assert!(index.list(cell).iter().all(|e| e.vector[0] < 50));
    }

    #[test]
    fn nearest_cells_count_is_nprobe() {
        let index = query_index();
        assert_eq!(index.nearest_cells(&[0; DIM], 1).len(), 1);
        assert_eq!(index.nearest_cells(&[0; DIM], 2).len(), 2);
        // clamp: nprobe beyond nlist returns all cells, not more.
        assert_eq!(index.nearest_cells(&[0; DIM], 99).len(), index.nlist());
    }

    #[test]
    fn query_matches_bruteforce_when_fully_probed() {
        let index = query_index();
        let all = index.entries.clone();
        let query = [1; DIM];
        // Probing every cell must reproduce the brute-force answer exactly.
        let ivf = index.fraud_score(&query, index.nlist());
        let brute = fraud_score_top5(&query, &all);
        assert_eq!(ivf, brute);
    }

    #[test]
    fn query_finds_fraud_cluster() {
        // a blob of 5 fraud vectors near 0; a query of 0s should score 1.0.
        let entries = vec![
            entry(0, true),
            entry(1, true),
            entry(2, true),
            entry(3, true),
            entry(4, true),
            entry(100, false),
            entry(101, false),
            entry(102, false),
        ];
        let index = IvfIndex::build(&entries, 2, 20);
        assert_eq!(index.fraud_score(&[0; DIM], 1), 1.0);
    }
}

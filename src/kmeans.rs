//! k-means clustering over the int8 reference vectors — the *build* half of IVF.
//!
//! You implement the two functions that ARE Lloyd's algorithm:
//!   * `nearest_centroid`  -> the ASSIGN step
//!   * `mean_centroid`     -> the UPDATE step
//!
//! `KMeans::fit` (the loop, init, empty-cluster handling, convergence) is written
//! for you to read. Run `cargo test kmeans` and make the 5 tests pass.
//!
//! See explainers/02-kmeans-in-rust.html for the walkthrough.

use crate::distance::squared_distance_i16_and_f32;

pub const DIM: usize = 14;

/// A reference point: same shape as the vectors in `ReferenceStore`.
pub type Point = [i16; DIM];

/// A centroid: the MEAN of points, so it must be `f32` — the average of int8s
/// is generally not an int8 (mean of 0 and 1 is 0.5).
pub type Centroid = [f32; DIM];

// =====================================================================
// EXERCISE 1 — the ASSIGN step.
//
// Return the INDEX of the centroid closest to `point`, by squared Euclidean
// distance (no sqrt needed — it doesn't change the ordering and costs more).
//
// Rules:
//   * Promote `i16 -> f32` BEFORE doing arithmetic (overflow trap, same as knn.rs).
//   * On a tie (equal distance), return the LOWEST index. (Hint: strict `<`.)
//   * `centroids` is always non-empty.
// =====================================================================
pub fn nearest_centroid(point: &Point, centroids: &[Centroid]) -> usize {
    let mut min_idx = 0;
    let mut min_dist = squared_distance_i16_and_f32(point, &centroids[0]);

    for (idx, centroid) in centroids.iter().enumerate().skip(1) {
        let dist = squared_distance_i16_and_f32(point, centroid);
        if dist < min_dist {
            min_dist = dist;
            min_idx = idx;
        }
    }

    min_idx
}

// =====================================================================
// EXERCISE 2 — the UPDATE step.
//
// Return the mean of `members` as a new centroid: sum each of the 14 dimensions,
// divide by the number of members.
//
// Rules:
//   * Accumulate in `f32` (or a wide int) — summing many int16s overflows i16.
//   * `members` is always non-empty when this is called (fit guarantees it).
// =====================================================================
pub fn mean_centroid(members: &[Point]) -> Centroid {
    let mut acc = [0.0f32; 14];

    for p in members {
        for i in 0..14 {
            acc[i] += p[i] as f32;
        }
    }

    let n = members.len() as f32;

    for i in 0..14 {
        acc[i] = acc[i] / n
    }

    return acc;
}

// ---------------------------------------------------------------------
// PROVIDED: the Lloyd loop. Read it — this is lesson 01 in Rust.
// ---------------------------------------------------------------------

#[derive(Debug)]
pub struct KMeans {
    /// `centroids[c]` is the f32 center of cluster `c`. There are `k` of them.
    pub centroids: Vec<Centroid>,
    /// `assignments[i]` is the cluster id of `points[i]`. `u16` keeps 3M ids in 6 MB.
    pub assignments: Vec<u16>,
}

impl KMeans {
    /// Cluster `points` into `k` groups, looping at most `max_iters` times.
    pub fn fit(points: &[Point], k: usize, max_iters: usize) -> Self {
        assert!(k >= 1, "need at least one cluster");
        assert!(k <= points.len(), "need at least k points");

        // INIT: first k points become the starting centroids (cast to f32).
        // Crude but deterministic -> stable tests. k-means++ is a later upgrade.
        let mut centroids: Vec<Centroid> = points[..k]
            .iter()
            .map(|p| {
                let mut c = [0.0f32; DIM];
                for i in 0..DIM {
                    c[i] = p[i] as f32;
                }
                c
            })
            .collect();

        let mut assignments = vec![0u16; points.len()];

        for _ in 0..max_iters {
            // ASSIGN: your nearest_centroid for every point.
            let mut changed = false;
            for (idx, p) in points.iter().enumerate() {
                let a = nearest_centroid(p, &centroids) as u16;
                if a != assignments[idx] {
                    changed = true;
                }
                assignments[idx] = a;
            }

            // UPDATE (single-pass): same math as `mean_centroid`, but accumulated
            // in ONE pass with no gathering. The old version built a
            // `Vec<Vec<Point>>` every iteration — 3M point copies + N allocations
            // per pass. This keeps one running sum + count per cluster instead.
            let mut sums = vec![[0.0f32; DIM]; k];
            let mut counts = vec![0u32; k];
            for (idx, p) in points.iter().enumerate() {
                let c = assignments[idx] as usize;
                let sum = &mut sums[c];
                for d in 0..DIM {
                    sum[d] += p[d] as f32;
                }
                counts[c] += 1;
            }
            for c in 0..k {
                if counts[c] > 0 {
                    let n = counts[c] as f32;
                    for d in 0..DIM {
                        centroids[c][d] = sums[c][d] / n;
                    }
                }
                // empty cluster -> no points to average, keep the old centroid.
            }

            // CONVERGE: assignments stable this round => done.
            if !changed {
                break;
            }
        }

        KMeans {
            centroids,
            assignments,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(v: i16) -> Point {
        [v; DIM]
    }
    fn ct(v: f32) -> Centroid {
        [v; DIM]
    }

    #[test]
    fn nearest_picks_closest() {
        let centroids = [ct(0.0), ct(10.0), ct(-5.0)];
        // point of 9s: closest to the centroid of 10s (index 1).
        assert_eq!(nearest_centroid(&pt(9), &centroids), 1);
    }

    #[test]
    fn nearest_breaks_ties_low() {
        // point 0 is equidistant from centroid +1 and centroid -1.
        let centroids = [ct(1.0), ct(-1.0)];
        assert_eq!(nearest_centroid(&pt(0), &centroids), 0);
    }

    #[test]
    fn mean_of_two_points() {
        // mean of 0 and 1 is 0.5 — must NOT be rounded to an integer.
        let members = [pt(0), pt(1)];
        let c = mean_centroid(&members);
        assert_eq!(c, ct(0.5));
    }

    #[test]
    fn mean_handles_negatives() {
        // three -32767s: the int sum (-98301) overflows i16, so you must accumulate wide.
        let members = [pt(-32767), pt(-32767), pt(-32767)];
        let c = mean_centroid(&members);
        assert_eq!(c, ct(-32767.0));
    }

    #[test]
    fn fit_separates_two_blobs() {
        // blob A near 0, blob B near 100. Note: init grabs the first two points,
        // BOTH from blob A — Lloyd must still recover and split the blobs.
        let points = [pt(0), pt(1), pt(2), pt(100), pt(101), pt(99)];
        let km = KMeans::fit(&points, 2, 20);

        let a = &km.assignments;
        // the three low points share a cluster...
        assert_eq!(a[0], a[1]);
        assert_eq!(a[1], a[2]);
        // ...the three high points share the other...
        assert_eq!(a[3], a[4]);
        assert_eq!(a[4], a[5]);
        // ...and the two groups are different clusters.
        assert_ne!(a[0], a[3]);
    }
}

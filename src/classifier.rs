use std::sync::Arc;

use crate::{
    dataset::{FeatureVector, LABEL_FRAUD, ReferenceDataset},
    distance::DistanceEngine,
    normalization::{FEATURE_DIM, normalize_request},
    payload::FraudRequest,
};

pub const K_NEIGHBORS: usize = 5;

#[derive(Clone)]
pub struct FraudDetector {
    dataset: Arc<ReferenceDataset>,
    distance: DistanceEngine,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FraudAnalysis {
    pub approved: bool,
    pub fraud_score: f32,
    pub neighbor_indices: [usize; K_NEIGHBORS],
}

impl FraudDetector {
    pub fn new(dataset: Arc<ReferenceDataset>, distance: DistanceEngine) -> Self {
        assert!(dataset.len() >= K_NEIGHBORS);
        Self { dataset, distance }
    }

    pub fn analyze(&self, req: &FraudRequest) -> FraudAnalysis {
        self.analyze_features(&normalize_request(req))
    }

    pub fn analyze_features(&self, features: &[f32; FEATURE_DIM]) -> FraudAnalysis {
        let query = FeatureVector::from_features(*features);
        let neighbors = self.closest_neighbors(&query);
        score_from_neighbor_indices(self.dataset.labels(), neighbors.indices)
    }

    #[inline(always)]
    fn closest_neighbors(&self, features: &FeatureVector) -> NeighborSet {
        let mut best_distances = [f32::INFINITY; K_NEIGHBORS];
        let mut best_indices = [usize::MAX; K_NEIGHBORS];
        let last = K_NEIGHBORS - 1;

        for (index, vector) in self.dataset.vectors().iter().enumerate() {
            let distance = self.distance.measure(features, vector);
            if !is_better(distance, index, best_distances[last], best_indices[last]) {
                continue;
            }

            insert_neighbor(distance, index, &mut best_distances, &mut best_indices);
        }

        NeighborSet {
            indices: best_indices,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct NeighborSet {
    indices: [usize; K_NEIGHBORS],
}

#[inline(always)]
fn insert_neighbor(
    distance: f32,
    index: usize,
    best_distances: &mut [f32; K_NEIGHBORS],
    best_indices: &mut [usize; K_NEIGHBORS],
) {
    let mut slot = K_NEIGHBORS;

    while slot > 0
        && is_better(
            distance,
            index,
            best_distances[slot - 1],
            best_indices[slot - 1],
        )
    {
        slot -= 1;
    }

    if slot == K_NEIGHBORS {
        return;
    }

    for idx in (slot + 1..K_NEIGHBORS).rev() {
        best_distances[idx] = best_distances[idx - 1];
        best_indices[idx] = best_indices[idx - 1];
    }

    best_distances[slot] = distance;
    best_indices[slot] = index;
}

#[inline(always)]
fn is_better(
    candidate_distance: f32,
    candidate_index: usize,
    best_distance: f32,
    best_index: usize,
) -> bool {
    candidate_distance < best_distance
        || (candidate_distance == best_distance && candidate_index < best_index)
}

#[inline(always)]
fn score_from_neighbor_indices(
    labels: &[u8],
    neighbor_indices: [usize; K_NEIGHBORS],
) -> FraudAnalysis {
    let fraud_count = neighbor_indices
        .into_iter()
        .filter(|&index| labels[index] == LABEL_FRAUD)
        .count();
    let fraud_score = fraud_count as f32 / K_NEIGHBORS as f32;

    FraudAnalysis {
        approved: fraud_score < 0.6,
        fraud_score,
        neighbor_indices,
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        sync::{Arc, OnceLock},
    };

    use chrono::{TimeZone, Utc};
    use serde::Deserialize;

    use super::{FraudDetector, K_NEIGHBORS, score_from_neighbor_indices};
    use crate::{
        dataset::{LABEL_FRAUD, LABEL_LEGIT, ReferenceDataset},
        distance::DistanceEngine,
        normalization::{FEATURE_DIM, normalize_request},
        payload::{Customer, FraudRequest, LastTransaction, Merchant, Terminal, Transaction},
    };

    #[derive(Deserialize)]
    struct ReferenceRecord {
        vector: [f32; FEATURE_DIM],
        label: String,
    }

    fn sample_request() -> FraudRequest {
        FraudRequest {
            id: "tx-1".into(),
            transaction: Transaction {
                amount: 41.12,
                installments: 2,
                requested_at: Utc.with_ymd_and_hms(2026, 3, 11, 18, 45, 53).unwrap(),
            },
            customer: Customer {
                avg_amount: 82.24,
                tx_count_24h: 3,
                known_merchants: vec!["MERC-003".into(), "MERC-016".into()],
            },
            merchant: Merchant {
                id: "MERC-016".into(),
                mcc: "5411".into(),
                avg_amount: 60.25,
            },
            terminal: Terminal {
                is_online: false,
                card_present: true,
                km_from_home: 29.2331036248,
            },
            last_transaction: Some(LastTransaction {
                timestamp: Utc.with_ymd_and_hms(2026, 3, 11, 14, 58, 35).unwrap(),
                km_from_current: 18.8626479774,
            }),
        }
    }

    fn load_fixture_payloads() -> &'static Vec<FraudRequest> {
        static PAYLOADS: OnceLock<Vec<FraudRequest>> = OnceLock::new();
        PAYLOADS.get_or_init(|| {
            let path = format!(
                "{}/resources/example-payloads.json",
                env!("CARGO_MANIFEST_DIR")
            );
            let body = fs::read_to_string(path).unwrap();
            serde_json::from_str(&body).unwrap()
        })
    }

    fn load_reference_records() -> &'static Vec<ReferenceRecord> {
        static RECORDS: OnceLock<Vec<ReferenceRecord>> = OnceLock::new();
        RECORDS.get_or_init(|| {
            let path = format!("{}/resources/references.json", env!("CARGO_MANIFEST_DIR"));
            let body = fs::read_to_string(path).unwrap();
            serde_json::from_str(&body).unwrap()
        })
    }

    fn reference_scan(
        features: &[f32; FEATURE_DIM],
        records: &[ReferenceRecord],
    ) -> ([usize; K_NEIGHBORS], f32) {
        let mut distances: Vec<(f32, usize, bool)> = records
            .iter()
            .enumerate()
            .map(|(index, record)| {
                let mut distance = 0.0_f32;
                for axis in 0..FEATURE_DIM {
                    let delta = features[axis] - record.vector[axis];
                    distance += delta * delta;
                }
                (distance, index, record.label == "fraud")
            })
            .collect();

        distances.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });

        let mut neighbor_indices = [usize::MAX; K_NEIGHBORS];
        let fraud_count =
            distances
                .iter()
                .take(K_NEIGHBORS)
                .enumerate()
                .fold(0usize, |acc, (offset, entry)| {
                    neighbor_indices[offset] = entry.1;
                    acc + usize::from(entry.2)
                });

        (neighbor_indices, fraud_count as f32 / K_NEIGHBORS as f32)
    }

    #[test]
    fn rejects_score_at_threshold() {
        let analysis = score_from_neighbor_indices(
            &[
                LABEL_FRAUD,
                LABEL_FRAUD,
                LABEL_FRAUD,
                LABEL_LEGIT,
                LABEL_LEGIT,
            ],
            [0, 1, 2, 3, 4],
        );

        assert_eq!(analysis.fraud_score, 0.6);
        assert!(!analysis.approved);
    }

    #[test]
    fn approves_below_threshold() {
        let analysis = score_from_neighbor_indices(
            &[
                LABEL_FRAUD,
                LABEL_FRAUD,
                LABEL_LEGIT,
                LABEL_LEGIT,
                LABEL_LEGIT,
            ],
            [0, 1, 2, 3, 4],
        );

        assert_eq!(analysis.fraud_score, 0.4);
        assert!(analysis.approved);
    }

    #[test]
    fn classifier_matches_reference_scan_for_fixture_payloads() {
        let dataset = Arc::new(ReferenceDataset::load_embedded().unwrap());
        let detector = FraudDetector::new(dataset, DistanceEngine::scalar());
        let records = load_reference_records();

        for payload in load_fixture_payloads() {
            let features = normalize_request(payload);
            let analysis = detector.analyze(payload);
            let (expected_neighbors, expected_score) = reference_scan(&features, records);

            assert_eq!(analysis.neighbor_indices, expected_neighbors);
            assert_eq!(analysis.fraud_score, expected_score);
            assert_eq!(analysis.approved, expected_score < 0.6);
        }
    }

    #[test]
    fn classifier_matches_reference_scan_for_dataset_samples() {
        let dataset = Arc::new(ReferenceDataset::load_embedded().unwrap());
        let detector = FraudDetector::new(dataset, DistanceEngine::scalar());
        let records = load_reference_records();

        for index in (0..records.len()).step_by(997).take(64) {
            let features = &records[index].vector;
            let analysis = detector.analyze_features(features);
            let (expected_neighbors, expected_score) = reference_scan(features, records);

            assert_eq!(analysis.neighbor_indices, expected_neighbors);
            assert_eq!(analysis.fraud_score, expected_score);
        }
    }

    #[test]
    fn simd_and_scalar_paths_return_identical_neighbors_when_available() {
        let Some(simd) = DistanceEngine::avx2() else {
            return;
        };

        let dataset = Arc::new(ReferenceDataset::load_embedded().unwrap());
        let scalar = FraudDetector::new(dataset.clone(), DistanceEngine::scalar());
        let simd = FraudDetector::new(dataset, simd);

        let mut queries: Vec<[f32; FEATURE_DIM]> = load_fixture_payloads()
            .iter()
            .map(normalize_request)
            .collect();
        queries.extend(
            load_reference_records()
                .iter()
                .step_by(1231)
                .take(64)
                .map(|record| record.vector),
        );

        for query in queries {
            let scalar_analysis = scalar.analyze_features(&query);
            let simd_analysis = simd.analyze_features(&query);

            assert_eq!(scalar_analysis, simd_analysis);
        }
    }

    #[test]
    fn sample_request_produces_valid_score() {
        let dataset = Arc::new(ReferenceDataset::load_embedded().unwrap());
        let detector = FraudDetector::new(dataset, DistanceEngine::scalar());

        let analysis = detector.analyze(&sample_request());

        assert!((0.0..=1.0).contains(&analysis.fraud_score));
    }
}

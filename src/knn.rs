#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReferenceEntry {
    pub vector: [i16; 14],
    pub is_fraud: bool,
}

pub fn fraud_score_top5(query: &[i16; 14], entries: &[ReferenceEntry]) -> f32 {
    let mut best = [(u64::MAX, false); 5];

    for entry in entries {
        let distance = squared_distance(query, &entry.vector);

        for pos in 0..best.len() {
            if distance < best[pos].0 {
                for shift in (pos + 1..best.len()).rev() {
                    best[shift] = best[shift - 1];
                }
                best[pos] = (distance, entry.is_fraud);
                break;
            }
        }
    }

    let frauds = best.iter().filter(|(_, is_fraud)| *is_fraud).count();
    frauds as f32 / 5.0
}

fn squared_distance(a: &[i16; 14], b: &[i16; 14]) -> u64 {
    a.iter()
        .zip(b.iter())
        .map(|(&left, &right)| {
            let diff = left as i64 - right as i64;
            (diff * diff) as u64
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(value: i16, is_fraud: bool) -> ReferenceEntry {
        ReferenceEntry {
            vector: [value; 14],
            is_fraud,
        }
    }

    #[test]
    fn returns_zero_when_no_nearest_neighbors_are_fraud() {
        let entries = [
            entry(0, false),
            entry(1, false),
            entry(2, false),
            entry(3, false),
            entry(4, false),
            entry(100, true),
        ];

        assert_eq!(fraud_score_top5(&[0; 14], &entries), 0.0);
    }

    #[test]
    fn returns_two_fifths_when_two_nearest_neighbors_are_fraud() {
        let entries = [
            entry(0, false),
            entry(1, true),
            entry(2, false),
            entry(3, true),
            entry(4, false),
            entry(100, true),
        ];

        assert_eq!(fraud_score_top5(&[0; 14], &entries), 0.4);
    }

    #[test]
    fn returns_three_fifths_at_rejection_threshold() {
        let score = fraud_score_top5(
            &[0; 14],
            &[
                entry(0, true),
                entry(1, true),
                entry(2, false),
                entry(3, true),
                entry(4, false),
                entry(100, false),
            ],
        );

        assert_eq!(score, 0.6);
        assert!(!(score < 0.6));
    }

    #[test]
    fn returns_one_when_all_nearest_neighbors_are_fraud() {
        let entries = [
            entry(0, true),
            entry(1, true),
            entry(2, true),
            entry(3, true),
            entry(4, true),
            entry(100, false),
        ];

        assert_eq!(fraud_score_top5(&[0; 14], &entries), 1.0);
    }
}

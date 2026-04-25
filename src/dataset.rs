use crate::normalization::FEATURE_DIM;
use thiserror::Error;

pub const DATASET_VERSION: u32 = 1;
pub const EXPECTED_RECORDS: usize = 100_000;
pub const LABEL_LEGIT: u8 = 0;
pub const LABEL_FRAUD: u8 = 1;
pub const HEADER_LEN: usize = 16;
pub const MAGIC: [u8; 4] = *b"CRAB";
pub const PADDED_FEATURE_DIM: usize = 16;

static EMBEDDED_DATASET: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/reference_dataset.bin"));

#[repr(C, align(32))]
#[derive(Debug, Clone, Copy)]
pub struct FeatureVector {
    lanes: [f32; PADDED_FEATURE_DIM],
}

impl FeatureVector {
    pub fn from_features(features: [f32; FEATURE_DIM]) -> Self {
        let mut lanes = [0.0_f32; PADDED_FEATURE_DIM];
        lanes[..FEATURE_DIM].copy_from_slice(&features);
        Self { lanes }
    }

    pub fn features(&self) -> &[f32] {
        &self.lanes[..FEATURE_DIM]
    }

    pub fn padded(&self) -> &[f32; PADDED_FEATURE_DIM] {
        &self.lanes
    }
}

#[derive(Debug)]
pub struct ReferenceDataset {
    vectors: Box<[FeatureVector]>,
    labels: Box<[u8]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetHeader {
    pub version: u32,
    pub count: u32,
    pub dimension: u32,
}

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("embedded dataset blob is truncated")]
    Truncated,
    #[error("invalid dataset magic: expected {expected:?}, found {found:?}")]
    InvalidMagic { expected: [u8; 4], found: [u8; 4] },
    #[error("invalid dataset version: expected {expected}, found {found}")]
    InvalidVersion { expected: u32, found: u32 },
    #[error("invalid dataset record count: expected {expected}, found {found}")]
    InvalidRecordCount { expected: usize, found: usize },
    #[error("invalid dataset dimension: expected {expected}, found {found}")]
    InvalidDimension { expected: usize, found: usize },
    #[error("invalid dataset label byte {0}")]
    InvalidLabel(u8),
}

impl ReferenceDataset {
    pub fn load_embedded() -> Result<Self, DatasetError> {
        let header = parse_header(EMBEDDED_DATASET)?;
        validate_header(header)?;

        let vectors_len =
            usize::try_from(header.count).unwrap() * FEATURE_DIM * std::mem::size_of::<f32>();
        let labels_len = usize::try_from(header.count).unwrap();
        let expected_total = HEADER_LEN + vectors_len + labels_len;

        if EMBEDDED_DATASET.len() != expected_total {
            return Err(DatasetError::Truncated);
        }

        let mut vectors = Vec::with_capacity(header.count as usize);
        let mut offset = HEADER_LEN;
        for _ in 0..header.count {
            let mut features = [0.0_f32; FEATURE_DIM];
            for value in &mut features {
                *value = read_f32(EMBEDDED_DATASET, &mut offset)?;
            }
            vectors.push(FeatureVector::from_features(features));
        }

        let mut labels = Vec::with_capacity(labels_len);
        for &label in &EMBEDDED_DATASET[offset..] {
            if label > LABEL_FRAUD {
                return Err(DatasetError::InvalidLabel(label));
            }
            labels.push(label);
        }

        Ok(Self {
            vectors: vectors.into_boxed_slice(),
            labels: labels.into_boxed_slice(),
        })
    }

    pub fn header() -> Result<DatasetHeader, DatasetError> {
        parse_header(EMBEDDED_DATASET)
    }

    pub fn vectors(&self) -> &[FeatureVector] {
        &self.vectors
    }

    pub fn labels(&self) -> &[u8] {
        &self.labels
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }
}

fn validate_header(header: DatasetHeader) -> Result<(), DatasetError> {
    if header.version != DATASET_VERSION {
        return Err(DatasetError::InvalidVersion {
            expected: DATASET_VERSION,
            found: header.version,
        });
    }

    if header.count as usize != EXPECTED_RECORDS {
        return Err(DatasetError::InvalidRecordCount {
            expected: EXPECTED_RECORDS,
            found: header.count as usize,
        });
    }

    if header.dimension as usize != FEATURE_DIM {
        return Err(DatasetError::InvalidDimension {
            expected: FEATURE_DIM,
            found: header.dimension as usize,
        });
    }

    Ok(())
}

fn parse_header(bytes: &[u8]) -> Result<DatasetHeader, DatasetError> {
    if bytes.len() < HEADER_LEN {
        return Err(DatasetError::Truncated);
    }

    let found_magic = bytes[..4].try_into().unwrap();
    if found_magic != MAGIC {
        return Err(DatasetError::InvalidMagic {
            expected: MAGIC,
            found: found_magic,
        });
    }

    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let dimension = u32::from_le_bytes(bytes[12..16].try_into().unwrap());

    Ok(DatasetHeader {
        version,
        count,
        dimension,
    })
}

fn read_f32(bytes: &[u8], offset: &mut usize) -> Result<f32, DatasetError> {
    let end = *offset + std::mem::size_of::<f32>();
    let slice = bytes.get(*offset..end).ok_or(DatasetError::Truncated)?;
    *offset = end;
    Ok(f32::from_le_bytes(slice.try_into().unwrap()))
}

#[cfg(test)]
mod tests {
    use std::{fs, sync::OnceLock};

    use serde::Deserialize;

    use super::{EXPECTED_RECORDS, LABEL_FRAUD, LABEL_LEGIT, ReferenceDataset};
    use crate::normalization::FEATURE_DIM;

    #[derive(Deserialize)]
    struct ReferenceRecord {
        vector: [f32; FEATURE_DIM],
        label: String,
    }

    fn load_reference_records() -> &'static Vec<ReferenceRecord> {
        static RECORDS: OnceLock<Vec<ReferenceRecord>> = OnceLock::new();
        RECORDS.get_or_init(|| {
            let path = format!("{}/resources/references.json", env!("CARGO_MANIFEST_DIR"));
            let body = fs::read_to_string(path).unwrap();
            serde_json::from_str(&body).unwrap()
        })
    }

    #[test]
    fn embedded_header_matches_expected_shape() {
        let header = ReferenceDataset::header().unwrap();

        assert_eq!(header.version, 1);
        assert_eq!(header.count as usize, EXPECTED_RECORDS);
        assert_eq!(header.dimension as usize, FEATURE_DIM);
    }

    #[test]
    fn embedded_dataset_matches_reference_json_exactly() {
        let dataset = ReferenceDataset::load_embedded().unwrap();
        let reference = load_reference_records();

        assert_eq!(dataset.len(), reference.len());

        for (index, record) in reference.iter().enumerate() {
            assert_eq!(dataset.vectors()[index].features(), record.vector);
            let expected_label = if record.label == "fraud" {
                LABEL_FRAUD
            } else {
                LABEL_LEGIT
            };
            assert_eq!(dataset.labels()[index], expected_label);
        }
    }

    #[test]
    fn embedded_vectors_keep_padding_zeroed() {
        let dataset = ReferenceDataset::load_embedded().unwrap();

        assert_eq!(dataset.vectors()[0].padded()[FEATURE_DIM], 0.0);
        assert_eq!(dataset.vectors()[0].padded()[FEATURE_DIM + 1], 0.0);
    }
}

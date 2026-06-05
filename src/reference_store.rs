use std::fmt;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use flate2::read::GzDecoder;
use serde::Deserialize;

use crate::ivf::IvfIndex;
use crate::knn::{ReferenceEntry, fraud_score_top5};
use crate::quantizer::quantize_int16;

const REFERENCES_JSON_GZ: &[u8] = include_bytes!("../resources/references.json.gz");
const MIN_REFERENCE_ENTRIES: usize = 5;

// IVF defaults. nlist is derived from N (~sqrt N); these bound the build cost and
// the speed/recall dial. Tune nprobe with the eval harness (see src/eval.rs).
const DEFAULT_NPROBE: usize = 16;
const DEFAULT_MAX_ITERS: usize = 10;
const DEFAULT_TRAIN_SAMPLE: usize = 100_000;

#[derive(Debug)]
pub struct ReferenceStore {
    index: IvfIndex,
    nprobe: usize,
}

impl ReferenceStore {
    pub fn from_embedded_gzip() -> Result<Self, ReferenceLoadError> {
        Self::from_gzip_reader(REFERENCES_JSON_GZ)
    }

    pub fn from_gzip_reader<R: Read>(reader: R) -> Result<Self, ReferenceLoadError> {
        Self::from_json_reader(GzDecoder::new(reader))
    }

    pub fn from_json_reader<R: Read>(reader: R) -> Result<Self, ReferenceLoadError> {
        let entries = parse_entries(reader)?;

        // Build the IVF index: train the coarse quantizer on a sample, bin all entries.
        let nlist = default_nlist(entries.len());
        let index =
            IvfIndex::build_sampled(&entries, nlist, DEFAULT_MAX_ITERS, DEFAULT_TRAIN_SAMPLE);

        Ok(Self {
            index,
            nprobe: DEFAULT_NPROBE.min(nlist),
        })
    }

    /// Persist the built index + nprobe to a `.bin` file (raw layout, see
    /// `IvfIndex::write_to`). nprobe is appended as a trailing u64.
    pub fn save_to_path(&self, path: &Path) -> Result<(), ReferenceLoadError> {
        let mut writer = BufWriter::new(File::create(path)?);
        self.index.write_to(&mut writer)?;
        writer.write_all(&(self.nprobe as u64).to_le_bytes())?;
        writer.flush()?;
        Ok(())
    }

    /// Load a prebuilt index from a `.bin` file. Fast path for boot — skips the
    /// gzip decode + k-means build entirely.
    pub fn from_bin_path(path: &Path) -> Result<Self, ReferenceLoadError> {
        let mut reader = BufReader::new(File::open(path)?);
        let index = IvfIndex::read_from(&mut reader)?;
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        let nprobe = u64::from_le_bytes(buf) as usize;
        Ok(Self { index, nprobe })
    }

    /// Approximate fraud score via IVF: probe `nprobe` cells, top-5, fraud ratio.
    pub fn fraud_score_for(&self, query: &[i16; 14]) -> f32 {
        self.index.fraud_score(query, self.nprobe)
    }

    /// Exact fraud score over every entry — the ground truth for recall measurement.
    pub fn fraud_score_bruteforce(&self, query: &[i16; 14]) -> f32 {
        fraud_score_top5(query, &self.index.entries)
    }

    pub fn nprobe(&self) -> usize {
        self.nprobe
    }

    /// Adjust the speed/recall dial (used by the eval harness to sweep nprobe).
    pub fn set_nprobe(&mut self, nprobe: usize) {
        self.nprobe = nprobe.clamp(1, self.index.nlist());
    }

    pub fn index(&self) -> &IvfIndex {
        &self.index
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.index.entries.len()
    }
}

/// Parse + validate the reference rows (order preserved — index build happens after).
fn parse_entries<R: Read>(reader: R) -> Result<Vec<ReferenceEntry>, ReferenceLoadError> {
    let entries: Vec<ReferenceEntry> = serde_json::from_reader(reader)?;

    if entries.len() < MIN_REFERENCE_ENTRIES {
        return Err(ReferenceLoadError::NotEnoughEntries { len: entries.len() });
    }

    Ok(entries)
}

/// Rule of thumb: nlist ≈ √N, clamped to [1, N].
fn default_nlist(n: usize) -> usize {
    ((n as f64).sqrt() as usize).clamp(1, n)
}

#[derive(Debug)]
pub enum ReferenceLoadError {
    Io(std::io::Error),
    Json(serde_json::Error),
    NotEnoughEntries { len: usize },
}

impl fmt::Display for ReferenceLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "failed to read reference dataset: {err}"),
            Self::Json(err) => write!(f, "failed to parse reference dataset: {err}"),
            Self::NotEnoughEntries { len } => {
                write!(
                    f,
                    "reference dataset has {len} entries; expected at least 5"
                )
            }
        }
    }
}

impl std::error::Error for ReferenceLoadError {}

impl From<std::io::Error> for ReferenceLoadError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<serde_json::Error> for ReferenceLoadError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err)
    }
}

impl<'de> Deserialize<'de> for ReferenceEntry {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawReferenceEntry {
            #[serde(deserialize_with = "deserialize_quantized_vector")]
            vector: [i16; 14],
            #[serde(rename = "label", deserialize_with = "deserialize_is_fraud")]
            is_fraud: bool,
        }

        let raw = RawReferenceEntry::deserialize(deserializer)?;

        Ok(Self {
            vector: raw.vector,
            is_fraud: raw.is_fraud,
        })
    }
}

fn deserialize_quantized_vector<'de, D>(deserializer: D) -> Result<[i16; 14], D::Error>
where
    D: serde::Deserializer<'de>,
{
    let vector = Vec::<f32>::deserialize(deserializer)?;

    let features: [f32; 14] = vector
        .try_into()
        .map_err(|vector: Vec<f32>| serde::de::Error::invalid_length(vector.len(), &"14"))?;

    Ok(quantize_int16(&features))
}

fn deserialize_is_fraud<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let label = String::deserialize(deserializer)?;

    match label.as_str() {
        "fraud" => Ok(true),
        "legit" => Ok(false),
        _ => Err(serde::de::Error::unknown_variant(
            &label,
            &["fraud", "legit"],
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::Compression;
    use flate2::write::GzEncoder;

    use super::*;

    const FIVE_ROWS_JSON: &[u8] = br#"[
        {"vector":[0,0,0,0,0,0,0,0,0,0,0,0,0,0],"label":"legit"},
        {"vector":[1,1,1,1,1,1,1,1,1,1,1,1,1,1],"label":"fraud"},
        {"vector":[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],"label":"legit"},
        {"vector":[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],"label":"fraud"},
        {"vector":[0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25],"label":"legit"}
    ]"#;

    #[test]
    fn parses_and_quantizes_reference_rows() {
        // Check parsing/quantization on the raw entries — the index reorders them,
        // so order-dependent assertions must run before the build.
        let entries = parse_entries(FIVE_ROWS_JSON).unwrap();

        assert_eq!(entries.len(), 5);
        assert_eq!(entries[0].vector, [0; 14]);
        assert!(!entries[0].is_fraud);
        assert_eq!(entries[1].vector, [32767; 14]);
        assert!(entries[1].is_fraud);
        assert_eq!(entries[2].vector, [16384; 14]);
        assert_eq!(entries[3].vector, [-32767; 14]);
    }

    #[test]
    fn decompresses_gzip_json() {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(FIVE_ROWS_JSON).unwrap();
        let compressed = encoder.finish().unwrap();

        let store = ReferenceStore::from_gzip_reader(compressed.as_slice()).unwrap();

        assert_eq!(store.len(), 5);
    }

    #[test]
    fn rejects_invalid_vector_length() {
        let json = br#"[{"vector":[0,1],"label":"legit"}]"#;
        let err = ReferenceStore::from_json_reader(json.as_slice()).unwrap_err();

        assert!(matches!(err, ReferenceLoadError::Json(_)));
    }

    #[test]
    fn rejects_invalid_label() {
        let json = br#"[{"vector":[0,0,0,0,0,0,0,0,0,0,0,0,0,0],"label":"unknown"}]"#;
        let err = ReferenceStore::from_json_reader(json.as_slice()).unwrap_err();

        assert!(matches!(err, ReferenceLoadError::Json(_)));
    }

    #[test]
    fn bin_roundtrip_preserves_scores() {
        let store = ReferenceStore::from_json_reader(FIVE_ROWS_JSON).unwrap();

        let mut buf = Vec::new();
        {
            let mut writer = std::io::BufWriter::new(&mut buf);
            store.index.write_to(&mut writer).unwrap();
            writer
                .write_all(&(store.nprobe as u64).to_le_bytes())
                .unwrap();
            writer.flush().unwrap();
        }

        let mut reader = std::io::Cursor::new(buf);
        let index = IvfIndex::read_from(&mut reader).unwrap();
        let mut tail = [0u8; 8];
        reader.read_exact(&mut tail).unwrap();
        let nprobe = u64::from_le_bytes(tail) as usize;
        let loaded = ReferenceStore { index, nprobe };

        assert_eq!(loaded.len(), store.len());
        assert_eq!(loaded.nprobe(), store.nprobe());
        let query = [0i16; 14];
        assert_eq!(
            loaded.fraud_score_for(&query),
            store.fraud_score_for(&query)
        );
    }

    #[test]
    #[ignore = "loads and parses the full embedded 3M-row dataset"]
    fn loads_embedded_reference_dataset() {
        let store = ReferenceStore::from_embedded_gzip().unwrap();

        assert_eq!(store.len(), 3_000_000);
    }
}

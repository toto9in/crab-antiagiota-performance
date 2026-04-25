use std::{
    env,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use serde::Deserialize;

const DATASET_VERSION: u32 = 1;
const FEATURE_DIM: usize = 14;
const EXPECTED_RECORDS: usize = 100_000;
const LABEL_LEGIT: u8 = 0;
const LABEL_FRAUD: u8 = 1;
const MAGIC: [u8; 4] = *b"CRAB";

#[derive(Deserialize)]
struct ReferenceRecord {
    vector: [f32; FEATURE_DIM],
    label: String,
}

fn main() {
    println!("cargo:rerun-if-changed=resources/references.json");
    println!("cargo:rerun-if-changed=build.rs");

    if let Err(error) = build_dataset_blob() {
        panic!("failed to build embedded dataset: {error}");
    }
}

fn build_dataset_blob() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")?;
    let input_path = Path::new(&manifest_dir).join("resources/references.json");
    let out_dir = env::var("OUT_DIR")?;
    let output_path = Path::new(&out_dir).join("reference_dataset.bin");

    let file = File::open(&input_path)?;
    let records: Vec<ReferenceRecord> = serde_json::from_reader(file)?;

    if records.len() != EXPECTED_RECORDS {
        return Err(format!(
            "expected {EXPECTED_RECORDS} records, found {}",
            records.len()
        )
        .into());
    }

    let mut writer = BufWriter::new(File::create(output_path)?);
    writer.write_all(&MAGIC)?;
    writer.write_all(&DATASET_VERSION.to_le_bytes())?;
    writer.write_all(&(records.len() as u32).to_le_bytes())?;
    writer.write_all(&(FEATURE_DIM as u32).to_le_bytes())?;

    for record in &records {
        for value in record.vector {
            writer.write_all(&value.to_le_bytes())?;
        }
    }

    for record in records {
        let label = match record.label.as_str() {
            "legit" => LABEL_LEGIT,
            "fraud" => LABEL_FRAUD,
            other => return Err(format!("unexpected label {other:?}").into()),
        };
        writer.write_all(&[label])?;
    }

    writer.flush()?;

    Ok(())
}

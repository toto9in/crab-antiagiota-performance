//! Offline index builder. Decodes the embedded reference dataset, trains the
//! IVF coarse quantizer, bins all entries, and writes the result to a `.bin`
//! the API loads at boot. Run once at image-build time so the API never pays
//! the k-means cost on startup.
//!
//! Usage: build-index [OUTPUT_PATH]   (default: index.bin)

use std::env;
use std::path::Path;

use crab_antiagiota_performance::reference_store::ReferenceStore;

fn main() {
    let out = env::args().nth(1).unwrap_or_else(|| "index.bin".into());

    eprintln!("building index from embedded dataset...");
    let store = ReferenceStore::from_embedded_gzip().expect("build index from dataset");

    store
        .save_to_path(Path::new(&out))
        .expect("write index .bin");

    eprintln!("wrote index to {out}");
}

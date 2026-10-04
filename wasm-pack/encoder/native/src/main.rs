//! Build-only encoder. Python owns publication; this process owns binary algorithms.
//! Interchange is limited to original bytes, final WXP1/PRD1 payloads and JSON metadata.
mod analysis;
mod delta;
mod matcher;
mod prediction;
use std::{fs, path::Path, time::Instant};
type Error = Box<dyn std::error::Error>;
type Result<T> = std::result::Result<T, Error>;
const MAX_BYTES: usize = 64 * 1024 * 1024;
fn put_u32(output: &mut Vec<u8>, value: usize) -> Result<()> {
    output.extend(u32::try_from(value)?.to_le_bytes());
    Ok(())
}

fn read_input(path: &Path) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() || metadata.len() > MAX_BYTES as u64 {
        return Err("input must be a regular file no larger than 64 MiB".into());
    }
    let bytes = fs::read(path)?;
    if bytes.len() > MAX_BYTES {
        return Err("input grew beyond limit".into());
    }
    Ok(bytes)
}

fn write_payload(directory: &Path, name: &str, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    if bytes.len() > MAX_BYTES {
        return Err("encoded payload exceeds limit".into());
    }
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join(name))?
        .write_all(bytes)?;
    Ok(())
}

fn execute() -> Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() == 3 && args[1] == "validate" {
        analysis::validate(&read_input(Path::new(&args[2]))?)?;
        return Ok(());
    }
    if args.len() != 5 || args[1] != "pair" {
        return Err(
            "usage: bicore-wasm-encoder validate INPUT | pair BASE TARGET NEW_OUTPUT_DIRECTORY"
                .into(),
        );
    }
    let started = Instant::now();
    let base = read_input(Path::new(&args[2]))?;
    let target = read_input(Path::new(&args[3]))?;
    let base_index = analysis::parse(&base)?;
    let target_index = analysis::parse(&target)?;
    let prediction = prediction::train(&base, &base_index, &target, &target_index)?;
    let analysis_seconds = started.elapsed().as_secs_f64();
    // Release parser state before allocating the matcher's history tables.
    drop(base_index);
    drop(target_index);
    let directory = Path::new(&args[4]);
    fs::create_dir(directory)?;
    // Refuse an existing directory; caller owns atomic publication.
    let plain = delta::encode(&base, &target, &matcher::find_matches(&base, &target)?)?;
    write_payload(directory, "plain.wxp", &plain)?;
    let mut candidates = vec![
        serde_json::json!({ "label" : "plain", "recipe" : "plain.wxp" })
    ];
    let mut rule_count = 0;
    if let Some(prediction) = prediction {
        let recipe = delta::encode(
            &prediction.dictionary,
            &target,
            &matcher::find_matches(&prediction.dictionary, &target)?,
        )?;
        write_payload(directory, "predicted.wxp", &recipe)?;
        write_payload(directory, "prediction.prd", &prediction.rules)?;
        // Build-only output for independent hashing. This dictionary is not distributed.
        write_payload(directory, "dictionary.bin", &prediction.dictionary)?;
        rule_count = prediction.rule_count;
        candidates
            .push(
                serde_json::json!(
                    { "label" : "predicted", "recipe" : "predicted.wxp", "prediction" :
                    "prediction.prd", "dictionary" : "dictionary.bin" }
                ),
            );
    }
    let report = serde_json::json!(
        { "formatVersion" : 1, "candidates" : candidates, "ruleCount" : rule_count,
        "analysisSeconds" : analysis_seconds, "seconds" : started.elapsed().as_secs_f64()
        }
    );
    write_payload(
        directory,
        "result.json",
        serde_json::to_string_pretty(&report)?.as_bytes(),
    )?;
    Ok(())
}

fn main() {
    if let Err(error) = execute() {
        eprintln!("Wasm encoder: {error}");
        std::process::exit(1);
    }
}

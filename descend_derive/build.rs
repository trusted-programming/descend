use std::fs;
use std::path::Path;

fn main() {
    // Tell Cargo to re-run this build script if any .desc files change
    let examples_dir = "examples/core";
    
    if Path::new(examples_dir).exists() {
        // Walk through the directory and tell Cargo to re-run if any .desc files change
        if let Ok(entries) = fs::read_dir(examples_dir) {
            for entry in entries.flatten() {
                if let Some(path) = entry.path().to_str() {
                    if path.ends_with(".desc") {
                        println!("cargo:rerun-if-changed={}", path);
                    }
                }
            }
        }
    }
    
    // Also watch the entire directory for new files
    println!("cargo:rerun-if-changed={}", examples_dir);
}

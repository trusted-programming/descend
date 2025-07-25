use clap::{Parser, ValueEnum};
use std::path::{PathBuf};
use std::fs;
use descend::{self, compile};

/// CLI for compiling Descend source files to selected backend
#[derive(Parser, Debug)]
#[command(version, about = "Descend compiler CLI")]
struct Args {
    /// Backend to use (cuda or ascendc)
    #[arg(value_enum)]
    backend: BackendArg,

    /// Path to Descend source file
    descend_file: PathBuf,

    /// Output directory (optional, default is current directory)
    #[arg(short, long, default_value = ".")]
    output_dir: PathBuf,
}

/// Backend selection passed via CLI
#[derive(Copy, Clone, Debug, ValueEnum)]
enum BackendArg {
    Cuda,
    AscendC,
}

impl From<BackendArg> for descend::Backend {
    fn from(arg: BackendArg) -> Self {
        match arg {
            BackendArg::Cuda => descend::Backend::Cuda,
            BackendArg::AscendC => descend::Backend::AscendC,
        }
    }
}

fn main() {
    let args = Args::parse();

    let backend = args.backend.into();
    let input_path = &args.descend_file;
    let output_dir = &args.output_dir;

    // Compile using Descend
    let output_string = match compile(&input_path.to_string_lossy(), backend) {
        Ok(output) => output,
        Err(e) => {
            eprintln!("Compilation failed: {:?}", e);
            std::process::exit(1);
        }
    };

    // Generate output file path (same name but with .out extension)
    let filename_stem = input_path.file_stem().unwrap_or_default();
    let output_file = output_dir.join(format!("{}.out", filename_stem.to_string_lossy()));

    // Write result to file
    if let Err(e) = fs::write(&output_file, output_string) {
        eprintln!("Failed to write output file: {}", e);
        std::process::exit(1);
    }

    println!("Compiled successfully: {}", output_file.display());
}


use clap::{Parser, ValueEnum};
use descend::{self, compile};
use std::fs;
use std::path::PathBuf;

/// CLI for compiling Descend source files to selected backend
#[derive(Parser, Debug)]
#[command(version, about = "Descend compiler CLI")]
struct Args {
    /// Path to Descend source file
    descend_file: PathBuf,

    /// Backend to use (cuda or mlir)
    #[arg(value_enum, default_value = "cuda")]
    backend: BackendArg,

    /// Output directory (optional, default is current directory)
    #[arg(short, long, default_value = ".")]
    output_dir: PathBuf,

    /// Print Ast
    #[arg(short, long)]
    print_ast: bool,
}

/// Backend selection passed via CLI
#[derive(Copy, Clone, Debug, ValueEnum)]
enum BackendArg {
    Cuda,
    Mlir,
}

impl From<BackendArg> for descend::Backend {
    fn from(arg: BackendArg) -> Self {
        match arg {
            BackendArg::Cuda => descend::Backend::Cuda,
            BackendArg::Mlir => descend::Backend::Mlir,
        }
    }
}

fn main() {
    let args = Args::parse();

    let backend = args.backend.into();
    let print_ast = args.print_ast.into();
    let input_path = &args.descend_file;
    let output_dir = &args.output_dir;

    // Compile using Descend
    let (code_string, ast_string) = match compile(&input_path.to_string_lossy(), backend) {
        Ok(output) => output,
        Err(e) => {
            eprintln!("Compilation failed: {:?}", e);
            std::process::exit(1);
        }
    };

    // Generate output file path with appropriate extension based on backend
    let filename_stem = input_path.file_stem().unwrap_or_default().to_string_lossy();
    let extension = match args.backend {
        BackendArg::Cuda => "cu",
        BackendArg::Mlir => "mlir",
    };
    let code_file = output_dir.join(format!("{}.{}", filename_stem, extension));
    if print_ast {
        let ast_file = output_dir.join(format!("{}.ast", filename_stem));
        // Write result to file
        if let Err(e) = fs::write(&ast_file, ast_string) {
            eprintln!("Failed to write output file: {}", e);
            std::process::exit(1);
        }
        println!("AST file written successfully to: {}", ast_file.display());
    }
    // Write result to file
    if let Err(e) = fs::write(&code_file, code_string) {
        eprintln!("Failed to write output file: {}", e);
        std::process::exit(1);
    }
    println!("code file written successfully to: {}", code_file.display());
}

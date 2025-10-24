use clap::Parser;
use descend;
use miette::IntoDiagnostic;
use std::fs;
use std::path::PathBuf;

/// CLI for compiling Descend source files to selected backend
#[derive(Parser, Debug)]
#[command(version, about = "Descend compiler CLI")]
struct Args {
    /// Path to Descend source file
    descend_file: PathBuf,

    /// Output directory (optional, default is current directory)
    #[arg(short, long, default_value = ".")]
    output_dir: PathBuf,

    /// Print Ast
    #[arg(short, long)]
    print_ast: bool,
}

fn main() -> miette::Result<()> {
    miette::set_panic_hook();

    let args = Args::parse();

    let print_ast = args.print_ast.into();
    let input_path = &args.descend_file;
    let output_dir = &args.output_dir;

    // Compile using Descend
    let (code_string, ast_string) = descend::compile_with_source(&input_path.to_string_lossy())?;

    // Generate output file path with appropriate extension
    let filename_stem = input_path.file_stem().unwrap_or_default().to_string_lossy();
    let extension = "mlir";

    let code_file = output_dir.join(format!("{}.{}", filename_stem, extension));
    if print_ast {
        let ast_file = output_dir.join(format!("{}.ast", filename_stem));
        // Write result to file
        fs::write(&ast_file, ast_string).into_diagnostic()?;
        println!("AST file written successfully to: {}", ast_file.display());
    }
    // Write result to file
    fs::write(&code_file, code_string).into_diagnostic()?;
    println!("code file written successfully to: {}", code_file.display());

    Ok(())
}

//! # `fzvk-shader`
//! Macros for compiling GLSL and HLSL shaders, generating types that provide
//! compile-time `fzvk` validations and cross-language type checking.
//!
//! ### Simple
//! ```rust,no_run
//! fzvk_shader::glsl!{
//!     r#"#version 460 core
//!     // Stage specified using pragma:
//!     #pragma shader_stage(compute)
//!     void main {
//!         //...
//!     }
//!     "#
//! }
//! ```
//! ### Advanced
//! ```rust,no_run
//! fzvk_shader::glsl!{
//!     // Specify a file to read the source code from, relative to workspace
//!     // root.
//!     file: "path/to/file.glsl",
//!     // Specify a string literal containing source code.
//!     source: "<some glsl source code>",
//!     // Specify defines applied to the source code.
//!     defines: {
//!         // Arbitrary strings are allowed.
//!         "hello": "world",
//!         // So are literals. Quotes around name are optional.
//!         number_of_horses: 2,
//!         // FIXME: not yet supported. use quotes around the value.
//!         "number_of_antihorses": "-2",
//!         // FIXME: not yet supported. use quotes around the value.
//!         "radians_per_hour": "2.598",
//!         "can_yippee": true,
//!         // Define without value, equivalent to setting it to "".
//!         defined,
//!     }
//! }
//! ```
use proc_macro::{TokenStream, TokenTree};
use rustc_hash::FxHashMap;

mod codegen;
mod module;

trait OperandExt {
    /// Unwrap a literal Bool32.
    fn unwrap_literal_bool(&self) -> bool;
}
impl OperandExt for rspirv::dr::Operand {
    fn unwrap_literal_bool(&self) -> bool {
        match self.unwrap_literal_bit32() {
            0 => false,
            1 => true,
            _ => panic!("invalid boolean literal"),
        }
    }
}

/// Where the shader code is extracted from.
enum Source {
    /// A file on disk, with the same root directory as `include_bytes`.
    File(std::path::PathBuf),
    /// A literal string containing all of the source code.
    Literal(String),
}
/// Parse the define syntax from a `{braced}` group. Defines are a comma
/// separated list of keys with optional values.
fn parse_defines(
    group: &proc_macro::Group,
) -> Result<FxHashMap<String, Option<String>>, CompilationSettingsError> {
    if !matches!(group.delimiter(), proc_macro::Delimiter::Brace) {
        return Err(CompilationSettingsError {
            error: "expected {",
            span: group.span(),
        });
    }
    let mut defines = FxHashMap::default();
    let mut tokens = group.stream().into_iter().peekable();
    loop {
        let Some(name) = tokens.next() else {
            // Done parsing.
            break;
        };
        // Accept a literal string or an ident (so, with or without quotes).
        let name_str = match &name {
            TokenTree::Ident(ident) => format!("{ident}"),
            TokenTree::Literal(lit) => match litrs::Literal::from(lit) {
                litrs::Literal::String(str) => str.value().to_owned(),
                _ => {
                    return Err(CompilationSettingsError {
                        error: "expected a string literal or identifier",
                        span: lit.span(),
                    });
                }
            },
            _ => {
                return Err(CompilationSettingsError {
                    error: "expected a string literal or identifier",
                    span: name.span(),
                });
            }
        };
        // Accept an optional ":" character.
        let Some(delim) = tokens.next() else {
            if defines.insert(name_str, Some(String::new())).is_some() {
                return Err(CompilationSettingsError {
                    error: "duplicate define",
                    span: name.span(),
                });
            }
            continue;
        };
        if is_punct(&delim, ':') {
            // Has a value, parse it!
            let value = tokens.next().ok_or(CompilationSettingsError {
                error: "expected value after delimeter",
                span: delim.span(),
            })?;
            let value: String = match &value {
                TokenTree::Ident(ident) => {
                    let name = format!("{ident}");
                    match name.as_str() {
                        "true" => "true".to_owned(),
                        "false" => "false".to_owned(),
                        _ => {
                            return Err(CompilationSettingsError {
                                error: "unexpected ident",
                                span: value.span(),
                            });
                        }
                    }
                }
                TokenTree::Literal(lit) => match litrs::Literal::from(lit) {
                    // FIXME: never called. `true` is an ident.
                    litrs::Literal::Bool(b) => b.as_str().to_owned(),
                    litrs::Literal::Float(_f) => unimplemented!(),
                    litrs::Literal::Integer(i) => format!("{}", i.value::<u128>().unwrap()),
                    litrs::Literal::String(s) => s.value().to_owned(),
                    litrs::Literal::Byte(b) => format!("{}", b.value().to_owned()),
                    litrs::Literal::Char(_) | litrs::Literal::ByteString(_) => {
                        return Err(CompilationSettingsError {
                            error: "unsupported literal type",
                            span: lit.span(),
                        });
                    }
                },
                tok => {
                    return Err(CompilationSettingsError {
                        error: "unexpected token",
                        span: tok.span(),
                    });
                }
            };
            if defines.insert(name_str, Some(value)).is_some() {
                return Err(CompilationSettingsError {
                    error: "duplicate define",
                    span: name.span(),
                });
            }
            let Some(comma) = tokens.next() else {
                break;
            };
            if !is_punct(&comma, ',') {
                return Err(CompilationSettingsError {
                    error: "expected comma",
                    span: comma.span(),
                });
            }
        } else if is_punct(&delim, ',') {
            // Does not have a value, write it with `None`.
            if defines.insert(name_str, None).is_some() {
                return Err(CompilationSettingsError {
                    error: "duplicate define",
                    span: name.span(),
                });
            }
            continue;
        }
    }
    Ok(defines)
}
/// Returns true if the provided tree is a single punctuation symbol of the
/// given character.
fn is_punct(tree: &proc_macro::TokenTree, punct: char) -> bool {
    matches!(tree, TokenTree::Punct(p) if p.as_char() == punct)
}
/// Compilation settings with incomplete fields, used during parsing.
#[derive(Default)]
struct PartialCompilationSettings {
    target_spirv_version: Option<shaderc::SpirvVersion>,
    target_environment_version: Option<shaderc::EnvVersion>,
    stage: Option<shaderc::ShaderKind>,
    source: Option<Source>,
    source_span: Option<proc_macro::Span>,
    language: Option<shaderc::SourceLanguage>,
    defines: FxHashMap<String, Option<String>>,
}
impl PartialCompilationSettings {
    /// Insert a key, parsing the value from a token tree.
    fn key_value(
        &mut self,
        key: proc_macro::Ident,
        value: TokenTree,
    ) -> Result<(), CompilationSettingsError> {
        // Parsing `Display` my beloathed (this is the only way to do it)
        let key_text = format!("{key}");
        // Many keys expect a string literal value, deduplicate that logic here:
        let string_literal_value = || {
            match &value {
                TokenTree::Literal(lit) => match litrs::Literal::from(lit) {
                    litrs::Literal::String(str) => Some(str.value().to_owned()),
                    _ => None,
                },
                _ => None,
            }
            .ok_or(CompilationSettingsError {
                error: "expected string literal",
                span: value.span(),
            })
        };
        let is_duplicate = match key_text.as_str() {
            "source" => {
                self.source_span = Some(value.span());
                self.source
                    .replace(Source::Literal(string_literal_value()?))
                    .is_some()
            }
            "file" => {
                self.source_span = Some(value.span());
                self.source
                    .replace(Source::File(string_literal_value()?.into()))
                    .is_some()
            }
            "stage" => todo!(),
            "defines" => {
                if !self.defines.is_empty() {
                    true
                } else {
                    let TokenTree::Group(group) = value else {
                        return Err(CompilationSettingsError {
                            error: "expected group",
                            span: key.span(),
                        });
                    };
                    self.defines = parse_defines(&group)?;
                    false
                }
            }
            "language" => todo!(),
            _ => {
                return Err(CompilationSettingsError {
                    error: "unrecognized key",
                    span: key.span(),
                });
            }
        };
        if is_duplicate {
            Err(CompilationSettingsError {
                error: "duplicate key",
                span: key.span(),
            })
        } else {
            Ok(())
        }
    }
    /// Finalize the settings. Missing fields are defaulted where possible, and
    /// where not possible an error is returned.
    fn make_whole(
        self,
        span: proc_macro::Span,
    ) -> Result<CompilationSettings, CompilationSettingsError> {
        Ok(CompilationSettings {
            target_spirv_version: self
                .target_spirv_version
                .unwrap_or(shaderc::SpirvVersion::V1_0),
            target_environment_version: self
                .target_environment_version
                .unwrap_or(shaderc::EnvVersion::Vulkan1_0),
            stage: self.stage.unwrap_or(shaderc::ShaderKind::InferFromSource),
            source: self.source.ok_or(CompilationSettingsError {
                error: "missing shader source",
                span,
            })?,
            source_span: self.source_span.unwrap(),
            language: self.language.ok_or(CompilationSettingsError {
                error: "missing source language",
                span,
            })?,
            defines: self.defines,
        })
    }
}
/// Finalized compilation settings.
struct CompilationSettings {
    target_spirv_version: shaderc::SpirvVersion,
    target_environment_version: shaderc::EnvVersion,
    stage: shaderc::ShaderKind,
    source: Source,
    source_span: proc_macro::Span,
    language: shaderc::SourceLanguage,
    defines: FxHashMap<String, Option<String>>,
}
struct CompilationSettingsError {
    error: &'static str,
    span: proc_macro::Span,
}
impl CompilationSettings {
    /// Parse a token stream into compilation settings, with an optional default
    /// language if the token stream does not specify otherwise.
    fn from_tokens(
        tokens: TokenStream,
        lang: Option<shaderc::SourceLanguage>,
    ) -> Result<Self, CompilationSettingsError> {
        use proc_macro::TokenTree;
        // Accept two syntaxes: if lang is Some, accept a string literal. Or,
        // always accept a key-value set of args.
        let mut tokens = tokens.into_iter().peekable();
        // Attempt to parse the string literal form:
        let root = tokens.peek().cloned().ok_or(CompilationSettingsError {
            error: "provide a string literal of source code or a set of key-value pairs with compilation settings",
            span: proc_macro::Span::call_site(),
        })?;
        if let TokenTree::Literal(lit) = &root {
            let _ = tokens.next();
            if let Some(tok) = tokens.next() {
                return Err(CompilationSettingsError {
                    error: "unexpected tokens",
                    span: tok.span(),
                });
            }
            let Some(lang) = lang else {
                return Err(CompilationSettingsError {
                    error: "language must be declared",
                    span: lit.span(),
                });
            };

            return match litrs::Literal::from(lit) {
                litrs::Literal::String(str) => Ok(Self {
                    language: lang,
                    source: Source::Literal(str.value().to_owned()),
                    source_span: lit.span(),
                    stage: shaderc::ShaderKind::InferFromSource,
                    defines: FxHashMap::default(),
                    target_environment_version: shaderc::EnvVersion::Vulkan1_0,
                    target_spirv_version: shaderc::SpirvVersion::V1_0,
                }),
                _ => Err(CompilationSettingsError {
                    error: "expected a string literal",
                    span: lit.span(),
                }),
            };
        }
        // Otherwise, attempt to read the key-value pairs.
        let mut partial = PartialCompilationSettings {
            language: lang,
            ..Default::default()
        };
        loop {
            // Read Key, colon, value tree, optional comma.
            let Some(key) = tokens.next() else {
                // No more pairs.
                break;
            };
            let TokenTree::Ident(key) = key else {
                return Err(CompilationSettingsError {
                    error: "expected ident",
                    span: key.span(),
                });
            };
            if !tokens.next().is_some_and(|colon| is_punct(&colon, ':')) {
                return Err(CompilationSettingsError {
                    error: "expected trailing colon",
                    span: key.span(),
                });
            }
            let value = tokens.next().ok_or(CompilationSettingsError {
                error: "expected value for key",
                span: key.span(),
            })?;
            partial.key_value(key, value)?;

            // Take comma, if any.
            let Some(comma) = tokens.next() else {
                break;
            };
            if !is_punct(&comma, ',') {
                return Err(CompilationSettingsError {
                    error: "expected comma",
                    span: comma.span(),
                });
            }
        }
        partial.make_whole(proc_macro::Span::call_site())
    }
}
/// Given compilation settings, compile the shader, analyze the SPIRV, and
/// return resulting codegen.
fn run_settings(settings: CompilationSettings) -> TokenStream {
    let (source_text, source_file, add_line_number) = match settings.source {
        Source::Literal(source) => (
            source,
            settings.source_span.file(),
            settings.source_span.line() - 1,
        ),
        Source::File(path) => {
            let path_name = path.to_string_lossy().into_owned();
            // Match behavior of include_bytes!, where path is relative to the
            // folder of the file containing the invocation.
            let mut invocation_path = proc_macro::Span::call_site().local_file().unwrap();
            // Parent directory.
            invocation_path.pop();
            // Append, or replace if the path is absolute.
            let path = invocation_path.join(&path);
            (
                // FIXME: This makes our macro incredibly non-pure, a big nono.
                // But there's no solution to this atm. See:
                // https://github.com/rust-lang/rust/issues/99515. Possible hack
                // is to emit a discarded include_bytes! in the output, creating
                // a dependency on that external file.
                std::fs::read_to_string(&path).unwrap_or_else(|err| {
                    proc_macro_error::abort!(
                        settings.source_span,
                        format!("{err} while trying to open {path:?}")
                    )
                }),
                path_name,
                0,
            )
        }
    };
    // Create a compiler with our CompilationSettings:
    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_source_language(settings.language);
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        settings.target_environment_version as _,
    );
    options.set_target_spirv(settings.target_spirv_version);
    for (key, value) in settings.defines {
        options.add_macro_definition(&key, value.as_deref());
    }
    // In the spirit of rust:tm:
    options.set_warnings_as_errors();

    let artifact = compiler
        .compile_into_spirv(
            &source_text,
            settings.stage,
            &source_file,
            "main",
            Some(&options),
        )
        .unwrap_or_else(|err| {
            proc_macro_error::abort!(
                settings.source_span,
                adjust_errors(&source_file, add_line_number, err)
            )
        });
    // ShaderC does not specify in what endian the bytes are returned.
    #[cfg(target_endian = "big")]
    unimplemented!("this is undocumented lol");
    // Run the codegen!
    let spirv = artifact.as_binary();
    codegen::from_spirv(spirv)
}
fn run_with_language(langauge: shaderc::SourceLanguage, stream: TokenStream) -> TokenStream {
    let settings = CompilationSettings::from_tokens(stream, Some(langauge))
        .unwrap_or_else(|err| proc_macro_error::abort!(err.span, err.error));
    run_settings(settings)
}
/// See [crate docs](crate) for usage.
#[proc_macro_error::proc_macro_error]
#[proc_macro]
pub fn glsl(input: TokenStream) -> TokenStream {
    run_with_language(shaderc::SourceLanguage::GLSL, input)
}
/// See [crate docs](crate) for usage.
#[proc_macro_error::proc_macro_error]
#[proc_macro]
pub fn hlsl(input: TokenStream) -> TokenStream {
    run_with_language(shaderc::SourceLanguage::HLSL, input)
}

/// Take a shaderc error, and adjust line/column numbers as necessary based on
/// the span of the input text.
fn adjust_errors(source_file: &str, add_lines: usize, error: shaderc::Error) -> String {
    let compilation_error = match error {
        shaderc::Error::CompilationError(_count, compilation_error) => compilation_error,
        // Do not unwrap the internal string, as it is often empty. Instead, the
        // message comes from the `Display` impl.
        x => return format!("{x}"),
    };

    let mut output = String::with_capacity(compilation_error.len());

    for line in compilation_error.lines() {
        // Modify the line, or returns None if failed.
        let try_parse = || -> Option<String> {
            // Parse lines of the format "/file/name:5: error text: etc" where 5
            // is the line number.
            let (line_number, line) = line
                .strip_prefix(source_file)?
                .strip_prefix(':')?
                .split_once(':')?;
            let line_number = line_number.parse::<usize>().ok()? + add_lines;

            Some(format!("{source_file}:{line_number}:{line}"))
        };

        if let Some(result) = try_parse() {
            output += &result;
        } else {
            // Fallback to outputting the line as-is
            output += line;
        }
        // GROG NOT CARE ABOUT PERFORMANCE HERE. THIS IS OFF-LINE CODE. it's out
        // of character but it's fine. this is fine :,3
        output += "\n";
    }
    output
}

/// Read a file as u32s, swapping endian if necessary based on the file magic.
fn read_spirv_words(path: impl AsRef<std::path::Path>) -> Vec<u32> {
    const MAGIC: u32 = rspirv::spirv::MAGIC_NUMBER;
    const SWAPPED_MAGIC: u32 = MAGIC.swap_bytes();

    let bytes = std::fs::read(path).unwrap();
    assert!(bytes.len().is_multiple_of(4));
    let mut words = vec![0u32; bytes.len() / 4];

    unsafe {
        // Have to copy bytes to bytes, not words to words, as `bytes` might not
        // be aligned!
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), words.as_mut_ptr().cast(), bytes.len())
    };

    match words.first().copied() {
        Some(MAGIC) => words,
        Some(SWAPPED_MAGIC) => {
            for word in &mut words {
                *word = word.swap_bytes();
            }
            words
        }
        _ => panic!("bad file magic"),
    }
}

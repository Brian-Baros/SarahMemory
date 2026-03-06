use pyo3::prelude::*;
use regex::Regex;

#[pyfunction]
fn token_count(text: &str) -> usize {
    text.split_whitespace().count()
}

#[pyfunction]
fn normalize_text(text: &str) -> String {
    let re = Regex::new(r"\s+").unwrap();
    re.replace_all(text.trim(), " ").to_string()
}

#[pyfunction]
fn compress_prompt(text: &str) -> String {
    let mut cleaned = text.trim().to_string();
    cleaned = cleaned.replace("\n", " ");
    cleaned = cleaned.replace("\t", " ");
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[pymodule]
fn sarahmemory_rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(token_count, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_text, m)?)?;
    m.add_function(wrap_pyfunction!(compress_prompt, m)?)?;
    Ok(())
}
use pyo3::prelude::*;

use web_audio_api_rs::context::{AudioContext, BaseAudioContext};
use web_audio_api_rs::node::{AudioNode, AudioScheduledSourceNode};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    let ctx = AudioContext::default();
    let mut osc = ctx.create_oscillator();
    osc.connect(&ctx.destination());
    osc.start();

    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn web_audio_api(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}

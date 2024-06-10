use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

use web_audio_api_rs::context::BaseAudioContext;
use web_audio_api_rs::node::{AudioNode as RsAudioNode, AudioScheduledSourceNode as _};

#[pyclass]
struct AudioContext(web_audio_api_rs::context::AudioContext);

#[pymethods]
impl AudioContext {
    #[new]
    fn new() -> Self {
        Self(Default::default())
    }

    fn destination(&self) -> AudioNode {
        let dest = self.0.destination();
        let node = Arc::new(Mutex::new(dest)) as Arc<Mutex<dyn RsAudioNode + Send + 'static>>;
        AudioNode(node)
    }
}

#[pyclass(subclass)]
struct AudioNode(Arc<Mutex<dyn RsAudioNode + Send + 'static>>);

#[pymethods]
impl AudioNode {
    fn connect(&self, other: &Self) {
        self.0.lock().unwrap().connect(&*other.0.lock().unwrap());
    }
    fn disconnect(&self, other: &Self) {
        self.0
            .lock()
            .unwrap()
            .disconnect_dest(&*other.0.lock().unwrap());
    }
}

#[pyclass]
struct AudioParam(web_audio_api_rs::AudioParam);

#[pymethods]
impl AudioParam {
    fn value(&self) -> f32 {
        self.0.value()
    }

    fn set_value(&self, value: f32) -> Self {
        Self(self.0.set_value(value).clone())
    }
}

#[pyclass(extends = AudioNode)]
struct OscillatorNode(Arc<Mutex<web_audio_api_rs::node::OscillatorNode>>);

#[pymethods]
impl OscillatorNode {
    #[new]
    fn new(ctx: &AudioContext) -> (Self, AudioNode) {
        let osc = ctx.0.create_oscillator();
        let node = Arc::new(Mutex::new(osc));
        let audio_node = Arc::clone(&node) as Arc<Mutex<dyn RsAudioNode + Send + 'static>>;
        (OscillatorNode(node), AudioNode(audio_node))
    }

    #[pyo3(signature = (when=0.0))]
    fn start(&mut self, when: f64) {
        self.0.lock().unwrap().start_at(when)
    }

    #[pyo3(signature = (when=0.0))]
    fn stop(&mut self, when: f64) {
        self.0.lock().unwrap().stop_at(when)
    }

    fn frequency(&self) -> AudioParam {
        AudioParam(self.0.lock().unwrap().frequency().clone())
    }

    fn detune(&self) -> AudioParam {
        AudioParam(self.0.lock().unwrap().detune().clone())
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn web_audio_api(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<AudioContext>()?;
    m.add_class::<AudioNode>()?;
    m.add_class::<OscillatorNode>()?;
    m.add_class::<AudioParam>()?;
    Ok(())
}

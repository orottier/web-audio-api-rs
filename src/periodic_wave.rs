//! PeriodicWave interface
use std::f32::consts::PI;
use std::sync::Arc;

use crate::node::TABLE_LENGTH_USIZE;
use crate::context::Context;

/// Options for constructing an `PeriodicWave`
pub struct PeriodicWaveOptions {
    /// The real parameter represents an array of cosine terms of Fourrier series.
    ///
    /// The first element (index 0) represents the DC-offset.
    /// This offset has to be given but will not be taken into account
    /// to build the custom periodic waveform.
    ///
    /// The following elements (index 1 and more) represent the fundamental and harmonics of the periodic waveform.
    pub real: Option<Vec<f32>>,
    /// The imag parameter represents an array of sine terms of Fourrier series.
    ///
    /// The first element (index 0) will not be taken into account
    /// to build the custom periodic waveform.
    ///
    /// The following elements (index 1 and more) represent the fundamental and harmonics of the periodic waveform.
    pub imag: Option<Vec<f32>>,
    /// By default PeriodicWave is build with normalization enabled (disable_normalization = false).
    /// In this case, a peak normalization is applied to the given custom periodic waveform.
    ///
    /// If disable_normalization is enabled (disable_normalization = true), the normalization is
    /// defined by the periodic waveform characteristics (img, and real fields).
    pub disable_normalization: Option<bool>,
}

/// `PeriodicWave` represents an arbitrary periodic waveform to be used with an `OscillatorNode`.
///
/// - MDN documentation: <https://developer.mozilla.org/en-US/docs/Web/API/PeriodicWave>
/// - specification: <https://webaudio.github.io/web-audio-api/#PeriodicWave>
/// - see also: [`Context::create_periodic_wave`](crate::context::Context::create_periodic_wave)
/// - see also: [`OscillatorNode`](crate::node::OscillatorNode)
///
/// # Usage
///
/// ```no_run
/// use web_audio_api::context::{Context, AudioContext};
/// use web_audio_api::periodic_wave::{PeriodicWave, PeriodicWaveOptions};
/// use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
///
/// let context = AudioContext::new(None);
///
/// // generate a simple waveform with 2 harmonics
/// let options = PeriodicWaveOptions {
///   real: Some(vec![0., 0., 0.]),
///   imag: Some(vec![0., 0.5, 0.5]),
///   disable_normalization: Some(false),
/// };
///
/// let periodic_wave = PeriodicWave::new(&context, Some(options));
///
/// let osc = context.create_oscillator();
/// osc.set_periodic_wave(periodic_wave);
/// osc.connect(&context.destination());
/// osc.start();
/// ```
/// # Examples
///
/// - `cargo run --release --example oscillators`
///
// Basically a wrapper around Arc<Vec<f32>>, so `PeriodicWave` are cheap to clone
#[derive(Debug, Clone)]
pub struct PeriodicWave {
    wavetable: Arc<Vec<f32>>,
}

impl PeriodicWave {
    /// Returns a `PeriodicWave`
    ///
    /// # Arguments
    ///
    /// * `real` - The real parameter represents an array of cosine terms of Fourrier series.
    /// * `imag` - The imag parameter represents an array of sine terms of Fourrier series.
    /// * `constraints` - The constraints parameter specifies the normalization mode of the `PeriodicWave`
    ///
    /// # Panics
    ///
    /// Will panic if:
    ///
    /// * `real` is defined and its length is less than 2
    /// * `imag` is defined and its length is less than 2
    /// * `real` and `imag` are defined and theirs lengths are not equal
    /// * `PeriodicWave` is more than 8192 components
    pub fn new<C: Context>(_context: &C, options: Option<PeriodicWaveOptions>) -> Self {
        let (real, imag, normalize) = if let Some(PeriodicWaveOptions {
            real,
            imag,
            disable_normalization,
        }) = options
        {
            let (real, imag) = match (real, imag) {
                (Some(r), Some(i)) => {
                    assert!(
                        r.len() >= 2,
                        "RangeError: Real field length should be at least 2"
                    );
                    assert!(
                        i.len() >= 2,
                        "RangeError: Imag field length should be at least 2",
                    );
                    assert!(
                        // the specs gives this number as a lower bound
                        // it is implemented here as a upper bound to enable required casting
                        // without loss of precision
                        r.len() <= 8192,
                        "NotSupported: periodic wave of more than 8192 components"
                    );
                    assert!(
                        r.len() == i.len(),
                        "RangeError: Imag and real field length should be equal"
                    );
                    (r, i)
                }
                (Some(r), None) => {
                    assert!(
                        r.len() >= 2,
                        "RangeError: Real field length should be at least 2"
                    );
                    assert!(
                        // the specs gives this number as a lower bound
                        // it is implemented here as a upper bound to enable required casting
                        // without loss of precision
                        r.len() <= 8192,
                        "NotSupported: periodic wave of more than 8192 components"
                    );
                    let r_len = r.len();
                    (r, vec![0.; r_len])
                }
                (None, Some(i)) => {
                    assert!(
                        i.len() >= 2,
                        "RangeError: Real field length should be at least 2"
                    );
                    assert!(
                        i.len() <= 8192,
                        // the specs gives this number as a lower bound
                        // it is implemented here as a upper bound to enable required casting
                        // without loss of precision
                        "NotSupported: periodic wave of more than 8192 components"
                    );
                    let i_len = i.len();
                    (vec![0.; i_len], i)
                }
                _ => (vec![0., 0.], vec![0., 1.]),
            };

            (real, imag, !disable_normalization.unwrap_or(false))
        } else {
            (vec![0., 0.], vec![0., 0.], true)
        };

        let wavetable = Self::generate_wavetable(&real, &imag, normalize, TABLE_LENGTH_USIZE);

        Self {
            wavetable: Arc::new(wavetable),
        }
    }

    pub(crate) fn as_slice(&self) -> &[f32] {
        &self.wavetable[..]
    }

    // cf. https://webaudio.github.io/web-audio-api/#waveform-generation
    // note that sines are in the imaginary components
    //
    // @note: Current implementation is naive and could be improved using inverse
    // FFT or table lookup on SINETABLE. Such performance improvements should be
    // however tested also against this implementation.
    fn generate_wavetable(reals: &[f32], imags: &[f32], normalize: bool, size: usize) -> Vec<f32> {
        let mut wavetable = Vec::with_capacity(size);
        let pi_2 = 2. * PI;

        for i in 0..size {
            let mut sample = 0.;
            let phase = pi_2 * i as f32 / size as f32;

            for j in 1..reals.len() {
                let freq = j as f32;
                let real = reals[j];
                let imag = imags[j];
                let rad = phase * freq;
                let contrib = real * rad.cos() + imag * rad.sin();
                sample += contrib;
            }

            wavetable.push(sample);
        }

        if normalize {
            Self::normalize(&mut wavetable);
        }

        wavetable
    }

    fn normalize(wavetable: &mut [f32]) {
        let mut max = 0.;

        for sample in wavetable.iter() {
            let abs = sample.abs();
            if abs > max {
                max = abs;
            }
        }

        // prevent division by 0. (nothing to normalize anyway...)
        if max > 0. {
            let norm_factor = 1. / max;

            for sample in wavetable.iter_mut() {
                *sample *= norm_factor;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;
    use std::f32::consts::PI;

    use crate::context::AudioContext;
    use crate::node::{TABLE_LENGTH_F32, TABLE_LENGTH_USIZE};

    use super::{PeriodicWave, PeriodicWaveOptions};

    #[test]
    #[should_panic]
    fn fails_to_build_when_real_is_too_short() {
        let context = AudioContext::new(None);

        let options = PeriodicWaveOptions {
            real: Some(vec![0.]),
            imag: Some(vec![0., 0., 0.]),
            disable_normalization: Some(false),
        };

        let _periodic_wave = PeriodicWave::new(&context, Some(options));
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_only_real_is_defined_and_too_short() {
        let context = AudioContext::new(None);

        let options = PeriodicWaveOptions {
            real: Some(vec![0.]),
            imag: None,
            disable_normalization: Some(false),
        };

        let _periodic_wave = PeriodicWave::new(&context, Some(options));
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_imag_is_too_short() {
        let context = AudioContext::new(None);

        let options = PeriodicWaveOptions {
            real: Some(vec![0., 0., 0.]),
            imag: Some(vec![0.]),
            disable_normalization: Some(false),
        };

        let _periodic_wave = PeriodicWave::new(&context, Some(options));
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_only_imag_is_defined_and_too_short() {
        let context = AudioContext::new(None);

        let options = PeriodicWaveOptions {
            real: None,
            imag: Some(vec![0.]),
            disable_normalization: Some(false),
        };

        let _periodic_wave = PeriodicWave::new(&context, Some(options));
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_imag_and_real_not_equal_length() {
        let context = AudioContext::new(None);

        let options = PeriodicWaveOptions {
            real: Some(vec![0., 0., 0.]),
            imag: Some(vec![0., 0.]),
            disable_normalization: Some(false),
        };

        let _periodic_wave = PeriodicWave::new(&context, Some(options));
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_imag_and_real_are_more_than_8192_comps() {
        let context = AudioContext::new(None);

        let options = PeriodicWaveOptions {
            real: Some(vec![0.; 8193]),
            imag: Some(vec![0.; 8193]),
            disable_normalization: Some(false),
        };

        let _periodic_wave = PeriodicWave::new(&context, Some(options));
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_real_is_more_than_8192_comps() {
        let context = AudioContext::new(None);

        let options = PeriodicWaveOptions {
            real: Some(vec![0.; 8193]),
            imag: None,
            disable_normalization: Some(false),
        };

        let _periodic_wave = PeriodicWave::new(&context, Some(options));
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_imag_is_more_than_8192_comps() {
        let context = AudioContext::new(None);

        let options = PeriodicWaveOptions {
            real: None,
            imag: Some(vec![0.; 8193]),
            disable_normalization: Some(false),
        };

        let _periodic_wave = PeriodicWave::new(&context, Some(options));
    }

    #[test]
    fn wavetable_generate_sine() {
        let reals = [0., 0.];
        let imags = [0., 1.];

        let result = PeriodicWave::generate_wavetable(&reals, &imags, true, TABLE_LENGTH_USIZE);
        let mut expected = Vec::new();

        for i in 0..TABLE_LENGTH_USIZE {
            let sample = (i as f32 / TABLE_LENGTH_F32 * 2. * PI).sin();
            expected.push(sample);
        }

        assert_float_eq!(result[..], expected[..], abs_all <= 0.);
    }

    #[test]
    fn wavetable_generate_2f_not_norm() {
        let reals = [0., 0., 0.];
        let imags = [0., 0.5, 0.5];

        let result = PeriodicWave::generate_wavetable(&reals, &imags, false, TABLE_LENGTH_USIZE);
        let mut expected = Vec::new();

        for i in 0..TABLE_LENGTH_USIZE {
            let mut sample = 0.;
            // fondamental frequency
            sample += 0.5 * (1. * i as f32 / TABLE_LENGTH_F32 * 2. * PI).sin();
            // 1rst partial
            sample += 0.5 * (2. * i as f32 / TABLE_LENGTH_F32 * 2. * PI).sin();

            expected.push(sample);
        }

        assert_float_eq!(result[..], expected[..], abs_all <= 0.);
    }

    #[test]
    fn normalize() {
        {
            let mut signal = [-0.5, 0.2];
            PeriodicWave::normalize(&mut signal);
            let expected = [-1., 0.4];

            assert_float_eq!(signal[..], expected[..], abs_all <= 0.);
        }

        {
            let mut signal = [0.5, -0.2];
            PeriodicWave::normalize(&mut signal);
            let expected = [1., -0.4];

            assert_float_eq!(signal[..], expected[..], abs_all <= 0.);
        }
    }

    #[test]
    fn wavetable_generate_2f_norm() {
        let reals = [0., 0., 0.];
        let imags = [0., 0.5, 0.5];

        let result = PeriodicWave::generate_wavetable(&reals, &imags, true, TABLE_LENGTH_USIZE);
        let mut expected = Vec::new();

        for i in 0..TABLE_LENGTH_USIZE {
            let mut sample = 0.;
            // fondamental frequency
            sample += 0.5 * (1. * i as f32 / TABLE_LENGTH_F32 * 2. * PI).sin();
            // 1rst partial
            sample += 0.5 * (2. * i as f32 / TABLE_LENGTH_F32 * 2. * PI).sin();

            expected.push(sample);
        }

        PeriodicWave::normalize(&mut expected);

        assert_float_eq!(result[..], expected[..], abs_all <= 0.);
    }
}

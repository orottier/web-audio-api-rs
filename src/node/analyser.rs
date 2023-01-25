// use std::cell::RefCell;
use std::sync::{Arc, RwLock};

use crate::analysis::{Analyser, AnalyserRingBuffer};
use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};

use super::{AudioNode, ChannelConfig, ChannelConfigOptions, ChannelInterpretation};

// use crossbeam_channel::{self, Receiver, Sender};

/// Options for constructing an [`AnalyserNode`]
// dictionary AnalyserOptions : AudioNodeOptions {
//   unsigned long fftSize = 2048;
//   double maxDecibels = -30;
//   double minDecibels = -100;
//   double smoothingTimeConstant = 0.8;
// };
#[derive(Clone, Debug)]
pub struct AnalyserOptions {
    pub fft_size: usize,
    pub max_decibels: f64,
    pub min_decibels: f64,
    pub smoothing_time_constant: f64,
    pub channel_config: ChannelConfigOptions,
}

impl Default for AnalyserOptions {
    fn default() -> Self {
        Self {
            fft_size: 2048,
            max_decibels: -30.,
            min_decibels: 100.,
            smoothing_time_constant: 0.8,
            channel_config: ChannelConfigOptions::default(),
        }
    }
}

/// Provides real-time frequency and time-domain analysis information
pub struct AnalyserNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    // needed to make the AnalyserNode API immutable
    analyser: Arc<RwLock<Analyser>>,
}

impl AudioNode for AnalyserNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> usize {
        1
    }

    fn number_of_outputs(&self) -> usize {
        1
    }
}

impl AnalyserNode {
    pub fn new<C: BaseAudioContext>(context: &C, options: AnalyserOptions) -> Self {
        context.register(move |registration| {
            // let fft_size = Arc::new(AtomicUsize::new(options.fft_size));
            // let smoothing_time_constant = Arc::new(AtomicU32::new(
            //     (options.smoothing_time_constant * 100.) as u32,
            // ));

            let analyser = Analyser::new();

            // apply options

            let render = AnalyserRenderer {
                ring_buffer: analyser.get_ring_buffer_clone(),
            };

            let node = AnalyserNode {
                registration,
                channel_config: options.channel_config.into(),
                analyser: Arc::new(RwLock::new(analyser)),
            };

            (node, Box::new(render))
        })
    }

    /// The size of the FFT used for frequency-domain analysis (in sample-frames)
    pub fn fft_size(&self) -> usize {
        self.analyser.read().unwrap().fft_size()
    }

    /// Set FFT size
    ///
    /// ## Panics
    ///
    /// This function panics if fft_size is not a power of two or not in the range [32, 32768]
    pub fn set_fft_size(&self, fft_size: usize) {
        self.analyser.write().unwrap().set_fft_size(fft_size);
    }

    /// Time averaging parameter with the last analysis frame.
    /// A value from 0 -> 1 where 0 represents no time averaging with the last
    /// analysis frame. The default value is 0.8.
    pub fn smoothing_time_constant(&self) -> f64 {
        self.analyser.read().unwrap().smoothing_time_constant()
    }

    /// Set smoothing time constant
    ///
    /// ## Panics
    ///
    /// This function panics if the value is set to a value less than 0 or more than 1.
    pub fn set_smoothing_time_constant(&self, value: f64) {
        self.analyser.write().unwrap().set_smoothing_time_constant(value);
    }


    /// Minimum power value in the scaling range for the FFT analysis data for
    /// conversion to unsigned byte values. The default value is -100.
    pub fn min_decibels(&self) -> f64 {
        self.analyser.read().unwrap().min_decibels()
    }

    /// Set min decibels
    ///
    /// ## Panics
    ///
    /// This function panics if the value is set to a value more than or equal
    /// to max decibels.
    pub fn set_min_decibels(&self, value: f64) {
        self.analyser.write().unwrap().set_min_decibels(value);
    }

    /// Maximum power value in the scaling range for the FFT analysis data for
    /// conversion to unsigned byte values. The default value is -30.
    pub fn max_decibels(&self) -> f64 {
        self.analyser.read().unwrap().max_decibels()
    }

    /// Set max decibels
    ///
    /// ## Panics
    ///
    /// This function panics if the value is set to a value less than or equal
    /// to min decibels.
    pub fn set_max_decibels(&self, value: f64) {
        self.analyser.write().unwrap().set_max_decibels(value);
    }

    /// Number of bins in the FFT results, is half the FFT size
    pub fn frequency_bin_count(&self) -> usize {
        self.analyser.read().unwrap().frequency_bin_count()
    }

    /// Copies the current time domain data (waveform data) into the provided buffer
    pub fn get_float_time_domain_data(&self, buffer: &mut [f32]) {
        self.analyser.write().unwrap().get_float_time_domain_data(buffer);
    }

    pub fn get_byte_time_domain_data(&self, buffer: &mut [u8]) {
        self.analyser.write().unwrap().get_byte_time_domain_data(buffer);
    }

    /// Copies the current frequency data into the provided buffer
    pub fn get_float_frequency_data(&self, buffer: &mut [f32]) {
        self.analyser.write().unwrap().get_float_frequency_data(buffer);
    }

    pub fn get_byte_frequency_data(&self, buffer: &mut [u8]) {
        self.analyser.write().unwrap().get_byte_frequency_data(buffer);
    }
}

struct AnalyserRenderer {
    ring_buffer: Arc<AnalyserRingBuffer>,
}

impl AudioProcessor for AnalyserRenderer {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues,
        _scope: &RenderScope,
    ) -> bool {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];

        // pass through input
        *output = input.clone();

        // down mix to mono
        let mut mono = input.clone();
        mono.mix(1, ChannelInterpretation::Speakers);

        // add current input to ring buffer
        let data = mono.channel_data(0).as_ref();
        self.ring_buffer.write(data);

        // @todo - review
        false
    }
}

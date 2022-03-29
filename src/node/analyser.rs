use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use crate::analysis::Analyser;
use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum};
use crate::SampleRate;

use super::{AudioNode, ChannelConfig, ChannelConfigOptions, ChannelInterpretation};

use crossbeam_channel::{self, Receiver, Sender};

/// Options for constructing an [`AnalyserNode`]
// dictionary AnalyserOptions : AudioNodeOptions {
//   unsigned long fftSize = 2048;
//   double maxDecibels = -30;
//   double minDecibels = -100;
//   double smoothingTimeConstant = 0.8;
// };
#[derive(Clone, Debug)]
pub struct AnalyserOptions {
    pub fft_size: u32,
    #[allow(dead_code)]
    pub max_decibels: f64,
    #[allow(dead_code)]
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

enum AnalyserRequest {
    FloatTime {
        sender: Sender<Vec<f32>>,
        buffer: Vec<f32>,
    },
    FloatFrequency {
        sender: Sender<Vec<f32>>,
        buffer: Vec<f32>,
    },
}

/// Provides real-time frequency and time-domain analysis information
pub struct AnalyserNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    fft_size: Arc<AtomicU32>,
    smoothing_time_constant: Arc<AtomicU32>,
    sender: Sender<AnalyserRequest>,
}

impl AudioNode for AnalyserNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> u32 {
        1
    }

    fn number_of_outputs(&self) -> u32 {
        1
    }
}

impl AnalyserNode {
    pub fn new<C: BaseAudioContext>(context: &C, options: AnalyserOptions) -> Self {
        context.base().register(move |registration| {
            let fft_size = Arc::new(AtomicU32::new(options.fft_size));
            let smoothing_time_constant = Arc::new(AtomicU32::new(
                (options.smoothing_time_constant * 100.) as u32,
            ));

            let (sender, receiver) = crossbeam_channel::bounded(0);

            let render = AnalyserRenderer {
                analyser: Analyser::new(options.fft_size as usize),
                fft_size: fft_size.clone(),
                smoothing_time_constant: smoothing_time_constant.clone(),
                receiver,
            };

            let node = AnalyserNode {
                registration,
                channel_config: options.channel_config.into(),
                fft_size,
                smoothing_time_constant,
                sender,
            };

            (node, Box::new(render))
        })
    }

    /// Half the FFT size
    pub fn frequency_bin_count(&self) -> u32 {
        self.fft_size.load(Ordering::SeqCst) / 2
    }

    /// The size of the FFT used for frequency-domain analysis (in sample-frames)
    pub fn fft_size(&self) -> u32 {
        self.fft_size.load(Ordering::SeqCst)
    }

    /// This MUST be a power of two in the range 32 to 32768
    pub fn set_fft_size(&self, fft_size: u32) {
        // todo assert size
        self.fft_size.store(fft_size, Ordering::SeqCst);
    }

    /// Time averaging parameter with the last analysis frame.
    pub fn smoothing_time_constant(&self) -> f64 {
        self.smoothing_time_constant.load(Ordering::SeqCst) as f64 / 100.
    }

    /// Set smoothing time constant, this MUST be a value between 0 and 1
    pub fn set_smoothing_time_constant(&self, value: f64) {
        // todo assert range
        self.smoothing_time_constant
            .store((value * 100.) as u32, Ordering::SeqCst);
    }

    /// Copies the current time domain data (waveform data) into the provided buffer
    // we can fix this panic cf issue #101
    #[allow(clippy::missing_panics_doc)]
    pub fn get_float_time_domain_data(&self, buffer: Vec<f32>) -> Vec<f32> {
        let (sender, receiver) = crossbeam_channel::bounded(0);
        let request = AnalyserRequest::FloatTime { sender, buffer };
        self.sender.send(request).unwrap();
        receiver.recv().unwrap()
    }

    /// Copies the current frequency data into the provided buffer
    // we can fix this panic cf issue #101
    #[allow(clippy::missing_panics_doc)]
    pub fn get_float_frequency_data(&self, buffer: Vec<f32>) -> Vec<f32> {
        let (sender, receiver) = crossbeam_channel::bounded(0);
        let request = AnalyserRequest::FloatFrequency { sender, buffer };
        self.sender.send(request).unwrap();
        receiver.recv().unwrap()
    }
}

struct AnalyserRenderer {
    pub analyser: Analyser,
    pub fft_size: Arc<AtomicU32>,
    pub smoothing_time_constant: Arc<AtomicU32>,
    pub receiver: Receiver<AnalyserRequest>,
}

// SAFETY:
// AudioBuffer is not Send, but the buffer Vec is empty when we move it to the render thread.
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for AnalyserRenderer {}

impl AudioProcessor for AnalyserRenderer {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues,
        _timestamp: f64,
        _sample_rate: SampleRate,
    ) -> bool {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];

        // pass through input
        *output = input.clone();

        // add current input to ring buffer
        let mut mono = input.clone();
        mono.mix(1, ChannelInterpretation::Speakers);
        let mono_data = mono.channel_data(0).clone();
        self.analyser.add_data(mono_data);

        // calculate frequency domain every `fft_size` samples
        let fft_size = self.fft_size.load(Ordering::Relaxed) as usize;
        let resized = self.analyser.current_fft_size() != fft_size;
        let complete_cycle = self.analyser.check_complete_cycle(fft_size);
        if resized || complete_cycle {
            let smoothing_time_constant =
                self.smoothing_time_constant.load(Ordering::Relaxed) as f32 / 100.;
            self.analyser
                .calculate_float_frequency(fft_size, smoothing_time_constant);
        }

        // check if any information was requested from the control thread
        if let Ok(request) = self.receiver.try_recv() {
            match request {
                AnalyserRequest::FloatTime { sender, mut buffer } => {
                    self.analyser.get_float_time(&mut buffer[..], fft_size);

                    // allow to fail when receiver is disconnected
                    let _ = sender.send(buffer);
                }
                AnalyserRequest::FloatFrequency { sender, mut buffer } => {
                    self.analyser.get_float_frequency(&mut buffer[..]);

                    // allow to fail when receiver is disconnected
                    let _ = sender.send(buffer);
                }
            }
        }

        false
    }
}

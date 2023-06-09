use web_audio_api::context::{
    AudioContextRegistration, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, ChannelConfig};
use web_audio_api::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct AtomicF64 {
    inner: AtomicU64,
}

impl AtomicF64 {
    pub fn new(v: f64) -> Self {
        Self {
            inner: AtomicU64::new(u64::from_ne_bytes(v.to_ne_bytes())),
        }
    }

    pub fn load(&self) -> f64 {
        f64::from_ne_bytes(self.inner.load(Ordering::SeqCst).to_ne_bytes())
    }

    pub fn store(&self, v: f64) {
        self.inner
            .store(u64::from_ne_bytes(v.to_ne_bytes()), Ordering::SeqCst)
    }
}

/// Audio
pub struct LatencyTesterNode {
    /// handle to the audio context, required for all audio nodes
    registration: AudioContextRegistration,
    /// channel configuration (for up/down-mixing of inputs), required for all audio nodes
    channel_config: ChannelConfig,

    calculated_delay: Arc<AtomicF64>,
}

// implement required methods for AudioNode trait
impl AudioNode for LatencyTesterNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config(&self) -> &ChannelConfig {
        &self.channel_config
    }

    // source nodes take no input
    fn number_of_inputs(&self) -> usize {
        1
    }

    // emit a single output
    fn number_of_outputs(&self) -> usize {
        1
    }
}

impl LatencyTesterNode {
    /// Construct a new LatencyTesterNode
    pub fn new<C: BaseAudioContext>(context: &C) -> Self {
        context.register(move |registration| {

            let calculated_delay = Arc::new(AtomicF64::new(0.));
            // setup the processor, this will run in the render thread
            let render = LatencyTesterProcessor {
                calculated_delay: calculated_delay.clone(),
                send_time: 0.,
            };

            // setup the audio node, this will live in the control thread (user facing)
            let node = LatencyTesterNode {
                registration,
                channel_config: ChannelConfig::default(),
                calculated_delay,
            };

            (node, Box::new(render))
        })
    }

    pub fn calculated_delay(&self) -> f64 {
        self.calculated_delay.load()
    }
}

struct LatencyTesterProcessor {
    calculated_delay: Arc<AtomicF64>,
    send_time: f64,
}

impl AudioProcessor for LatencyTesterProcessor {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues,
        scope: &RenderScope,
    ) -> bool {
        // send a dirac every second
        // 48000 / 128 = 375
        let output = &mut outputs[0];

        if (scope.current_frame / 128) % 375 == 0 {
            output.channels_mut().iter_mut().for_each(|channel| {
                channel[0] = 1.
            });

            self.send_time = scope.current_time;
        } else {
            output.make_silent();
        }

        // check input for dirac
        let input = &inputs[0];
        let sample_rate = scope.sample_rate;
        let sample_duration = 1. / sample_rate as f64;
        let dirac_found = false;

        input.channel_data(0)
            .iter()
            .enumerate()
            .for_each(|(i, s)| {
                if !dirac_found {
                    if *s > 0.5 {
                        // @todo - add smaple count
                        let now = scope.current_time + (i as f64 * sample_duration);
                        let diff = now - self.send_time;
                        self.calculated_delay.store(diff);
                    }
                }
            });

        true
    }
}


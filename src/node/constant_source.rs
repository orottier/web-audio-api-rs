use crate::context::{AudioContextRegistration, AudioParamId, BaseAudioContext};
use crate::param::{AudioParam, AudioParamDescriptor, AutomationRate};
use crate::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};
use crate::RENDER_QUANTUM_SIZE;

use super::{AudioNode, AudioScheduledSourceNode, ChannelConfig};

/// Options for constructing an [`ConstantSourceNode`]
// dictionary ConstantSourceOptions {
//   float offset = 1;
// };
#[derive(Clone, Debug)]
pub struct ConstantSourceOptions {
    pub offset: f32,
}

impl Default for ConstantSourceOptions {
    fn default() -> Self {
        Self { offset: 1. }
    }
}

/// Instructions to start or stop processing
#[derive(Debug, Copy, Clone)]
enum Schedule {
    Start(f64),
    Stop(f64),
}

/// Audio source whose output is nominally a constant value. A `ConstantSourceNode`
/// can be used as a constructible `AudioParam` by automating the value of its offset.
///
/// - MDN documentation: <https://developer.mozilla.org/en-US/docs/Web/API/ConstantSourceNode>
/// - specification: <https://webaudio.github.io/web-audio-api/#ConstantSourceNode>
/// - see also: [`BaseAudioContext::create_constant_source`](crate::context::BaseAudioContext::create_constant_source)
///
/// # Usage
///
/// ```no_run
/// use web_audio_api::context::{BaseAudioContext, AudioContext};
/// use web_audio_api::node::AudioNode;
///
/// let audio_context = AudioContext::default();
///
/// let gain1 = audio_context.create_gain();
/// gain1.gain().set_value(0.);
///
/// let gain2 = audio_context.create_gain();
/// gain2.gain().set_value(0.);
///
/// let automation = audio_context.create_constant_source();
/// automation.offset().set_value(0.);
/// automation.connect(gain1.gain());
/// automation.connect(gain2.gain());
///
/// // control both `GainNode`s with 1 automation
/// automation.offset().set_target_at_time(1., audio_context.current_time(), 0.1);
/// ```
///
/// # Example
///
/// - `cargo run --release --example constant_source`
///
pub struct ConstantSourceNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    offset: AudioParam,
}

impl AudioNode for ConstantSourceNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> usize {
        0
    }

    fn number_of_outputs(&self) -> usize {
        1
    }
}

impl AudioScheduledSourceNode for ConstantSourceNode {
    fn start(&self) {
        let when = self.registration.context().current_time();
        self.start_at(when);
    }

    fn start_at(&self, when: f64) {
        self.registration.post_message(Schedule::Start(when));
    }

    fn stop(&self) {
        let when = self.registration.context().current_time();
        self.stop_at(when);
    }

    fn stop_at(&self, when: f64) {
        self.registration.post_message(Schedule::Stop(when));
    }
}

impl ConstantSourceNode {
    pub fn new<C: BaseAudioContext>(context: &C, options: ConstantSourceOptions) -> Self {
        context.register(move |registration| {
            let param_opts = AudioParamDescriptor {
                min_value: f32::MIN,
                max_value: f32::MAX,
                default_value: 1.,
                automation_rate: AutomationRate::A,
            };
            let (param, proc) = context.create_audio_param(param_opts, &registration);
            param.set_value(options.offset);

            let render = ConstantSourceRenderer {
                offset: proc,
                start_time: f64::MAX,
                stop_time: f64::MAX,
                ended_triggered: false,
            };

            let node = ConstantSourceNode {
                registration,
                channel_config: ChannelConfig::default(),
                offset: param,
            };

            (node, Box::new(render))
        })
    }

    pub fn offset(&self) -> &AudioParam {
        &self.offset
    }
}

struct ConstantSourceRenderer {
    offset: AudioParamId,
    start_time: f64,
    stop_time: f64,
    ended_triggered: bool,
}

impl AudioProcessor for ConstantSourceRenderer {
    fn process(
        &mut self,
        _inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        params: AudioParamValues<'_>,
        scope: &RenderScope,
    ) -> bool {
        // single output node
        let output = &mut outputs[0];

        let dt = 1. / scope.sample_rate as f64;
        let next_block_time = scope.current_time + dt * RENDER_QUANTUM_SIZE as f64;

        if self.start_time >= next_block_time {
            output.make_silent();
            return true;
        }

        output.force_mono();

        let offset = params.get(&self.offset);
        let output_channel = output.channel_data_mut(0);
        let mut current_time = scope.current_time;

        output_channel
            .iter_mut()
            .zip(offset.iter().cycle())
            .for_each(|(o, &value)| {
                if current_time < self.start_time || current_time >= self.stop_time {
                    *o = 0.;
                } else {
                    // as we pick values directly from the offset param which is already
                    // computed at sub-sample accuracy, we don't need to do more than
                    // copying the values to their right place.
                    *o = value;
                }

                current_time += dt;
            });

        // tail_time false when output has ended this quantum
        let still_running = self.stop_time >= next_block_time;

        if !still_running {
            // @note: we need this check because this is called a until the program
            // ends, such as if the node was never removed from the graph
            if !self.ended_triggered {
                scope.send_ended_event();
                self.ended_triggered = true;
            }
        }

        still_running
    }

    fn onmessage(&mut self, msg: Box<dyn std::any::Any + Send + 'static>) {
        if let Ok(schedule) = msg.downcast::<Schedule>() {
            match *schedule {
                Schedule::Start(v) => self.start_time = v,
                Schedule::Stop(v) => self.stop_time = v,
            }
        }

        log::warn!("ConstantSourceRenderer: Ignoring incoming message");
    }
}

#[cfg(test)]
mod tests {
    use crate::context::{BaseAudioContext, OfflineAudioContext};
    use crate::node::{AudioNode, AudioScheduledSourceNode};

    use float_eq::assert_float_eq;

    #[test]
    fn test_start_stop() {
        let sample_rate = 48000.;
        let start_in_samples = (128 + 1) as f64; // start rendering in 2d block
        let stop_in_samples = (256 + 1) as f64; // stop rendering of 3rd block
        let context = OfflineAudioContext::new(1, 128 * 4, sample_rate);

        let src = context.create_constant_source();
        src.connect(&context.destination());

        src.start_at(start_in_samples / sample_rate as f64);
        src.stop_at(stop_in_samples / sample_rate as f64);

        let buffer = context.start_rendering_sync();
        let channel = buffer.get_channel_data(0);

        // 1rst block should be silence
        assert_float_eq!(channel[0..128], vec![0.; 128][..], abs_all <= 0.);

        // 2d block - start at second frame
        let mut res = vec![1.; 128];
        res[0] = 0.;
        assert_float_eq!(channel[128..256], res[..], abs_all <= 0.);

        // 3rd block - stop at second frame
        let mut res = vec![0.; 128];
        res[0] = 1.;
        assert_float_eq!(channel[256..384], res[..], abs_all <= 0.);

        // 4th block is silence
        assert_float_eq!(channel[384..512], vec![0.; 128][..], abs_all <= 0.);
    }

    #[test]
    fn test_start_in_the_past() {
        let context = OfflineAudioContext::new(1, 128, 48000.);

        let src = context.create_constant_source();
        src.connect(&context.destination());
        src.start_at(-1.);

        let buffer = context.start_rendering_sync();
        let channel = buffer.get_channel_data(0);

        // 1rst block should be silence
        assert_float_eq!(channel[0..128], vec![1.; 128][..], abs_all <= 0.);
    }
}

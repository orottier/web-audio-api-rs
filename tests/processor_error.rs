use float_eq::assert_float_eq;
use web_audio_api::context::{AudioContextRegistration, BaseAudioContext, OfflineAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, ChannelConfig};
use web_audio_api::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};
use web_audio_api::RENDER_QUANTUM_SIZE;

struct PanicNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
}

impl AudioNode for PanicNode {
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

impl PanicNode {
    /// Construct a new WhiteNoiseNode
    fn new<C: BaseAudioContext>(context: &C) -> Self {
        context.register(move |registration| {
            let render = PanicProcessor {};

            let node = PanicNode {
                registration,
                channel_config: ChannelConfig::default(),
            };

            (node, Box::new(render))
        })
    }
}

struct PanicProcessor {}

impl AudioProcessor for PanicProcessor {
    fn process(
        &mut self,
        _inputs: &[AudioRenderQuantum],
        _outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues,
        _scope: &RenderScope,
    ) -> bool {
        panic!("panic message");
    }
}

#[test]
fn test_processor_error() {
    let context = OfflineAudioContext::new(1, RENDER_QUANTUM_SIZE, 48000.);

    {
        // create constant source with value 1, connect to destination
        let source1 = context.create_constant_source();
        source1.offset().set_value(1.);
        source1.connect(&context.destination());
        source1.start();

        // create constant source with value 2, connect to error processor
        let source2 = context.create_constant_source();
        source2.offset().set_value(2.);
        let panic = PanicNode::new(&context);
        source2.connect(&panic);
        panic.connect(&context.destination());
        source2.start();
    }

    let output = context.start_rendering_sync();
    // error branch should be muted, and other source should be processed
    assert_float_eq!(
        output.get_channel_data(0),
        &[1.; RENDER_QUANTUM_SIZE][..],
        abs_all <= 0.
    );
}

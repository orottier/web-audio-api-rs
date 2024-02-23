use float_eq::assert_float_eq;

use web_audio_api::context::{BaseAudioContext, OfflineAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
use web_audio_api::worklet::{
    AudioParamValues, AudioWorkletGlobalScope, AudioWorkletNode, AudioWorkletNodeOptions,
    AudioWorkletProcessor,
};

struct PanicProcessor;

impl AudioWorkletProcessor for PanicProcessor {
    type ProcessorOptions = ();

    fn constructor(_opts: Self::ProcessorOptions) -> Self {
        Self {}
    }

    fn process<'a, 'b>(
        &mut self,
        _inputs: &'b [&'a [&'a [f32]]],
        _outputs: &'b mut [&'a mut [&'a mut [f32]]],
        _params: AudioParamValues<'b>,
        _scope: &'b AudioWorkletGlobalScope,
    ) -> bool {
        panic!("panic message");
    }
}

#[test]
fn test_processor_error() {
    let mut context = OfflineAudioContext::new(1, 128, 48000.);

    {
        // create constant source with value 1, connect to destination
        let mut source1 = context.create_constant_source();
        source1.offset().set_value(1.);
        source1.connect(&context.destination());
        source1.start();

        // create constant source with value 2, connect to error processor
        let mut source2 = context.create_constant_source();
        source2.offset().set_value(2.);
        let options = AudioWorkletNodeOptions::default();
        let panic = AudioWorkletNode::new::<PanicProcessor>(&context, options);
        source2.connect(&panic);
        panic.connect(&context.destination());
        source2.start();
    }

    let output = context.start_rendering_sync();
    // error branch should be muted, and other source should be processed
    assert_float_eq!(output.get_channel_data(0), &[1.; 128][..], abs_all <= 0.);
}

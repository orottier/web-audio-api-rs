use web_audio_api::node::worklet::{AudioParamValues, AudioWorkletProcessor};
use web_audio_api::render::RenderScope;
use web_audio_api::{AudioParamDescriptor, AutomationRate};

pub struct GainProcessor;

impl AudioWorkletProcessor for GainProcessor {
    type ProcessorOptions = ();

    fn constructor(_opts: Self::ProcessorOptions) -> Self {
        Self {}
    }

    fn parameter_descriptors() -> Vec<AudioParamDescriptor>
    where
        Self: Sized,
    {
        vec![AudioParamDescriptor {
            name: String::from("gain"),
            min_value: f32::MIN,
            max_value: f32::MAX,
            default_value: 1.,
            automation_rate: AutomationRate::A,
        }]
    }

    fn process<'a, 'b>(
        &mut self,
        _scope: &'b RenderScope,
        inputs: &'b [&'b [&'a [f32]]],
        outputs: &'b mut [&'b mut [&'a mut [f32]]],
        params: AudioParamValues<'b>,
    ) -> bool {
        // passthrough with gain
        inputs[0]
            .iter()
            .zip(outputs[0].iter_mut())
            .for_each(|(ic, oc)| {
                let gain = params.get("gain");
                for ((is, os), g) in ic.iter().zip(oc.iter_mut()).zip(gain.iter().cycle()) {
                    *os = is * g;
                }
            });

        false
    }
}

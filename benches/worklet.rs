use web_audio_api::worklet::{AudioParamValues, AudioWorkletGlobalScope, AudioWorkletProcessor};
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
        inputs: &'b [&'a [&'a [f32]]],
        outputs: &'b mut [&'a mut [&'a mut [f32]]],
        params: AudioParamValues<'b>,
        _scope: &'b AudioWorkletGlobalScope,
    ) -> bool {
        let gain = params.get("gain");
        let io_zip = inputs[0].iter().zip(outputs[0].iter_mut());
        if gain.len() == 1 {
            let gain = gain[0];
            io_zip.for_each(|(ic, oc)| {
                for (is, os) in ic.iter().zip(oc.iter_mut()) {
                    *os = is * gain;
                }
            });
        } else {
            io_zip.for_each(|(ic, oc)| {
                for ((is, os), g) in ic.iter().zip(oc.iter_mut()).zip(gain.iter().cycle()) {
                    *os = is * g;
                }
            });
        }

        false
    }
}

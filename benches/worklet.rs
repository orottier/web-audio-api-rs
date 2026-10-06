use web_audio_api::context::AudioParamId;
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

/// Parameter-table size for the lookup benchmarks: representative of synth
/// instruments / plugin hosts, where every parameter is read once per render
/// quantum.
pub const MANY_PARAMS: usize = 64;

pub fn many_param_descriptors() -> Vec<AudioParamDescriptor> {
    (0..MANY_PARAMS)
        .map(|i| AudioParamDescriptor {
            name: format!("param_{i}"),
            min_value: f32::MIN,
            max_value: f32::MAX,
            default_value: 1.,
            automation_rate: AutomationRate::A,
        })
        .collect()
}

fn many_param_names() -> Vec<String> {
    (0..MANY_PARAMS).map(|i| format!("param_{i}")).collect()
}

/// Reads all `MANY_PARAMS` parameters each quantum through `get(name)` - one
/// name hash per parameter per quantum. Names are pre-formatted so the
/// benchmark isolates the lookup, not string construction.
pub struct ManyParamsByNameProcessor {
    names: Vec<String>,
}

impl AudioWorkletProcessor for ManyParamsByNameProcessor {
    type ProcessorOptions = ();

    fn constructor(_opts: Self::ProcessorOptions) -> Self {
        Self {
            names: many_param_names(),
        }
    }

    fn parameter_descriptors() -> Vec<AudioParamDescriptor>
    where
        Self: Sized,
    {
        many_param_descriptors()
    }

    fn process<'a, 'b>(
        &mut self,
        _inputs: &'b [&'a [&'a [f32]]],
        outputs: &'b mut [&'a mut [&'a mut [f32]]],
        params: AudioParamValues<'b>,
        _scope: &'b AudioWorkletGlobalScope,
    ) -> bool {
        let mut acc = 0.;
        for name in &self.names {
            acc += params.get(name)[0];
        }
        for channel in outputs[0].iter_mut() {
            channel.fill(acc / MANY_PARAMS as f32);
        }
        true
    }
}

/// Same workload as [`ManyParamsByNameProcessor`], but the names are resolved
/// to [`AudioParamId`]s once on the first quantum and every subsequent read
/// goes through `get_by_id` - no hashing on the audio thread.
pub struct ManyParamsByIdProcessor {
    ids: Option<Vec<AudioParamId>>,
}

impl AudioWorkletProcessor for ManyParamsByIdProcessor {
    type ProcessorOptions = ();

    fn constructor(_opts: Self::ProcessorOptions) -> Self {
        Self { ids: None }
    }

    fn parameter_descriptors() -> Vec<AudioParamDescriptor>
    where
        Self: Sized,
    {
        many_param_descriptors()
    }

    fn process<'a, 'b>(
        &mut self,
        _inputs: &'b [&'a [&'a [f32]]],
        outputs: &'b mut [&'a mut [&'a mut [f32]]],
        params: AudioParamValues<'b>,
        _scope: &'b AudioWorkletGlobalScope,
    ) -> bool {
        let ids = self.ids.get_or_insert_with(|| {
            many_param_names()
                .iter()
                .map(|name| params.id(name).unwrap())
                .collect()
        });
        let mut acc = 0.;
        for &id in ids.iter() {
            acc += params.get_by_id(id)[0];
        }
        for channel in outputs[0].iter_mut() {
            channel.fill(acc / MANY_PARAMS as f32);
        }
        true
    }
}

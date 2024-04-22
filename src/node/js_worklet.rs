use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::js_runtime::NodeRuntime;
use crate::node::{AudioNode, ChannelConfig};
use crate::param::{AudioParam, AudioParamDescriptor};
use crate::worklet::{
    AudioParamValues, AudioWorkletGlobalScope, AudioWorkletNode, AudioWorkletNodeOptions,
    AudioWorkletProcessor,
};
use crate::MessagePort;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

fn js_runtime() -> &'static Mutex<NodeRuntime> {
    static INSTANCE: OnceLock<Mutex<NodeRuntime>> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        let mut runtime = NodeRuntime::new().unwrap();
        let init_code = "class AudioWorkletProcessor { }\nfunction registerProcessor(name, cls) { console.log('REGISTER', name, cls.name); }\n";
        runtime.eval(init_code).unwrap();
        Mutex::new(runtime)
    })
}

// Horrible hack to use a singleton for all AudioParamDescriptors - TODO
fn dynamic_param_descriptors() -> &'static Mutex<Vec<AudioParamDescriptor>> {
    static INSTANCE: OnceLock<Mutex<Vec<AudioParamDescriptor>>> = OnceLock::new();
    INSTANCE.get_or_init(Default::default)
}

// todo, this should be handled per AudioContext, not globally
fn registered_processors() -> &'static Mutex<HashMap<String, String>> {
    static INSTANCE: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();
    INSTANCE.get_or_init(Default::default)
}

fn incremental_id() -> u32 {
    static INSTANCE: OnceLock<AtomicU32> = OnceLock::new();
    INSTANCE
        .get_or_init(|| AtomicU32::new(0))
        .fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug)]
pub struct JsWorkletNode {
    node: AudioWorkletNode,
}

impl JsWorkletNode {
    #[allow(clippy::missing_panics_doc)]
    pub fn add_module(module: &str) {
        let mut runtime = js_runtime().lock().unwrap();
        runtime.eval_file(module).unwrap();
        runtime.eval("console.log('Done999')\n").unwrap();

        'outer: loop {
            for o in runtime.output() {
                println!("{o}");
                if o.starts_with("> REGISTER") {
                    let mut pieces = o.split(' ');
                    pieces.next().unwrap();
                    pieces.next().unwrap();
                    let register_name = pieces.next().unwrap().to_string();
                    let register_class = pieces.next().unwrap().to_string();
                    registered_processors()
                        .lock()
                        .unwrap()
                        .insert(register_name, register_class);
                }
                if o == "> > Done999" {
                    break 'outer;
                }
            }
        }
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn new(
        context: &impl BaseAudioContext,
        node_name: &str,
        options: AudioWorkletNodeOptions<()>,
    ) -> Self {
        let mut runtime = js_runtime().lock().unwrap();
        let id = incremental_id();

        let class = registered_processors()
            .lock()
            .unwrap()
            .get(node_name)
            .unwrap_or_else(|| panic!("Unknown node {}, not registered", node_name))
            .clone();

        let code = format!(
            "const proc{} = new {}();
            console.log(JSON.stringify({}.parameterDescriptors));
            console.log('Done123');\n",
            id, class, class
        );
        runtime.eval(&code).unwrap();

        let mut prev = String::new();
        let mut pprev = String::new();
        'outer: loop {
            for o in runtime.output() {
                println!("{o}");
                if o == "> Done123" {
                    break 'outer;
                }
                std::mem::swap(&mut prev, &mut pprev);
                prev = o;
            }
        }

        let descriptors_js = if pprev == "> undefined" {
            "[]"
        } else {
            &pprev[2..]
        };
        let params: Vec<AudioParamDescriptor> = serde_json::from_str(descriptors_js).unwrap();
        let param_names: Vec<_> = params.iter().map(|d| &d.name).cloned().collect();
        dbg!(&param_names);
        *dynamic_param_descriptors().lock().unwrap() = params;

        // Remap the constructor options to include our processor options
        let AudioWorkletNodeOptions {
            number_of_inputs,
            number_of_outputs,
            output_channel_count,
            parameter_data,
            processor_options: _processor_options,
            audio_node_options,
        } = options;
        let options = AudioWorkletNodeOptions {
            number_of_inputs,
            number_of_outputs,
            output_channel_count,
            parameter_data,
            audio_node_options,
            processor_options: (id, param_names),
        };

        let node = AudioWorkletNode::new::<JsWorkletProcessor>(context, options);

        Self { node }
    }

    /// Collection of AudioParam objects with associated names of this node
    ///
    /// This map is populated from a list of [`AudioParamDescriptor`]s in the
    /// [`AudioWorkletProcessor`] class constructor at the instantiation.
    pub fn parameters(&self) -> &HashMap<String, AudioParam> {
        self.node.parameters()
    }

    /// Message port to the processor in the render thread
    ///
    /// Every AudioWorkletNode has an associated port which is the [`MessagePort`]. It is connected
    /// to the port on the corresponding [`AudioWorkletProcessor`] object allowing bidirectional
    /// communication between the AudioWorkletNode and its AudioWorkletProcessor.
    pub fn port(&self) -> MessagePort<'_> {
        self.node.port()
    }
}

impl AudioNode for JsWorkletNode {
    fn registration(&self) -> &AudioContextRegistration {
        self.node.registration()
    }

    fn channel_config(&self) -> &ChannelConfig {
        self.node.channel_config()
    }

    fn number_of_inputs(&self) -> usize {
        self.node.number_of_inputs()
    }

    fn number_of_outputs(&self) -> usize {
        self.node.number_of_outputs()
    }
}

struct JsWorkletProcessor {
    id: u32,
    param_names: Vec<String>,
}

impl AudioWorkletProcessor for JsWorkletProcessor {
    type ProcessorOptions = (u32, Vec<String>);

    fn constructor(opts: Self::ProcessorOptions) -> Self {
        Self {
            id: opts.0,
            param_names: opts.1,
        }
    }

    fn parameter_descriptors() -> Vec<AudioParamDescriptor>
    where
        Self: Sized,
    {
        dynamic_param_descriptors().lock().unwrap().clone()
    }

    fn process<'a, 'b>(
        &mut self,
        inputs: &'b [&'a [&'a [f32]]],
        outputs: &'b mut [&'a mut [&'a mut [f32]]],
        params: AudioParamValues<'b>,
        scope: &'b AudioWorkletGlobalScope,
    ) -> bool {
        let input_json = serde_json::to_string(&inputs).unwrap();
        let output_json = serde_json::to_string(&outputs).unwrap();
        let params: HashMap<String, Vec<f32>> = self
            .param_names
            .iter()
            .map(|n| (n.clone(), params.get(n).to_vec()))
            .collect();
        let params_json = serde_json::to_string(&params).unwrap();
        let code = format!(
            // the last value is echoed back by the repl, so put some scalar value last:
            "inputs = {}, outputs = {}, params = {}, currentFrame = {}, currentTime = {}, sampleRate = {};
proc{}.process(inputs, outputs, params);
console.log(JSON.stringify(outputs));
console.log('Done123');
",
            input_json,
            output_json,
            params_json,
            scope.current_time,
            scope.current_frame,
            scope.sample_rate,
            self.id
        );

        // println!(&code);
        let mut runtime = js_runtime().lock().unwrap();
        runtime.eval(&code).unwrap();

        let mut prev = String::new();
        let mut pprev = String::new();
        'outer: loop {
            for o in runtime.output() {
                // println!("{o}");
                if o == "> Done123" {
                    break 'outer;
                }
                std::mem::swap(&mut prev, &mut pprev);
                prev = o;
            }
        }

        // dbg!(&pprev[2..]);
        let node_outputs: Vec<Vec<Vec<f32>>> = serde_json::from_str(&pprev[2..]).unwrap();

        for i in 0..outputs.len() {
            for c in 0..outputs[i].len() {
                outputs[i][c].copy_from_slice(&node_outputs[i][c]);
            }
        }

        true
    }
}

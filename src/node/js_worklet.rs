use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::js_runtime::NodeRuntime;
use crate::worklet::{
    AudioParamValues, AudioWorkletGlobalScope, AudioWorkletNode, AudioWorkletNodeOptions,
    AudioWorkletProcessor,
};

use crate::node::{AudioNode, ChannelConfig};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

fn js_runtime() -> &'static Mutex<NodeRuntime> {
    static INSTANCE: OnceLock<Mutex<NodeRuntime>> = OnceLock::new();
    INSTANCE.get_or_init(|| Mutex::new(NodeRuntime::new().unwrap()))
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
    id: u32,
}

impl JsWorkletNode {
    pub fn new(
        context: &impl BaseAudioContext,
        module: &str,
        options: AudioWorkletNodeOptions<()>,
    ) -> Self {
        let node = AudioWorkletNode::new::<JsWorkletProcessor>(context, options);
        let id = incremental_id();

        let mut runtime = js_runtime().lock().unwrap();
        runtime.eval_file(module).unwrap();
        let code = format!(
            "const proc{} = new WhiteNoiseProcessor(); console.log('Done123');\n",
            id
        );
        runtime.eval(&code).unwrap();

        'outer: loop {
            for o in runtime.output() {
                println!("{o}");
                if o.contains("> Done123") {
                    break 'outer;
                }
            }
        }

        Self { node, id }
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
}

impl AudioWorkletProcessor for JsWorkletProcessor {
    type ProcessorOptions = ();

    fn constructor(_opts: Self::ProcessorOptions) -> Self {
        Self { id: 0 } // TODO
    }

    fn process<'a, 'b>(
        &mut self,
        inputs: &'b [&'a [&'a [f32]]],
        outputs: &'b mut [&'a mut [&'a mut [f32]]],
        _params: AudioParamValues<'b>,
        scope: &'b AudioWorkletGlobalScope,
    ) -> bool {
        let input_json = serde_json::to_string(&inputs).unwrap();
        let output_json = serde_json::to_string(&outputs).unwrap();
        let params = "{}"; // TODO
        let code = format!(
            "
currentFrame = {}, currentTime = {}, sampleRate = {}, inputs = {}, outputs = {}, params = {};
proc{}.process(inputs, outputs, params);
console.log(JSON.stringify(outputs));
console.log('Done123');
",
            scope.current_time,
            scope.current_frame,
            scope.sample_rate,
            input_json,
            output_json,
            params,
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
                if o.contains("> Done123") {
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

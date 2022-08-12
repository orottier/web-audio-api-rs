use web_audio_api::context::{AudioContextRegistration, BaseAudioContext, OfflineAudioContext};
use web_audio_api::node::{AudioNode, ChannelConfig};
use web_audio_api::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};
use web_audio_api::RENDER_QUANTUM_SIZE;

use std::sync::{Arc, Mutex};

struct DebugNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
}

impl AudioNode for DebugNode {
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

impl DebugNode {
    fn new<C: BaseAudioContext>(
        context: &C,
        name: impl ToString,
        collect: Arc<Mutex<Vec<String>>>,
    ) -> Self {
        context.register(move |registration| {
            let render = DebugProcessor {
                name: name.to_string(),
                collect,
            };

            let node = DebugNode {
                registration,
                channel_config: ChannelConfig::default(),
            };

            (node, Box::new(render))
        })
    }
}

struct DebugProcessor {
    name: String,
    collect: Arc<Mutex<Vec<String>>>,
}

impl AudioProcessor for DebugProcessor {
    fn process(
        &mut self,
        _inputs: &[AudioRenderQuantum],
        _outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues,
        _scope: &RenderScope,
    ) -> bool {
        self.collect.lock().unwrap().push(self.name.clone());
        true
    }
}

/*
 +---+     +---+     +---+     +---+
 | 1 | --> | 2 | --> | 3 | --> | D |
 +---+     +---+     +---+     +---+
*/
#[test]
fn sort_linear() {
    let context = OfflineAudioContext::new(1, RENDER_QUANTUM_SIZE, 44_100.);
    let collect = Arc::new(Mutex::new(vec![]));

    let first = DebugNode::new(&context, "1", collect.clone());
    let second = DebugNode::new(&context, "2", collect.clone());
    let third = DebugNode::new(&context, "3", collect.clone());

    first.connect(&second);
    second.connect(&third);

    let _ = context.start_rendering_sync();
    assert_eq!(
        &collect.lock().unwrap()[..],
        ["1".to_string(), "2".to_string(), "3".to_string(),]
    );
}

/*
 +----+     +----+     +---+     +---+
 | 1A | --> | 1B | --> | 3 | --> | D |
 +----+     +----+     +---+     +---+
                         ^
 +----+     +----+       |
 | 2A | --> | 2B | ------+
 +----+     +----+
*/
#[test]
fn sort_fork() {
    let context = OfflineAudioContext::new(1, RENDER_QUANTUM_SIZE, 44_100.);
    let collect = Arc::new(Mutex::new(vec![]));

    let one_a = DebugNode::new(&context, "1A", collect.clone());
    let one_b = DebugNode::new(&context, "1B", collect.clone());
    let two_a = DebugNode::new(&context, "2A", collect.clone());
    let two_b = DebugNode::new(&context, "2B", collect.clone());
    let three = DebugNode::new(&context, "3", collect.clone());

    one_a.connect(&one_b);
    one_b.connect(&three);
    two_a.connect(&two_b);
    two_b.connect(&three);

    let _ = context.start_rendering_sync();
    assert_eq!(
        &collect.lock().unwrap()[..],
        [
            "1A".to_string(),
            "1B".to_string(),
            "2A".to_string(),
            "2B".to_string(),
            "3".to_string(),
        ]
    );
}

/*
             +---------+
             v         |
 +---+     +---+     +---+
 | 1 | --> | 2 | --> | 3 |
 +---+     +---+     +---+
             |
             v
           +---+
           | 4 |
           +---+
*/
#[test]
fn sort_mute_cycle() {
    let context = OfflineAudioContext::new(1, RENDER_QUANTUM_SIZE, 44_100.);
    let collect = Arc::new(Mutex::new(vec![]));

    let one = DebugNode::new(&context, "1", collect.clone());
    let two = DebugNode::new(&context, "2", collect.clone());
    let three = DebugNode::new(&context, "3", collect.clone());
    let four = DebugNode::new(&context, "4", collect.clone());

    one.connect(&two);
    two.connect(&three);
    three.connect(&two);
    two.connect(&four);

    let _ = context.start_rendering_sync();
    assert_eq!(
        &collect.lock().unwrap()[..],
        ["1".to_string(), "4".to_string(),]
    );
}

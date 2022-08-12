use web_audio_api::context::{AudioContextRegistration, BaseAudioContext, OfflineAudioContext};
use web_audio_api::node::{AudioNode, ChannelConfig};
use web_audio_api::render::{AudioParamValues, AudioProcessor, AudioRenderQuantum, RenderScope};
use web_audio_api::RENDER_QUANTUM_SIZE;

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use rand::seq::SliceRandom;
use rand::thread_rng;

type Label = u32;

fn test_ordering(nodes: &[Label], edges: &[[Label; 2]], test: impl Fn(Vec<Label>)) {
    test_ordering_with_cycle_breakers(nodes, &[], edges, test);
}

fn test_ordering_with_cycle_breakers(
    nodes: &[Label],
    cycle_breakers: &[Label],
    edges: &[[Label; 2]],
    test: impl Fn(Vec<Label>),
) {
    for _ in 0..10 {
        // shuffle inputs because graph ordering may depend on initial ordering
        let mut nodes = nodes.to_vec();
        let mut edges = edges.to_vec();
        let mut rng = thread_rng();
        nodes.shuffle(&mut rng);
        edges.shuffle(&mut rng);

        let context = OfflineAudioContext::new(1, RENDER_QUANTUM_SIZE, 44_100.);
        let collect = Arc::new(Mutex::new(vec![]));

        let map: HashMap<_, _> = nodes
            .into_iter()
            .map(|label| {
                let cycle_breaker = cycle_breakers.iter().any(|&c| c == label);
                (
                    label,
                    DebugNode::new(&context, label, collect.clone(), cycle_breaker),
                )
            })
            .collect();

        edges.iter().for_each(|[s, d]| {
            let source = map.get(s).unwrap();
            let dest = map.get(d).unwrap();
            source.connect(dest);
        });

        let _ = context.start_rendering_sync();
        let results = Arc::try_unwrap(collect).unwrap().into_inner().unwrap();
        dbg!(&results);
        (test)(results);
    }
}

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
        name: Label,
        collect: Arc<Mutex<Vec<Label>>>,
        cycle_breaker: bool,
    ) -> Self {
        let node = context.register(move |registration| {
            let render = DebugProcessor { name, collect };

            let node = DebugNode {
                registration,
                channel_config: ChannelConfig::default(),
            };

            (node, Box::new(render))
        });

        if cycle_breaker {
            let in_cycle = Arc::new(AtomicBool::new(false));
            context
                .base()
                .mark_cycle_breaker(node.registration(), in_cycle);
        }

        node
    }
}

struct DebugProcessor {
    name: Label,
    collect: Arc<Mutex<Vec<Label>>>,
}

impl AudioProcessor for DebugProcessor {
    fn process(
        &mut self,
        _inputs: &[AudioRenderQuantum],
        _outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues,
        _scope: &RenderScope,
    ) -> bool {
        self.collect.lock().unwrap().push(self.name);
        true
    }
}

fn pos(v: Label, list: &[Label]) -> usize {
    list.iter().position(|&x| x == v).unwrap()
}

/*
 +---+     +---+     +---+
 | 1 | --> | 2 | --> | 3 |
 +---+     +---+     +---+
*/
#[test]
fn sort_linear() {
    test_ordering(&[1, 2, 3], &[[1, 2], [2, 3]], |result| {
        assert_eq!(result, [1, 2, 3])
    });
}

/*
 +----+     +----+     +---+
 | 10 | --> | 11 | --> | 3 |
 +----+     +----+     +---+
                         ^
 +----+     +----+       |
 | 20 | --> | 21 | ------+
 +----+     +----+
*/
#[test]
fn sort_fork() {
    test_ordering(
        &[10, 11, 20, 21, 3],
        &[[10, 11], [11, 3], [20, 21], [21, 3]],
        |result| {
            let pos10 = pos(10, &result);
            let pos11 = pos(11, &result);
            let pos20 = pos(20, &result);
            let pos21 = pos(21, &result);
            let pos3 = pos(3, &result);

            assert!(pos10 < pos11);
            assert!(pos20 < pos21);
            assert!(pos11 < pos3);
            assert!(pos21 < pos3);
        },
    );
}

/*
   +-------------------+
   |                   v
 +---+     +---+     +---+
 | 1 | --> | 2 | --> | 3 |
 +---+     +---+     +---+
*/
#[test]
fn sort_no_cyle() {
    test_ordering(&[1, 2, 3], &[[1, 2], [2, 3], [1, 3]], |result| {
        assert_eq!(result, [1, 2, 3])
    });
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
    test_ordering(&[1, 2, 3, 4], &[[1, 2], [2, 3], [3, 2], [2, 4]], |result| {
        assert_eq!(result, &[1, 4])
    });
}

/*
            +---------+
            v         |
+---+     +---+     +----------+
| 1 | --> | 2 | --> | 3: delay |
+---+     +---+     +----------+
            |
            v
          +---+
          | 4 |
          +---+
*/
#[test]
fn sort_cycle_breaker() {
    test_ordering_with_cycle_breakers(
        &[1, 2, 3, 4],
        &[3],
        &[[1, 2], [2, 3], [3, 2], [2, 4]],
        |result| {
            let pos1 = pos(1, &result);
            let pos2 = pos(2, &result);
            let pos3 = pos(3, &result);
            let pos4 = pos(4, &result);

            // cycle is broken, which clears the edge 3 -> 2
            assert!(pos1 < pos2);
            assert!(pos2 < pos3);
            assert!(pos2 < pos4);
        },
    );
}

/*
 +---+     +----------+     +---+
 | 1 | --> | 2: delay | --> | 3 |
 +---+     +----------+     +---+
*/
#[test]
fn sort_dont_break_cycle_if_possible() {
    test_ordering_with_cycle_breakers(&[1, 2, 3], &[2], &[[1, 2], [2, 3]], |result| {
        assert_eq!(result, [1, 2, 3])
    });
}

/*
+---+     +----------+     +----------+
| 1 | --> |          | --> | 3: delay |
+---+     |          |     +----------+
          |    2     |       |
          |          | <-----+
+---+     |          |
| 5 | <-- |          | <+
+---+     +----------+  |
            |           |
            |           |
            v           |
          +----------+  |
          | 4: delay | -+
          +----------+
*/
#[test]
fn sort_two_cycles() {
    test_ordering_with_cycle_breakers(
        &[1, 2, 3, 4, 5],
        &[3, 4],
        &[[1, 2], [2, 3], [3, 2], [2, 5], [2, 4], [4, 2]],
        |result| {
            // cycle is broken, which clears the edges 3 -> 2 and 4 -> 2
            assert_eq!(result[0], 1);
            assert_eq!(result[1], 2);
            assert_eq!(result.len(), 5); // all nodes present
        },
    );
}

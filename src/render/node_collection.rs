use crate::context::{AudioNodeId, DESTINATION_NODE_ID};
use crate::render::graph::Node;

use std::cell::RefCell;
use std::ops::{Index, IndexMut};

pub(crate) struct NodeCollection {
    nodes: Vec<Option<RefCell<Node>>>,
}

impl NodeCollection {
    pub fn new() -> Self {
        let mut instance = Self {
            nodes: Vec::with_capacity(64),
        };
        instance.ensure_capacity(64);
        instance
    }

    // NodeCollection is considered empty until the destination is set up
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        let destination_id = DESTINATION_NODE_ID.0 as usize;
        self.nodes[destination_id].is_none()
    }

    #[inline(always)]
    fn ensure_capacity(&mut self, new_len: usize) {
        self.nodes
            .resize_with(new_len.max(self.nodes.len()), || None);
    }

    #[inline(always)]
    pub fn insert(&mut self, index: AudioNodeId, value: RefCell<Node>) {
        let index = index.0 as usize;
        self.ensure_capacity(index + 1);
        self.nodes[index] = Some(value);
    }

    #[inline(always)]
    pub fn remove(&mut self, index: AudioNodeId) -> RefCell<Node> {
        self.nodes[index.0 as usize]
            .take()
            .expect("Unable to remove non-existing Node in NodeCollection")
    }

    #[inline(always)]
    pub fn keys(&self) -> impl Iterator<Item = AudioNodeId> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.as_ref().and(Some(AudioNodeId(i as u64))))
    }

    #[inline(always)]
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut RefCell<Node>> {
        self.nodes.iter_mut().filter_map(Option::as_mut)
    }

    #[inline(always)]
    pub fn get(&self, index: AudioNodeId) -> Option<&RefCell<Node>> {
        self.nodes[index.0 as usize].as_ref()
    }

    #[inline(always)]
    pub fn get_mut(&mut self, index: AudioNodeId) -> Option<&mut RefCell<Node>> {
        self.nodes[index.0 as usize].as_mut()
    }

    #[inline(always)]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(AudioNodeId, &mut RefCell<Node>) -> bool,
    {
        self.nodes.iter_mut().enumerate().for_each(|(i, opt)| {
            if let Some(v) = opt.as_mut() {
                if !f(AudioNodeId(i as u64), v) {
                    *opt = None;
                }
            }
        })
    }
}

impl Index<AudioNodeId> for NodeCollection {
    type Output = RefCell<Node>;

    #[track_caller]
    #[inline(always)]
    fn index(&self, index: AudioNodeId) -> &Self::Output {
        self.nodes
            .get(index.0 as usize)
            .unwrap_or_else(|| panic!("Unexpected index {} for NodeCollection", index.0))
            .as_ref()
            .unwrap_or_else(|| panic!("Index {} for dropped Node in NodeCollection", index.0))
    }
}

impl IndexMut<AudioNodeId> for NodeCollection {
    #[track_caller]
    #[inline(always)]
    fn index_mut(&mut self, index: AudioNodeId) -> &mut Self::Output {
        self.nodes
            .get_mut(index.0 as usize)
            .unwrap_or_else(|| panic!("Unexpected index {} for NodeCollection", index.0))
            .as_mut()
            .unwrap_or_else(|| panic!("Index {} for dropped Node in NodeCollection", index.0))
    }
}

use crate::render::graph::Node;
use std::cell::RefCell;
use std::ops::{Index, IndexMut};

pub struct NodeCollection {
    nodes: Vec<Option<RefCell<Node>>>,
}

impl NodeCollection {
    pub fn with_capacity(capacity: usize) -> Self {
        let mut instance = Self {
            nodes: Vec::with_capacity(capacity),
        };
        instance.ensure_capacity(capacity);
        instance
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    #[inline(always)]
    fn ensure_capacity(&mut self, new_len: usize) {
        self.nodes
            .resize_with(new_len.max(self.nodes.len()), || None);
    }

    #[inline(always)]
    pub fn insert(&mut self, index: usize, value: RefCell<Node>) {
        self.ensure_capacity(index + 1);
        self.nodes[index] = Some(value);
    }

    #[inline(always)]
    pub fn remove(&mut self, index: usize) -> RefCell<Node> {
        self.nodes
            .get_mut(index)
            .expect("Unexpected remove index for NodeCollection")
            .take()
            .expect("Unable to remove non-existing Node in NodeCollection")
    }

    #[inline(always)]
    pub fn keys(&self) -> impl Iterator<Item = usize> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.as_ref().and(Some(i)))
    }

    #[inline(always)]
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut RefCell<Node>> {
        self.nodes.iter_mut().filter_map(Option::as_mut)
    }

    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<&RefCell<Node>> {
        self.nodes[index].as_ref()
    }

    #[inline(always)]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut RefCell<Node>> {
        self.nodes[index].as_mut()
    }

    #[inline(always)]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, &mut RefCell<Node>) -> bool,
    {
        self.nodes.iter_mut().enumerate().for_each(|(i, opt)| {
            if let Some(v) = opt.as_mut() {
                if !f(i, v) {
                    *opt = None;
                }
            }
        })
    }
}

impl Index<usize> for NodeCollection {
    type Output = RefCell<Node>;

    #[track_caller]
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        self.nodes
            .get(index)
            .unwrap_or_else(|| panic!("Unexpected index {} for NodeCollection", index))
            .as_ref()
            .unwrap_or_else(|| panic!("Index {} for dropped Node in NodeCollection", index))
    }
}

impl IndexMut<usize> for NodeCollection {
    #[track_caller]
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.nodes
            .get_mut(index)
            .unwrap_or_else(|| panic!("Unexpected index {} for NodeCollection", index))
            .as_mut()
            .unwrap_or_else(|| panic!("Index {} for dropped Node in NodeCollection", index))
    }
}

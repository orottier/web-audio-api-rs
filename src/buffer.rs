//! Audio signal data structures

/// Memory-resident audio asset, basically wraps a Vec<f32> (no channel support yet)
#[derive(Clone, Debug)]
pub struct AudioBuffer {
    data: Box<[f32]>,
}

impl AudioBuffer {
    pub fn new(length: usize) -> Self {
        Self {
            data: vec![0.; length].into_boxed_slice(),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.data.iter_mut()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data[..]
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data[..]
    }

    // cannot use AddAssign, since other is a ref
    pub fn add(&mut self, other: &Self) {
        self.iter_mut().zip(other.iter()).for_each(|(a, b)| *a += b)
    }
}

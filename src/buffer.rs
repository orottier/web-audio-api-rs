//! Audio signal data structures

use std::sync::Arc;

/// Memory-resident audio asset, basically a matrix of channels * samples
#[derive(Clone, Debug)]
pub struct AudioBuffer {
    channels: Box<[ChannelData]>,
}

impl AudioBuffer {
    pub fn new(channels: usize, length: usize) -> Self {
        let single = ChannelData::new(length);
        let channels = vec![single; channels].into_boxed_slice();

        Self { channels }
    }

    pub fn from(data: ChannelData) -> Self {
        let channels = vec![data].into_boxed_slice();
        Self { channels }
    }

    pub fn mix(&self, channels: usize) -> Self {
        match (self.number_of_channels(), channels) {
            (1, 1) => self.clone(),
            (1, 2) => Self {
                channels: vec![self.channels[0].clone(); 2].into_boxed_slice(),
            },
            (2, 1) => {
                let mut l = self.channels[0].clone();
                l.iter_mut()
                    .zip(self.channels[1].iter())
                    .for_each(|(l, r)| *l = 0.5 * (*l + r));

                Self {
                    channels: vec![l].into_boxed_slice(),
                }
            }
            _ => todo!(),
        }
    }

    pub fn number_of_channels(&self) -> usize {
        self.channels.len()
    }
}

/// Single channel audio samples, basically wraps a Vec<f32>
#[derive(Clone, Debug)]
pub struct ChannelData {
    data: Arc<Box<[f32]>>,
}

impl ChannelData {
    pub fn new(length: usize) -> Self {
        let buffer = vec![0.; length].into_boxed_slice();
        let data = Arc::new(buffer);

        Self { data }
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
        Arc::make_mut(&mut self.data).iter_mut()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data[..]
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut Arc::make_mut(&mut self.data)[..]
    }

    // cannot use AddAssign, since other is a ref
    pub fn add(&mut self, other: &Self) {
        self.iter_mut().zip(other.iter()).for_each(|(a, b)| *a += b)
    }
}

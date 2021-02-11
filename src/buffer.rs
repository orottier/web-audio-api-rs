//! Audio signal data structures

use std::sync::Arc;

/// Memory-resident audio asset, basically a matrix of channels * samples
///
/// An AudioBuffer has copy-on-write semantics, so it is cheap to clone.
#[derive(Clone, Debug)]
pub struct AudioBuffer {
    inner: AudioBufferType,
}
#[derive(Clone, Debug)]
enum AudioBufferType {
    /// n channels with m samples of silence
    Silence(usize, usize),
    /// n identical channels
    Mono(ChannelData, usize),
    /// n different channels
    Multi(Box<[ChannelData]>), // todo, make [ChannelData; 32] fixed size array
}

use AudioBufferType::*;

impl AudioBuffer {
    /// Create a silent audiobuffer with given channel and samples count.
    ///
    /// This function does not allocate.
    pub fn new(channels: usize, len: usize) -> Self {
        Self {
            inner: Silence(channels, len),
        }
    }

    /// Create a mono audiobuffer (single channel)
    pub fn from_mono(data: ChannelData, channels: usize) -> Self {
        Self {
            inner: Mono(data, channels),
        }
    }

    /// Create a multi-channel audiobuffer
    pub fn from_channels(data: Vec<ChannelData>) -> Self {
        Self {
            inner: Multi(data.into_boxed_slice()),
        }
    }

    /// Up/Down-mix to the desired number of channels
    pub fn mix(&self, channels: usize) -> Self {
        // short circuit silence and mono
        let data = match &self.inner {
            Silence(_, len) => {
                return Self {
                    inner: Silence(channels, *len),
                }
            }
            Mono(data, _) => {
                return Self {
                    inner: Mono(data.clone(), channels),
                }
            }
            Multi(data) => data,
        };

        match (data.len(), channels) {
            (n, m) if n == m => self.clone(),
            (1, c) => Self {
                inner: Mono(data[0].clone(), c),
            },
            (2, 1) => {
                let mut l = data[0].clone();
                l.iter_mut()
                    .zip(data[1].iter())
                    .for_each(|(l, r)| *l = 0.5 * (*l + r));

                Self { inner: Mono(l, 1) }
            }
            _ => todo!(),
        }
    }

    /// Number of channels in this AudioBuffer
    pub fn number_of_channels(&self) -> usize {
        match &self.inner {
            Silence(c, _) => *c,
            Mono(_, c) => *c,
            Multi(data) => data.len(),
        }
    }

    /// Number of samples per channel in this AudioBuffer
    pub fn sample_len(&self) -> usize {
        match &self.inner {
            Silence(_, len) => *len,
            Mono(data, _) => data.len(),
            Multi(data) => data[0].len(),
        }
    }

    /// Get the samples from this specific channel
    pub fn channel_data(&self, channel: usize) -> Option<&ChannelData> {
        match &self.inner {
            Silence(_, _) => None,
            Mono(data, _) => Some(data),
            Multi(data) => data.get(channel),
        }
    }

    /// Convert this buffer to silence, maintaining the channel and sample counts
    pub fn make_silent(&mut self) {
        match &mut self.inner {
            Silence(_, _) => (),
            Mono(data, channels) => self.inner = Silence(*channels, data.len()),
            Multi(data) => self.inner = Silence(data.len(), data[0].len()),
        }
    }

    /// Convert this buffer to a mono sound, maintaining the channel and sample counts.
    pub fn make_mono(&mut self) {
        let len = self.sample_len();
        let channels = self.number_of_channels();

        match &mut self.inner {
            Silence(_, _) => {
                self.inner = Mono(ChannelData::new(len), channels);
            }
            Mono(_data, _) => (),
            Multi(data) => {
                self.inner = Mono(data[0].clone(), channels);
            }
        }
    }

    /// Modify every channel in the same way
    pub fn modify_channels<F: Fn(&mut ChannelData)>(&mut self, fun: F) {
        if matches!(&self.inner, Silence(_, _)) {
            self.make_mono();
        }

        match &mut self.inner {
            Silence(_, _) => unreachable!(),
            Mono(data, _) => (fun)(data),
            Multi(data) => data.iter_mut().for_each(fun),
        }
    }
}

impl std::ops::Add for AudioBuffer {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        // mix buffers to the max channel count
        let channels_self = self.number_of_channels();
        let channels_other = other.number_of_channels();
        let channels = channels_self.max(channels_other);

        if channels_self > channels_other {
            other.mix(channels_self);
        }
        if channels_self < channels_other {
            self.mix(channels_other);
        }

        // early exit for simple cases, or determine which signal is Multi
        let (mut multi, other) = match (self.inner, other.inner) {
            (Silence(_, _), inner) => return Self { inner },
            (inner, Silence(_, _)) => return Self { inner },
            (Mono(mut s, _), Mono(o, _)) => {
                s.add(&o);
                return Self {
                    inner: Mono(s, channels),
                };
            }
            (Multi(data), inner) => (data, Self { inner }),
            (inner, Multi(data)) => (data, Self { inner }),
        };

        // mutate the Multi signal with values from the other
        (0..channels).for_each(|i| {
            if let Some(data) = other.channel_data(i) {
                multi[i].add(data)
            }
        });

        Self {
            inner: Multi(multi),
        }
    }
}

impl std::ops::AddAssign for AudioBuffer {
    fn add_assign(&mut self, other: Self) {
        *self = self.clone() + other;
    }
}

/// Single channel audio samples, basically wraps a `Arc<Vec<f32>>`
///
/// ChannelData has copy-on-write semantics, so it is cheap to clone.
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

    pub fn add(&mut self, other: &Self) {
        self.iter_mut().zip(other.iter()).for_each(|(a, b)| *a += b)
    }
}

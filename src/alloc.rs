//! Optimized audio signal data structures, used in AudioProcessors

use arrayvec::ArrayVec;
use std::cell::RefCell;
use std::rc::Rc;

use crate::buffer::ChannelInterpretation;

const LEN: usize = crate::BUFFER_SIZE as usize;
use crate::MAX_CHANNELS;

pub(crate) struct Alloc {
    inner: Rc<AllocInner>,
}

struct AllocInner {
    pool: RefCell<Vec<Rc<[f32; LEN]>>>,
    zeroes: Rc<[f32; LEN]>,
}

impl Alloc {
    pub fn with_capacity(n: usize) -> Self {
        let pool: Vec<_> = (0..n).map(|_| Rc::new([0.; LEN])).collect();
        let zeroes = Rc::new([0.; LEN]);

        let inner = AllocInner {
            pool: RefCell::new(pool),
            zeroes,
        };

        Self {
            inner: Rc::new(inner),
        }
    }

    pub fn allocate(&self) -> ChannelData {
        ChannelData {
            data: self.inner.allocate(),
            alloc: Rc::clone(&self.inner),
        }
    }

    pub fn silence(&self) -> ChannelData {
        ChannelData {
            data: Rc::clone(&self.inner.zeroes),
            alloc: Rc::clone(&self.inner),
        }
    }

    #[cfg(test)]
    pub fn pool_size(&self) -> usize {
        self.inner.pool.borrow().len()
    }
}

impl AllocInner {
    fn allocate(&self) -> Rc<[f32; LEN]> {
        if let Some(rc) = self.pool.borrow_mut().pop() {
            // re-use from pool
            rc
        } else {
            // allocate
            Rc::new([0.; 128])
        }
    }

    fn push(&self, data: Rc<[f32; LEN]>) {
        self.pool
            .borrow_mut() // infallible when single threaded
            .push(data);
    }
}

/// Single channel audio samples, basically wraps a `Rc<[f32; BUFFER_SIZE]>`
///
/// ChannelData has copy-on-write semantics, so it is cheap to clone.
#[derive(Clone)]
pub struct ChannelData {
    data: Rc<[f32; LEN]>,
    alloc: Rc<AllocInner>,
}

impl ChannelData {
    fn make_mut(&mut self) -> &mut [f32; LEN] {
        if Rc::strong_count(&self.data) != 1 {
            let mut new = self.alloc.allocate();
            Rc::make_mut(&mut new).copy_from_slice(self.data.deref());
            self.data = new;
        }

        Rc::make_mut(&mut self.data)
    }

    /// `O(1)` check if this buffer is equal to the 'silence buffer'
    ///
    /// If this function returns false, it is still possible for all samples to be zero.
    pub fn is_silent(&self) -> bool {
        Rc::ptr_eq(&self.data, &self.alloc.zeroes)
    }

    /// Sum two channels
    pub fn add(&mut self, other: &Self) {
        if self.is_silent() {
            *self = other.clone();
        } else if !other.is_silent() {
            self.iter_mut().zip(other.iter()).for_each(|(a, b)| *a += b)
        }
    }

    pub fn silence(&self) -> Self {
        ChannelData {
            data: self.alloc.zeroes.clone(),
            alloc: Rc::clone(&self.alloc),
        }
    }
}

use std::ops::{Deref, DerefMut};

impl Deref for ChannelData {
    type Target = [f32; LEN];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for ChannelData {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.make_mut()
    }
}

impl std::ops::Drop for ChannelData {
    fn drop(&mut self) {
        if Rc::strong_count(&self.data) == 1 {
            let rc = std::mem::replace(&mut self.data, self.alloc.zeroes.clone());
            self.alloc.push(rc);
        }
    }
}

/// Memory-resident audio asset, basically a matrix of channels * samples
///
/// An AudioBuffer has copy-on-write semantics, so it is cheap to clone.
#[derive(Clone)]
pub struct AudioBuffer {
    channels: ArrayVec<[ChannelData; MAX_CHANNELS]>,
}

impl AudioBuffer {
    pub fn new(channel: ChannelData) -> Self {
        let mut channels = ArrayVec::new();
        channels.push(channel);

        Self { channels }
    }

    /// Number of channels in this AudioBuffer
    pub fn number_of_channels(&self) -> usize {
        self.channels.len()
    }

    /// Set number of channels in this AudioBuffer
    ///
    /// Note: if the new number is higher than the previous, the new channels will be filled with
    /// garbage.
    pub fn set_number_of_channels(&mut self, n: usize) {
        assert!(n <= MAX_CHANNELS);
        for _ in self.number_of_channels()..n {
            self.channels.push(self.channels[0].clone());
        }
        self.channels.truncate(n);
    }

    /// Get the samples from this specific channel.
    pub fn channel_data(&self, channel: usize) -> &ChannelData {
        &self.channels[channel]
    }

    /// Get the samples from this specific channel (mutable).
    pub fn channel_data_mut(&mut self, channel: usize) -> &mut ChannelData {
        &mut self.channels[channel]
    }

    /// Up/Down-mix to the desired number of channels
    pub fn mix(&mut self, channels: usize, interpretation: ChannelInterpretation) {
        assert!(channels < MAX_CHANNELS);

        if self.number_of_channels() == channels {
            return;
        }

        let silence = self.channels[0].silence();

        // handle discrete interpretation
        if interpretation == ChannelInterpretation::Discrete {
            // upmix by filling with silence
            for _ in self.number_of_channels()..channels {
                self.channels.push(silence.clone());
            }

            // downmix by truncating
            self.channels.truncate(channels);

            return;
        }

        match (self.number_of_channels(), channels) {
            (1, 2) => {
                self.channels.push(self.channels[0].clone());
            }
            (1, 4) => {
                self.channels.push(self.channels[0].clone());
                self.channels.push(silence.clone());
                self.channels.push(silence);
            }
            (1, 6) => {
                let main = std::mem::replace(&mut self.channels[0], silence.clone());
                self.channels.push(silence.clone());
                self.channels.push(main);
                self.channels.push(silence.clone());
                self.channels.push(silence);
            }
            (2, 1) => {
                let right = self.channels[1].clone();
                self.channels[0]
                    .iter_mut()
                    .zip(right.iter())
                    .for_each(|(l, r)| *l = (*l + *r) / 2.);
                self.channels.truncate(1);
            }
            _ => todo!(),
        }
    }

    /// Convert this buffer to silence
    pub fn make_silent(&mut self) {
        let silence = self.channels[0].silence();

        self.channels[0] = silence;
        self.channels.truncate(1);
    }

    /// Convert to a single channel buffer, dropping excess channels
    pub fn force_mono(&mut self) {
        self.channels.truncate(1);
    }

    /// Modify every channel in the same way
    pub fn modify_channels<F: Fn(&mut ChannelData)>(&mut self, fun: F) {
        self.channels.iter_mut().for_each(fun)
    }

    /// Sum two AudioBuffers
    ///
    /// If the channel counts differ, the buffer with lower count will be upmixed.
    #[must_use]
    pub fn add(&self, other: &Self, interpretation: ChannelInterpretation) -> Self {
        // mix buffers to the max channel count
        let channels_self = self.number_of_channels();
        let channels_other = other.number_of_channels();
        let channels = channels_self.max(channels_other);

        let mut self_mixed = self.clone();
        self_mixed.mix(channels, interpretation);

        let mut other_mixed = other.clone();
        other_mixed.mix(channels, interpretation);

        self_mixed
            .channels
            .iter_mut()
            .zip(other_mixed.channels.iter())
            .take(channels)
            .for_each(|(s, o)| s.add(o));

        self_mixed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool() {
        // Create pool of size 2
        let alloc = Alloc::with_capacity(2);
        assert_eq!(alloc.pool_size(), 2);

        alloc_counter::deny_alloc(|| {
            {
                // take a buffer out of the pool
                let a = alloc.allocate();
                assert_eq!(*a.as_ref(), [0.; LEN]);
                assert_eq!(alloc.pool_size(), 1);

                // mutating this buffer will not allocate
                let mut a = a;
                a.iter_mut().for_each(|v| *v += 1.);
                assert_eq!(*a.as_ref(), [1.; LEN]);
                assert_eq!(alloc.pool_size(), 1);

                // clone this buffer, should not allocate
                let mut b: ChannelData = a.clone();
                assert_eq!(alloc.pool_size(), 1);

                // mutate cloned buffer, this will allocate
                b.iter_mut().for_each(|v| *v += 1.);
                assert_eq!(alloc.pool_size(), 0);
            }

            // all buffers are reclaimed
            assert_eq!(alloc.pool_size(), 2);

            let c = {
                let a = alloc.allocate();
                let b = alloc.allocate();

                let c = alloc_counter::allow_alloc(|| {
                    // we can allocate beyond the pool size
                    let c = alloc.allocate();
                    assert_eq!(alloc.pool_size(), 0);
                    c
                });

                // dirty allocations
                assert_eq!(*a.as_ref(), [1.; LEN]);
                assert_eq!(*b.as_ref(), [2.; LEN]);
                assert_eq!(*c.as_ref(), [0.; LEN]); // this one is fresh

                c
            };

            // dropping c will cause a re-allocation: the pool capacity is extended
            alloc_counter::allow_alloc(move || {
                std::mem::drop(c);
            });

            // pool size is now 3 due to extra allocations
            assert_eq!(alloc.pool_size(), 3);

            {
                // silence will not allocate at first
                let mut a = alloc.silence();
                assert!(a.is_silent());
                assert_eq!(alloc.pool_size(), 3);

                // deref mut will allocate
                let a_vals = a.deref_mut();
                assert_eq!(alloc.pool_size(), 2);

                // but should be silent, even though a dirty buffer is taken
                assert_eq!(*a_vals, [0.; LEN]);
                assert_eq!(*a_vals, [0.; LEN]);

                // is_silent is a superficial ptr check
                assert_eq!(a.is_silent(), false);
            }
        });
    }

    #[test]
    fn test_silence() {
        let alloc = Alloc::with_capacity(1);
        let silence = alloc.silence();
        assert_eq!(&silence[..], &[0.; LEN]);
        assert!(silence.is_silent());

        // changing silence is possible
        let mut changed = silence;
        changed.iter_mut().for_each(|v| *v = 1.);
        assert_eq!(&changed[..], &[1.; LEN]);
        assert!(!changed.is_silent());

        // but should not alter new silence
        let silence = alloc.silence();
        assert_eq!(&silence[..], &[0.; LEN]);
        assert!(silence.is_silent());

        // can also create silence from ChannelData
        let from_channel = silence.silence();
        assert_eq!(&from_channel[..], &[0.; LEN]);
        assert!(from_channel.is_silent());
    }

    #[test]
    fn test_channel_add() {
        let alloc = Alloc::with_capacity(1);
        let silence = alloc.silence();

        let mut signal1 = alloc.silence();
        signal1.copy_from_slice(&[1.; LEN]);

        let mut signal2 = alloc.allocate();
        signal2.copy_from_slice(&[2.; LEN]);

        // test add silence to signal
        signal1.add(&silence);
        assert_eq!(&signal1[..], &[1.; LEN]);

        // test add signal to silence
        let mut sum = alloc.silence();
        sum.add(&signal1);
        assert_eq!(&sum[..], &[1.; LEN]);

        // test add two signals
        signal1.add(&signal2);
        assert_eq!(&signal1[..], &[3.; LEN]);
    }

    #[test]
    fn test_audiobuffer_channels() {
        let alloc = Alloc::with_capacity(1);
        let silence = alloc.silence();

        let buffer = AudioBuffer::new(silence);
        assert_eq!(buffer.number_of_channels(), 1);

        let mut buffer = buffer;
        buffer.set_number_of_channels(5);
        assert_eq!(buffer.number_of_channels(), 5);
        let _ = buffer.channel_data(4); // no panic

        buffer.set_number_of_channels(2);
        assert_eq!(buffer.number_of_channels(), 2);
    }

    #[test]
    fn test_audiobuffer_mix_discrete() {
        let alloc = Alloc::with_capacity(1);

        let mut signal = alloc.silence();
        signal.copy_from_slice(&[1.; LEN]);

        let mut buffer = AudioBuffer::new(signal);

        buffer.mix(1, ChannelInterpretation::Discrete);
        assert_eq!(buffer.number_of_channels(), 1);
        assert_eq!(&buffer.channel_data(0)[..], &[1.; LEN]);

        buffer.mix(2, ChannelInterpretation::Discrete);
        assert_eq!(buffer.number_of_channels(), 2);
        // first channel unchanged, second channel silent
        assert_eq!(&buffer.channel_data(0)[..], &[1.; LEN]);
        assert_eq!(&buffer.channel_data(1)[..], &[0.; LEN]);

        buffer.mix(1, ChannelInterpretation::Discrete);
        assert_eq!(buffer.number_of_channels(), 1);
        assert_eq!(&buffer.channel_data(0)[..], &[1.; LEN]);
    }

    #[test]
    fn test_audiobuffer_mix_speakers() {
        let alloc = Alloc::with_capacity(1);

        let mut signal = alloc.silence();
        signal.copy_from_slice(&[1.; LEN]);

        let mut buffer = AudioBuffer::new(signal);

        buffer.mix(1, ChannelInterpretation::Speakers);
        assert_eq!(buffer.number_of_channels(), 1);
        assert_eq!(&buffer.channel_data(0)[..], &[1.; LEN]);

        buffer.mix(2, ChannelInterpretation::Speakers);
        assert_eq!(buffer.number_of_channels(), 2);
        // left and right equal
        assert_eq!(&buffer.channel_data(0)[..], &[1.; LEN]);
        assert_eq!(&buffer.channel_data(1)[..], &[1.; LEN]);

        buffer.mix(1, ChannelInterpretation::Speakers);
        assert_eq!(buffer.number_of_channels(), 1);
        assert_eq!(&buffer.channel_data(0)[..], &[1.; LEN]);
    }

    #[test]
    fn test_audiobuffer_add() {
        let alloc = Alloc::with_capacity(1);

        let mut signal = alloc.silence();
        signal.copy_from_slice(&[1.; LEN]);
        let mut buffer = AudioBuffer::new(signal);
        buffer.mix(2, ChannelInterpretation::Speakers);

        let mut signal2 = alloc.silence();
        signal2.copy_from_slice(&[2.; LEN]);
        let buffer2 = AudioBuffer::new(signal2);

        let sum = buffer.add(&buffer2, ChannelInterpretation::Discrete);
        assert_eq!(sum.number_of_channels(), 2);
        assert_eq!(&sum.channel_data(0)[..], &[3.; LEN]);
        assert_eq!(&sum.channel_data(1)[..], &[1.; LEN]);
    }
}

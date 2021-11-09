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

#[derive(Debug)]
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

    #[cfg(test)]
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
            Rc::new([0.; LEN])
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
#[derive(Clone, Debug)]
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

impl AsRef<[f32]> for ChannelData {
    fn as_ref(&self) -> &[f32] {
        &self.data[..]
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

/// Fixed length audio asset, basically a matrix of `channels * [f32; BUFFER_SIZE]`
///
/// An AudioBuffer has copy-on-write semantics, so it is cheap to clone.
#[derive(Clone, Debug)]
pub struct AudioBuffer {
    channels: ArrayVec<ChannelData, MAX_CHANNELS>,
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
    ///
    /// Panics if the index is greater than the available number of channels
    pub fn channel_data(&self, index: usize) -> &ChannelData {
        &self.channels[index]
    }

    /// Get the samples (mutable) from this specific channel.
    ///
    /// Panics if the index is greater than the available number of channels
    pub fn channel_data_mut(&mut self, index: usize) -> &mut ChannelData {
        &mut self.channels[index]
    }

    /// Channel data as slice
    pub fn channels(&self) -> &[ChannelData] {
        &self.channels[..]
    }

    /// Channel data as slice (mutable)
    pub fn channels_mut(&mut self) -> &mut [ChannelData] {
        &mut self.channels[..]
    }

    /// Up/Down-mix to the desired number of channels
    pub fn mix(&mut self, channels: usize, interpretation: ChannelInterpretation) {
        assert!(channels < MAX_CHANNELS);

        if self.number_of_channels() == channels {
            return;
        }

        let silence = self.channels[0].silence();

        // cf. https://www.w3.org/TR/webaudio/#channel-up-mixing-and-down-mixing
        // handle discrete interpretation
        if interpretation == ChannelInterpretation::Discrete {
            // upmix by filling with silence
            for _ in self.number_of_channels()..channels {
                self.channels.push(silence.clone());
            }

            // downmix by truncating
            self.channels.truncate(channels);

        } else if interpretation == ChannelInterpretation::Speakers {
            match (self.number_of_channels(), channels) {
                // ------------------------------------------
                // UP MIX
                // https://www.w3.org/TR/webaudio/#UpMix-sub
                // ------------------------------------------
                // 1 -> 2 : up-mix from mono to stereo
                //   output.L = input;
                //   output.R = input;
                (1, 2) => {
                    self.channels.push(self.channels[0].clone());
                }
                  // 1 -> 4 : up-mix from mono to quad
                  //   output.L = input;
                  //   output.R = input;
                  //   output.SL = 0;
                  //   output.SR = 0;
                (1, 4) => {
                    self.channels.push(self.channels[0].clone());
                    self.channels.push(silence.clone());
                    self.channels.push(silence);
                }
                // 1 -> 5.1 : up-mix from mono to 5.1
                //   output.L = 0;
                //   output.R = 0;
                //   output.C = input; // put in center channel
                //   output.LFE = 0;
                //   output.SL = 0;
                //   output.SR = 0;
                (1, 6) => {
                    let main = std::mem::replace(&mut self.channels[0], silence.clone());
                    self.channels.push(silence.clone());
                    self.channels.push(main);
                    self.channels.push(silence.clone());
                    self.channels.push(silence.clone());
                    self.channels.push(silence);
                }
                // 2 -> 4 : up-mix from stereo to quad
                //   output.L = input.L;
                //   output.R = input.R;
                //   output.SL = 0;
                //   output.SR = 0;
                (2, 4) => {
                    self.channels.push(silence.clone());
                    self.channels.push(silence);
                }
                // 2 -> 5.1 : up-mix from stereo to 5.1
                //   output.L = input.L;
                //   output.R = input.R;
                //   output.C = 0;
                //   output.LFE = 0;
                //   output.SL = 0;
                //   output.SR = 0;
                (2, 6) => {
                    self.channels.push(silence.clone());
                    self.channels.push(silence.clone());
                    self.channels.push(silence.clone());
                    self.channels.push(silence);
                }
                // 4 -> 5.1 : up-mix from quad to 5.1
                //   output.L = input.L;
                //   output.R = input.R;
                //   output.C = 0;
                //   output.LFE = 0;
                //   output.SL = input.SL;
                //   output.SR = input.SR;
                (4, 6) => {
                    let sl = std::mem::replace(&mut self.channels[2], silence.clone());
                    let sr = std::mem::replace(&mut self.channels[3], silence);
                    self.channels.push(sl);
                    self.channels.push(sr);
                }
                // ------------------------------------------
                // DOWN MIX
                // https://www.w3.org/TR/webaudio/#down-mix
                // ------------------------------------------
                // 2 -> 1 : stereo to mono
                //   output = 0.5 * (input.L + input.R);
                (2, 1) => {
                    let right = self.channels[1].clone();

                    self.channels[0]
                        .iter_mut()
                        .zip(right.iter())
                        .for_each(|(l, r)| *l = 0.5 * (*l + *r));

                    self.channels.truncate(1);
                }
                // 4 -> 1 : quad to mono
                //   output = 0.25 * (input.L + input.R + input.SL + input.SR);
                (4, 1) => {
                    let right = self.channels[1].clone();
                    let s_left = self.channels[2].clone();
                    let s_right = self.channels[3].clone();

                    self.channels[0]
                        .iter_mut()
                        .zip(right.iter())
                        .zip(s_left.iter())
                        .zip(s_right.iter())
                        .for_each(|(((l, r), sl), sr)| *l = 0.25 * (*l + *r + *sl + *sr));

                    self.channels.truncate(1);
                }
                // 5.1 -> 1 : 5.1 to mono
                //   output = sqrt(0.5) * (input.L + input.R) + input.C + 0.5 * (input.SL + input.SR)
                (6, 1) => {
                    let right = self.channels[1].clone();
                    let center = self.channels[2].clone();
                    let s_left = self.channels[4].clone();
                    let s_right = self.channels[5].clone();
                    let sqrt05 = (0.5_f32).sqrt();

                    self.channels[0]
                        .iter_mut()
                        .zip(right.iter())
                        .zip(center.iter())
                        .zip(s_left.iter())
                        .zip(s_right.iter())
                        .for_each(|((((l, r), c), sl), sr)| *l = sqrt05 * (*l + *r) + *c + 0.5 * (*sl + *sr));

                    self.channels.truncate(1);
                }
                // 4 -> 2 : quad to stereo
                //   output.L = 0.5 * (input.L + input.SL);
                //   output.R = 0.5 * (input.R + input.SR);
                (4, 2) => {
                    let s_left = self.channels[2].clone();
                    let s_right = self.channels[3].clone();

                    self.channels[0]
                        .iter_mut()
                        .zip(s_left.iter())
                        .for_each(|(l, sl)| *l = 0.5 * (*l + *sl));

                    self.channels[1]
                        .iter_mut()
                        .zip(s_right.iter())
                        .for_each(|(r, sr)| *r = 0.5 * (*r + *sr));

                    self.channels.truncate(2);
                }
                // 5.1 -> 2 : 5.1 to stereo
                //   output.L = L + sqrt(0.5) * (input.C + input.SL)
                //   output.R = R + sqrt(0.5) * (input.C + input.SR)
                (6, 2) => {
                    let center = self.channels[2].clone();
                    let s_left = self.channels[4].clone();
                    let s_right = self.channels[5].clone();
                    let sqrt05 = (0.5_f32).sqrt();

                    self.channels[0]
                        .iter_mut()
                        .zip(center.iter())
                        .zip(s_left.iter())
                        .for_each(|((l, c), sl)| *l = *l + sqrt05 * (*c + *sl));

                    self.channels[1]
                        .iter_mut()
                        .zip(center.iter())
                        .zip(s_right.iter())
                        .for_each(|((r, c), sr)| *r = *r + sqrt05 * (*c + *sr));

                    self.channels.truncate(2)
                }
                // 5.1 -> 4 : 5.1 to quad
                //   output.L = L + sqrt(0.5) * input.C
                //   output.R = R + sqrt(0.5) * input.C
                //   output.SL = input.SL
                //   output.SR = input.SR
                (6, 4) => {
                    let _low_f = self.channels.swap_remove(3); // swap lr to index 3
                    let center = self.channels.swap_remove(2); // swap lf to index 2
                    let sqrt05 = (0.5_f32).sqrt();

                    self.channels[0]
                        .iter_mut()
                        .zip(center.iter())
                        .for_each(|(l, c)| *l = *l + sqrt05 * c);

                    self.channels[1]
                        .iter_mut()
                        .zip(center.iter())
                        .for_each(|(r, c)| *r = *r + sqrt05 * c);
                }

                _ => panic!("{mixing} from {from} to {to} channels not supported",
                    mixing = if self.number_of_channels() < channels { "Up-mixing" } else { "Down-mixing" },
                    from = self.number_of_channels(),
                    to = channels,
                ),
            }
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
        // todo, optimize for Rcs that are equal
        self.channels.iter_mut().for_each(fun)
    }

    /// Sum two AudioBuffers
    ///
    /// If the channel counts differ, the buffer with lower count will be upmixed.
    pub fn add(&mut self, other: &Self, interpretation: ChannelInterpretation) {
        // mix buffers to the max channel count
        let channels_self = self.number_of_channels();
        let channels_other = other.number_of_channels();
        let channels = channels_self.max(channels_other);

        self.mix(channels, interpretation);

        let mut other_mixed = other.clone();
        other_mixed.mix(channels, interpretation);

        self.channels
            .iter_mut()
            .zip(other_mixed.channels.iter())
            .take(channels)
            .for_each(|(s, o)| s.add(o));
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

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

                assert_float_eq!(&a[..], &[0.; LEN][..], ulps_all <= 0);
                assert_eq!(alloc.pool_size(), 1);

                // mutating this buffer will not allocate
                let mut a = a;
                a.iter_mut().for_each(|v| *v += 1.);
                assert_float_eq!(&a[..], &[1.; LEN][..], ulps_all <= 0);
                assert_eq!(alloc.pool_size(), 1);

                // clone this buffer, should not allocate
                #[allow(clippy::redundant_clone)]
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
                assert_float_eq!(&a[..], &[1.; LEN][..], ulps_all <= 0);
                assert_float_eq!(&b[..], &[2.; LEN][..], ulps_all <= 0);
                assert_float_eq!(&c[..], &[0.; LEN][..], ulps_all <= 0);

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
                assert_float_eq!(&a_vals[..], &[0.; LEN][..], ulps_all <= 0);

                // is_silent is a superficial ptr check
                assert!(!a.is_silent());
            }
        });
    }

    #[test]
    fn test_silence() {
        let alloc = Alloc::with_capacity(1);
        let silence = alloc.silence();

        assert_float_eq!(&silence[..], &[0.; LEN][..], ulps_all <= 0);
        assert!(silence.is_silent());

        // changing silence is possible
        let mut changed = silence;
        changed.iter_mut().for_each(|v| *v = 1.);
        assert_float_eq!(&changed[..], &[1.; LEN][..], ulps_all <= 0);
        assert!(!changed.is_silent());

        // but should not alter new silence
        let silence = alloc.silence();
        assert_float_eq!(&silence[..], &[0.; LEN][..], ulps_all <= 0);
        assert!(silence.is_silent());

        // can also create silence from ChannelData
        let from_channel = silence.silence();
        assert_float_eq!(&from_channel[..], &[0.; LEN][..], ulps_all <= 0);
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
        assert_float_eq!(&signal1[..], &[1.; LEN][..], ulps_all <= 0);

        // test add signal to silence
        let mut silence = alloc.silence();
        silence.add(&signal1);
        assert_float_eq!(&silence[..], &[1.; LEN][..], ulps_all <= 0);

        // test add two signals
        signal1.add(&signal2);
        assert_float_eq!(&signal1[..], &[3.; LEN][..], ulps_all <= 0);
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
        assert_float_eq!(&buffer.channel_data(0)[..], &[1.; LEN][..], ulps_all <= 0);

        buffer.mix(2, ChannelInterpretation::Discrete);
        assert_eq!(buffer.number_of_channels(), 2);

        // first channel unchanged, second channel silent
        assert_float_eq!(&buffer.channel_data(0)[..], &[1.; LEN][..], ulps_all <= 0);
        assert_float_eq!(&buffer.channel_data(1)[..], &[0.; LEN][..], ulps_all <= 0);

        buffer.mix(1, ChannelInterpretation::Discrete);
        assert_eq!(buffer.number_of_channels(), 1);
        assert_float_eq!(&buffer.channel_data(0)[..], &[1.; LEN][..], ulps_all <= 0);
    }

    #[test]
    fn test_audiobuffer_upmix_speakers() {
        let alloc = Alloc::with_capacity(1);

        { // 1 -> 2
            let mut signal = alloc.silence();
            signal.copy_from_slice(&[1.; LEN]);

            let mut buffer = AudioBuffer::new(signal.clone());

            // make sure 1 -> 1 does nothing
            buffer.mix(1, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 1);
            assert_float_eq!(&buffer.channel_data(0)[..], &[1.; LEN][..], ulps_all <= 0);

            buffer.mix(2, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 2);

            // left and right equal
            assert_float_eq!(&buffer.channel_data(0)[..], &[1.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(1)[..], &[1.; LEN][..], ulps_all <= 0);
        }

        { // 1 -> 4
            let mut signal = alloc.silence();
            signal.copy_from_slice(&[1.; LEN]);

            let mut buffer = AudioBuffer::new(signal.clone());

            buffer.mix(4, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 4);

            // left and right equal
            assert_float_eq!(&buffer.channel_data(0)[..], &[1.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(1)[..], &[1.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(2)[..], &[0.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(3)[..], &[0.; LEN][..], ulps_all <= 0);
        }

        { // 1 -> 6
            let mut signal = alloc.silence();
            signal.copy_from_slice(&[1.; LEN]);

            let mut buffer = AudioBuffer::new(signal.clone());

            buffer.mix(6, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 6);

            // left and right equal
            assert_float_eq!(&buffer.channel_data(0)[..], &[0.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(1)[..], &[0.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(2)[..], &[1.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(3)[..], &[0.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(4)[..], &[0.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(5)[..], &[0.; LEN][..], ulps_all <= 0);
        }

        { // 2 -> 4
            let mut left_signal = alloc.silence();
            left_signal.copy_from_slice(&[1.; LEN]);
            let mut right_signal = alloc.silence();
            right_signal.copy_from_slice(&[0.5; LEN]);

            let mut buffer = AudioBuffer::new(left_signal.clone());
            buffer.channels.push(right_signal.clone());

            assert_eq!(buffer.number_of_channels(), 2);
            assert_float_eq!(&buffer.channel_data(0)[..], &[1.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(1)[..], &[0.5; LEN][..], ulps_all <= 0);

            buffer.mix(4, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 4);
            assert_float_eq!(&buffer.channel_data(0)[..], &[1.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(1)[..], &[0.5; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(2)[..], &[0.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(3)[..], &[0.; LEN][..], ulps_all <= 0);
        }

        { // 2 -> 5.1
            let mut left_signal = alloc.silence();
            left_signal.copy_from_slice(&[1.; LEN]);
            let mut right_signal = alloc.silence();
            right_signal.copy_from_slice(&[0.5; LEN]);

            let mut buffer = AudioBuffer::new(left_signal.clone());
            buffer.channels.push(right_signal.clone());

            assert_eq!(buffer.number_of_channels(), 2);
            assert_float_eq!(&buffer.channel_data(0)[..], &[1.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(1)[..], &[0.5; LEN][..], ulps_all <= 0);

            buffer.mix(6, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 6);
            assert_float_eq!(&buffer.channel_data(0)[..], &[1.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(1)[..], &[0.5; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(2)[..], &[0.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(3)[..], &[0.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(4)[..], &[0.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(5)[..], &[0.; LEN][..], ulps_all <= 0);
        }

        { // 4 -> 5.1
            let mut left_signal = alloc.silence();
            left_signal.copy_from_slice(&[0.25; LEN]);
            let mut right_signal = alloc.silence();
            right_signal.copy_from_slice(&[0.5; LEN]);
            let mut s_left_signal = alloc.silence();
            s_left_signal.copy_from_slice(&[0.75; LEN]);
            let mut s_right_signal = alloc.silence();
            s_right_signal.copy_from_slice(&[1.; LEN]);

            let mut buffer = AudioBuffer::new(left_signal.clone());
            buffer.channels.push(right_signal.clone());
            buffer.channels.push(s_left_signal.clone());
            buffer.channels.push(s_right_signal.clone());

            assert_eq!(buffer.number_of_channels(), 4);
            assert_float_eq!(&buffer.channel_data(0)[..], &[0.25; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(1)[..], &[0.5; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(2)[..], &[0.75; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(3)[..], &[1.; LEN][..], ulps_all <= 0);

            buffer.mix(6, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 6);
            assert_float_eq!(&buffer.channel_data(0)[..], &[0.25; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(1)[..], &[0.5; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(2)[..], &[0.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(3)[..], &[0.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(4)[..], &[0.75; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(5)[..], &[1.; LEN][..], ulps_all <= 0);
        }
    }

    #[test]
    fn test_audiobuffer_downmix_speakers() {
        let alloc = Alloc::with_capacity(1);

        { // 2 -> 1
            let mut left_signal = alloc.silence();
            left_signal.copy_from_slice(&[1.; LEN]);
            let mut right_signal = alloc.silence();
            right_signal.copy_from_slice(&[0.5; LEN]);

            let mut buffer = AudioBuffer::new(left_signal.clone());
            buffer.channels.push(right_signal.clone());

            assert_eq!(buffer.number_of_channels(), 2);
            assert_float_eq!(&buffer.channel_data(0)[..], &[1.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(1)[..], &[0.5; LEN][..], ulps_all <= 0);

            buffer.mix(1, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 1);
            assert_float_eq!(&buffer.channel_data(0)[..], &[0.75; LEN][..], ulps_all <= 0);
        }

        { // 4 -> 1
            let mut left_signal = alloc.silence();
            left_signal.copy_from_slice(&[1.; LEN]);
            let mut right_signal = alloc.silence();
            right_signal.copy_from_slice(&[0.75; LEN]);
            let mut s_left_signal = alloc.silence();
            s_left_signal.copy_from_slice(&[0.5; LEN]);
            let mut s_right_signal = alloc.silence();
            s_right_signal.copy_from_slice(&[0.25; LEN]);

            let mut buffer = AudioBuffer::new(left_signal.clone());
            buffer.channels.push(right_signal.clone());
            buffer.channels.push(s_left_signal.clone());
            buffer.channels.push(s_right_signal.clone());

            assert_eq!(buffer.number_of_channels(), 4);
            assert_float_eq!(&buffer.channel_data(0)[..], &[1.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(1)[..], &[0.75; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(2)[..], &[0.5; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(3)[..], &[0.25; LEN][..], ulps_all <= 0);

            buffer.mix(1, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 1);
            assert_float_eq!(&buffer.channel_data(0)[..], &[0.625; LEN][..], ulps_all <= 0);
        }

        { // 6 -> 1
            let mut left_signal = alloc.silence();
            left_signal.copy_from_slice(&[1.; LEN]);
            let mut right_signal = alloc.silence();
            right_signal.copy_from_slice(&[0.9; LEN]);
            let mut center_signal = alloc.silence();
            center_signal.copy_from_slice(&[0.8; LEN]);
            let mut low_freq_signal = alloc.silence();
            low_freq_signal.copy_from_slice(&[0.7; LEN]);
            let mut s_left_signal = alloc.silence();
            s_left_signal.copy_from_slice(&[0.6; LEN]);
            let mut s_right_signal = alloc.silence();
            s_right_signal.copy_from_slice(&[0.5; LEN]);

            let mut buffer = AudioBuffer::new(left_signal.clone());
            buffer.channels.push(right_signal.clone());
            buffer.channels.push(center_signal.clone());
            buffer.channels.push(low_freq_signal.clone());
            buffer.channels.push(s_left_signal.clone());
            buffer.channels.push(s_right_signal.clone());

            assert_eq!(buffer.number_of_channels(), 6);
            assert_float_eq!(&buffer.channel_data(0)[..], &[1.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(1)[..], &[0.9; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(2)[..], &[0.8; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(3)[..], &[0.7; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(4)[..], &[0.6; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(5)[..], &[0.5; LEN][..], ulps_all <= 0);

            buffer.mix(1, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 1);
            // output = sqrt(0.5) * (input.L + input.R) + input.C + 0.5 * (input.SL + input.SR)
            let res = (0.5_f32).sqrt() * (1. + 0.9) + 0.8 + 0.5 * (0.6 + 0.5);
            assert_float_eq!(&buffer.channel_data(0)[..], &[res; LEN][..], ulps_all <= 0);
        }

        { // 4 -> 2
            let mut left_signal = alloc.silence();
            left_signal.copy_from_slice(&[0.25; LEN]);
            let mut right_signal = alloc.silence();
            right_signal.copy_from_slice(&[0.5; LEN]);
            let mut s_left_signal = alloc.silence();
            s_left_signal.copy_from_slice(&[0.75; LEN]);
            let mut s_right_signal = alloc.silence();
            s_right_signal.copy_from_slice(&[1.; LEN]);

            let mut buffer = AudioBuffer::new(left_signal.clone());
            buffer.channels.push(right_signal.clone());
            buffer.channels.push(s_left_signal.clone());
            buffer.channels.push(s_right_signal.clone());

            assert_eq!(buffer.number_of_channels(), 4);
            assert_float_eq!(&buffer.channel_data(0)[..], &[0.25; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(1)[..], &[0.5; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(2)[..], &[0.75; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(3)[..], &[1.; LEN][..], ulps_all <= 0);

            buffer.mix(2, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 2);
            assert_float_eq!(&buffer.channel_data(0)[..], &[0.5; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(1)[..], &[0.75; LEN][..], ulps_all <= 0);
        }

        { // 6 -> 2
            let mut left_signal = alloc.silence();
            left_signal.copy_from_slice(&[1.; LEN]);
            let mut right_signal = alloc.silence();
            right_signal.copy_from_slice(&[0.9; LEN]);
            let mut center_signal = alloc.silence();
            center_signal.copy_from_slice(&[0.8; LEN]);
            let mut low_freq_signal = alloc.silence();
            low_freq_signal.copy_from_slice(&[0.7; LEN]);
            let mut s_left_signal = alloc.silence();
            s_left_signal.copy_from_slice(&[0.6; LEN]);
            let mut s_right_signal = alloc.silence();
            s_right_signal.copy_from_slice(&[0.5; LEN]);

            let mut buffer = AudioBuffer::new(left_signal.clone());
            buffer.channels.push(right_signal.clone());
            buffer.channels.push(center_signal.clone());
            buffer.channels.push(low_freq_signal.clone());
            buffer.channels.push(s_left_signal.clone());
            buffer.channels.push(s_right_signal.clone());

            assert_eq!(buffer.number_of_channels(), 6);
            assert_float_eq!(&buffer.channel_data(0)[..], &[1.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(1)[..], &[0.9; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(2)[..], &[0.8; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(3)[..], &[0.7; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(4)[..], &[0.6; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(5)[..], &[0.5; LEN][..], ulps_all <= 0);

            buffer.mix(2, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 2);

            let res_left = 1. + (0.5_f32).sqrt() * (0.8 + 0.6);
            let res_right = 0.9 + (0.5_f32).sqrt() * (0.8 + 0.5);
            assert_float_eq!(&buffer.channel_data(0)[..], &[res_left; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(1)[..], &[res_right; LEN][..], ulps_all <= 0);
        }

        { // 6 -> 4
            let mut left_signal = alloc.silence();
            left_signal.copy_from_slice(&[1.; LEN]);
            let mut right_signal = alloc.silence();
            right_signal.copy_from_slice(&[0.9; LEN]);
            let mut center_signal = alloc.silence();
            center_signal.copy_from_slice(&[0.8; LEN]);
            let mut low_freq_signal = alloc.silence();
            low_freq_signal.copy_from_slice(&[0.7; LEN]);
            let mut s_left_signal = alloc.silence();
            s_left_signal.copy_from_slice(&[0.6; LEN]);
            let mut s_right_signal = alloc.silence();
            s_right_signal.copy_from_slice(&[0.5; LEN]);

            // create stereo AudioBuffer
            // @note - not sure that's the proper way this
            let mut buffer = AudioBuffer::new(left_signal.clone());
            buffer.channels.push(right_signal.clone());
            buffer.channels.push(center_signal.clone());
            buffer.channels.push(low_freq_signal.clone());
            buffer.channels.push(s_left_signal.clone());
            buffer.channels.push(s_right_signal.clone());

            assert_eq!(buffer.number_of_channels(), 6);
            assert_float_eq!(&buffer.channel_data(0)[..], &[1.; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(1)[..], &[0.9; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(2)[..], &[0.8; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(3)[..], &[0.7; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(4)[..], &[0.6; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(5)[..], &[0.5; LEN][..], ulps_all <= 0);

            buffer.mix(4, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 4);

            let res_left = 1. + (0.5_f32).sqrt() * 0.8;
            let res_right = 0.9 + (0.5_f32).sqrt() * 0.8;
            assert_float_eq!(&buffer.channel_data(0)[..], &[res_left; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(1)[..], &[res_right; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(2)[..], &[0.6; LEN][..], ulps_all <= 0);
            assert_float_eq!(&buffer.channel_data(3)[..], &[0.5; LEN][..], ulps_all <= 0);
        }
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

        buffer.add(&buffer2, ChannelInterpretation::Discrete);

        assert_eq!(buffer.number_of_channels(), 2);
        assert_float_eq!(&buffer.channel_data(0)[..], &[3.; LEN][..], ulps_all <= 0);
        assert_float_eq!(&buffer.channel_data(1)[..], &[1.; LEN][..], ulps_all <= 0);
    }
}

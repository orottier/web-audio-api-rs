//! Optimized audio signal data structures, used in `AudioProcessors`
use arrayvec::ArrayVec;
use std::rc::Rc;

use crate::node::{ChannelConfigInner, ChannelCountMode, ChannelInterpretation};

use crate::assert_valid_number_of_channels;
use crate::{MAX_CHANNELS, RENDER_QUANTUM_SIZE};

/// A sample slice containing only zeroes
const ZEROES: [f32; RENDER_QUANTUM_SIZE] = [0.; RENDER_QUANTUM_SIZE];

mod pool {
    //! Thread local allocator for sample slices

    use super::*;
    use std::cell::RefCell;

    thread_local! {
        /// Thread local pool of pre-allocated sample slice that can be reused after being dropped
        static POOL: RefCell<Vec<Rc<[f32; RENDER_QUANTUM_SIZE]>>> = {
            log::debug!("Setting up a new thread local pool of audio quanta");

            const SIZE: usize = 128; // TODO - leak more memory or risk allocations at runtime?
            let pool: Vec<_> = (0..SIZE).map(|_| Rc::new(ZEROES)).collect();
            RefCell::new(pool)
        };
    }

    #[cfg(test)]
    /// Number of pre-allocated items available in the pool
    pub(super) fn size() -> usize {
        POOL.with_borrow(|p| p.len())
    }

    /// Obtain a new sample slice, either reused from the pool, or newly allocated
    pub(super) fn allocate() -> Rc<[f32; RENDER_QUANTUM_SIZE]> {
        if let Some(rc) = POOL.with_borrow_mut(|p| p.pop()) {
            // reuse from pool
            rc
        } else {
            // allocate
            Rc::new(ZEROES)
        }
    }

    /// Reclaim a sample slice into the thread local pool
    pub(super) fn push(data: Rc<[f32; RENDER_QUANTUM_SIZE]>) {
        POOL.with_borrow_mut(|p| p.push(data));
    }
}

/// Render thread channel buffer
///
/// Basically wraps a `Rc<[f32; render_quantum_size]>`, which means it derefs to a (mutable) slice
/// of `[f32]` sample values. Plus it has copy-on-write semantics, so it is cheap to clone.
///
/// The `render_quantum_size` is equal to 128 by default, but in future versions it may be equal to
/// the hardware preferred render quantum size or any other value.
///
/// # Usage
///
/// Audio buffers are managed with a dedicated allocator per render thread, hence there are no
/// public constructors available. If you must create a new instance, copy an existing one and
/// mutate it from there.
#[derive(Clone, Debug)]
pub struct AudioRenderQuantumChannel {
    data: Option<Rc<[f32; RENDER_QUANTUM_SIZE]>>,
}

impl AudioRenderQuantumChannel {
    fn make_mut(&mut self) -> &mut [f32; RENDER_QUANTUM_SIZE] {
        if self.data.is_none() {
            self.data = Some(pool::allocate());
            Rc::make_mut(self.data.as_mut().unwrap()).copy_from_slice(&ZEROES);
        }

        match &mut self.data {
            Some(data) => {
                if Rc::strong_count(data) != 1 {
                    let mut new = pool::allocate();
                    Rc::make_mut(&mut new).copy_from_slice(&data[..]);
                    *data = new;
                }

                return Rc::make_mut(data);
            }
            None => unreachable!(),
        }
    }

    /// `O(1)` check if this buffer is equal to the 'silence buffer'
    ///
    /// If this function returns false, it is still possible for all samples to be zero.
    pub(crate) fn is_silent(&self) -> bool {
        self.data.is_none()
    }

    /// Sum two channels
    pub(crate) fn add(&mut self, other: &Self) {
        if self.is_silent() {
            *self = other.clone();
        } else if !other.is_silent() {
            self.iter_mut().zip(other.iter()).for_each(|(a, b)| *a += b)
        }
    }

    pub(crate) fn silence() -> Self {
        Self { data: None }
    }
}

use std::ops::{Deref, DerefMut};

impl Deref for AudioRenderQuantumChannel {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        match &self.data {
            Some(data) => data.as_slice(),
            None => &ZEROES,
        }
    }
}

impl DerefMut for AudioRenderQuantumChannel {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.make_mut()
    }
}

impl AsRef<[f32]> for AudioRenderQuantumChannel {
    fn as_ref(&self) -> &[f32] {
        match &self.data {
            Some(data) => data.as_slice(),
            None => &ZEROES,
        }
    }
}

impl std::ops::Drop for AudioRenderQuantumChannel {
    fn drop(&mut self) {
        let data = match self.data.take() {
            None => return,
            Some(data) => data,
        };

        if Rc::strong_count(&data) == 1 {
            pool::push(data);
        }
    }
}

/// Render thread audio buffer, consisting of multiple channel buffers
///
/// This is a  fixed length audio asset of `render_quantum_size` sample frames for block rendering,
/// basically a list of [`AudioRenderQuantumChannel`]s cf.
/// <https://webaudio.github.io/web-audio-api/#render-quantum>
///
/// The `render_quantum_size` is equal to 128 by default, but in future versions it may be equal to
/// the hardware preferred render quantum size or any other value.
///
/// An `AudioRenderQuantum` has copy-on-write semantics, so it is cheap to clone.
///
/// # Usage
///
/// Audio buffers are managed with a dedicated allocator per render thread, hence there are no
/// public constructors available. If you must create a new instance, copy an existing one and
/// mutate it from there.
#[derive(Clone, Debug)]
pub struct AudioRenderQuantum {
    channels: ArrayVec<AudioRenderQuantumChannel, MAX_CHANNELS>,
    // this field is only used by AudioParam so that when we know the param is
    // constant for a render_quantum it return a slice of length 1 instead of 128
    single_valued: bool,
}

impl AudioRenderQuantum {
    /// Create a new `AudioRenderQuantum` with a single channel of silence
    pub(crate) fn new() -> Self {
        let mut channels = ArrayVec::new();
        channels.push(AudioRenderQuantumChannel::silence());

        Self {
            channels,
            single_valued: false,
        }
    }

    /// Create a new `AudioRenderQuantum` from a single channel buffer
    pub(crate) fn from(channel: AudioRenderQuantumChannel) -> Self {
        let mut channels = ArrayVec::new();
        channels.push(channel);

        Self {
            channels,
            single_valued: false,
        }
    }

    pub(crate) fn single_valued(&self) -> bool {
        self.single_valued
    }

    pub(crate) fn set_single_valued(&mut self, value: bool) {
        self.single_valued = value;
    }

    /// Number of channels in this AudioRenderQuantum
    pub fn number_of_channels(&self) -> usize {
        self.channels.len()
    }

    /// Set number of channels in this AudioRenderQuantum
    ///
    /// Note: if the new number is higher than the previous, the new channels will be filled with
    /// garbage.
    ///
    /// # Panics
    ///
    /// This function will panic if the given number of channels is outside the [1, 32] range, 32
    /// being defined by the MAX_CHANNELS constant.
    pub fn set_number_of_channels(&mut self, n: usize) {
        assert_valid_number_of_channels(n);
        for _ in self.number_of_channels()..n {
            self.channels.push(self.channels[0].clone());
        }
        self.channels.truncate(n);
    }

    /// Get the samples from this specific channel.
    ///
    /// # Panics
    /// Panics if the index is greater than the available number of channels
    pub fn channel_data(&self, index: usize) -> &AudioRenderQuantumChannel {
        &self.channels[index]
    }

    /// Get the samples (mutable) from this specific channel.
    ///
    /// # Panics
    /// Panics if the index is greater than the available number of channels
    pub fn channel_data_mut(&mut self, index: usize) -> &mut AudioRenderQuantumChannel {
        &mut self.channels[index]
    }

    /// Channel data as slice
    pub fn channels(&self) -> &[AudioRenderQuantumChannel] {
        &self.channels[..]
    }

    /// Channel data as slice (mutable)
    pub fn channels_mut(&mut self) -> &mut [AudioRenderQuantumChannel] {
        &mut self.channels[..]
    }

    /// `O(1)` check if this buffer is equal to the 'silence buffer'
    ///
    /// If this function returns false, it is still possible for all samples to be zero.
    pub fn is_silent(&self) -> bool {
        !self.channels.iter().any(|channel| !channel.is_silent())
    }

    pub(crate) fn stereo_mut(&mut self) -> [&mut AudioRenderQuantumChannel; 2] {
        assert_eq!(self.number_of_channels(), 2);
        let (ls, rs) = self.channels_mut().split_at_mut(1);
        [&mut ls[0], &mut rs[0]]
    }

    /// Up/Down-mix to the desired number of channels
    ///
    /// # Panics
    ///
    /// This function will panic if the given number of channels is outside the [1, 32] range, 32
    /// being defined by the MAX_CHANNELS constant.
    #[inline(always)]
    pub(crate) fn mix(
        &mut self,
        computed_number_of_channels: usize,
        interpretation: ChannelInterpretation,
    ) {
        if self.number_of_channels() == computed_number_of_channels {
            return;
        }
        self.mix_inner(computed_number_of_channels, interpretation)
    }

    fn mix_inner(
        &mut self,
        computed_number_of_channels: usize,
        interpretation: ChannelInterpretation,
    ) {
        // cf. https://www.w3.org/TR/webaudio/#channel-up-mixing-and-down-mixing
        assert_valid_number_of_channels(computed_number_of_channels);
        let silence = AudioRenderQuantumChannel::silence();

        // Handle discrete interpretation or speaker layouts where the initial or desired number of
        // channels is larger than 6 (undefined by the specification)
        if interpretation == ChannelInterpretation::Discrete
            || self.number_of_channels() > 6
            || computed_number_of_channels > 6
        {
            // upmix by filling with silence
            for _ in self.number_of_channels()..computed_number_of_channels {
                self.channels.push(silence.clone());
            }

            // downmix by truncating
            self.channels.truncate(computed_number_of_channels);
        } else {
            match (self.number_of_channels(), computed_number_of_channels) {
                // ------------------------------------------
                // UP MIX
                // https://www.w3.org/TR/webaudio/#UpMix-sub
                // ------------------------------------------
                (1, 2) => {
                    // output.L = input;
                    // output.R = input;
                    self.channels.push(self.channels[0].clone());
                }
                (1, 3) => {
                    self.channels.push(silence.clone());
                    self.channels.push(silence);
                }
                (1, 4) => {
                    // output.L = input;
                    // output.R = input;
                    // output.SL = 0;
                    // output.SR = 0;
                    self.channels.push(self.channels[0].clone());
                    self.channels.push(silence.clone());
                    self.channels.push(silence);
                }
                (1, 5) => {
                    // output.C = input;
                    // output.L = 0;
                    // output.R = 0;
                    // output.SL = 0;
                    // output.SR = 0;
                    self.channels.push(silence.clone());
                    self.channels.push(silence.clone());
                    self.channels.push(silence.clone());
                    self.channels.push(silence);
                }
                (1, 6) => {
                    // output.L = 0;
                    // output.R = 0;
                    // output.C = input; // put in center channel
                    // output.LFE = 0;
                    // output.SL = 0;
                    // output.SR = 0;
                    let main = std::mem::replace(&mut self.channels[0], silence.clone());
                    self.channels.push(silence.clone());
                    self.channels.push(main);
                    self.channels.push(silence.clone());
                    self.channels.push(silence.clone());
                    self.channels.push(silence);
                }
                (2, 3) => {
                    let left = std::mem::replace(&mut self.channels[0], silence);
                    let right = std::mem::replace(&mut self.channels[1], left);
                    self.channels.push(right);
                }
                (2, 4) => {
                    // output.L = input.L;
                    // output.R = input.R;
                    // output.SL = 0;
                    // output.SR = 0;
                    self.channels.push(silence.clone());
                    self.channels.push(silence);
                }
                (2, 5) => {
                    // output.L = input.L;
                    // output.R = input.R;
                    // output.C = 0;
                    // output.SL = 0;
                    // output.SR = 0;
                    self.channels.push(silence.clone());
                    self.channels.push(silence.clone());
                    self.channels.push(silence);
                }
                (2, 6) => {
                    // output.L = input.L;
                    // output.R = input.R;
                    // output.C = 0;
                    // output.LFE = 0;
                    // output.SL = 0;
                    // output.SR = 0;
                    self.channels.push(silence.clone());
                    self.channels.push(silence.clone());
                    self.channels.push(silence.clone());
                    self.channels.push(silence);
                }
                (3, 4) => {
                    self.channels.push(silence);
                }
                (3, 5) => {
                    self.channels.push(silence.clone());
                    self.channels.push(silence);
                }
                (3, 6) => {
                    self.channels.push(silence.clone());
                    self.channels.push(silence.clone());
                    self.channels.push(silence);
                }
                (4, 5) => {
                    // output.L = input.L;
                    // output.R = input.R;
                    // output.C = 0;
                    // output.SL = input.SL;
                    // output.SR = input.SR;
                    let sl = std::mem::replace(&mut self.channels[2], silence.clone());
                    let sr = std::mem::replace(&mut self.channels[3], sl);
                    self.channels.push(sr);
                }
                (4, 6) => {
                    // output.L = input.L;
                    // output.R = input.R;
                    // output.C = 0;
                    // output.LFE = 0;
                    // output.SL = input.SL;
                    // output.SR = input.SR;
                    let sl = std::mem::replace(&mut self.channels[2], silence.clone());
                    let sr = std::mem::replace(&mut self.channels[3], silence);
                    self.channels.push(sl);
                    self.channels.push(sr);
                }
                (5, 6) => {
                    // output.L = input.L;
                    // output.R = input.R;
                    // output.C = 0;
                    // output.LFE = 0;
                    // output.SL = input.SL;
                    // output.SR = input.SR;
                    self.channels.push(silence);
                }
                // ------------------------------------------
                // DOWN MIX
                // https://www.w3.org/TR/webaudio/#down-mix
                // ------------------------------------------
                (2, 1) => {
                    // M = 0.5 * (input.L + input.R);
                    let right = self.channels[1].clone();

                    self.channels[0]
                        .iter_mut()
                        .zip(right.iter())
                        .for_each(|(l, r)| *l = 0.5 * (*l + *r));

                    self.channels.truncate(1);
                }
                (3, 1) => {
                    // M = C;
                    self.channels.truncate(1);
                }
                (4, 1) => {
                    // M = 0.25 * (input.L + input.R + input.SL + input.SR);
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
                (5, 1) => {
                    // M = C;
                    let c = std::mem::replace(&mut self.channels[2], silence);
                    self.channels[0] = c;
                    self.channels.truncate(1);
                }
                (6, 1) => {
                    // output = sqrt(0.5) * (input.L + input.R) + input.C + 0.5 * (input.SL + input.SR)
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
                        .for_each(|((((l, r), c), sl), sr)| {
                            *l = sqrt05.mul_add(*l + *r, 0.5f32.mul_add(*sl + *sr, *c))
                        });

                    self.channels.truncate(1);
                }
                (3, 2) => {
                    self.channels.truncate(2);
                }
                (4, 2) => {
                    // output.L = 0.5 * (input.L + input.SL);
                    // output.R = 0.5 * (input.R + input.SR);
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
                (5, 2) => {
                    self.channels.truncate(2);
                }
                (6, 2) => {
                    // output.L = L + sqrt(0.5) * (input.C + input.SL)
                    // output.R = R + sqrt(0.5) * (input.C + input.SR)
                    let center = self.channels[2].clone();
                    let s_left = self.channels[4].clone();
                    let s_right = self.channels[5].clone();
                    let sqrt05 = (0.5_f32).sqrt();

                    self.channels[0]
                        .iter_mut()
                        .zip(center.iter())
                        .zip(s_left.iter())
                        .for_each(|((l, c), sl)| *l += sqrt05 * (*c + *sl));

                    self.channels[1]
                        .iter_mut()
                        .zip(center.iter())
                        .zip(s_right.iter())
                        .for_each(|((r, c), sr)| *r += sqrt05 * (*c + *sr));

                    self.channels.truncate(2)
                }
                (4, 3) => {
                    self.channels.truncate(3);
                }
                (5, 3) => {
                    self.channels.truncate(3);
                }
                (6, 3) => {
                    self.channels.truncate(3);
                }
                (5, 4) => {
                    self.channels.truncate(4);
                }
                (6, 4) => {
                    // output.L = L + sqrt(0.5) * input.C
                    // output.R = R + sqrt(0.5) * input.C
                    // output.SL = input.SL
                    // output.SR = input.SR
                    let _low_f = self.channels.swap_remove(3); // swap lr to index 3
                    let center = self.channels.swap_remove(2); // swap lf to index 2
                    let sqrt05 = (0.5_f32).sqrt();

                    self.channels[0]
                        .iter_mut()
                        .zip(center.iter())
                        .for_each(|(l, c)| *l += sqrt05 * c);

                    self.channels[1]
                        .iter_mut()
                        .zip(center.iter())
                        .for_each(|(r, c)| *r += sqrt05 * c);
                }
                (6, 5) => {
                    self.channels.truncate(5);
                }
                _ => unreachable!(),
            }
        }
        debug_assert_eq!(self.number_of_channels(), computed_number_of_channels);
    }

    /// Convert this buffer to silence
    ///
    /// `O(1)` operation to convert this buffer to the 'silence buffer' which will enable some
    /// optimizations in the graph rendering.
    pub fn make_silent(&mut self) {
        self.channels[0] = AudioRenderQuantumChannel::silence();
        self.channels.truncate(1);
    }

    /// Convert to a single channel buffer, dropping excess channels
    pub(crate) fn force_mono(&mut self) {
        self.channels.truncate(1);
    }

    /// Modify every channel in the same way
    pub(crate) fn modify_channels<F: Fn(&mut AudioRenderQuantumChannel)>(&mut self, fun: F) {
        // todo, optimize for Rcs that are equal
        self.channels.iter_mut().for_each(fun)
    }

    /// Sum two `AudioRenderQuantum`s
    ///
    /// Both buffers will be mixed up front according to the supplied `channel_config`
    pub(crate) fn add(&mut self, other: &Self, channel_config: &ChannelConfigInner) {
        // gather initial channel counts
        let channels_self = self.number_of_channels();
        let channels_other = other.number_of_channels();
        let max_channels = channels_self.max(channels_other);

        // up/down-mix the to the desired channel count for the receiving node
        let interpretation = channel_config.interpretation;
        let mode = channel_config.count_mode;
        let count = channel_config.count;

        let new_channels = match mode {
            ChannelCountMode::Max => max_channels,
            ChannelCountMode::Explicit => count,
            ChannelCountMode::ClampedMax => max_channels.min(count),
        };

        // fast path if both buffers are upmixed mono signals
        if interpretation == ChannelInterpretation::Speakers
            && self.all_channels_identical()
            && other.all_channels_identical()
        {
            self.channels.truncate(1);
            self.channels[0].add(&other.channels[0]);
            self.mix(new_channels, interpretation);
            return;
        }

        self.mix(new_channels, interpretation);

        let mut other_mixed = other.clone();
        other_mixed.mix(new_channels, interpretation);

        self.channels
            .iter_mut()
            .zip(other_mixed.channels.iter())
            .for_each(|(s, o)| s.add(o));
    }

    /// Determine if all channels are identical (by pointer)
    ///
    /// This is often the case for upmixed buffers. When all channels are identical, modifications
    /// only need to be applied once.
    fn all_channels_identical(&self) -> bool {
        let mut channels = self.channels.iter();
        let first = channels.next().unwrap();
        for c in channels {
            match (&first.data, &c.data) {
                (None, None) => (),
                (None, _) => return false,
                (_, None) => return false,
                (Some(d1), Some(d2)) => {
                    if !Rc::ptr_eq(d1, d2) {
                        return false;
                    }
                }
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use super::*;

    #[test]
    fn test_pool() {
        {
            // allocate and drop
            let mut a = AudioRenderQuantumChannel::silence();
            let _a = a.make_mut();
            let mut b = AudioRenderQuantumChannel::silence();
            let _b = b.make_mut();
        }

        let pool_size_start = pool::size();
        assert!(pool_size_start >= 2); // due to allocation, pool size is nonzero

        {
            // take a buffer out of the pool
            let a = AudioRenderQuantumChannel::silence();

            assert_float_eq!(&a[..], &ZEROES[..], abs_all <= 0.);
            assert_eq!(pool::size(), pool_size_start);

            // mutating this buffer will take from the pool
            let mut a = a;
            a.iter_mut().for_each(|v| *v += 1.);
            assert_float_eq!(&a[..], &[1.; RENDER_QUANTUM_SIZE][..], abs_all <= 0.);
            assert_eq!(pool::size(), pool_size_start - 1);

            // clone this buffer, should not allocate
            #[allow(clippy::redundant_clone)]
            let mut b: AudioRenderQuantumChannel = a.clone();
            assert_eq!(pool::size(), pool_size_start - 1);

            // mutate cloned buffer, this will allocate
            b.iter_mut().for_each(|v| *v += 1.);
            assert_eq!(pool::size(), pool_size_start - 2);
        }

        // all buffers are reclaimed
        assert_eq!(pool::size(), pool_size_start);
    }

    #[test]
    fn test_silence() {
        let silence = AudioRenderQuantumChannel::silence();

        assert_float_eq!(&silence[..], &ZEROES[..], abs_all <= 0.);
        assert!(silence.is_silent());

        // changing silence is possible
        let mut changed = silence;
        changed.iter_mut().for_each(|v| *v = 1.);
        assert_float_eq!(&changed[..], &[1.; RENDER_QUANTUM_SIZE][..], abs_all <= 0.);
        assert!(!changed.is_silent());

        // but should not alter new silence
        let silence = AudioRenderQuantumChannel::silence();
        assert_float_eq!(&silence[..], &ZEROES[..], abs_all <= 0.);
        assert!(silence.is_silent());
    }

    #[test]
    fn test_channel_add() {
        let silence = AudioRenderQuantumChannel::silence();

        let mut signal1 = AudioRenderQuantumChannel::silence();
        signal1.copy_from_slice(&[1.; RENDER_QUANTUM_SIZE]);

        let mut signal2 = AudioRenderQuantumChannel::silence();
        signal2.copy_from_slice(&[2.; RENDER_QUANTUM_SIZE]);

        // test add silence to signal
        signal1.add(&silence);
        assert_float_eq!(&signal1[..], &[1.; RENDER_QUANTUM_SIZE][..], abs_all <= 0.);

        // test add signal to silence
        let mut silence = AudioRenderQuantumChannel::silence();
        silence.add(&signal1);
        assert_float_eq!(&silence[..], &[1.; RENDER_QUANTUM_SIZE][..], abs_all <= 0.);

        // test add two signals
        signal1.add(&signal2);
        assert_float_eq!(&signal1[..], &[3.; RENDER_QUANTUM_SIZE][..], abs_all <= 0.);
    }

    #[test]
    fn test_audiobuffer_channels() {
        let silence = AudioRenderQuantumChannel::silence();

        let buffer = AudioRenderQuantum::from(silence);
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
        let mut signal = AudioRenderQuantumChannel::silence();
        signal.copy_from_slice(&[1.; RENDER_QUANTUM_SIZE]);

        let mut buffer = AudioRenderQuantum::from(signal);

        buffer.mix(1, ChannelInterpretation::Discrete);

        assert_eq!(buffer.number_of_channels(), 1);
        assert_float_eq!(
            &buffer.channel_data(0)[..],
            &[1.; RENDER_QUANTUM_SIZE][..],
            abs_all <= 0.
        );

        buffer.mix(2, ChannelInterpretation::Discrete);
        assert_eq!(buffer.number_of_channels(), 2);

        // first channel unchanged, second channel silent
        assert_float_eq!(
            &buffer.channel_data(0)[..],
            &[1.; RENDER_QUANTUM_SIZE][..],
            abs_all <= 0.
        );
        assert_float_eq!(&buffer.channel_data(1)[..], &ZEROES[..], abs_all <= 0.);

        buffer.mix(1, ChannelInterpretation::Discrete);
        assert_eq!(buffer.number_of_channels(), 1);
        assert_float_eq!(
            &buffer.channel_data(0)[..],
            &[1.; RENDER_QUANTUM_SIZE][..],
            abs_all <= 0.
        );
    }

    #[test]
    fn test_audiobuffer_mix_speakers_all() {
        let signal = AudioRenderQuantumChannel::silence();
        let mut buffer = AudioRenderQuantum::from(signal);

        for i in 1..MAX_CHANNELS {
            buffer.set_number_of_channels(i);
            assert_eq!(buffer.number_of_channels(), i);
            for j in 1..MAX_CHANNELS {
                buffer.mix(j, ChannelInterpretation::Speakers);
                assert_eq!(buffer.number_of_channels(), j);
            }
        }
    }

    #[test]
    fn test_audiobuffer_upmix_speakers() {
        {
            // 1 -> 2
            let mut signal = AudioRenderQuantumChannel::silence();
            signal.copy_from_slice(&[1.; RENDER_QUANTUM_SIZE]);

            let mut buffer = AudioRenderQuantum::from(signal);

            // make sure 1 -> 1 does nothing
            buffer.mix(1, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 1);
            assert_float_eq!(
                &buffer.channel_data(0)[..],
                &[1.; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );

            buffer.mix(2, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 2);

            // left and right equal
            assert_float_eq!(
                &buffer.channel_data(0)[..],
                &[1.; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(1)[..],
                &[1.; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
        }

        {
            // 1 -> 4
            let mut signal = AudioRenderQuantumChannel::silence();
            signal.copy_from_slice(&[1.; RENDER_QUANTUM_SIZE]);

            let mut buffer = AudioRenderQuantum::from(signal);

            buffer.mix(4, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 4);

            // left and right equal
            assert_float_eq!(
                &buffer.channel_data(0)[..],
                &[1.; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(1)[..],
                &[1.; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(&buffer.channel_data(2)[..], &ZEROES[..], abs_all <= 0.);
            assert_float_eq!(&buffer.channel_data(3)[..], &ZEROES[..], abs_all <= 0.);
        }

        {
            // 1 -> 6
            let mut signal = AudioRenderQuantumChannel::silence();
            signal.copy_from_slice(&[1.; RENDER_QUANTUM_SIZE]);

            let mut buffer = AudioRenderQuantum::from(signal);

            buffer.mix(6, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 6);

            // left and right equal
            assert_float_eq!(&buffer.channel_data(0)[..], &ZEROES[..], abs_all <= 0.);
            assert_float_eq!(&buffer.channel_data(1)[..], &ZEROES[..], abs_all <= 0.);
            assert_float_eq!(
                &buffer.channel_data(2)[..],
                &[1.; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(&buffer.channel_data(3)[..], &ZEROES[..], abs_all <= 0.);
            assert_float_eq!(&buffer.channel_data(4)[..], &ZEROES[..], abs_all <= 0.);
            assert_float_eq!(&buffer.channel_data(5)[..], &ZEROES[..], abs_all <= 0.);
        }

        {
            // 2 -> 4
            let mut left_signal = AudioRenderQuantumChannel::silence();
            left_signal.copy_from_slice(&[1.; RENDER_QUANTUM_SIZE]);
            let mut right_signal = AudioRenderQuantumChannel::silence();
            right_signal.copy_from_slice(&[0.5; RENDER_QUANTUM_SIZE]);

            let mut buffer = AudioRenderQuantum::from(left_signal);
            buffer.channels.push(right_signal);

            assert_eq!(buffer.number_of_channels(), 2);
            assert_float_eq!(
                &buffer.channel_data(0)[..],
                &[1.; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(1)[..],
                &[0.5; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );

            buffer.mix(4, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 4);
            assert_float_eq!(
                &buffer.channel_data(0)[..],
                &[1.; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(1)[..],
                &[0.5; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(&buffer.channel_data(2)[..], &ZEROES[..], abs_all <= 0.);
            assert_float_eq!(&buffer.channel_data(3)[..], &ZEROES[..], abs_all <= 0.);
        }

        {
            // 2 -> 5.1
            let mut left_signal = AudioRenderQuantumChannel::silence();
            left_signal.copy_from_slice(&[1.; RENDER_QUANTUM_SIZE]);
            let mut right_signal = AudioRenderQuantumChannel::silence();
            right_signal.copy_from_slice(&[0.5; RENDER_QUANTUM_SIZE]);

            let mut buffer = AudioRenderQuantum::from(left_signal);
            buffer.channels.push(right_signal);

            assert_eq!(buffer.number_of_channels(), 2);
            assert_float_eq!(
                &buffer.channel_data(0)[..],
                &[1.; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(1)[..],
                &[0.5; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );

            buffer.mix(6, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 6);
            assert_float_eq!(
                &buffer.channel_data(0)[..],
                &[1.; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(1)[..],
                &[0.5; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(&buffer.channel_data(2)[..], &ZEROES[..], abs_all <= 0.);
            assert_float_eq!(&buffer.channel_data(3)[..], &ZEROES[..], abs_all <= 0.);
            assert_float_eq!(&buffer.channel_data(4)[..], &ZEROES[..], abs_all <= 0.);
            assert_float_eq!(&buffer.channel_data(5)[..], &ZEROES[..], abs_all <= 0.);
        }

        {
            // 4 -> 5.1
            let mut left_signal = AudioRenderQuantumChannel::silence();
            left_signal.copy_from_slice(&[0.25; RENDER_QUANTUM_SIZE]);
            let mut right_signal = AudioRenderQuantumChannel::silence();
            right_signal.copy_from_slice(&[0.5; RENDER_QUANTUM_SIZE]);
            let mut s_left_signal = AudioRenderQuantumChannel::silence();
            s_left_signal.copy_from_slice(&[0.75; RENDER_QUANTUM_SIZE]);
            let mut s_right_signal = AudioRenderQuantumChannel::silence();
            s_right_signal.copy_from_slice(&[1.; RENDER_QUANTUM_SIZE]);

            let mut buffer = AudioRenderQuantum::from(left_signal);
            buffer.channels.push(right_signal);
            buffer.channels.push(s_left_signal);
            buffer.channels.push(s_right_signal);

            assert_eq!(buffer.number_of_channels(), 4);
            assert_float_eq!(
                &buffer.channel_data(0)[..],
                &[0.25; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(1)[..],
                &[0.5; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(2)[..],
                &[0.75; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(3)[..],
                &[1.; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );

            buffer.mix(6, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 6);
            assert_float_eq!(
                &buffer.channel_data(0)[..],
                &[0.25; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(1)[..],
                &[0.5; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(&buffer.channel_data(2)[..], &ZEROES[..], abs_all <= 0.);
            assert_float_eq!(&buffer.channel_data(3)[..], &ZEROES[..], abs_all <= 0.);
            assert_float_eq!(
                &buffer.channel_data(4)[..],
                &[0.75; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(5)[..],
                &[1.; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
        }
    }

    #[test]
    fn test_audiobuffer_downmix_speakers() {
        {
            // 2 -> 1
            let mut left_signal = AudioRenderQuantumChannel::silence();
            left_signal.copy_from_slice(&[1.; RENDER_QUANTUM_SIZE]);
            let mut right_signal = AudioRenderQuantumChannel::silence();
            right_signal.copy_from_slice(&[0.5; RENDER_QUANTUM_SIZE]);

            let mut buffer = AudioRenderQuantum::from(left_signal);
            buffer.channels.push(right_signal);

            assert_eq!(buffer.number_of_channels(), 2);
            assert_float_eq!(
                &buffer.channel_data(0)[..],
                &[1.; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(1)[..],
                &[0.5; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );

            buffer.mix(1, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 1);
            assert_float_eq!(
                &buffer.channel_data(0)[..],
                &[0.75; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
        }

        {
            // 4 -> 1
            let mut left_signal = AudioRenderQuantumChannel::silence();
            left_signal.copy_from_slice(&[1.; RENDER_QUANTUM_SIZE]);
            let mut right_signal = AudioRenderQuantumChannel::silence();
            right_signal.copy_from_slice(&[0.75; RENDER_QUANTUM_SIZE]);
            let mut s_left_signal = AudioRenderQuantumChannel::silence();
            s_left_signal.copy_from_slice(&[0.5; RENDER_QUANTUM_SIZE]);
            let mut s_right_signal = AudioRenderQuantumChannel::silence();
            s_right_signal.copy_from_slice(&[0.25; RENDER_QUANTUM_SIZE]);

            let mut buffer = AudioRenderQuantum::from(left_signal);
            buffer.channels.push(right_signal);
            buffer.channels.push(s_left_signal);
            buffer.channels.push(s_right_signal);

            assert_eq!(buffer.number_of_channels(), 4);
            assert_float_eq!(
                &buffer.channel_data(0)[..],
                &[1.; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(1)[..],
                &[0.75; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(2)[..],
                &[0.5; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(3)[..],
                &[0.25; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );

            buffer.mix(1, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 1);
            assert_float_eq!(
                &buffer.channel_data(0)[..],
                &[0.625; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
        }

        {
            // 6 -> 1
            let mut left_signal = AudioRenderQuantumChannel::silence();
            left_signal.copy_from_slice(&[1.; RENDER_QUANTUM_SIZE]);
            let mut right_signal = AudioRenderQuantumChannel::silence();
            right_signal.copy_from_slice(&[0.9; RENDER_QUANTUM_SIZE]);
            let mut center_signal = AudioRenderQuantumChannel::silence();
            center_signal.copy_from_slice(&[0.8; RENDER_QUANTUM_SIZE]);
            let mut low_freq_signal = AudioRenderQuantumChannel::silence();
            low_freq_signal.copy_from_slice(&[0.7; RENDER_QUANTUM_SIZE]);
            let mut s_left_signal = AudioRenderQuantumChannel::silence();
            s_left_signal.copy_from_slice(&[0.6; RENDER_QUANTUM_SIZE]);
            let mut s_right_signal = AudioRenderQuantumChannel::silence();
            s_right_signal.copy_from_slice(&[0.5; RENDER_QUANTUM_SIZE]);

            let mut buffer = AudioRenderQuantum::from(left_signal);
            buffer.channels.push(right_signal);
            buffer.channels.push(center_signal);
            buffer.channels.push(low_freq_signal);
            buffer.channels.push(s_left_signal);
            buffer.channels.push(s_right_signal);

            assert_eq!(buffer.number_of_channels(), 6);
            assert_float_eq!(
                &buffer.channel_data(0)[..],
                &[1.; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(1)[..],
                &[0.9; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(2)[..],
                &[0.8; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(3)[..],
                &[0.7; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(4)[..],
                &[0.6; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(5)[..],
                &[0.5; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );

            buffer.mix(1, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 1);
            // output = sqrt(0.5) * (input.L + input.R) + input.C + 0.5 * (input.SL + input.SR)
            let res = (0.5_f32).sqrt() * (1. + 0.9) + 0.8 + 0.5 * (0.6 + 0.5);
            assert_float_eq!(
                &buffer.channel_data(0)[..],
                &[res; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
        }

        {
            // 4 -> 2
            let mut left_signal = AudioRenderQuantumChannel::silence();
            left_signal.copy_from_slice(&[0.25; RENDER_QUANTUM_SIZE]);
            let mut right_signal = AudioRenderQuantumChannel::silence();
            right_signal.copy_from_slice(&[0.5; RENDER_QUANTUM_SIZE]);
            let mut s_left_signal = AudioRenderQuantumChannel::silence();
            s_left_signal.copy_from_slice(&[0.75; RENDER_QUANTUM_SIZE]);
            let mut s_right_signal = AudioRenderQuantumChannel::silence();
            s_right_signal.copy_from_slice(&[1.; RENDER_QUANTUM_SIZE]);

            let mut buffer = AudioRenderQuantum::from(left_signal);
            buffer.channels.push(right_signal);
            buffer.channels.push(s_left_signal);
            buffer.channels.push(s_right_signal);

            assert_eq!(buffer.number_of_channels(), 4);
            assert_float_eq!(
                &buffer.channel_data(0)[..],
                &[0.25; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(1)[..],
                &[0.5; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(2)[..],
                &[0.75; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(3)[..],
                &[1.; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );

            buffer.mix(2, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 2);
            assert_float_eq!(
                &buffer.channel_data(0)[..],
                &[0.5; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(1)[..],
                &[0.75; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
        }

        {
            // 6 -> 2
            let mut left_signal = AudioRenderQuantumChannel::silence();
            left_signal.copy_from_slice(&[1.; RENDER_QUANTUM_SIZE]);
            let mut right_signal = AudioRenderQuantumChannel::silence();
            right_signal.copy_from_slice(&[0.9; RENDER_QUANTUM_SIZE]);
            let mut center_signal = AudioRenderQuantumChannel::silence();
            center_signal.copy_from_slice(&[0.8; RENDER_QUANTUM_SIZE]);
            let mut low_freq_signal = AudioRenderQuantumChannel::silence();
            low_freq_signal.copy_from_slice(&[0.7; RENDER_QUANTUM_SIZE]);
            let mut s_left_signal = AudioRenderQuantumChannel::silence();
            s_left_signal.copy_from_slice(&[0.6; RENDER_QUANTUM_SIZE]);
            let mut s_right_signal = AudioRenderQuantumChannel::silence();
            s_right_signal.copy_from_slice(&[0.5; RENDER_QUANTUM_SIZE]);

            let mut buffer = AudioRenderQuantum::from(left_signal);
            buffer.channels.push(right_signal);
            buffer.channels.push(center_signal);
            buffer.channels.push(low_freq_signal);
            buffer.channels.push(s_left_signal);
            buffer.channels.push(s_right_signal);

            assert_eq!(buffer.number_of_channels(), 6);
            assert_float_eq!(
                &buffer.channel_data(0)[..],
                &[1.; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(1)[..],
                &[0.9; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(2)[..],
                &[0.8; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(3)[..],
                &[0.7; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(4)[..],
                &[0.6; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(5)[..],
                &[0.5; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );

            buffer.mix(2, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 2);

            let res_left = 1. + (0.5_f32).sqrt() * (0.8 + 0.6);
            let res_right = 0.9 + (0.5_f32).sqrt() * (0.8 + 0.5);
            assert_float_eq!(
                &buffer.channel_data(0)[..],
                &[res_left; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(1)[..],
                &[res_right; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
        }

        {
            // 6 -> 4
            let mut left_signal = AudioRenderQuantumChannel::silence();
            left_signal.copy_from_slice(&[1.; RENDER_QUANTUM_SIZE]);
            let mut right_signal = AudioRenderQuantumChannel::silence();
            right_signal.copy_from_slice(&[0.9; RENDER_QUANTUM_SIZE]);
            let mut center_signal = AudioRenderQuantumChannel::silence();
            center_signal.copy_from_slice(&[0.8; RENDER_QUANTUM_SIZE]);
            let mut low_freq_signal = AudioRenderQuantumChannel::silence();
            low_freq_signal.copy_from_slice(&[0.7; RENDER_QUANTUM_SIZE]);
            let mut s_left_signal = AudioRenderQuantumChannel::silence();
            s_left_signal.copy_from_slice(&[0.6; RENDER_QUANTUM_SIZE]);
            let mut s_right_signal = AudioRenderQuantumChannel::silence();
            s_right_signal.copy_from_slice(&[0.5; RENDER_QUANTUM_SIZE]);

            let mut buffer = AudioRenderQuantum::from(left_signal);
            buffer.channels.push(right_signal);
            buffer.channels.push(center_signal);
            buffer.channels.push(low_freq_signal);
            buffer.channels.push(s_left_signal);
            buffer.channels.push(s_right_signal);

            assert_eq!(buffer.number_of_channels(), 6);
            assert_float_eq!(
                &buffer.channel_data(0)[..],
                &[1.; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(1)[..],
                &[0.9; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(2)[..],
                &[0.8; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(3)[..],
                &[0.7; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(4)[..],
                &[0.6; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(5)[..],
                &[0.5; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );

            buffer.mix(4, ChannelInterpretation::Speakers);
            assert_eq!(buffer.number_of_channels(), 4);

            let res_left = 1. + (0.5_f32).sqrt() * 0.8;
            let res_right = 0.9 + (0.5_f32).sqrt() * 0.8;
            assert_float_eq!(
                &buffer.channel_data(0)[..],
                &[res_left; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(1)[..],
                &[res_right; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(2)[..],
                &[0.6; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
            assert_float_eq!(
                &buffer.channel_data(3)[..],
                &[0.5; RENDER_QUANTUM_SIZE][..],
                abs_all <= 0.
            );
        }
    }

    #[test]
    fn test_audiobuffer_add() {
        let mut signal = AudioRenderQuantumChannel::silence();
        signal.copy_from_slice(&[1.; RENDER_QUANTUM_SIZE]);
        let mut buffer = AudioRenderQuantum::from(signal);
        buffer.mix(2, ChannelInterpretation::Speakers);

        let mut signal2 = AudioRenderQuantumChannel::silence();
        signal2.copy_from_slice(&[2.; RENDER_QUANTUM_SIZE]);
        let buffer2 = AudioRenderQuantum::from(signal2);

        let channel_config = ChannelConfigInner {
            count: 2,
            count_mode: ChannelCountMode::Explicit,
            interpretation: ChannelInterpretation::Discrete,
        };

        buffer.add(&buffer2, &channel_config);

        assert_eq!(buffer.number_of_channels(), 2);
        assert_float_eq!(
            &buffer.channel_data(0)[..],
            &[3.; RENDER_QUANTUM_SIZE][..],
            abs_all <= 0.
        );
        assert_float_eq!(
            &buffer.channel_data(1)[..],
            &[1.; RENDER_QUANTUM_SIZE][..],
            abs_all <= 0.
        );
    }

    #[test]
    fn test_is_silent_quantum() {
        // create 2 channel silent buffer
        let signal = AudioRenderQuantumChannel::silence();
        let mut buffer = AudioRenderQuantum::from(signal);
        buffer.mix(2, ChannelInterpretation::Speakers);

        assert_float_eq!(&buffer.channel_data(0)[..], &ZEROES[..], abs_all <= 0.);
        assert_float_eq!(&buffer.channel_data(1)[..], &ZEROES[..], abs_all <= 0.);

        assert!(buffer.is_silent());
    }

    #[test]
    fn test_is_not_silent_quantum() {
        // create 2 channel silent buffer
        let mut signal = AudioRenderQuantumChannel::silence();
        signal.copy_from_slice(&[1.; RENDER_QUANTUM_SIZE]);
        let mut buffer = AudioRenderQuantum::from(signal);
        buffer.mix(2, ChannelInterpretation::Discrete);

        assert_float_eq!(
            &buffer.channel_data(0)[..],
            &[1.; RENDER_QUANTUM_SIZE][..],
            abs_all <= 0.
        );
        assert_float_eq!(&buffer.channel_data(1)[..], &ZEROES[..], abs_all <= 0.);
        assert!(!buffer.is_silent());
    }
}

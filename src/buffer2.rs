use std::cell::RefCell;
use std::rc::Rc;

use crate::buffer::ChannelInterpretation;

const LEN: usize = crate::BUFFER_SIZE as usize;
const MAX_CHANNELS: usize = 32;

pub(crate) struct Alloc {
    pool: RefCell<Vec<Rc<[f32; LEN]>>>,
    zeroes: Rc<[f32; LEN]>,
}

impl Alloc {
    pub fn with_capacity(n: usize) -> Self {
        let pool: Vec<_> = (0..n).map(|_| Rc::new([0.; LEN])).collect();

        Self {
            pool: RefCell::new(pool),
            zeroes: Rc::new([0.; LEN]),
        }
    }

    fn push(&self, data: Rc<[f32; LEN]>) {
        self.pool
            .borrow_mut() // infallible when single threaded
            .push(data);
    }

    pub fn silence(&self) -> ChannelData<'_> {
        ChannelData {
            data: self.zeroes.clone(),
            alloc: &self,
        }
    }

    pub fn allocate(&self) -> ChannelData<'_> {
        ChannelData {
            data: self.allocate_inner(),
            alloc: &self,
        }
    }

    fn allocate_inner(&self) -> Rc<[f32; LEN]> {
        if let Some(rc) = self.pool.borrow_mut().pop() {
            // re-use from pool
            rc
        } else {
            // allocate
            Rc::new([0.; 128])
        }
    }

    pub fn pool_size(&self) -> usize {
        self.pool.borrow().len()
    }
}

#[derive(Clone)]
pub struct ChannelData<'a> {
    data: Rc<[f32; LEN]>,
    alloc: &'a Alloc,
}

impl<'a> ChannelData<'a> {
    fn make_mut(&mut self) -> &mut [f32; LEN] {
        if Rc::strong_count(&self.data) != 1 {
            let mut new = self.alloc.allocate_inner();
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
        self.iter_mut().zip(other.iter()).for_each(|(a, b)| *a += b)
    }
}

use std::ops::{Deref, DerefMut};

impl<'a> Deref for ChannelData<'a> {
    type Target = [f32; LEN];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'a> DerefMut for ChannelData<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.make_mut()
    }
}

impl<'a> std::ops::Drop for ChannelData<'a> {
    fn drop(&mut self) {
        if Rc::strong_count(&self.data) == 1 {
            let rc = std::mem::replace(&mut self.data, self.alloc.zeroes.clone());
            self.alloc.push(rc);
        }
    }
}

#[derive(Clone)]
pub struct AudioBuffer<'a> {
    channels: [ChannelData<'a>; MAX_CHANNELS],
    channel_count: u8,
}

impl<'a> AudioBuffer<'a> {
    pub fn new(channel: ChannelData<'a>) -> Self {
        // sorry..
        let channels = [
            channel.clone(), channel.clone(), channel.clone(), channel.clone(),
            channel.clone(), channel.clone(), channel.clone(), channel.clone(),
            channel.clone(), channel.clone(), channel.clone(), channel.clone(),
            channel.clone(), channel.clone(), channel.clone(), channel.clone(),
            channel.clone(), channel.clone(), channel.clone(), channel.clone(),
            channel.clone(), channel.clone(), channel.clone(), channel.clone(),
            channel.clone(), channel.clone(), channel.clone(), channel.clone(),
            channel.clone(), channel.clone(), channel.clone(), channel.clone(),
        ];
        Self {
            channels,
            channel_count: 1,
        }
    }

    /// Number of channels in this AudioBuffer
    pub fn number_of_channels(&self) -> usize {
        self.channel_count as _
    }

    /// Get the samples from this specific channel.
    pub fn channel_data(&self, channel: usize) -> &ChannelData {
        &self.channels[channel]
    }

    /// Get the samples from this specific channel (mutable).
    pub fn channel_data_mut(&mut self, channel: usize) -> &mut ChannelData<'a> {
        &mut self.channels[channel]
    }

    /// Up/Down-mix to the desired number of channels
    pub fn mix(&self, channels: usize, interpretation: ChannelInterpretation) -> Self {
        assert!(channels < MAX_CHANNELS);

        if self.number_of_channels() == channels {
            return self.clone();
        }

        let silence = self.channels[0].alloc.silence();

        // handle discrete interpretation
        if interpretation == ChannelInterpretation::Discrete {
            let mut new = self.clone();

            // downmix by setting channel_count
            new.channel_count = channels as _;

            // upmix by filling with silence
            for i in (self.channel_count as usize)..channels {
                new.channels[i] = silence.clone();
            }

            return new;
        }

        match (self.number_of_channels(), channels) {
            (1, 2) => {
                let mut new = self.clone();
                new.channel_count = 2;
                new.channels[1] = new.channels[0].clone();
                new
            }
            (1, 4) => {
                let mut new = self.clone();
                new.channel_count = 4;
                new.channels[1] = new.channels[0].clone();
                new.channels[2] = silence.clone();
                new.channels[3] = silence.clone();
                new
            }
            (1, 6) => {
                let mut new = self.clone();
                new.channels[2] = new.channels[0].clone();
                new.channels[0] = silence.clone();
                new.channels[1] = silence.clone();
                new.channels[3] = silence.clone();
                new.channels[4] = silence.clone();
                new
            }
            (2, 1) => {
                let mut new = self.clone();
                let right = new.channels[1].clone();
                new.channel_count = 1;
                new.channels[0]
                    .iter_mut()
                    .zip(right.iter())
                    .for_each(|(l, r)| *l = (*l + *r) / 2.);
                new
            }
            _ => todo!(),
        }
    }

    /// Convert this buffer to silence
    pub fn make_silent(&mut self) {
        let silence = self.channels[0].alloc.silence();

        self.channel_count = 1;
        self.channels[0] = silence;
    }

    /// Convert to a single channel buffer, dropping excess channels
    pub fn force_mono(&mut self) {
        self.channel_count = 1;
    }

    /// Modify every channel in the same way
    pub fn modify_channels<F: Fn(&mut ChannelData)>(&mut self, fun: F) {
        self.channels
            .iter_mut()
            .take(self.channel_count as usize)
            .for_each(fun)
    }

    /// Sum two AudioBuffers
    ///
    /// If the channel counts differ, the buffer with lower count will be upmixed.
    pub fn add(&self, other: &Self, interpretation: ChannelInterpretation) -> Self {
        // mix buffers to the max channel count
        let channels_self = self.number_of_channels();
        let channels_other = other.number_of_channels();
        let channels = channels_self.max(channels_other);
        let mut self_mixed = self.mix(channels, interpretation);
        let other_mixed = other.mix(channels, interpretation);

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
}

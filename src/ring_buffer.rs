#[derive(Debug, Clone, Copy)]
pub struct RingBuffer<const N: usize> {
    start: usize,
    end: usize,
    data: [f32; N],
}

impl<const N: usize> RingBuffer<N> {
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    pub fn is_full(&self) -> bool {
        self.end == (self.start - 1)
    }

    pub fn len(&self) -> usize {
        if self.start < self.end {
            self.end - self.start
        } else {
            self.data.len() - (self.start - self.end)
        }
    }

    pub fn push_slice(&mut self, data: &[f32]) {
        for &datum in data {
            self.data[self.end] = datum;
            self.end = (self.end + 1) % self.data.len();
        }
    }

    pub fn push(&mut self, datum: f32) {
        self.data[self.end] = datum;
        self.end = (self.end + 1) % self.data.len();
    }

    pub fn pop(&mut self) -> f32 {
        let res = self.data[self.start];
        self.start += 1;
        self.start %= self.data.len();
        res
    }
}

impl<const N: usize> Default for RingBuffer<N> {
    fn default() -> Self {
        Self {
            start: 0,
            end: 0,
            data: [0.; N],
        }
    }
}

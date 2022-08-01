// taken from cpal/samples_formats.rs

/// Trait for containers that contain PCM data.
pub trait Sample: Copy + Clone {
    /// Turns the sample into its equivalent as a floating-point.
    fn to_f32(&self) -> f32;
    /// Converts this sample into a standard i16 sample.
    fn to_i16(&self) -> i16;
    /// Converts this sample into a standard u16 sample.
    fn to_u16(&self) -> u16;

    /// Converts any sample type to this one by calling `to_i16`, `to_u16` or `to_f32`.
    fn from<S>(s: &S) -> Self
    where
        S: Sample;
}

impl Sample for u16 {
    #[inline]
    fn to_f32(&self) -> f32 {
        self.to_i16().to_f32()
    }

    #[inline]
    fn to_i16(&self) -> i16 {
        (*self as i16).wrapping_add(i16::MIN)
    }

    #[inline]
    fn to_u16(&self) -> u16 {
        *self
    }

    #[inline]
    fn from<S>(sample: &S) -> Self
    where
        S: Sample,
    {
        sample.to_u16()
    }
}

impl Sample for i16 {
    #[inline]
    fn to_f32(&self) -> f32 {
        if *self < 0 {
            *self as f32 / -(i16::MIN as f32)
        } else {
            *self as f32 / i16::MAX as f32
        }
    }

    #[inline]
    fn to_i16(&self) -> i16 {
        *self
    }

    #[inline]
    fn to_u16(&self) -> u16 {
        self.wrapping_add(i16::MIN) as u16
    }

    #[inline]
    fn from<S>(sample: &S) -> Self
    where
        S: Sample,
    {
        sample.to_i16()
    }
}
const F32_TO_16BIT_INT_MULTIPLIER: f32 = u16::MAX as f32 * 0.5;
impl Sample for f32 {
    #[inline]
    fn to_f32(&self) -> f32 {
        *self
    }

    #[inline]
    fn to_i16(&self) -> i16 {
        if *self >= 0.0 {
            (*self * i16::MAX as f32) as i16
        } else {
            (-*self * i16::MIN as f32) as i16
        }
    }

    #[inline]
    fn to_u16(&self) -> u16 {
        self.mul_add(F32_TO_16BIT_INT_MULTIPLIER, F32_TO_16BIT_INT_MULTIPLIER)
            .round() as u16
    }

    #[inline]
    fn from<S>(sample: &S) -> Self
    where
        S: Sample,
    {
        sample.to_f32()
    }
}

#[cfg(test)]
mod test {
    use super::Sample;

    #[test]
    fn i16_to_i16() {
        assert_eq!(0i16.to_i16(), 0);
        assert_eq!((-467i16).to_i16(), -467);
        assert_eq!(32767i16.to_i16(), 32767);
        assert_eq!((-32768i16).to_i16(), -32768);
    }

    #[test]
    fn i16_to_u16() {
        assert_eq!(0i16.to_u16(), 32768);
        assert_eq!((-16384i16).to_u16(), 16384);
        assert_eq!(32767i16.to_u16(), 65535);
        assert_eq!((-32768i16).to_u16(), 0);
    }

    #[test]
    fn i16_to_f32() {
        assert_eq!(0i16.to_f32(), 0.0);
        assert_eq!((-16384i16).to_f32(), -0.5);
        assert_eq!(32767i16.to_f32(), 1.0);
        assert_eq!((-32768i16).to_f32(), -1.0);
    }

    #[test]
    fn u16_to_i16() {
        assert_eq!(32768u16.to_i16(), 0);
        assert_eq!(16384u16.to_i16(), -16384);
        assert_eq!(65535u16.to_i16(), 32767);
        assert_eq!(0u16.to_i16(), -32768);
    }

    #[test]
    fn u16_to_u16() {
        assert_eq!(0u16.to_u16(), 0);
        assert_eq!(467u16.to_u16(), 467);
        assert_eq!(32767u16.to_u16(), 32767);
        assert_eq!(65535u16.to_u16(), 65535);
    }

    #[test]
    fn u16_to_f32() {
        assert_eq!(0u16.to_f32(), -1.0);
        assert_eq!(32768u16.to_f32(), 0.0);
        assert_eq!(65535u16.to_f32(), 1.0);
    }

    #[test]
    fn f32_to_i16() {
        assert_eq!(0.0f32.to_i16(), 0);
        assert_eq!((-0.5f32).to_i16(), i16::MIN / 2);
        assert_eq!(1.0f32.to_i16(), i16::MAX);
        assert_eq!((-1.0f32).to_i16(), i16::MIN);
    }

    #[test]
    fn f32_to_u16() {
        assert_eq!((-1.0f32).to_u16(), 0);
        assert_eq!(0.0f32.to_u16(), 32768);
        assert_eq!(1.0f32.to_u16(), 65535);
    }

    #[test]
    fn f32_to_f32() {
        assert_eq!(0.1f32.to_f32(), 0.1);
        assert_eq!((-0.7f32).to_f32(), -0.7);
        assert_eq!(1.0f32.to_f32(), 1.0);
    }
}

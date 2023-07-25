//! Type-safe wrappers and their values for the types used in the library

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

/// Signed offset in seconds value
pub type OffsetSamplesValue = f64;

/// Signed offset in seconds
///
/// Relative position from some origin in the time domain.
#[derive(Debug, Clone, Copy, Default, PartialEq, PartialOrd, derive_more::Display)]
#[repr(transparent)]
pub struct OffsetSamples(OffsetSamplesValue);

impl OffsetSamples {
    pub const ZERO: Self = Self(0.0);
    pub const MAX: Self = Self(OffsetSamplesValue::MAX);

    /// Create a new offset from samples
    #[must_use]
    pub const fn new(value: OffsetSamplesValue) -> Self {
        Self(value)
    }

    /// Get the value of the offset in samples
    #[must_use]
    pub const fn value(self) -> OffsetSamplesValue {
        self.0
    }

    /// Determine the minimum of two offsets
    #[must_use]
    pub fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    /// Determine the maximum of two offsets
    #[must_use]
    pub fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    #[must_use]
    pub fn floor(self) -> Self {
        Self(self.0.floor())
    }

    #[must_use]
    pub fn ceil(self) -> Self {
        Self(self.0.ceil())
    }

    #[must_use]
    pub fn round(self) -> Self {
        Self(self.0.round())
    }
}

impl Sub for OffsetSamples {
    type Output = DurationSecs;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::Output::new(self.value() - rhs.value())
    }
}

impl Add<DurationSecs> for OffsetSamples {
    type Output = Self;

    fn add(self, rhs: DurationSecs) -> Self::Output {
        Self::Output::new(self.value() + rhs.value())
    }
}

impl AddAssign<DurationSecs> for OffsetSamples {
    fn add_assign(&mut self, rhs: DurationSecs) {
        *self = *self + rhs;
    }
}

impl Sub<DurationSecs> for OffsetSamples {
    type Output = Self;

    fn sub(self, rhs: DurationSecs) -> Self::Output {
        Self::Output::new(self.value() - rhs.value())
    }
}

impl SubAssign<DurationSecs> for OffsetSamples {
    fn sub_assign(&mut self, rhs: DurationSecs) {
        *self = *self - rhs;
    }
}

impl<T> Mul<T> for OffsetSamples
where
    T: Into<OffsetSamplesValue>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self::Output::new(self.value() * rhs.into())
    }
}

impl<T> MulAssign<T> for OffsetSamples
where
    T: Into<OffsetSamplesValue>,
{
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

/// Signed offset in seconds value
pub type OffsetSecsValue = f64;

/// Signed offset in seconds
///
/// Relative position from some origin in the time domain.
#[derive(Debug, Clone, Copy, Default, PartialEq, PartialOrd, derive_more::Display)]
#[repr(transparent)]
pub struct OffsetSecs(OffsetSecsValue);

impl OffsetSecs {
    pub const ZERO: Self = Self(0.0);
    pub const MAX: Self = Self(OffsetSecsValue::MAX);

    /// Create a new offset from seconds
    #[must_use]
    pub const fn new(value: OffsetSecsValue) -> Self {
        Self(value)
    }

    /// Get the value of the offset in seconds
    #[must_use]
    pub const fn value(self) -> OffsetSecsValue {
        self.0
    }

    /// Determine the minimum of two offsets
    #[must_use]
    pub fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    /// Determine the maximum of two offsets
    #[must_use]
    pub fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }
}

impl Sub for OffsetSecs {
    type Output = DurationSecs;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::Output::new(self.value() - rhs.value())
    }
}

impl Add<DurationSecs> for OffsetSecs {
    type Output = Self;

    fn add(self, rhs: DurationSecs) -> Self::Output {
        Self::Output::new(self.value() + rhs.value())
    }
}

impl AddAssign<DurationSecs> for OffsetSecs {
    fn add_assign(&mut self, rhs: DurationSecs) {
        *self = *self + rhs;
    }
}

impl Sub<DurationSecs> for OffsetSecs {
    type Output = Self;

    fn sub(self, rhs: DurationSecs) -> Self::Output {
        Self::Output::new(self.value() - rhs.value())
    }
}

impl SubAssign<DurationSecs> for OffsetSecs {
    fn sub_assign(&mut self, rhs: DurationSecs) {
        *self = *self - rhs;
    }
}

impl<T> Mul<T> for OffsetSecs
where
    T: Into<OffsetSecsValue>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self::Output::new(self.value() * rhs.into())
    }
}

impl<T> MulAssign<T> for OffsetSecs
where
    T: Into<OffsetSecsValue>,
{
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

impl Rem<DurationSecs> for OffsetSecs {
    type Output = Self;

    fn rem(self, rhs: DurationSecs) -> Self::Output {
        debug_assert!(rhs > DurationSecs::ZERO);
        Self(self.value() % rhs.value())
    }
}

impl RemAssign<DurationSecs> for OffsetSecs {
    fn rem_assign(&mut self, rhs: DurationSecs) {
        *self = *self % rhs;
    }
}

/// Signed duration in seconds value
pub type DurationSecsValue = f64;

/// Signed duration in seconds
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    PartialOrd,
    derive_more::Add,
    derive_more::AddAssign,
    derive_more::Sub,
    derive_more::SubAssign,
    derive_more::Display,
)]
#[repr(transparent)]
pub struct DurationSecs(DurationSecsValue);

impl DurationSecs {
    pub const ZERO: Self = Self(0.0);
    pub const MAX: Self = Self(DurationSecsValue::MAX);

    /// Create a new duration from seconds
    #[must_use]
    pub const fn new(value: DurationSecsValue) -> Self {
        Self(value)
    }

    /// Get the value of the duration in seconds
    #[must_use]
    pub const fn value(self) -> DurationSecsValue {
        self.0
    }

    /// Determine the minimum of two durations
    #[must_use]
    pub fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    /// Determine the maximum of two durations
    #[must_use]
    pub fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }
}

impl<T> Mul<T> for DurationSecs
where
    T: Into<DurationSecsValue>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self::Output::new(self.value() * rhs.into())
    }
}

impl<T> MulAssign<T> for DurationSecs
where
    T: Into<DurationSecsValue>,
{
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

impl<T> Div<T> for DurationSecs
where
    T: Into<DurationSecsValue>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self::Output::new(self.value() / rhs.into())
    }
}

impl<T> DivAssign<T> for DurationSecs
where
    T: Into<DurationSecsValue>,
{
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}

/// Sample rate in Hz value
///
/// Stored as a single-precision floating point value. Calculations involving
/// a sample rate are done using double-precision floating point values.
pub type SampleRateHzValue = f32;

/// Sample rate in Hz
#[derive(Debug, Clone, Copy, Default, PartialEq, PartialOrd, derive_more::Display)]
#[repr(transparent)]
pub struct SampleRateHz(f32);

impl SampleRateHz {
    pub const ZERO: Self = Self(0.0);

    /// Create a new sample rate from Hz
    #[must_use]
    pub const fn new(value: SampleRateHzValue) -> Self {
        Self(value)
    }

    /// Get the value of the sample rate in Hz
    #[must_use]
    pub const fn value(self) -> SampleRateHzValue {
        self.0
    }

    /// Determine the minimum of two sample rates
    #[must_use]
    pub fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    /// Determine the maximum of two sample rates
    #[must_use]
    pub fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    /// Convert frequency to duration
    ///
    /// Convert the frequency into the corresponding cycle duration.
    #[must_use]
    pub fn period(self) -> DurationSecs {
        debug_assert!(self > Self::ZERO);
        DurationSecs::new(1. / DurationSecsValue::from(self.value()))
    }
}

impl Mul<SampleRateHz> for OffsetSecs {
    type Output = OffsetSamples;

    fn mul(self, rhs: SampleRateHz) -> Self::Output {
        debug_assert!(rhs > SampleRateHz::ZERO);
        Self::Output::new(self.value() * OffsetSamplesValue::from(rhs.value()))
    }
}

impl Div<SampleRateHz> for OffsetSamples {
    type Output = OffsetSecs;

    fn div(self, rhs: SampleRateHz) -> Self::Output {
        debug_assert!(rhs > SampleRateHz::ZERO);
        Self::Output::new(self.value() / OffsetSecsValue::from(rhs.value()))
    }
}

impl Div for SampleRateHz {
    type Output = f64;

    fn div(self, rhs: Self) -> Self::Output {
        debug_assert!(rhs > Self::ZERO);
        self.value() as f64 / rhs.value() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_offset_in_seconds_is_zero() {
        assert_eq!(OffsetSecs::ZERO, Default::default());
    }

    #[test]
    fn default_duration_in_seconds_is_zero() {
        assert_eq!(DurationSecs::ZERO, Default::default());
    }
}

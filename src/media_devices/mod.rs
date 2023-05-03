//! Primitives of the MediaDevices API
//!
//! The MediaDevices interface provides access to connected media input devices like microphones.
//!
//! <https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices>

/// List the available media output devices, such as speakers, headsets, loopbacks, etc
///
/// The media device_id can be used to specify the [`sink_id` of the `AudioContext`](crate::context::AudioContextOptions::sink_id)
///
/// ```no_run
/// use web_audio_api::media_devices::{enumerate_devices, MediaDeviceInfoKind};
///
/// let devices = enumerate_devices();
/// assert_eq!(devices[0].device_id(), "1");
/// assert_eq!(devices[0].group_id(), None);
/// assert_eq!(devices[0].kind(), MediaDeviceInfoKind::AudioOutput);
/// assert_eq!(devices[0].label(), "Macbook Pro Builtin Speakers");
/// ```
pub fn enumerate_devices() -> Vec<MediaDeviceInfo> {
    crate::io::enumerate_devices()
}

/// Describes input/output type of a media device
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MediaDeviceInfoKind {
    VideoInput,
    AudioInput,
    AudioOutput,
}

/// Describes a single media input or output device
///
/// Call [`enumerate_devices`] to obtain a list of devices for your hardware.
#[derive(Debug)]
pub struct MediaDeviceInfo {
    device_id: String,
    group_id: Option<String>,
    kind: MediaDeviceInfoKind,
    label: String,
    device: Box<dyn std::any::Any>,
}

impl MediaDeviceInfo {
    pub(crate) fn new(
        device_id: String,
        group_id: Option<String>,
        kind: MediaDeviceInfoKind,
        label: String,
        device: Box<dyn std::any::Any>,
    ) -> Self {
        Self {
            device_id,
            group_id,
            kind,
            label,
            device,
        }
    }

    /// Identifier for the represented device
    ///
    /// The current implementation is not stable across sessions so you should not persist this
    /// value
    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    /// Two devices have the same group identifier if they belong to the same physical device
    pub fn group_id(&self) -> Option<&str> {
        self.group_id.as_deref()
    }

    /// Enumerated value that is either "videoinput", "audioinput" or "audiooutput".
    pub fn kind(&self) -> MediaDeviceInfoKind {
        self.kind
    }

    /// Friendly label describing this device
    pub fn label(&self) -> &str {
        &self.label
    }

    pub(crate) fn device(self) -> Box<dyn std::any::Any> {
        self.device
    }
}

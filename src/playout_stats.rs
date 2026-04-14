use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::context::{BaseAudioContext, ConcreteBaseAudioContext};
use crate::stats::AudioStats;

const UPDATE_INTERVAL: Duration = Duration::from_secs(1);

/// Snapshot of [`AudioPlayoutStats`] values.
#[derive(Clone, Debug, Default)]
pub struct AudioPlayoutStatsSnapshot {
    /// Total duration of fallback frames in milliseconds.
    pub fallback_frames_duration: f64,
    /// Number of fallback events.
    pub fallback_frames_events: u64,
    /// Total duration of all rendered output frames in milliseconds.
    pub total_frames_duration: f64,
    /// Average output latency in milliseconds since the last latency reset.
    pub average_latency: f64,
    /// Minimum output latency in milliseconds since the last latency reset.
    pub minimum_latency: f64,
    /// Maximum output latency in milliseconds since the last latency reset.
    pub maximum_latency: f64,
}

#[derive(Debug, Default)]
struct ExposedStats {
    values: AudioPlayoutStatsSnapshot,
    last_update: Option<Instant>,
}

/// Playout statistics for an [`AudioContext`](crate::context::AudioContext).
///
/// The fallback-frame counters are currently based on render callbacks that miss their realtime
/// deadline. Backends that expose device-level underrun counters can feed more precise values into
/// the shared stats layer in the future.
#[derive(Clone, Debug)]
pub struct AudioPlayoutStats {
    context: ConcreteBaseAudioContext,
    stats: AudioStats,
    exposed: Arc<Mutex<ExposedStats>>,
}

impl AudioPlayoutStats {
    pub(crate) fn new(context: ConcreteBaseAudioContext, stats: AudioStats) -> Self {
        let instance = Self {
            context,
            stats,
            exposed: Arc::new(Mutex::new(ExposedStats::default())),
        };
        instance.stats.reset_latency();
        instance
    }

    /// Total duration of fallback frames in milliseconds.
    #[must_use]
    pub fn fallback_frames_duration(&self) -> f64 {
        self.current().fallback_frames_duration
    }

    /// Number of fallback events.
    #[must_use]
    pub fn fallback_frames_events(&self) -> u64 {
        self.current().fallback_frames_events
    }

    /// Total duration of all rendered output frames in milliseconds.
    #[must_use]
    pub fn total_frames_duration(&self) -> f64 {
        self.current().total_frames_duration
    }

    /// Average output latency in milliseconds since the last latency reset.
    #[must_use]
    pub fn average_latency(&self) -> f64 {
        self.current().average_latency
    }

    /// Minimum output latency in milliseconds since the last latency reset.
    #[must_use]
    pub fn minimum_latency(&self) -> f64 {
        self.current().minimum_latency
    }

    /// Maximum output latency in milliseconds since the last latency reset.
    #[must_use]
    pub fn maximum_latency(&self) -> f64 {
        self.current().maximum_latency
    }

    /// Reset the tracked latency interval.
    pub fn reset_latency(&self) {
        self.stats.reset_latency();
        self.refresh();
    }

    /// Return the currently exposed values as a plain Rust snapshot.
    #[must_use]
    pub fn to_json(&self) -> AudioPlayoutStatsSnapshot {
        self.current()
    }

    fn current(&self) -> AudioPlayoutStatsSnapshot {
        let mut exposed = self.exposed.lock().unwrap();
        let should_update = exposed
            .last_update
            .is_none_or(|last_update| last_update.elapsed() >= UPDATE_INTERVAL);
        if should_update {
            exposed.values = self.read_current_values();
            exposed.last_update = Some(Instant::now());
        }
        exposed.values.clone()
    }

    fn refresh(&self) {
        let mut exposed = self.exposed.lock().unwrap();
        exposed.values = self.read_current_values();
        exposed.last_update = Some(Instant::now());
    }

    fn read_current_values(&self) -> AudioPlayoutStatsSnapshot {
        let snapshot = self.stats.snapshot();
        let sample_rate = self.context.sample_rate();

        AudioPlayoutStatsSnapshot {
            fallback_frames_duration: snapshot.fallback_frames_duration_ms(sample_rate),
            fallback_frames_events: snapshot.fallback_events_total,
            total_frames_duration: snapshot.total_frames_duration_ms(sample_rate),
            average_latency: snapshot.average_latency_ms(),
            minimum_latency: snapshot.minimum_latency_ms(),
            maximum_latency: snapshot.maximum_latency_ms(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::context::{AudioContext, AudioContextOptions};
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn test_same_instance() {
        let options = AudioContextOptions {
            sink_id: "none".into(),
            ..AudioContextOptions::default()
        };
        let context = AudioContext::new(options);

        let stats1 = context.playout_stats();
        let stats2 = context.playout_stats();
        let stats3 = stats2.clone();

        assert!(Arc::ptr_eq(&stats1.exposed, &stats2.exposed));
        assert!(Arc::ptr_eq(&stats1.exposed, &stats3.exposed));
    }

    #[test]
    fn test_playout_stats() {
        let options = AudioContextOptions {
            sink_id: "none".into(),
            ..AudioContextOptions::default()
        };
        let context = AudioContext::new(options);
        let stats = context.playout_stats();

        std::thread::sleep(Duration::from_millis(50));

        let snapshot = stats.to_json();
        assert!(snapshot.total_frames_duration > 0.);
        assert!(snapshot.fallback_frames_duration >= 0.);
        assert_eq!(
            stats.fallback_frames_events(),
            snapshot.fallback_frames_events
        );
        assert!(stats.average_latency().is_finite());
        assert!(stats.minimum_latency().is_finite());
        assert!(stats.maximum_latency().is_finite());

        stats.reset_latency();
        assert_eq!(stats.average_latency(), 0.);
        assert_eq!(stats.minimum_latency(), 0.);
        assert_eq!(stats.maximum_latency(), 0.);
    }
}

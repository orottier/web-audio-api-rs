use crossbeam_channel::{Receiver, Sender};
use std::sync::{Arc, Mutex};

#[derive(Copy, Clone)]
pub(crate) struct LoadValueData {
    pub render_timestamp: f64,
    pub load_value: f64,
}

/// Options for constructing an `AudioRenderCapacity`
pub struct AudioRenderCapacityOptions {
    /// An update interval (in seconds) for dispaching [`AudioRenderCapacityEvent`]s
    pub update_interval: f64,
}

impl Default for AudioRenderCapacityOptions {
    fn default() -> Self {
        Self {
            update_interval: 1.,
        }
    }
}

/// Performance metrics of the rendering thread
#[derive(Clone, Debug)]
pub struct AudioRenderCapacityEvent {
    timestamp: f64,
    average_load: f64,
    peak_load: f64,
    underrun_ratio: f64,
}

impl AudioRenderCapacityEvent {
    fn new(timestamp: f64, average_load: f64, peak_load: f64, underrun_ratio: f64) -> Self {
        Self {
            timestamp,
            average_load: (average_load * 100.).round() / 100.,
            peak_load: (peak_load * 100.).round() / 100.,
            underrun_ratio: (underrun_ratio * 100.).ceil() / 100.,
        }
    }

    /// The start time of the data collection period in terms of the associated AudioContext's currentTime
    pub fn timestamp(&self) -> f64 {
        self.timestamp
    }

    /// An average of collected load values over the given update interval
    pub fn average_load(&self) -> f64 {
        self.average_load
    }

    /// A maximum value from collected load values over the given update interval.
    pub fn peak_load(&self) -> f64 {
        self.peak_load
    }

    /// A ratio between the number of buffer underruns and the total number of system-level audio callbacks over the given update interval.
    pub fn underrun_ratio(&self) -> f64 {
        self.underrun_ratio
    }
}

type EventHandler = Box<dyn FnMut(AudioRenderCapacityEvent) + Send + 'static>;

/// Provider for rendering performance metrics
///
/// A load value is computed for each system-level audio callback, by dividing its execution
/// duration by the system-level audio callback buffer size divided by the sample rate.
///
/// Ideally the load value is below 1.0, meaning that it took less time to render the audio than it
/// took to play it out. An audio buffer underrun happens when this load value is greater than 1.0: the
/// system could not render audio fast enough for real-time.
pub struct AudioRenderCapacity {
    sender: Sender<LoadValueData>,
    receiver: Option<Receiver<LoadValueData>>,
    callback: Arc<Mutex<Option<EventHandler>>>,
}

impl AudioRenderCapacity {
    pub(crate) fn new(sender: Sender<LoadValueData>, receiver: Receiver<LoadValueData>) -> Self {
        let callback = Arc::new(Mutex::new(None));

        Self {
            sender,
            receiver: Some(receiver),
            callback,
        }
    }

    /// Start metric collection and analysis
    #[allow(clippy::missing_panics_doc)]
    pub fn start(&mut self, options: AudioRenderCapacityOptions) {
        let receiver = match self.receiver.take() {
            None => return,
            Some(receiver) => receiver,
        };

        let callback = self.callback.clone();

        let mut timestamp: f64 = 0.;
        let mut load_sum: f64 = 0.;
        let mut counter = 0;
        let mut peak_load: f64 = 0.;
        let mut underrun_sum = 0;

        let mut next_checkpoint = options.update_interval;
        std::thread::spawn(move || {
            for item in receiver {
                let LoadValueData {
                    render_timestamp,
                    load_value,
                } = item;

                // check for signal to stop
                if render_timestamp.is_nan() && load_value.is_nan() {
                    return; // stop thread
                };

                counter += 1;
                load_sum += load_value;
                peak_load = peak_load.max(load_value);
                if load_value > 1. {
                    underrun_sum += 1;
                }

                if render_timestamp >= next_checkpoint {
                    let event = AudioRenderCapacityEvent::new(
                        timestamp,
                        load_sum / counter as f64,
                        peak_load,
                        underrun_sum as f64 / counter as f64,
                    );
                    if let Some(f) = &mut *callback.lock().unwrap() {
                        (f)(event);
                    }

                    next_checkpoint += options.update_interval;
                    timestamp = render_timestamp;
                    load_sum = 0.;
                    counter = 0;
                    peak_load = 0.;
                    underrun_sum = 0;
                }
            }
        });
    }

    /// Stop metric collection and analysis
    pub fn stop(self) {
        // halt callback thread
        let signal = LoadValueData {
            render_timestamp: f64::NAN,
            load_value: f64::NAN,
        };
        let _ = self.sender.send(signal);
    }

    /// An EventHandler for [`AudioRenderCapacityEvent`].
    #[allow(clippy::missing_panics_doc)]
    pub fn onupdate<F: FnMut(AudioRenderCapacityEvent) + Send + 'static>(&mut self, callback: F) {
        *self.callback.lock().unwrap() = Some(Box::new(callback));
    }
}

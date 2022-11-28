use crossbeam_channel::{Receiver, Sender};
use std::sync::{Arc, Mutex};

use crate::context::{BaseAudioContext, ConcreteBaseAudioContext};

#[derive(Copy, Clone)]
pub(crate) struct AudioRenderCapacityLoad {
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
        // We are limiting the precision here conform
        // https://webaudio.github.io/web-audio-api/#dom-audiorendercapacityevent-averageload
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
    context: ConcreteBaseAudioContext,
    receiver: Receiver<AudioRenderCapacityLoad>,
    callback: Arc<Mutex<Option<EventHandler>>>,
    stop_send: Arc<Mutex<Option<Sender<()>>>>,
}

impl AudioRenderCapacity {
    pub(crate) fn new(
        context: ConcreteBaseAudioContext,
        receiver: Receiver<AudioRenderCapacityLoad>,
    ) -> Self {
        let callback = Arc::new(Mutex::new(None));
        let stop_send = Arc::new(Mutex::new(None));

        Self {
            context,
            receiver,
            callback,
            stop_send,
        }
    }

    /// Start metric collection and analysis
    #[allow(clippy::missing_panics_doc)]
    pub fn start(&self, options: AudioRenderCapacityOptions) {
        // stop current metric collection, if any
        self.stop();

        let callback = self.callback.clone();
        let receiver = self.receiver.clone();
        let (stop_send, stop_recv) = crossbeam_channel::bounded(0);
        *self.stop_send.lock().unwrap() = Some(stop_send);

        let mut timestamp: f64 = self.context.current_time();
        let mut load_sum: f64 = 0.;
        let mut counter = 0;
        let mut peak_load: f64 = 0.;
        let mut underrun_sum = 0;

        let mut next_checkpoint = timestamp + options.update_interval;
        std::thread::spawn(move || loop {
            let try_item = crossbeam_channel::select! {
                recv(receiver) -> item => item,
                recv(stop_recv) -> _ => return,
            };

            // stop thread when render thread has shut down
            let item = match try_item {
                Err(_) => return,
                Ok(item) => item,
            };

            let AudioRenderCapacityLoad {
                render_timestamp,
                load_value,
            } = item;

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
        });
    }

    /// Stop metric collection and analysis
    #[allow(clippy::missing_panics_doc)]
    pub fn stop(&self) {
        // halt callback thread
        if let Some(stop_send) = self.stop_send.lock().unwrap().take() {
            let _ = stop_send.send(());
        }
    }

    /// The EventHandler for [`AudioRenderCapacityEvent`].
    ///
    /// Only a single event handler is active at any time. Calling this method multiple times will
    /// disable the previous event handlers.
    #[allow(clippy::missing_panics_doc)]
    pub fn onupdate<F: FnMut(AudioRenderCapacityEvent) + Send + 'static>(&self, callback: F) {
        *self.callback.lock().unwrap() = Some(Box::new(callback));
    }
}

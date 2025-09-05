use std::collections::VecDeque;
use std::sync::{Arc, Mutex, Weak};

/// Global echo reference manager to share audio output with echo cancellers
pub struct EchoReferenceManager {
    reference_buffers: Arc<Mutex<Vec<Weak<Mutex<VecDeque<f32>>>>>>,
}

impl EchoReferenceManager {
    pub fn new() -> Self {
        Self {
            reference_buffers: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    /// Register a reference buffer for echo cancellation
    pub fn register_buffer(&self, buffer: Weak<Mutex<VecDeque<f32>>>) {
        let mut buffers = self.reference_buffers.lock().unwrap();
        // Clean up any dead weak references
        buffers.retain(|b| b.upgrade().is_some());
        buffers.push(buffer);
    }
    
    /// Send reference audio to all registered echo cancellers
    pub fn send_reference(&self, audio: &[f32], channels: usize) {
        let mut buffers = self.reference_buffers.lock().unwrap();
        
        // Clean up dead references and send to alive ones
        buffers.retain(|weak_buffer| {
            if let Some(buffer) = weak_buffer.upgrade() {
                if let Ok(mut buf) = buffer.lock() {
                    // Mix to mono if needed
                    if channels > 1 {
                        for frame in audio.chunks(channels) {
                            let mono_sample: f32 = frame.iter().sum::<f32>() / channels as f32;
                            buf.push_back(mono_sample);
                            
                            // Keep buffer size reasonable (max 1 second at 48kHz)
                            if buf.len() > 48000 {
                                buf.pop_front();
                            }
                        }
                    } else {
                        buf.extend(audio);
                        // Keep buffer size reasonable
                        while buf.len() > 48000 {
                            buf.pop_front();
                        }
                    }
                }
                true
            } else {
                false
            }
        });
    }
}

// Global instance
lazy_static::lazy_static! {
    pub static ref ECHO_REFERENCE_MANAGER: EchoReferenceManager = EchoReferenceManager::new();
}
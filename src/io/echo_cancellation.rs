use crate::buffer::AudioBuffer;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use super::echo_reference::ECHO_REFERENCE_MANAGER;
use std::cell::RefCell;

// Wrapper to make EchoCanceller Send + Sync
pub struct EchoCanceller {
    inner: RefCell<EchoCancellerInner>,
    reference_buffer: Arc<Mutex<VecDeque<f32>>>,
}

struct EchoCancellerInner {
    aec: aec_rs::Aec,
    config: aec_rs::AecConfig,
    frame_buffer: Vec<f32>,
    reference_frame_buffer: Vec<f32>,
}

// SAFETY: EchoCanceller is only used from the audio thread
unsafe impl Send for EchoCanceller {}
unsafe impl Sync for EchoCanceller {}

impl EchoCanceller {
    pub fn new(sample_rate: f32, frame_size: usize) -> Self {
        let config = aec_rs::AecConfig {
            sample_rate: sample_rate as u32,
            filter_length: (sample_rate * 0.1) as i32, // 0.1s filter
            frame_size,
            enable_preprocess: true, // Enable denoising
        };
        
        let aec = aec_rs::Aec::new(&config);
        
        let reference_buffer = Arc::new(Mutex::new(VecDeque::new()));
        
        // Register this buffer with the global echo reference manager
        ECHO_REFERENCE_MANAGER.register_buffer(Arc::downgrade(&reference_buffer));
        
        let inner = EchoCancellerInner {
            aec,
            config,
            frame_buffer: vec![0.0; frame_size],
            reference_frame_buffer: vec![0.0; frame_size],
        };
        
        Self {
            inner: RefCell::new(inner),
            reference_buffer,
        }
    }
    
    pub fn get_reference_buffer(&self) -> Arc<Mutex<VecDeque<f32>>> {
        Arc::clone(&self.reference_buffer)
    }
    
    pub fn process(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; input.len()];
        let mut inner = self.inner.borrow_mut();
        
        // Process in frame-sized chunks
        for (chunk_idx, input_chunk) in input.chunks(inner.config.frame_size).enumerate() {
            let chunk_len = input_chunk.len();
            
            // If chunk is smaller than frame size, we need to pad
            if chunk_len < inner.config.frame_size {
                inner.frame_buffer[..chunk_len].copy_from_slice(input_chunk);
                inner.frame_buffer[chunk_len..].fill(0.0);
            } else {
                inner.frame_buffer.copy_from_slice(input_chunk);
            }
            
            // Get reference samples from the buffer
            {
                let mut ref_buffer = self.reference_buffer.lock().unwrap();
                for i in 0..inner.config.frame_size {
                    inner.reference_frame_buffer[i] = ref_buffer.pop_front().unwrap_or(0.0);
                }
            }
            
            // Convert f32 to i16 for AEC processing
            let mut input_i16: Vec<i16> = inner.frame_buffer
                .iter()
                .map(|&x| (x * 32767.0).clamp(-32768.0, 32767.0) as i16)
                .collect();
                
            let mut reference_i16: Vec<i16> = inner.reference_frame_buffer
                .iter()
                .map(|&x| (x * 32767.0).clamp(-32768.0, 32767.0) as i16)
                .collect();
                
            let mut output_i16 = vec![0i16; inner.config.frame_size];
            
            // Apply echo cancellation
            inner.aec.cancel_echo(&mut input_i16, &mut reference_i16, &mut output_i16);
            
            // Convert back to f32 and copy to output
            let start_idx = chunk_idx * inner.config.frame_size;
            for (i, &sample) in output_i16.iter().take(chunk_len).enumerate() {
                if start_idx + i < output.len() {
                    output[start_idx + i] = sample as f32 / 32767.0;
                }
            }
        }
        
        output
    }
    
    pub fn add_reference_audio(&self, audio: &AudioBuffer) {
        let mut ref_buffer = self.reference_buffer.lock().unwrap();
        let inner = self.inner.borrow();
        
        // Mix all channels to mono for reference
        let num_samples = audio.length();
        let num_channels = audio.number_of_channels();
        
        for sample_idx in 0..num_samples {
            let mut mixed_sample = 0.0;
            for ch in 0..num_channels {
                mixed_sample += audio.get_channel_data(ch)[sample_idx];
            }
            mixed_sample /= num_channels as f32;
            
            ref_buffer.push_back(mixed_sample);
            
            // Keep buffer size reasonable (max 1 second)
            if ref_buffer.len() > inner.config.sample_rate as usize {
                ref_buffer.pop_front();
            }
        }
    }
}
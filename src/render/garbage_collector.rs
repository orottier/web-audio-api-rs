use std::cell::RefCell;
use std::rc::Rc;

use std::any::Any;
use std::time::Duration;

type AnyChannel = llq::Producer<Box<dyn Any + Send>>;

#[derive(Clone, Default)]
pub(crate) struct GarbageCollector {
    gc: Option<Rc<RefCell<AnyChannel>>>,
}

impl GarbageCollector {
    pub fn deallocate_async(&self, value: llq::Node<Box<dyn Any + Send>>) {
        if let Some(gc) = self.gc.as_ref() {
            gc.borrow_mut().push(value);
        }
    }

    pub fn spawn_garbage_collector_thread(&mut self) {
        if self.gc.is_none() {
            let (gc_producer, gc_consumer) = llq::Queue::new().split();
            spawn_garbage_collector_thread(gc_consumer);
            self.gc = Some(Rc::new(RefCell::new(gc_producer)));
        }
    }
}

// Controls the polling frequency of the garbage collector thread.
const GARBAGE_COLLECTOR_THREAD_TIMEOUT: Duration = Duration::from_millis(100);

// Poison pill that terminates the garbage collector thread.
#[derive(Debug)]
pub(crate) struct TerminateGarbageCollectorThread;

// Spawns a sidecar thread of the `RenderThread` for dropping resources.
fn spawn_garbage_collector_thread(consumer: llq::Consumer<Box<dyn Any + Send>>) {
    let _join_handle = std::thread::spawn(move || run_garbage_collector_thread(consumer));
}

fn run_garbage_collector_thread(mut consumer: llq::Consumer<Box<dyn Any + Send>>) {
    log::debug!("Entering garbage collector thread");
    loop {
        if let Some(node) = consumer.pop() {
            if node
                .as_ref()
                .downcast_ref::<TerminateGarbageCollectorThread>()
                .is_some()
            {
                log::info!("Terminating garbage collector thread");
                break;
            }
            // Implicitly drop the received node.
        } else {
            std::thread::sleep(GARBAGE_COLLECTOR_THREAD_TIMEOUT);
        }
    }
    log::info!("Exiting garbage collector thread");
}

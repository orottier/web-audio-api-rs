pub const BUFFER_SIZE: u32 = 512;

pub mod context;
pub mod node;

pub(crate) mod control;
pub(crate) mod graph;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

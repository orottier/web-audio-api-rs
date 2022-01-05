use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub struct SnapShot {
    pub data: Vec<f32>,
}

pub fn read<P: AsRef<Path>>(path: P) -> Result<SnapShot, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let data = reader
        .lines()
        .map(|l| l.unwrap().parse::<f32>().unwrap())
        .collect();

    Ok(SnapShot { data })
}

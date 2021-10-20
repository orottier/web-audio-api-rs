use serde::Deserialize;
use std::{error::Error, fs::File, io::BufReader, path::Path};

#[derive(Deserialize, Debug)]
pub struct SnapShot {
    pub data: Vec<f32>,
}

pub fn read<P: AsRef<Path>>(path: P) -> Result<SnapShot, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let snp = serde_json::from_reader(reader)?;

    Ok(snp)
}

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

// keep that around, usefull to write data into files and plot long signals
// @todo - `write`
// use std::fs::File;
// use std::io::Write;

// let mut file = File::create("_signal-expected.txt").unwrap();
// for i in expected.iter() {
//     let mut tmp = String::from(i.to_string());
//     tmp += ",\n";
//     file.write(tmp.to_string().as_bytes()).unwrap();
// }

// let mut file = File::create("_signal-result.txt").unwrap();
// for i in result.iter() {
//     let mut tmp = String::from(i.to_string());
//     tmp += ",\n";
//     file.write(tmp.to_string().as_bytes()).unwrap();
// }

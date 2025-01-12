use aplot;
use hound;
use aplot::{Segments, Config};

const SVG_PATH: &'static str = "M 100 880 Q 525 100, 950 800 T 1800 800";

fn main() {
    let config = Config {
        sample_rate: 48_000,
        time_seconds: 2.5,
        min_val: 100.0,
        max_val: 1500.0,
        min_pitch: 800.0,
        max_pitch: 2300.0,
        reverse: true,
        start: 0.5,
        len: 0.2,
    };
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: config.sample_rate as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let segs = Segments::from_path(SVG_PATH).unwrap();
    let mut writer = hound::WavWriter::create("out.wav", spec).unwrap();
    let _ = segs.all_samples(config)
        .for_each(|samp| {
            writer.write_sample(samp).unwrap();
        });
    writer.finalize().unwrap();
}

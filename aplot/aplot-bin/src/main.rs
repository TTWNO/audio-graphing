use aplot;
use hound;
use aplot::{Interpolate, Point};

fn main() {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let quad = aplot::Quadradic {
        p1: Point { x: 10.0, y: 80.0*8.0 },
        p2: Point { x: 52.5, y: 10.0*8.0 },
        p3: Point { x: 10.0, y: 80.0*8.0 },
    };
    let mut writer = hound::WavWriter::create("out.wav", spec).unwrap();
    for sample in quad.samples(48_000.0, 2.0) {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
}

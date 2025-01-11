use aplot;
use hound;

fn main() {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create("out.wav", spec).unwrap();
    for sample in aplot::from_q3_points(640.0, 80.0, 640.0, 48_000.0, 2.0) {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
}

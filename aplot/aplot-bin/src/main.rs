use aplot;
use hound;
use aplot::{Segments, Config, PlottingFunc};

const SVG_PATH: &'static str = "M 10 80 Q 52.5 10, 55 80 T 180 80";
const EXPR: &'static str = "sin(x/5) * 5 + 100";

fn main() {
    let config = Config {
        sample_rate: 48_000,
        time_seconds: 2.5,
        min_val: 50.0,
        max_val: 150.0,
        min_pitch: 100.0,
        max_pitch: 2000.0,
        reverse: true,
        start: 0.0,
        len: 1.0,
    };
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: config.sample_rate as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create("out.wav", spec).unwrap();

    /// Segments from SVG path
    //let segs = Segments::from_path(SVG_PATH).unwrap();
    //let _ = segs.all_samples(config)
    //    .for_each(|samp| {
    //        writer.write_sample(samp).unwrap();
    //    });
    
    /// From math expression to SVG path and audio samples
    let expr = PlottingFunc::from_str(EXPR).unwrap();
    println!("<path d=\"{}\" stroke=\"navy\" fill=\"transparent\"/>", expr.to_path_segments(100, 200.0).unwrap());
    expr.all_samples(config, 200.0)
        .unwrap()
        .for_each(|samp| {
            writer.write_sample(samp).unwrap();
        });
    writer.finalize().unwrap();
}

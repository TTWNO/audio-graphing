use aplot;
use hound;
use aplot::{
    //Segments,
    Config, PlottingFunc};

const SVG_PATH: &'static str = "M 10 80 Q 52.5 10, 55 80 T 180 80";
const EXPR: &'static str = "x^2";

use fasteval::{Evaler, Compiler, Slab, Parser, Instruction};

pub struct DynamicPlottingFunc {
    slab: Slab,
    inner: Instruction,
}

impl DynamicPlottingFunc {
    pub fn from_str(s: &str) -> Result<Self, fasteval::Error> {
        let parser = Parser::new();
        let mut slab = Slab::new();
        let compiled = parser.parse(s, &mut slab.ps)?.from(&slab.ps).compile(&slab.ps, &mut slab.cs);
        Ok(Self { inner: compiled, slab })
    }
    pub fn fn_mut(&mut self) -> impl FnMut(f64) -> Result<f64, fasteval::Error>  + use<'_> {
        |x: f64| {
         let mut cb = move |name:&str, _args:Vec<f64>| -> Option<f64> {
             match name {
                 "x" => Some(x),
                 _ => None,
             }
         };
        self.inner.eval(&self.slab, &mut cb)
        }
    }
}

fn main() {
    let config = Config {
        sample_rate: 48_000,
        time_seconds: 2.5,
        x_start: -100.0,
        x_end: 100.0,
        min_val: -5.0,
        max_val: 5.0,
        min_pitch: 200.0,
        max_pitch: 1000.0,
        min_y: -100.0,
        max_y: 100.0,
        reverse: false,
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
    let mut expr_ = DynamicPlottingFunc::from_str(EXPR).unwrap();
    let mut expr = PlottingFunc::new(expr_.fn_mut());
    //println!("<path id=\"p2\" transform=\"translate(100, 100)\" d=\"{}\" stroke=\"navy\" fill=\"transparent\"/>", expr.to_path_segments(config.clone(), 250, 1.0, 1.0).unwrap());
    expr.all_samples(config.clone(), 1.0)
        .unwrap()
        .for_each(|samp| {
            writer.write_sample(samp).unwrap();
        });
    writer.finalize().unwrap();
}

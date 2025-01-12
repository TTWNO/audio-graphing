//! `aplot`
//!
//! Audio-plotting (sonification) utilities.

use core::f64::consts::PI;
use itertools::Itertools;

#[derive(Debug)]
pub struct Point {
    pub x: f64,
    pub y: f64
}

pub struct Config {
    /// Sample rate (in Hz)
    pub sample_rate: usize,
    /// Length of all segments (in seconds)
    pub time_seconds: f64,
    /// Minimum value (corresponds to minimum pitch)
    pub min_val: f64,
    /// Maximum value (corresponds to maximum pitch)
    pub max_val: f64,
    /// Minimum pitch (corresponding to minimum value)
    pub min_pitch: f64,
    /// Maximum pitch (corresponding to maximum value)
    pub max_pitch: f64,
    /// Reverses the pitch direction of the y axis, making low values high pitches and high values
    /// low pitched.
    pub reverse: bool,
    /// Starting point as a decimal percentage.
    pub start: f64,
    /// Length to render in decimal percentage.
    pub len: f64,
}

trait PathSegExt {
    fn endx(&self) -> f64;
    fn endy(&self) -> f64;
    fn endpoint(&self) -> Point {
        Point { x: self.endx(), y: self.endy() }
    }
}
impl PathSegExt for svgtypes::PathSegment {
    /// 0 sentinal value, is this correct?
    fn endx(&self) -> f64 {
        match self {
            PathSegment::MoveTo { x, .. } => *x,
            PathSegment::LineTo { x, .. } => *x,
            PathSegment::HorizontalLineTo { x, .. } => *x,
            PathSegment::VerticalLineTo { .. } => 0.0,
            PathSegment::CurveTo { x, .. } => *x,
            PathSegment::SmoothCurveTo { x, .. } => *x,
            PathSegment::Quadratic { x, .. } => *x,
            PathSegment::SmoothQuadratic { x, .. } => *x,
            PathSegment::EllipticalArc { x, .. } => *x,
            PathSegment::ClosePath { .. } => 0.0,
        }
    }
    fn endy(&self) -> f64 {
        match self {
            PathSegment::MoveTo { y, .. } => *y,
            PathSegment::LineTo { y, .. } => *y,
            PathSegment::HorizontalLineTo { .. } => 0.0,
            PathSegment::VerticalLineTo { y, .. } => *y,
            PathSegment::CurveTo { y, .. } => *y,
            PathSegment::SmoothCurveTo { y, .. } => *y,
            PathSegment::Quadratic { y, .. } => *y,
            PathSegment::SmoothQuadratic { y, .. } => *y,
            PathSegment::EllipticalArc { y, .. } => *y,
            PathSegment::ClosePath { .. } => 0.0,
        }
    }
}
use svgtypes::PathSegment;

pub struct Segments {
    inner: Vec<Segment>,
}
impl Segments {
    pub fn from_path(s: &str) -> Result<Self, svgtypes::Error> {
        let segs = svgtypes::PathParser::from(s)
            .collect::<Result<Vec<PathSegment>, svgtypes::Error>>()?
            .into_iter()
            .tuple_windows()
            .map(|(seg1, seg2)| {
                match (seg1, seg2) {
                    (start, PathSegment::MoveTo { x, y, .. }) => Segment::Line(Linear {
                        p1: start.endpoint(),
                        p2: Point { x, y},
                    }),
                    (start, PathSegment::LineTo { x, y, .. }) => Segment::Line(Linear {
                        p1: start.endpoint(),
                        p2: Point { x, y},
                    }),
                    (start, PathSegment::Quadratic { x1, y1, x, y, .. }) => Segment::Quad(Quadradic {
                        p1: start.endpoint(),
                        p2: Point { x: x1, y: y1 },
                        p3: Point { x, y },
                    }),
                    (PathSegment::Quadratic {x: xs, y: ys, x1, y1, .. }, PathSegment::SmoothQuadratic { x, y, .. }) => Segment::Quad(Quadradic {
                        p1: Point { x: xs, y: ys },
                        p2: Point { x: x+(xs-x1), y: y+(ys-y1) },
                        p3: Point { x, y },
                    }),
                    _ => panic!("I'm afraid I can't do that, Dave!"),
                }
            })
            .collect::<Vec<Segment>>();
        Ok(Self { inner: segs })
    }
    pub fn all_samples(&self, c: Config) -> impl Iterator<Item = i16> + use<'_> {
        let time_per = c.time_seconds / self.inner.len() as f64;
        let all_samples = c.sample_rate as f64 * c.time_seconds;
        let start_sample = (c.start * all_samples) as usize;
        let end_sample = start_sample + (c.len * all_samples) as usize;
        self.inner.iter()
            .map(move |seg| {
                microsteps(c.sample_rate as f64, time_per).interpolate(seg)
            })
            .flatten()
            .enumerate()
            .filter_map(move |(i,samp)| {
                if i >= start_sample && i <= end_sample { Some(samp) } else { None }
            })
            .normalize_1_0(c.min_val, c.max_val)
            .reverse(c.reverse)
            .apply_pitch_params(c.min_pitch, c.max_pitch)
            .cumsum()
            .normalize(c.sample_rate as f64)
            .amplitude(AMPLITUDE)
            .as_i16()
    }
}

#[derive(Debug)]
pub enum Segment {
    Line(Linear),
    Quad(Quadradic),
    Cub(Cubic),
}
impl Interpolate for Segment {
    fn expr(&self, t: f64) -> f64  {
        match self {
            Self::Line(l) => l.expr(t),
            Self::Quad(q) => q.expr(t),
            Self::Cub(c) => c.expr(t),
        }
    }
}

#[derive(Debug)]
pub struct Linear {
    pub p1: Point,
    pub p2: Point,
}
#[derive(Debug)]
pub struct Quadradic {
    pub p1: Point,
    pub p2: Point,
    pub p3: Point,
}
#[derive(Debug)]
pub struct Cubic {
    p1: Point,
    p2: Point,
    p3: Point,
    p4: Point,
}
impl Interpolate for Linear {
    fn expr(&self, t: f64) -> f64 {
        linear(t, self.p1.y, self.p2.y)
    }
}
impl Interpolate for Quadradic {
    fn expr(&self, t: f64) -> f64 {
        quadradic_bezier(t, self.p1.y, self.p2.y, self.p3.y)
    }
}
impl Interpolate for Cubic {
    fn expr(&self, t: f64) -> f64 {
        cubic_bezier(t, self.p1.y, self.p2.y, self.p3.y, self.p4.y)
    }
}

/// Flip a points' x and y axes.
pub trait Flip {
    fn flip(&self) -> Self;
}

impl Flip for Point {
    fn flip(&self) -> Self {
        Point {
            x: self.y,
            y: self.x,
        }
    }
}
impl Flip for Linear {
    fn flip(&self) -> Self {
        Linear {
            p1: self.p1.flip(),
            p2: self.p2.flip(),
        }
    }
}
impl Flip for Quadradic {
    fn flip(&self) -> Self {
        Quadradic {
            p1: self.p1.flip(),
            p2: self.p2.flip(),
            p3: self.p3.flip(),
        }
    }
}
impl Flip for Cubic {
    fn flip(&self) -> Self {
        Cubic {
            p1: self.p1.flip(),
            p2: self.p2.flip(),
            p3: self.p3.flip(),
            p4: self.p4.flip(),
        }
    }
}

/// Amplitude for 16 bit WAV sound: (2^15)-1
const AMPLITUDE: f64 = 32767.0;

pub trait SampleIter: Iterator<Item = f64> {
    /// Interpolate value given a interpolatable value (inter) and a series of points in time as
    /// `f64`s (self).
    fn interpolate<T>(self, inter: &T) -> impl Iterator<Item = f64> 
    where Self: Sized,
          T: Interpolate {
        self.map(move |f| inter.expr(f))
    }
    /// Re-normalize a value (reversed): x = abs(1-x)
    fn reverse(self, rev: bool) -> impl Iterator<Item = f64> 
    where Self: Sized {
        let diff = if rev { 1.0 } else { 0.0 };
        self.map(move |v| (diff-v).abs())
    }
    /// normalize value (-1 < x < 1)
    fn normalize(self, sample_rate: f64) -> impl Iterator<Item = f64> 
    where Self: Sized {
        self.map(move |freq| {
                normalize(freq, sample_rate)
            })
    }
    /// apply amplitude (loudness)
    fn amplitude(self, amp: f64) -> impl Iterator<Item = f64> 
    where Self: Sized {
        self.map(move |norm_freq| {
                amp * norm_freq
            })
    }
    /// cumsum - flatten out the changes to produce smoother transitions
    fn cumsum(self) -> impl Iterator<Item = f64> 
    where Self: Sized {
        self.scan(0.0, |acc, freq| {
                *acc += freq;
                Some(*acc)
            })
    }
    fn as_i16(self) -> impl Iterator<Item = i16> 
    where Self: Sized {
            self.map(|f| f as i16)
    }
    /// TODO: optimize cloning
    fn auto_normalize_1_0(self) -> impl Iterator<Item = f64> 
    where Self: Sized {
        let new = self.collect::<Vec<f64>>();
        let minmax = new.iter().cloned().minmax();
        let itertools::MinMaxResult::MinMax(lo, hi) = minmax else {
            panic!("This can not be called on a list < length of 2");
        };
        let diff = hi - lo;
        let abs_diff = diff.abs();
        new.into_iter()
        .map(move |v| {
            if abs_diff < f64::EPSILON { 0.5 } else { (v-lo)/diff }
        })
    }
    fn normalize_1_0(self, lo: f64, hi: f64) -> impl Iterator<Item = f64> 
    where Self: Sized {
        let diff = hi - lo;
        let abs_diff = diff.abs();
        self.map(move |v| {
            if abs_diff < f64::EPSILON { 0.5 } else { (v-lo)/diff }
        })
    }
    /// NOTE: all values must be 0.0 <= x <= 1.0
    fn apply_pitch_params(self, min_pitch: f64, max_pitch: f64) -> impl Iterator<Item = f64> 
    where Self: Sized {
        let diff = max_pitch - min_pitch;
        self.map(move |v| min_pitch + v*diff)
    }
}
impl<I> SampleIter for I
where I: Iterator<Item = f64> {}

/// Converts a frequency to a signed normalized value (-1 <= x <= 1).
fn normalize(freq: f64, sample_rate: f64) -> f64 {
    (freq * PI / sample_rate).sin()
}

pub trait Interpolate {
    /// Interpolate the y value of the curve at point t (between 0 and 1)
    fn expr(&self, t: f64) -> f64;
    /// Create audio samples
    fn samples(&self, sample_rate: f64, len_seconds: f64) -> impl Iterator<Item = i16> 
    where Self: Sized {
        microsteps(sample_rate, len_seconds)
            .interpolate(self)
            .cumsum()
            .normalize(sample_rate)
            .amplitude(AMPLITUDE)
            .as_i16()
    }
}

/// NOTE: `t` must be between 0 and 1.
fn linear(t: f64, p1: f64, p2: f64) -> f64 {
    p1 + ((p2-p1) * t)
}

/// NOTE: `t` must be between 0 and 1.
fn quadradic_bezier(t: f64, p1: f64, p2: f64, p3: f64) -> f64 {
    let t2 = t*t;
    let mt = 1.0-t;
    let mt2 = mt*mt;
    (p1 * mt2) + (p2 * 2.0 * mt * t) + (p3*t2)
}

/// NOTE: `t` must be between 0 and 1.
fn cubic_bezier(t: f64, p1: f64, p2: f64, p3: f64, p4: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    let mt = 1.0-t;
    let mt2 = mt * mt;
    let mt3 = mt2 * mt;
    p1*mt3 + 3.0*p2*mt2*t + 3.0*p3*mt*t2 + p4*t3
}

#[test]
fn test_microsteps() {
    let mut ms = microsteps(48_000.0, 2.0);
    // NOTE: that these two functions are actually really fickle.
    // But if they're close enough, it's fine, feel free to remove some digits of precision if
    // needed.
    assert!(ms.nth(10).unwrap() - 0.00010416666666666666 < f64::EPSILON);
    assert!(ms.nth(95980).unwrap() - 0.99990625 < f64::EPSILON);
}

#[test]
fn test_main_func() {
    // samples and values taken from the Python impl
    let correct_values = (
        -24919, 32021, -14044, -27476
    );
    // These were the points used in the Python impl.
    let quad = Quadradic { 
        p1: Point { x: 0.0, y: 80.0},
        p2: Point { x: 0.0, y: 10.0},
        p3: Point { x: 0.0, y: 80.0},
    };
    let ans: Vec<i16> = quad.samples(48_000.0, 2.0).collect();
    assert_eq!(
        (ans[2000], ans[4000], ans[90000], ans[95000]),
        correct_values
    );
}

fn microsteps(sample_rate: f64, length_seconds: f64) -> impl Iterator<Item = f64> {
    let max = (sample_rate * length_seconds).round() as u32;
    let over1 = (sample_rate * length_seconds).recip();
    (0..=max)
        .map(move |i: u32| (i as f64) * over1)
}

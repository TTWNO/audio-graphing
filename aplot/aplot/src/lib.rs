//! `aplot`
//!
//! Audio-plotting (sonification) utilities.

use core::f64::consts::PI;

pub struct Point {
    pub x: f64,
    pub y: f64
}

pub struct Linear {
    pub p1: Point,
    pub p2: Point,
}
pub struct Quadradic {
    pub p1: Point,
    pub p2: Point,
    pub p3: Point,
}
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

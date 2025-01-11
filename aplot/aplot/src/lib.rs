//! `aplot`
//!
//! Audio-plotting (sonification) utilities.

use core::f64::consts::PI;

/// NOTE: `t` must be between 0 and 1.
fn quadradic_bezier(t: f64, p1: f64, p2: f64, p3: f64) -> f64 {
    let t2 = t*t;
    let mt = 1.0-t;
    let mt2 = mt*mt;
    (p1 * mt2) + (p2 * 2.0 * mt * t) + (p3*t2)
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
    let (p1, p2, p3) = (80.0, 10.0, 80.0);
    let ans: Vec<i16> = from_q3_points(p1, p2, p3, 48_000.0, 2.0).collect();
    assert_eq!(
        (ans[2000], ans[4000], ans[90000], ans[95000]),
        correct_values
    );
}

fn microsteps(sample_rate: f64, length_seconds: f64) -> impl Iterator<Item = f64> {
    let max = sample_rate.round() as u32 * length_seconds.round() as u32;
    let over1 = (sample_rate * length_seconds).recip();
    (0..=max)
        .map(move |i: u32| (i as f64) * over1)
}

pub fn from_q3_points(p1: f64, p2: f64, p3: f64, sample_rate: f64, len_seconds: f64) -> impl Iterator<Item = i16> {
    let amplitude = 2.0f64.powi(15) - 1.0;
    microsteps(sample_rate, len_seconds)
        .map(move |t| {
            quadradic_bezier(t, p1, p2, p3)
        })
        // cumsum - flatten out the changes to produce smoother transitions
        .scan(0.0, |acc, freq| {
            *acc += freq;
            Some(*acc)
        })
        // normalize value (-1 < x < 1)
        .map(move |mm_freq| {
            (mm_freq * PI / sample_rate).sin()
        })
        // apply amplitude (loudness)
        .map(move |norm_freq| {
            amplitude * norm_freq
        })
        .map(move |f| f as i16)
}


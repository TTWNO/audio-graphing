#![no_std]
#![no_main]

use aplot::{PlottingFunc, Config};
use core::hint::black_box;
use core::iter::Iterator;
use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

fn double_x(x: f64) -> f64 {
    x*2.0
}

#[no_mangle]
pub extern "C" fn _start() {
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
    let mut expr = PlottingFunc::<_, ()>::new(double_x);
    expr.all_samples_infallible(config, 1.0)
        .for_each(|sampl| { let _ = black_box(sampl); });
}

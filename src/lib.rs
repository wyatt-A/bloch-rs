use std::io::Write;
use std::fs::File;
use std::io::Read;
use bytemuck::cast_slice;
use num_complex::{Complex, Complex64, ComplexFloat};
use crate::operators::{apply_relaxation, apply_rotation, relaxation_operator, rotation_operator};
use rayon::prelude::*;
pub mod operators;



// isochromat is a "spin" in this context. We can assign multiple isochromats to a location to model
// partial volume effects, off-resonance, chemical shift,... etc


#[test]
fn test() {

    // 20mm sphere diameter with 100um spacing
    let positions = Positions::sphere(10.,0.1);
    println!("generated {} spins",positions.len());

    let cf_hz = 300e6;
    let gamma = 42.58e6;
    let t1 = 100e-3;
    let t2 = 30e-3;
    let m0 = 1.;
    let ppm = 1.;
    let rf_scale = 5e-3;

    let mut spins = Isochromats::uniform(gamma,t1,t2,m0,positions.len());
    let offres = OffResonance::uniform(ppm,positions.len());
    let tx = TxSensitivity::uniform(rf_scale,positions.len());
    let rx = RxSensitivity::uniform(positions.len());

    let mut f = File::open("/Users/Wyatt/seq-lib/rf_cal.ps").unwrap();
    let mut file_contents = vec![];
    f.read_to_end(&mut file_contents).unwrap();
    assert_eq!(file_contents.len() % 8, 0, "Length must be multiple of 8");
    // Convert Vec<u8> → &[u8] → &[f64]
    let floats: &[f64] = cast_slice(&file_contents);
    // Copy into a new Vec<f64> (still cheap)
    let pulse_sequence = floats.to_vec();


    let mut signal = vec![];

    let time_steps = pulse_sequence.chunks_exact(7).collect::<Vec<_>>();


    time_steps.windows(2).enumerate().for_each(|(i,t)|{

        println!("working on time step {i} of {}",time_steps.len());

        // time step in seconds
        let tau = (t[1][0] - t[0][0]) * 0.1e-6;

        let gx = t[0][1];
        let gy = t[0][2];
        let gz = t[0][3];
        let rf_r = t[0][4];
        let rf_i = t[0][5];
        let acq = t[0][6];

        update(
            &mut signal,
            &mut spins,
            &positions,
            &tx,
            &rx,
            &offres,
            gx,
            gy,
            gz,
            rf_r,
            rf_i,
            acq,
            tau,
            cf_hz
        );
    });

    let mut f = File::create("out.txt").unwrap();
    for s in signal {
        write!(f,"{},{}\n",s.re,s.im).unwrap();
    }

}





// isochromat position
pub struct Positions {
    x:Vec<f64>,
    y:Vec<f64>,
    z:Vec<f64>,
}

impl Positions {

    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn sphere(radius_mm:f64, spacing_mm:f64) -> Positions {

        let mut x = vec![];
        let mut y = vec![];
        let mut z = vec![];

        let n = (radius_mm / spacing_mm).floor() as i32;

        for _x in -n..=n {
            for _y in -n..=n {
                for _z in -n..=n {
                    let r = _x*_x + _y*_y + _z*_z;
                    if r as f64 * spacing_mm <= radius_mm {
                        x.push(_x as f64);
                        y.push(_y as f64);
                        z.push(_z as f64);
                    }
                }
            }
        }

        Positions {
            x,y,z
        }

    }


}




pub struct Isochromats {
    gamma:Vec<f64>,
    m0:Vec<f64>,
    mx:Vec<f64>,
    my:Vec<f64>,
    mz:Vec<f64>,
    t1:Vec<f64>,
    t2:Vec<f64>,
}

impl Isochromats {
    pub fn uniform(gamma:f64,t1:f64,t2:f64,m0:f64,n:usize) -> Isochromats {
        Isochromats {
            gamma:vec![gamma;n],
            m0:vec![m0;n],
            mx:vec![0.;n],
            my:vec![0.;n],
            mz:vec![1.;n],
            t1:vec![t1;n],
            t2:vec![t2;n],
        }
    }
}

pub struct OffResonance {
    // off-resonance from center frequency
    db0:Vec<f64>,
}

impl OffResonance {
    pub fn uniform(ppm: f64, n:usize) -> OffResonance {
        OffResonance {
            db0:vec![ppm * 1e-6;n]
        }
    }
}

/// convert relative rf strength (dB) to field strength T
pub struct TxSensitivity {
    c:Vec<f64>,
}

impl TxSensitivity {
    pub fn uniform(scale_t:f64,n:usize) -> TxSensitivity {
        TxSensitivity {
            c:vec![scale_t;n],
        }
    }
}

/// convert net m0 to signal strength
pub struct RxSensitivity {
    c:Vec<f64>,
}

impl RxSensitivity {
    pub fn uniform(n:usize) -> RxSensitivity {
        RxSensitivity {
            c:vec![1.;n],
        }
    }
}

pub fn update(signal:&mut Vec<Complex<f64>>, isochromats:&mut Isochromats, positions:&Positions, tx:&TxSensitivity, rx:&RxSensitivity, offres:&OffResonance, gx:f64, gy:f64, gz:f64, rf_r:f64, rf_i:f64, acq:f64, tau:f64, cf_hz:f64 ) {


    //println!("g = [{gx},{gy},{gz}]");

    // // generate signal
    if !acq.is_nan() {
        let net_mx = isochromats.mx.par_iter().zip(rx.c.par_iter()).zip(isochromats.m0.par_iter()).map(|((&mx,&c),&m0)| c * mx * m0).sum::<f64>();
        let net_my = isochromats.my.par_iter().zip(rx.c.par_iter()).zip(isochromats.m0.par_iter()).map(|((&my,&c),&m0)| c * my * m0).sum::<f64>();
        // apply acq phase
        signal.push(
            Complex64::new(net_mx, net_my) * Complex64::from_polar(1.,acq)
        )
    }

    // let net_mx = isochromats.mx.par_iter().zip(rx.c.par_iter()).zip(isochromats.m0.par_iter()).map(|((&mx,&c),&m0)| c * mx * m0).sum::<f64>();
    // let net_my = isochromats.my.par_iter().zip(rx.c.par_iter()).zip(isochromats.m0.par_iter()).map(|((&my,&c),&m0)| c * my * m0).sum::<f64>();
    // // apply acq phase
    // signal.push(
    //     Complex64::new(net_mx, net_my) * Complex64::from_polar(1.,0.)
    // );

    // apply spin state update
    isochromats.mx.par_iter_mut()
        .zip(isochromats.my.par_iter_mut())
        .zip(isochromats.mz.par_iter_mut())
        .zip(positions.x.par_iter())
        .zip(positions.y.par_iter())
        .zip(positions.z.par_iter())
        .zip(isochromats.m0.par_iter())
        .zip(isochromats.t1.par_iter())
        .zip(isochromats.t2.par_iter())
        .zip(isochromats.gamma.par_iter())
        .zip(offres.db0.par_iter())
        .zip(tx.c.par_iter())
        .for_each(|(((((((((((mx,my),mz),&x),&y),&z),&m0),&t1),&t2),&gamma),&db0),&c)| {

        let bx = c * rf_r;
        let by = c * rf_i;
        let bz = gx * x + gy * y + gz * z + (cf_hz * db0 / gamma);

        // half step for strang splitting
        let rel_op = relaxation_operator(t1, t2, tau/2.);

        // full step
        let rot_op = rotation_operator(bx, by, bz, tau, gamma);

        // half-step relax
        apply_relaxation(
            mx,
            my,
            mz,
            m0,
            &rel_op
        );

        // full step rotation
        apply_rotation(
            mx,
            my,
            mz,
            &rot_op
        );

        // half-step relax
        apply_relaxation(
            mx,
            my,
            mz,
            m0,
            &rel_op
        );

    });

}
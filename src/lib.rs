use std::f64::consts::PI;
use std::io::Write;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use bytemuck::cast_slice;
use num_complex::{Complex, Complex64, ComplexFloat};
use rand::distr::Distribution;
use rand::rng;
use rand_distr::Normal;
use rand_distr::num_traits::abs_sub;
use crate::operators::{apply_relax, apply_rotation, compute_relax_coeffs, compute_rotation_coeffs};
use rayon::prelude::*;
pub mod operators;



// isochromat is a "spin" in this context. We can assign multiple isochromats to a location to model
// partial volume effects, off-resonance, chemical shift,... etc


#[test]
fn test() {

    // 20mm sphere diameter with 100um spacing
    let positions = Positions::sphere(10.,0.1);
    println!("generated {} spins",positions.len());

    let gamma = 42.58e6 * 2. * PI;
    let t1 = 100e-3;
    let t2 = 30e-3;
    let m0 = 1.;
    let ppm = 7.0 * 2e-6; // ppm in tesla
    let rf_scale = 8e-6;

    let mut spins = Isochromats::uniform(gamma,t1,t2,m0,positions.len());
    let offres = OffResonance::normal_dist(ppm,positions.len());
    let tx = TxSensitivity::uniform(rf_scale,positions.len());
    let rx = RxSensitivity::uniform(positions.len());

    let mut f = File::open("C:/Users/waust/seq-lib/output.ps").unwrap();
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

    pub fn from_file(file:impl AsRef<Path>) -> Positions {
        let mut f = File::open(file.as_ref()).unwrap();
        let mut bytes = vec![];
        f.read_to_end(&mut bytes).unwrap();
        // number of 8-byte values should be divisible by 7
        assert_eq!(bytes.len() % (8*3), 0);
        let data:&[f64] = cast_slice(&bytes);
        let n = data.len() / 3;
        let chunks:Vec<_> = data.chunks_exact(n).collect();
        Positions {
            x: chunks[0].to_vec(),
            y: chunks[1].to_vec(),
            z: chunks[2].to_vec(),
        }
    }

    pub fn to_file(&self,file:impl AsRef<Path>) {
        let mut f = File::create(file.as_ref()).unwrap();
        f.write_all(cast_slice(self.x.as_slice())).unwrap();
        f.write_all(cast_slice(self.y.as_slice())).unwrap();
        f.write_all(cast_slice(self.z.as_slice())).unwrap();
    }

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
                        x.push(_x as f64 * spacing_mm * 1e-3);
                        y.push(_y as f64 * spacing_mm * 1e-3);
                        z.push(_z as f64 * spacing_mm * 1e-3);
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
    mx:Vec<f64>,
    my:Vec<f64>,
    mz:Vec<f64>,
    gamma:Vec<f64>,
    m0:Vec<f64>,
    t1:Vec<f64>,
    t2:Vec<f64>,
}

impl Isochromats {

    pub fn from_file(file:impl AsRef<Path>) -> Isochromats {
        let mut f = File::open(file.as_ref()).unwrap();
        let mut bytes = vec![];
        f.read_to_end(&mut bytes).unwrap();
        // number of 8-byte values should be divisible by 7
        assert_eq!(bytes.len() % (8*7), 0);
        let data:&[f64] = cast_slice(&bytes);
        let n = data.len() / 7;
        let chunks:Vec<_> = data.chunks_exact(n).collect();
        Isochromats {
            mx: chunks[0].to_vec(),
            my: chunks[1].to_vec(),
            mz: chunks[2].to_vec(),
            gamma: chunks[3].to_vec(),
            m0: chunks[4].to_vec(),
            t1: chunks[5].to_vec(),
            t2: chunks[6].to_vec(),
        }
    }

    pub fn to_file(&self,file:impl AsRef<Path>) {
        let mut f = File::create(file.as_ref()).unwrap();
        f.write_all(cast_slice(self.mx.as_slice())).unwrap();
        f.write_all(cast_slice(self.my.as_slice())).unwrap();
        f.write_all(cast_slice(self.mz.as_slice())).unwrap();
        f.write_all(cast_slice(self.gamma.as_slice())).unwrap();
        f.write_all(cast_slice(self.m0.as_slice())).unwrap();
        f.write_all(cast_slice(self.t1.as_slice())).unwrap();
        f.write_all(cast_slice(self.t2.as_slice())).unwrap();
    }

    pub fn len(&self) -> usize {
        self.mx.len()
    }

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

    pub fn len(&self) -> usize {
        self.db0.len()
    }

    pub fn from_file(file:impl AsRef<Path>) -> OffResonance {
        let mut f = File::open(file.as_ref()).unwrap();
        let mut bytes = vec![];
        f.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes.len() % 8, 0);
        let values:&[f64] = cast_slice(&bytes);
        OffResonance {
            db0: values.to_vec()
        }
    }

    pub fn to_file(&self,file:impl AsRef<Path>) {
        let mut f = File::create(file.as_ref()).unwrap();
        f.write_all(cast_slice(self.db0.as_slice())).unwrap();
    }

    pub fn uniform(tesla: f64, n:usize) -> OffResonance {
        OffResonance {
            db0:vec![tesla;n]
        }
    }

    pub fn normal_dist(tesla_std: f64, n:usize) -> OffResonance {

        let normal = Normal::new(0.0, tesla_std).unwrap();
        let mut rng = rng();

        // Generate a single Gaussian f64

        // Generate a vector of Gaussian samples
        let samples: Vec<f64> = (0..n)
            .map(|_| {
                let val = normal.sample(&mut rng);
                if val == 0.0 {
                    f64::EPSILON
                }else {
                    val
                }
            } )
            .collect();

        OffResonance {
            db0:samples
        }
    }
}

/// convert relative rf strength (dB) to field strength T
pub struct TxSensitivity {
    c:Vec<f64>,
}

impl TxSensitivity {

    pub fn scale(self,rf_scale_tesla:f64) -> Self {
        let c = self.c.into_par_iter().map(|x| x * rf_scale_tesla).collect();
        Self {
            c
        }
    }

    pub fn len(&self) -> usize {
        self.c.len()
    }

    pub fn from_file(file:impl AsRef<Path>) -> TxSensitivity {
        let mut f = File::open(file.as_ref()).unwrap();
        let mut bytes = vec![];
        f.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes.len() % 8, 0);
        let values:&[f64] = cast_slice(&bytes);
        TxSensitivity {
            c: values.to_vec()
        }
    }

    pub fn to_file(&self,file:impl AsRef<Path>) {
        let mut f = File::create(file.as_ref()).unwrap();
        f.write_all(cast_slice(self.c.as_slice())).unwrap();
    }

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

    pub fn len(&self) -> usize {
        self.c.len()
    }

    pub fn from_file(file:impl AsRef<Path>) -> RxSensitivity {
        let mut f = File::open(file.as_ref()).unwrap();
        let mut bytes = vec![];
        f.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes.len() % 8, 0);
        let values:&[f64] = cast_slice(&bytes);
        RxSensitivity {
            c: values.to_vec()
        }
    }

    pub fn to_file(&self,file:impl AsRef<Path>) {
        let mut f = File::create(file.as_ref()).unwrap();
        f.write_all(cast_slice(self.c.as_slice())).unwrap();
    }

    pub fn uniform(n:usize) -> RxSensitivity {
        RxSensitivity {
            c:vec![1.;n],
        }
    }
}

pub fn update(signal:&mut Vec<Complex<f64>>, isochromats:&mut Isochromats, positions:&Positions, tx:&TxSensitivity, rx:&RxSensitivity, offres:&OffResonance, gx:f64, gy:f64, gz:f64, rf_r:f64, rf_i:f64, acq:f64, tau:f64 ) {


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
            let relax_coeffs = compute_relax_coeffs(t1, t2, m0, tau/2.);

            // half-step relax
            apply_relax(
                mx,
                my,
                mz,
                &relax_coeffs
            );

            // full step rotation
            if let Some(rot_coeffs) = compute_rotation_coeffs(x, y, z, gx, gy, gz, bx, by, db0, gamma, tau) {
                apply_rotation(
                    mx,
                    my,
                    mz,
                    &rot_coeffs
                );
            }

            // half-step relax
            apply_relax(
                mx,
                my,
                mz,
                &relax_coeffs
            );

    });

}
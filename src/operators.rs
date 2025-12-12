



/// defines a set of coefficient to rotate the magnetization vector
#[derive(Clone, Copy, Debug)]
pub struct RotCoeffs {
    pub ux: f64,
    pub uy: f64,
    pub uz: f64,
    pub cos_theta: f64,
    pub sin_theta: f64,
    pub one_minus_cos: f64,
}

pub fn compute_rotation_coeffs(
    rx:f64,ry:f64,rz:f64,
    gx:f64,gy:f64,gz:f64,
    bx:f64,by:f64,db0:f64,
    gamma:f64,dt:f64) -> Option<RotCoeffs> {

    // Bz from off-resonance + gradients
    let bz = db0 + gx * rx + gy * ry + gz * rz;

    // magnitude^2 of the field
    let b2 = bx * bx + by * by + bz * bz;
    if b2 == 0.0 {
        // No effective field; no rotation.
        return None;
    }

    // magnitude of field
    let b_mag = b2.sqrt();

    // Rotation angle θ = γ |B_eff| Δt
    let theta = gamma * b_mag * dt;

    // Unit axis u = B_eff / |B_eff|
    let ux = bx / b_mag;
    let uy = by / b_mag;
    let uz = bz / b_mag;

    // sin and cos of the angle
    let (sin_theta, cos_theta) = theta.sin_cos();
    let one_minus_cos = 1.0 - cos_theta;

    Some(RotCoeffs {
        ux,
        uy,
        uz,
        cos_theta,
        sin_theta,
        one_minus_cos,
    })

}

pub fn apply_rotation(mx: &mut f64, my: &mut f64, mz: &mut f64, rot_coeffs: &RotCoeffs) {
    let RotCoeffs {
        ux,
        uy,
        uz,
        cos_theta: ct,
        sin_theta: st,
        one_minus_cos,
    } = *rot_coeffs;

    // Copy original magnetization
    let mx0 = *mx;
    let my0 = *my;
    let mz0 = *mz;

    // u × M
    let cx = uy * mz0 - uz * my0;
    let cy = uz * mx0 - ux * mz0;
    let cz = ux * my0 - uy * mx0;

    // u · M
    let u_dot_m = ux * mx0 + uy * my0 + uz * mz0;

    // Rodrigues' rotation formula:
    // M' = M cosθ + (u × M) sinθ + u (u · M) (1 - cosθ)
    let mx_new = mx0 * ct + cx * st + ux * u_dot_m * one_minus_cos;
    let my_new = my0 * ct + cy * st + uy * u_dot_m * one_minus_cos;
    let mz_new = mz0 * ct + cz * st + uz * u_dot_m * one_minus_cos;

    *mx = mx_new;
    *my = my_new;
    *mz = mz_new;
}


#[derive(Clone, Copy, Debug)]
pub struct RelaxCoeffs {
    pub e1: f64,  // exp(-dt / T1)
    pub e2: f64,  // exp(-dt / T2)
    pub m0: f64,  // equilibrium Mz
}

/// Precompute relaxation coefficients for a given dt, T1, T2, and M0.
pub fn compute_relax_coeffs(t1: f64, t2: f64, m0: f64, dt: f64) -> RelaxCoeffs {
    let e1 = if t1.is_finite() && t1 > 0.0 {
        (-dt / t1).exp()
    } else {
        1.0
    };

    let e2 = if t2.is_finite() && t2 > 0.0 {
        (-dt / t2).exp()
    } else {
        1.0
    };

    RelaxCoeffs { e1, e2, m0 }
}

pub fn apply_relax(
    mx: &mut f64,
    my: &mut f64,
    mz: &mut f64,
    coeffs: &RelaxCoeffs,
) {
    let RelaxCoeffs { e1, e2, m0 } = *coeffs;
    *mx *= e2;
    *my *= e2;
    *mz = m0 + (*mz - m0) * e1;
}

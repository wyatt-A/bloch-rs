
/// returns a rotation matrix for a spin group given a magnetic flux density (T),
/// time step (seconds), and gyromagnetic ratio (rad s^-1 T^-1)
pub fn rotation_operator(bx:f64, by:f64, bz:f64, tau: f64, gamma: f64) -> [f64;9] {
    let norm = (bx*bx + by*by + bz*bz).sqrt();
    let s = 1./norm;
    let nx = bx * s;
    let ny = by * s;
    let nz = bz * s;

    let phi_rad = tau * gamma * norm;
    let cosphi = phi_rad.cos();
    let onemcosphi = 1. - cosphi;
    let sinphi = phi_rad.sin();
    let nxsinphi = nx * sinphi;
    let nysinphi = ny * sinphi;
    let nzsinphi = nz * sinphi;
    let nxnx = nx * nx;
    let nyny = ny * ny;
    let nznz = nz * nz;
    let nxny = nx * ny;
    let nxnz = nx * nz;
    let nynz = ny * nz;

    // column-major rotation matrix
    let rot = [
        nxnx + (1. - nxnx) * cosphi,
        nxny * (onemcosphi) + nzsinphi,
        nxnz * onemcosphi - nysinphi,
        nxny * onemcosphi - nzsinphi,
        nyny + (1. - nyny) * cosphi,
        nynz * onemcosphi + nxsinphi,
        nxnz * onemcosphi + nysinphi,
        nynz * onemcosphi - nxsinphi,
        nznz + (1. - nznz) * cosphi,
    ];

    rot

}

/// calculates the relaxation operator
pub fn relaxation_operator(t1: f64, t2: f64, tau: f64) -> [f64;3] {
    let t_rel = (-tau / t2).exp();
    let l_rel = (-tau / t1).exp();
    [t_rel,t_rel,l_rel]
}

/// applies the relaxation operator to the magnetization vector, returning a new vectosr
pub fn apply_relaxation(mx: &mut f64, my: &mut f64, mz: &mut f64, m0: f64, relaxation_op: &[f64]) {
    let _mz = *mz * relaxation_op[2];
    *mx = *mx * relaxation_op[0];
    *my = *my * relaxation_op[1];
    *mz = _mz + m0 * (1. - relaxation_op[2]);
}

/// Applies rotation matrix to a magnetization vector, returning a new vector. The rotation matrix
/// is assumed to be in col-maj layout
pub fn apply_rotation(mx:&mut f64, my:&mut f64, mz:&mut f64, rotation_op: &[f64]) {

    // matrix-vector multiply

    let _mx = *mx * rotation_op[0] + *my * rotation_op[3] + *mz * rotation_op[6];
    let _my = *mx * rotation_op[1] + *my * rotation_op[4] + *mz * rotation_op[7];
    let _mz = *mx * rotation_op[2] + *my * rotation_op[5] + *mz * rotation_op[8];

    *mx = _mx;
    *my = _my;
    *mz = _mz;

}
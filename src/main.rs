use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use bytemuck::cast_slice;
use clap::Parser;
use bloch_rs::{update, Isochromats, OffResonance, Positions, RxSensitivity, TxSensitivity};

#[derive(Parser)]
struct Args {
    pulse_sequence: PathBuf,
    spins: PathBuf,
    positions: PathBuf,
    off_resonance: PathBuf,
    output: PathBuf,
    rf_scale: Option<f64>,
    tx: Option<PathBuf>,
    rx: Option<PathBuf>,
}


fn main() {

    let args = Args::parse();

    let mut output_file = File::create(&args.output).unwrap();

    let ps = load_pulse_sequence(&args.pulse_sequence);
    println!("loaded {} time points from {}",ps.len() / 7, args.pulse_sequence.display());

    let mut spins = Isochromats::from_file(&args.spins);
    println!("loaded {} spins from {}",spins.len(),args.spins.display());

    let positions = Positions::from_file(&args.positions);
    println!("loaded {} positions from {}",positions.len(),args.positions.display());

    let off_resonance = OffResonance::from_file(&args.off_resonance);
    println!("loaded {} delta B0 values from {}",off_resonance.len(),args.off_resonance.display());

    let tx = args.tx.map(|tx|{
        TxSensitivity::from_file(tx).scale(args.rf_scale.unwrap_or(1.0e-6))
    }).unwrap_or(
        TxSensitivity::uniform(1.,spins.len()).scale(args.rf_scale.unwrap_or(1.0e-6))
    );

    let rx = args.rx.map(|rx|{
        RxSensitivity::from_file(rx)
    }).unwrap_or(
        RxSensitivity::uniform(spins.len())
    );

    let mut signal = vec![];
    let time_steps = ps.chunks_exact(7).collect::<Vec<_>>();
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
            &off_resonance,
            gx,
            gy,
            gz,
            rf_r,
            rf_i,
            acq,
            tau,
        );
    });

    let mut signal_floats = vec![];
    signal.into_iter().for_each(|signal| {
        signal_floats.push(signal.re);
        signal_floats.push(signal.im);
    });
    output_file.write_all(cast_slice(&signal_floats)).unwrap();

}


fn load_pulse_sequence(file:impl AsRef<Path>) -> Vec<f64> {
    let mut f = File::open(file.as_ref()).unwrap();
    let mut file_contents = vec![];
    f.read_to_end(&mut file_contents).unwrap();
    assert_eq!(file_contents.len() % 8, 0, "Length must be multiple of 8");
    // Convert Vec<u8> → &[u8] → &[f64]
    let floats: &[f64] = cast_slice(&file_contents);
    // Copy into a new Vec<f64> (still cheap)
    floats.to_vec()
}
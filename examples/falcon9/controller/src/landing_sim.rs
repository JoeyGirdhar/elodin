//! Landing-burn-in-the-loop tests (WHITEPAPER 11.5, Exercise 3).
//!
//! The full SITL flight is ~470 s of 1000 Hz plant time, so terminal descent
//! regressions only showed up in long Monte Carlo runs. This module flies the
//! real `Fsw::step` from landing-burn ignition to leg contact against a reduced
//! plant that keeps the parts of `sim.py` the burn actually exercises:
//!
//! - engine arming, TEA-TEB charges and the three spool time constants
//!   (`engine_dynamics`), with the center three engines on their last charge;
//! - mass flow from `propulsion.cluster_mdot`;
//! - the 1000 Hz quaternion-error attitude loop executed through rate- and
//!   angle-limited gimbals (`attitude_control`, `tvc_actuators`);
//! - engines-first axial drag with plume dominance (`aero.body_aero_wrench`);
//! - 25 Hz GPS, the radar altimeter, and an ideal IMU;
//! - four-pad contact latched at the first pad to reach the ground.
//!
//! Everything runs in rotating ECEF with the same gravity and fictitious
//! terms the navigator uses, so the FSW sees a consistent world.

use super::*;

const DT: f64 = 0.001;
const GUIDANCE_EVERY: usize = 10; // 100 Hz guidance, as in main.py
const GPS_EVERY: usize = 40; // 25 Hz

// Plant values, mirrored from constants.py / sim.py / aero.py.
const DRY_KG: f64 = 25_600.0;
const OF_RATIO: f64 = 2.33;
const G0: f64 = 9.806_65;
const ISP_VAC_S: f64 = 282.0 * T_VAC_PER_ENGINE / 760_000.0;
const SPINUP_TAU_S: f64 = 1.5;
const SHUTDOWN_TAU_S: f64 = 0.35;
const THROTTLE_TAU_S: f64 = 0.15;
const TVC_MAX_RAD: f64 = 5.0 * std::f64::consts::PI / 180.0;
const TVC_RATE_RADPS: f64 = 20.0 * std::f64::consts::PI / 180.0;
const TVC_TAU_S: f64 = 0.030;
const ATT_WN: f64 = 1.7;
const ATT_ZETA: f64 = 0.9;
const STAGE_LENGTH_M: f64 = 47.0;
const STAGE_RADIUS_M: f64 = 1.83;
const S_REF_M2: f64 = std::f64::consts::PI * 1.83 * 1.83;
const CA_DESCENT_LOW_MACH: f64 = 1.9;
const CA_SCALE: f64 = 0.9574; // calibrated default in main.py
const LEG_RADIUS_M: f64 = 10.0;
const RADAR_FOV_COS: f64 = 0.819_152; // cos(35 deg)
const RADAR_MAX_RANGE_M: f64 = 500.0;
const N_LIT: usize = 3;

/// Transverse inertia about the CG, `propulsion.stack_mass_props` after
/// separation: dry stage as a uniform rod plus the two propellant columns.
fn transverse_inertia(lox: f64, rp1: f64) -> f64 {
    let area = std::f64::consts::PI * STAGE_RADIUS_M * STAGE_RADIUS_M;
    let column = |m: f64, rho: f64, bottom: f64| {
        let len = m / (rho * area);
        (
            bottom + 0.5 * len,
            m * (len * len / 12.0 + STAGE_RADIUS_M * STAGE_RADIUS_M / 4.0),
        )
    };
    let (cg_lox, i_lox) = column(lox, 1220.0, 17.5);
    let (cg_rp1, i_rp1) = column(rp1, 830.0, 3.0);
    let cg = cg_station_m(lox, rp1);
    DRY_KG * STAGE_LENGTH_M * STAGE_LENGTH_M / 12.0
        + DRY_KG * (18.8 - cg).powi(2)
        + i_lox
        + lox * (cg_lox - cg).powi(2)
        + i_rp1
        + rp1 * (cg_rp1 - cg).powi(2)
}

/// Calibrated guidance defaults from main.py (only the fields the landing
/// burn reads matter; the rest keep the packet well formed).
fn default_params(values: &mut [f64; STATE_FLOATS]) {
    let params = [
        6.17, 7.81, 11.74, 0.7105, 18_942.0, 30_000.0, 1_645.1, 47.67, -1_450.0, 1_297.2, 350.0,
        1.273, 0.9969, 35.27, 0.5626, 0.5725, 5_630.0, 49_618.0, 41.44, 1.0,
    ];
    values[22..22 + params.len()].copy_from_slice(&params);
    values[46] = 1.5;
    values[47] = 34.0;
    values[48] = 0.21;
}

/// Initial conditions at landing-burn ignition, in the pad's local frame.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Ignition {
    pub height_m: f64,
    pub vdown_mps: f64,
    /// Horizontal offset from LZ-1 (east, north) in metres.
    pub offset_en_m: [f64; 2],
    /// Horizontal velocity (east, north) in m/s.
    pub vel_en_mps: [f64; 2],
    pub lox_kg: f64,
    pub rp1_kg: f64,
    /// Plant thrust relative to the FSW's model (main.py `thrust_scale`).
    pub thrust_scale: f64,
}

impl Default for Ignition {
    fn default() -> Self {
        Ignition {
            height_m: 1_800.0,
            vdown_mps: 230.0,
            offset_en_m: [40.0, -25.0],
            vel_en_mps: [-4.0, 3.0],
            lox_kg: 5_600.0,
            rp1_kg: 2_400.0,
            thrust_scale: 1.0323,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Contact {
    pub vertical_mps: f64,
    pub lateral_mps: f64,
    pub impact_mps: f64,
    pub tilt_deg: f64,
    pub rate_dps: f64,
    pub pos_err_m: f64,
    pub prop_kg: f64,
    /// An engine went from burning to disarmed before contact.
    pub engine_lost_in_burn: bool,
}

struct Plant {
    t: f64,
    r: V3,
    v: V3,
    q: Quat,
    w_body: V3,
    gimbal: [f64; 2],
    spool: [f64; N_LIT],
    armed: [bool; N_LIT],
    charges: [f64; N_LIT],
    lox: f64,
    rp1: f64,
    thrust_scale: f64,
    gps_count: f64,
    ground_m: f64,
    burned: [bool; N_LIT],
    engine_lost_in_burn: bool,
}

impl Plant {
    fn mass(&self) -> f64 {
        DRY_KG + self.lox + self.rp1
    }

    fn up(&self) -> V3 {
        let (lat, lon, _) = ecef_to_geodetic(self.r);
        scale(ned_basis(lat, lon)[2], -1.0)
    }

    fn i_trans(&self) -> f64 {
        transverse_inertia(self.lox, self.rp1)
    }

    fn ambient_pa(&self) -> f64 {
        let h = ecef_to_geodetic(self.r).2.max(0.0);
        101_325.0 * (-h / 8_434.0).exp()
    }

    /// One 1 ms engine step, `sim.py::engine_dynamics` for the lit engines.
    fn step_engines(&mut self, cmd: &Command) -> (f64, f64) {
        let feed_open = cmd.valves[V_MAIN_LOX] > 0.5 && cmd.valves[V_MAIN_RP1] > 0.5;
        let teateb_open = cmd.valves[V_TEATEB] > 0.5;
        let prop_ok = self.lox > 0.0 && self.rp1 > 0.0;
        let p_amb = self.ambient_pa();
        let (mut thrust, mut mdot) = (0.0, 0.0);
        for i in 0..N_LIT {
            let u = cmd.engines[i].clamp(0.0, 1.0);
            let on = u >= THROTTLE_MIN * 0.5;
            let lighting = on
                && !self.armed[i]
                && self.charges[i] >= 1.0
                && feed_open
                && teateb_open
                && prop_ok;
            if lighting {
                self.charges[i] -= 1.0;
            }
            self.armed[i] = on && (self.armed[i] || lighting);
            let target = if self.armed[i] && feed_open && prop_ok {
                u.max(THROTTLE_MIN)
            } else {
                0.0
            };
            let running = self.spool[i] > 0.5 * THROTTLE_MIN;
            let tau = if target > self.spool[i] {
                if running {
                    THROTTLE_TAU_S
                } else {
                    SPINUP_TAU_S
                }
            } else {
                SHUTDOWN_TAU_S
            };
            self.spool[i] += (target - self.spool[i]) * (1.0 - (-DT / tau).exp());
            if self.spool[i] > 1e-3 {
                self.burned[i] |= self.spool[i] > THROTTLE_MIN * 0.9;
                thrust += (self.spool[i] * T_VAC_PER_ENGINE - p_amb * A_E_M2).max(0.0)
                    * self.thrust_scale;
                mdot += self.spool[i] * T_VAC_PER_ENGINE / (ISP_VAC_S * G0) * self.thrust_scale
                    / 1.0215;
            }
        }
        (thrust, mdot)
    }

    /// `sim.py::attitude_control` + `tvc_actuators`: PD torque executed
    /// through gimbal angle/rate limits and a first-order actuator lag.
    fn step_attitude(&mut self, setpoint: Quat, thrust: f64, cg: f64) -> V3 {
        let q_err = quat_mul(quat_conj(self.q), setpoint);
        let sign = if q_err[3] >= 0.0 { 1.0 } else { -1.0 };
        let err = [sign * q_err[0], sign * q_err[1], sign * q_err[2]];
        let torque_des: V3 = std::array::from_fn(|k| {
            self.i_trans() * (ATT_WN * ATT_WN * err[k] - 2.0 * ATT_ZETA * ATT_WN * self.w_body[k])
        });
        let lever = (cg * thrust).max(1.0);
        let cmd = if thrust > 2.0e5 {
            [-torque_des[1] / lever, -torque_des[2] / lever]
        } else {
            [0.0, 0.0]
        };
        for (k, c) in cmd.iter().enumerate() {
            let target = c.clamp(-TVC_MAX_RAD, TVC_MAX_RAD);
            let step = (target - self.gimbal[k]) * (1.0 - (-DT / TVC_TAU_S).exp());
            self.gimbal[k] += step.clamp(-TVC_RATE_RADPS * DT, TVC_RATE_RADPS * DT);
        }
        // Engine force in body axes; tau_y = -cg T dp, tau_z = -cg T dy.
        [thrust, thrust * self.gimbal[1], -thrust * self.gimbal[0]]
    }

    fn pad_heights(&self) -> [f64; 4] {
        std::array::from_fn(|i| {
            let a = (i as f64 + 0.5) * 0.5 * std::f64::consts::PI;
            let pad = quat_rotate(
                self.q,
                [0.0, LEG_RADIUS_M * a.cos(), LEG_RADIUS_M * a.sin()],
            );
            ecef_to_geodetic(add(self.r, pad)).2 - self.ground_m
        })
    }

    fn sensors(&self, f_body: V3, values: &mut [f64; STATE_FLOATS]) {
        values[0] = self.t;
        values[1..4].copy_from_slice(&f_body);
        let earth_rate = quat_rotate_inv(self.q, [0.0, 0.0, OMEGA_EARTH]);
        values[4..7].copy_from_slice(&add(self.w_body, earth_rate));
        values[13] = self.gps_count;
        let up = self.up();
        let cos_tilt = dot(quat_rotate(self.q, [1.0, 0.0, 0.0]), up);
        let h = ecef_to_geodetic(self.r).2 - self.ground_m;
        let slant = h / cos_tilt.max(1e-3);
        values[14] = if cos_tilt > RADAR_FOV_COS && slant <= RADAR_MAX_RANGE_M {
            slant
        } else {
            -1.0
        };
        values[20] = self.lox;
        values[21] = self.rp1;
    }
}

/// Fly the landing burn from `ign` until the first pad touches the ground.
pub(crate) fn fly_landing_burn(ign: Ignition) -> Contact {
    let mut fsw = Fsw::new("photo3d").expect("photo3d site data");
    let ground_m = fsw.nav.ground.landing_m;
    let (lat, lon, _) = ecef_to_geodetic(fsw.lz1_pos);
    let ned = ned_basis(lat, lon);
    let (north, east, up) = (ned[0], ned[1], scale(ned[2], -1.0));
    let horiz = |en: [f64; 2]| add(scale(east, en[0]), scale(north, en[1]));

    let r0 = add(
        add(fsw.lz1_pos, horiz(ign.offset_en_m)),
        scale(up, ign.height_m),
    );
    let v0 = add(horiz(ign.vel_en_mps), scale(up, -ign.vdown_mps));
    let mut plant = Plant {
        t: 0.0,
        r: r0,
        v: v0,
        q: quat_between([1.0, 0.0, 0.0], scale(normalize(v0), -1.0)),
        w_body: [0.0; 3],
        gimbal: [0.0; 2],
        spool: [0.0; N_LIT],
        armed: [false; N_LIT],
        charges: [1.0; N_LIT], // the fourth and last TEA-TEB charge
        lox: ign.lox_kg,
        rp1: ign.rp1_kg,
        thrust_scale: ign.thrust_scale,
        gps_count: 1.0,
        ground_m,
        burned: [false; N_LIT],
        engine_lost_in_burn: false,
    };

    // FSW at the AeroDescent -> LandingBurn handoff with a converged navigator.
    fsw.phase = Phase::LandingBurn;
    fsw.landing_escalated = true;
    fsw.landing_thrust_est_after = THRUST_EST_SETTLE_IGNITION_S;
    fsw.nav.pos = plant.r;
    fsw.nav.vel = plant.v;
    fsw.nav.att = plant.q;
    fsw.nav.last_gps_count = plant.gps_count;
    fsw.nav.initialized = true;
    fsw.att_cmd = plant.q;
    fsw.att_cmd_init = true;

    let mut values = [0.0; STATE_FLOATS];
    default_params(&mut values);
    let mut cmd = Command::default();
    let mut f_body = [0.0; 3];
    for tick in 0..120_000usize {
        if tick % GPS_EVERY == 0 && tick > 0 {
            plant.gps_count += 1.0;
            values[7..10].copy_from_slice(&plant.r);
            values[10..13].copy_from_slice(&plant.v);
        }
        if tick % GUIDANCE_EVERY == 0 {
            if tick == 0 {
                values[7..10].copy_from_slice(&plant.r);
                values[10..13].copy_from_slice(&plant.v);
            }
            plant.sensors(f_body, &mut values);
            fsw.nav.last_t = if tick == 0 { plant.t } else { fsw.nav.last_t };
            cmd = fsw.step(&SensorPacket::parse(&values));
            // The center engine shut down while the FSW still thinks it is
            // flying the burn: the stop-short guard fired.
            if fsw.phase == Phase::LandingBurn && plant.burned[0] && cmd.engines[0] == 0.0 {
                plant.engine_lost_in_burn = true;
            }
        }

        let (thrust, mdot) = plant.step_engines(&cmd);
        let cg = cg_station_m(plant.lox, plant.rp1);
        let engine_body = plant.step_attitude(cmd.attitude, thrust, cg);

        // Engines-first axial drag with plume dominance (aero.py).
        let speed = norm(plant.v);
        let qbar = 0.5 * density(ecef_to_geodetic(plant.r).2) * speed * speed;
        let c_t = thrust / (qbar * S_REF_M2).max(1.0);
        let kappa = c_t / (c_t + 1.0);
        let drag_mag = qbar * S_REF_M2 * CA_DESCENT_LOW_MACH * CA_SCALE * (1.0 - kappa);
        let drag_ecef = scale(normalize(plant.v), -drag_mag);

        let m = plant.mass();
        let force_ecef = add(quat_rotate(plant.q, engine_body), drag_ecef);
        f_body = scale(quat_rotate_inv(plant.q, force_ecef), 1.0 / m);
        let accel = add(
            add(scale(force_ecef, 1.0 / m), gravity(plant.r)),
            frame_accel(plant.r, plant.v),
        );

        // Pitch/yaw torque from the gimbaled engine below the CG.
        let torque = [0.0, cg * engine_body[2], -cg * engine_body[1]];
        let alpha = scale(torque, 1.0 / plant.i_trans());
        plant.w_body = add(plant.w_body, scale(alpha, DT));
        plant.w_body[0] = 0.0; // RCS holds roll
        plant.q = quat_integrate(plant.q, plant.w_body, DT);
        plant.v = add(plant.v, scale(accel, DT));
        plant.r = add(plant.r, scale(plant.v, DT));
        let burned = mdot * DT;
        plant.lox = (plant.lox - burned * OF_RATIO / (1.0 + OF_RATIO)).max(0.0);
        plant.rp1 = (plant.rp1 - burned / (1.0 + OF_RATIO)).max(0.0);
        plant.t += DT;

        if plant.pad_heights().iter().any(|h| *h <= 0.0) {
            let up = plant.up();
            let v_up = dot(plant.v, up);
            let lateral = norm(sub(plant.v, scale(up, v_up)));
            let d = sub(plant.r, fsw.lz1_pos);
            let miss = norm(sub(d, scale(up, dot(d, up))));
            let tilt = dot(quat_rotate(plant.q, [1.0, 0.0, 0.0]), up)
                .clamp(-1.0, 1.0)
                .acos();
            return Contact {
                vertical_mps: v_up.abs(),
                lateral_mps: lateral,
                impact_mps: norm(plant.v),
                tilt_deg: tilt.to_degrees(),
                rate_dps: norm(plant.w_body).to_degrees(),
                pos_err_m: miss,
                prop_kg: plant.lox + plant.rp1,
                engine_lost_in_burn: plant.engine_lost_in_burn,
            };
        }
    }
    panic!("no ground contact within 120 s of landing-burn ignition");
}

/// Soft-landing gates, mirrored from hooks/mc_metrics.py.
fn assert_soft(c: &Contact, ign: &Ignition) {
    let gates = [
        ("vertical speed (m/s)", c.vertical_mps, 2.0),
        ("impact speed (m/s)", c.impact_mps, 2.0),
        ("lateral speed (m/s)", c.lateral_mps, 1.5),
        ("tilt (deg)", c.tilt_deg, 2.0),
        ("body rate (deg/s)", c.rate_dps, 1.0),
        ("miss (m)", c.pos_err_m, 5.0),
    ];
    for (gate, value, limit) in gates {
        assert!(
            value <= limit,
            "{gate} {value:.2} over {limit} from {ign:?}"
        );
    }
    assert!(c.prop_kg > 0.0, "propellant ran out from {ign:?}");
    assert!(
        !c.engine_lost_in_burn,
        "center engine shut down before contact from {ign:?}"
    );
}

/// Deterministic dispersion set around the nominal ignition state: ignition
/// height and speed, a divert left over from the aero descent, propellant
/// load, and the campaign's thrust uncertainty.
fn dispersed_ignitions(n: usize) -> Vec<Ignition> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut uniform = move |lo: f64, hi: f64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        lo + (hi - lo) * ((state >> 11) as f64 / (1u64 << 53) as f64)
    };
    (0..n)
        .map(|_| Ignition {
            height_m: uniform(1_500.0, 2_200.0),
            vdown_mps: uniform(200.0, 260.0),
            offset_en_m: [uniform(-80.0, 80.0), uniform(-80.0, 80.0)],
            vel_en_mps: [uniform(-8.0, 8.0), uniform(-8.0, 8.0)],
            lox_kg: uniform(4_500.0, 7_000.0),
            rp1_kg: uniform(1_900.0, 3_000.0),
            thrust_scale: uniform(0.94, 1.14),
        })
        .collect()
}

#[test]
fn nominal_burn_touches_down_at_the_design_speed() {
    let ign = Ignition::default();
    let c = fly_landing_burn(ign);
    assert_soft(&c, &ign);
    assert!(
        (c.vertical_mps - ZEM_V_TD_MPS).abs() < 0.2,
        "touched down at {:.2} m/s",
        c.vertical_mps
    );
    assert!(c.pos_err_m < 1.0, "missed LZ-1 by {:.2} m", c.pos_err_m);
}

/// WHITEPAPER Exercise 3. Across the campaign's whole `thrust_scale` range
/// the burn must neither stop short on hot engines nor arrive fast on weak
/// ones; the in-flight thrust estimate should make touchdown speed nearly
/// independent of it.
#[test]
fn thrust_dispersion_barely_moves_touchdown_speed() {
    let speeds: Vec<f64> = [0.94, 0.97, 1.0, 1.0323, 1.07, 1.10, 1.14]
        .into_iter()
        .map(|thrust_scale| {
            let ign = Ignition {
                thrust_scale,
                ..Ignition::default()
            };
            let c = fly_landing_burn(ign);
            assert_soft(&c, &ign);
            c.vertical_mps
        })
        .collect();
    let spread = speeds.iter().cloned().fold(f64::MIN, f64::max)
        - speeds.iter().cloned().fold(f64::MAX, f64::min);
    assert!(spread < 0.3, "touchdown speeds {speeds:.2?}");
}

#[test]
fn dispersed_ignitions_all_land_soft() {
    for ign in dispersed_ignitions(100) {
        assert_soft(&fly_landing_burn(ign), &ign);
    }
}

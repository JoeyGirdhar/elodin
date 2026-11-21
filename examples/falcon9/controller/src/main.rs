//! Falcon 9 booster flight software (WHITEPAPER 11).
//!
//! A standalone process: reads a sensor packet over UDP, runs estimation,
//! phase management, and guidance, and replies with actuator/valve commands.
//! It never sees truth state — only the simulated sensors (WHITEPAPER 12).

#[cfg(test)]
mod landing_sim;
mod math;
mod profile;

use math::*;
use profile::AscentProfile;
use std::net::UdpSocket;

const STATE_FLOATS: usize = 49;
const CMD_FLOATS: usize = 27;
const N_ENGINES: usize = 9;
const N_VALVES: usize = 8;
/// Max attitude-setpoint slew (rad/s) — smooths AeroDescent→LandingBurn handoff.
/// Fast enough that a ~20° ignition step settles in <1 s (avoids a long AoA spike).
const ATT_SLEW_RADPS: f64 = 0.70;
/// Fin effectiveness priors matching aero.py (EST).
const FIN_S_M2: f64 = 1.5;
const FIN_CN_DELTA: f64 = 1.2;
const FIN_LEVER_M: f64 = 22.0;
const FIN_I_TRANS: f64 = 1.5e7; // kg·m² pitch/yaw inertia proxy at landing mass

// Valve indices (mirror sim.py).
const V_HE_LOX: usize = 0;
const V_HE_RP1: usize = 2;
const V_MAIN_LOX: usize = 4;
const V_MAIN_RP1: usize = 5;
const V_TEATEB: usize = 6;
const V_PURGE: usize = 7;

const THROTTLE_MIN: f64 = 0.57;
const T_VAC_PER_ENGINE: f64 = 829_000.0; // matches plant constants
const A_E_M2: f64 = 0.681;
const P_SL_PA: f64 = 101_325.0;
const DRY_MASS_KG: f64 = 25_600.0;
/// ZEM/ZEV terminal guidance (Guo/Hawkins/Wie) — LandingBurn.
const ZEM_V_TD_MPS: f64 = 1.2;
const ZEM_TILT_CAP_RAD: f64 = 0.25;
/// Merlin shutdown time constant (plant ENGINE_SHUTDOWN_TAU_S, EST). An
/// engine commanded off keeps pushing for ~T·τ of impulse while it spools
/// down, which the 3 -> 1 handover has to count.
const MERLIN_SHUTDOWN_TAU_S: f64 = 0.35;
/// Where in the single-engine deceleration band the handover aims (0 = min
/// throttle, 1 = full). The middle leaves room for thrust dispersion and
/// tracking error in both directions.
const SINGLE_ENGINE_BAND_AIM: f64 = 0.5;
/// Below this height the landing burn stops steering for position and only
/// nulls horizontal velocity (Apollo P66-style terminal phase).
const TERMINAL_GATE_M: f64 = 30.0;
/// Lateral velocity loop time constant, slow enough for the gimbal-limited
/// attitude response.
const LATERAL_VEL_TAU_S: f64 = 1.5;
/// Gimbal-limited attitude response at landing mass (~1.5 s): the lateral
/// loops lead by the drift the current lean will add meanwhile.
const ATTITUDE_LAG_S: f64 = 1.5;
/// Leg pads sit this far off the centerline at the engine plane (plant
/// LEG_RADIUS_M), so a leaning booster touches down on its low pad first.
const LEG_PAD_RADIUS_M: f64 = 10.0;
/// Thrust tilt allowance: full divert authority high up, tapering to a small
/// trim allowance at the terminal gate and to near-vertical at touchdown.
const TILT_TAPER_START_M: f64 = 150.0;
/// Share of the tilt-limited lateral acceleration the approach may plan to
/// brake with. The rest is headroom for the attitude lag and for wind.
const DIVERT_BRAKE_MARGIN: f64 = 0.6;
const TERMINAL_TILT_CAP_RAD: f64 = 0.07;
const TOUCHDOWN_TILT_CAP_RAD: f64 = 0.02;
/// Merlin throttle response once running (plant ENGINE_THROTTLE_TAU_S, EST).
const MERLIN_THROTTLE_TAU_S: f64 = 0.15;
/// In-burn thrust estimate: low-pass time constant and the settling time
/// after a spool-up or an engine shutdown before the accelerometer is
/// trusted to reflect commanded thrust (plant spin-up τ = 1.5 s, shutdown
/// τ = 0.35 s).
const THRUST_EST_TAU_S: f64 = 1.0;
const THRUST_EST_SETTLE_IGNITION_S: f64 = 4.0;
const THRUST_EST_SETTLE_HANDOVER_S: f64 = 1.5;
const GUIDANCE_DT_S: f64 = 0.01;

/// Shared with the plant through constants.py; heights are WGS84 ellipsoidal.
#[derive(Clone, Copy, serde::Deserialize)]
struct GroundHeights {
    launch_m: f64,
    landing_m: f64,
}

#[derive(serde::Deserialize)]
struct SiteCoordinates {
    latitude_degrees: f64,
    longitude_degrees: f64,
}

#[derive(serde::Deserialize)]
struct Sites {
    landing: SiteCoordinates,
    legacy: GroundHeights,
    aerial: GroundHeights,
    photo3d: GroundHeights,
}

impl Sites {
    fn load() -> Self {
        serde_json::from_str(include_str!("../../data/sites.json")).expect("valid Falcon 9 sites")
    }
}

impl GroundHeights {
    fn load(sites: &Sites, mode: &str) -> std::io::Result<Self> {
        let heights = match mode {
            "legacy" => sites.legacy,
            "aerial" => sites.aerial,
            "photo3d" => sites.photo3d,
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "terrain mode must be legacy, aerial, or photo3d",
                ));
            }
        };
        if !heights.launch_m.is_finite() || !heights.landing_m.is_finite() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid site heights",
            ));
        }
        Ok(heights)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Phase {
    PadPress = 0,
    VerticalRise = 1,
    PitchKick = 2,
    GravityTurn = 3,
    Meco = 4,
    Flip = 5,
    Boostback = 6,
    Coast = 7,
    EntryBurn = 8,
    AeroDescent = 9,
    LandingBurn = 10,
    Touchdown = 11,
}

/// Sensor packet layout (sim -> FSW), little-endian f64.
struct SensorPacket {
    t: f64,
    imu_accel: V3,
    imu_gyro: V3,
    gps_pos: V3,
    gps_vel: V3,
    gps_count: f64,
    /// Terminal-descent refinement (Phase 7); -1 when invalid.
    #[allow(dead_code)]
    radar_range: f64,
    /// Tank/inlet monitoring (fault logic is a stretch goal).
    #[allow(dead_code)]
    pressures: [f64; 4],
    lox_kg: f64,
    rp1_kg: f64,
    params: GuidanceParams,
    landed: f64,
}

/// Campaign-tunable guidance parameters, sent each exchange (apollo pattern).
#[derive(Clone, Copy, Debug)]
struct GuidanceParams {
    kick_deg: f64,
    kick_start_s: f64,
    kick_ramp_s: f64,
    bucket_throttle: f64,
    bucket_q_on_pa: f64,
    /// Reserved for a predictor-based bucket exit (packet layout stability).
    #[allow(dead_code)]
    bucket_q_off_pa: f64,
    meco_speed_mps: f64,
    azimuth_deg: f64,
    boostback_overshoot_m: f64,
    entry_ignite_speed_mps: f64,
    entry_dv_mps: f64,
    landing_accel_margin: f64,
    ascent_throttle: f64,
    /// Flight-path angle from horizontal commanded at MECO speed (deg).
    meco_fpa_deg: f64,
    /// Pitch-program shape exponent in speed space.
    pitch_exp: f64,
    /// Entry-burn engine throttle (the recorded 26 m/s^2 mean decel implies
    /// a throttled three-engine burn).
    entry_throttle: f64,
    /// The hoverslam solve only arms below this altitude — above it the
    /// aerodynamic descent does the braking (WHITEPAPER 11.4-11.5).
    landing_arm_alt_m: f64,
    /// Entry-burn ignition altitude gate (recorded: ~49.8 km).
    entry_ignite_alt_m: f64,
    /// FSW onboard aero database: engines-first CA * S_ref (m^2) used by the
    /// impact-point predictor. A real navigator carries a calibrated drag
    /// model; the campaign calibrates ours.
    fsw_cd_s_m2: f64,
    /// Boostback three-engine throttle (the recorded burn decelerates at
    /// ~27 m/s^2 — a throttled burn, not full thrust).
    boostback_throttle: f64,
    /// Fin attitude-loop natural frequency (rad/s); torque is divided by
    /// q̄·effectiveness so closed-loop bandwidth is q̄-invariant.
    fin_wn: f64,
    /// Landing-burn lateral approach-speed cap (m/s); retained for MC packet
    /// layout (ZEM/ZEV no longer reads it).
    #[allow(dead_code)]
    divert_speed_cap: f64,
    /// Aero-descent cross-track tilt cap (rad) at low q̄; scheduled down as q̄ rises.
    steer_tilt_cap: f64,
}

impl SensorPacket {
    fn parse(v: &[f64; STATE_FLOATS]) -> Self {
        SensorPacket {
            t: v[0],
            imu_accel: [v[1], v[2], v[3]],
            imu_gyro: [v[4], v[5], v[6]],
            gps_pos: [v[7], v[8], v[9]],
            gps_vel: [v[10], v[11], v[12]],
            gps_count: v[13],
            radar_range: v[14],
            pressures: [v[16], v[17], v[18], v[19]],
            lox_kg: v[20],
            rp1_kg: v[21],
            params: GuidanceParams {
                kick_deg: v[22],
                kick_start_s: v[23],
                kick_ramp_s: v[24],
                bucket_throttle: v[25],
                bucket_q_on_pa: v[26],
                bucket_q_off_pa: v[27],
                meco_speed_mps: v[28],
                azimuth_deg: v[29],
                boostback_overshoot_m: v[30],
                entry_ignite_speed_mps: v[31],
                entry_dv_mps: v[32],
                landing_accel_margin: v[33],
                ascent_throttle: v[34],
                meco_fpa_deg: v[35],
                pitch_exp: v[36],
                entry_throttle: v[37],
                landing_arm_alt_m: v[38],
                entry_ignite_alt_m: v[39],
                fsw_cd_s_m2: v[40],
                boostback_throttle: v[41],
                fin_wn: if v[46] > 0.1 { v[46] } else { 1.4 },
                divert_speed_cap: if v[47] > 1.0 { v[47] } else { 35.0 },
                steer_tilt_cap: if v[48] > 0.01 { v[48] } else { 0.15 },
            },
            landed: v[43],
        }
    }
}

#[derive(Default)]
struct Command {
    engines: [f64; N_ENGINES],
    valves: [f64; N_VALVES],
    attitude: Quat,
    tvc_enable: f64,
    rcs_enable: f64,
    fins: [f64; 3],
    phase: f64,
}

impl Command {
    fn pack(&self) -> [f64; CMD_FLOATS] {
        let mut out = [0.0; CMD_FLOATS];
        out[..9].copy_from_slice(&self.engines);
        out[9..17].copy_from_slice(&self.valves);
        out[17..21].copy_from_slice(&self.attitude);
        out[21] = self.tvc_enable;
        out[22] = self.rcs_enable;
        out[23..26].copy_from_slice(&self.fins);
        out[26] = self.phase;
        out
    }
}

/// CoM station from the engine-plane antenna (plant `stack_mass_props`, m_upper=0).
fn cg_station_m(lox_kg: f64, rp1_kg: f64) -> f64 {
    let area = std::f64::consts::PI * 1.83 * 1.83;
    let cg_lox = 17.5 + 0.5 * lox_kg / (1220.0 * area);
    let cg_rp1 = 3.0 + 0.5 * rp1_kg / (830.0 * area);
    (25_600.0 * 18.8 + lox_kg * cg_lox + rp1_kg * cg_rp1) / (25_600.0 + lox_kg + rp1_kg)
}

/// IMU + GPS navigator with the rotating-frame mechanization (WHITEPAPER 12).
struct Navigator {
    ground: GroundHeights,
    landing_pos: V3,
    pos: V3,
    vel: V3,
    att: Quat, // body -> ECEF
    last_gps_count: f64,
    last_t: f64,
    initialized: bool,
    /// Steady wind estimate (ECEF m/s) from GPS–IMU innovation.
    wind_est: V3,
    /// Radar-smoothed antenna height above terrain (m); <0 when invalid.
    radar_agl_m: f64,
    /// Last radar sample folded in (the bridge holds it between 40 Hz updates).
    last_radar_range: f64,
    /// CoM above the engine-plane antenna (m), from current propellant.
    station_m: f64,
}

impl Navigator {
    fn new(ground: GroundHeights, landing_pos: V3) -> Self {
        Navigator {
            ground,
            landing_pos,
            pos: [0.0; 3],
            vel: [0.0; 3],
            att: QUAT_IDENT,
            last_gps_count: 0.0,
            last_t: 0.0,
            initialized: false,
            wind_est: [0.0; 3],
            radar_agl_m: -1.0,
            last_radar_range: -1.0,
            station_m: 18.8,
        }
    }

    fn site_alt(&self) -> f64 {
        if norm(sub(self.pos, self.landing_pos)) < 5_000.0 {
            self.ground.landing_m
        } else {
            self.ground.launch_m
        }
    }

    fn init(&mut self, s: &SensorPacket) {
        // Known pad state: GPS position, zero ECEF velocity, body +X up.
        self.pos = s.gps_pos;
        self.vel = [0.0; 3];
        let (lat, lon, _) = ecef_to_geodetic(self.pos);
        let up = scale(ned_basis(lat, lon)[2], -1.0);
        self.att = quat_between([1.0, 0.0, 0.0], up);
        self.last_gps_count = s.gps_count;
        self.last_t = s.t;
        self.initialized = true;
        self.wind_est = [0.0; 3];
        self.radar_agl_m = -1.0;
        self.last_radar_range = -1.0;
        self.station_m = cg_station_m(s.lox_kg, s.rp1_kg);
    }

    fn step(&mut self, s: &SensorPacket) {
        self.station_m = cg_station_m(s.lox_kg, s.rp1_kg);
        if !self.initialized {
            if s.gps_count > 0.0 {
                self.init(s);
            }
            return;
        }
        let dt = (s.t - self.last_t).clamp(0.0, 0.1);
        self.last_t = s.t;
        // Attitude: integrate the gyro, removing the Earth-rate term the
        // gyro measures but the ECEF-referenced attitude must not integrate.
        let omega_e_body = quat_rotate_inv(self.att, [0.0, 0.0, OMEGA_EARTH]);
        let omega_frame = sub(s.imu_gyro, omega_e_body);
        self.att = quat_integrate(self.att, omega_frame, dt);
        // Translation: specific force + gravity + fictitious terms.
        let f_e = quat_rotate(self.att, s.imu_accel);
        let a = add(add(f_e, gravity(self.pos)), frame_accel(self.pos, self.vel));
        self.vel = add(self.vel, scale(a, dt));
        self.pos = add(self.pos, scale(self.vel, dt));
        // Complementary GPS blend (no snap) — kills metre-class steps into
        // terminal guidance. Position gain ~0.2/fix, velocity ~0.5/fix.
        if s.gps_count > self.last_gps_count {
            let innov_v = sub(s.gps_vel, self.vel);
            self.wind_est = add(scale(self.wind_est, 0.95), scale(innov_v, 0.05));
            self.pos = add(self.pos, scale(sub(s.gps_pos, self.pos), 0.20));
            self.vel = add(self.vel, scale(sub(s.gps_vel, self.vel), 0.50));
            self.last_gps_count = s.gps_count;
        }
        // Radar altimeter below 500 m: slant → antenna AGL, then reconstruct CoM.
        if s.radar_range >= 0.0 && s.radar_range < 500.0 {
            let (lat, lon, geo_alt) = ecef_to_geodetic(self.pos);
            let up = scale(ned_basis(lat, lon)[2], -1.0);
            // Carry the height forward with the navigated vertical rate and
            // pull it toward each fresh sample. A plain moving average of the
            // 40 Hz radar trails a descending booster by tens of
            // milliseconds, which near the pad is decimetres of height the
            // landing burn never sees.
            if self.radar_agl_m >= 0.0 {
                self.radar_agl_m += dot(self.vel, up) * dt;
            }
            if s.radar_range != self.last_radar_range {
                let body_x = quat_rotate(self.att, [1.0, 0.0, 0.0]);
                let cos_tilt = dot(body_x, up).max(0.6);
                let h_agl = s.radar_range * cos_tilt;
                if self.radar_agl_m < 0.0 {
                    self.radar_agl_m = h_agl;
                } else {
                    self.radar_agl_m += 0.3 * (h_agl - self.radar_agl_m);
                }
                self.last_radar_range = s.radar_range;
            }
            let dh = (self.radar_agl_m + self.station_m + self.site_alt()) - geo_alt;
            if dh.abs() < 50.0 {
                self.pos = add(self.pos, scale(up, 0.35 * dh));
            }
        } else {
            self.radar_agl_m = -1.0;
            self.last_radar_range = -1.0;
        }
    }

    fn altitude(&self) -> f64 {
        if self.radar_agl_m >= 0.0 {
            self.radar_agl_m + self.station_m + self.site_alt()
        } else {
            ecef_to_geodetic(self.pos).2
        }
    }

    fn height_agl(&self) -> f64 {
        if self.radar_agl_m >= 0.0 {
            self.radar_agl_m
        } else {
            ecef_to_geodetic(self.pos).2 - self.station_m - self.site_alt()
        }
    }

    fn speed(&self) -> f64 {
        norm(self.vel)
    }
}

struct Fsw {
    nav: Navigator,
    phase: Phase,
    phase_t0: f64,
    meco_speed_reached: bool,
    purge_until: f64,
    entry_v0: f64,
    pad_pos: V3,
    lz1_pos: V3,
    up_pad: V3,
    track_dir: V3,
    /// 1-3-1 landing-burn engine latch: (escalated to 3, back down to 1).
    landing_escalated: bool,
    landing_deescalated: bool,
    /// Last landing-burn throttle command, for the handover tail-off estimate.
    landing_throttle: f64,
    /// Throttle the engines are actually at: the command through the Merlin
    /// throttle lag. The thrust estimate compares against this, not the
    /// command, so throttle motion does not read as a thrust error.
    landing_spool: f64,
    /// Measured / modelled thrust during the landing burn. The plant's
    /// engines are dispersed from the FSW model (main.py `thrust_scale`);
    /// without this the burn over-brakes, the hoverslam solution drifts below
    /// min throttle, and the booster stops above the pad.
    landing_thrust_ratio: f64,
    /// Earliest time the accelerometer reflects the current engine set.
    landing_thrust_est_after: f64,
    /// The designed ascent reference: the recorded webcast profile.
    ascent_profile: Option<AscentProfile>,
    t_liftoff: f64,
    /// Previous attitude command (for slew limiting).
    att_cmd: Quat,
    att_cmd_init: bool,
}

impl Fsw {
    fn new(mode: &str) -> std::io::Result<Self> {
        let sites = Sites::load();
        let ground = GroundHeights::load(&sites, mode)?;
        let lz1 = geodetic_to_ecef(
            sites.landing.latitude_degrees.to_radians(),
            sites.landing.longitude_degrees.to_radians(),
            ground.landing_m,
        );
        Ok(Fsw {
            nav: Navigator::new(ground, lz1),
            phase: Phase::PadPress,
            phase_t0: 0.0,
            meco_speed_reached: false,
            purge_until: -1.0,
            entry_v0: 0.0,
            pad_pos: [0.0; 3],
            lz1_pos: lz1,
            up_pad: [0.0, 0.0, 1.0],
            track_dir: [1.0, 0.0, 0.0],
            landing_escalated: false,
            landing_deescalated: false,
            landing_throttle: THROTTLE_MIN,
            landing_spool: THROTTLE_MIN,
            landing_thrust_ratio: 1.0,
            landing_thrust_est_after: f64::INFINITY,
            ascent_profile: std::env::var("ELODIN_F9_PROFILE")
                .ok()
                .and_then(|p| AscentProfile::load(&p)),
            t_liftoff: -1.0,
            att_cmd: QUAT_IDENT,
            att_cmd_init: false,
        })
    }

    /// Slew-limit the attitude setpoint toward `desired` (max ATT_SLEW_RADPS).
    fn slew_attitude(&mut self, desired: Quat, dt: f64) -> Quat {
        if !self.att_cmd_init {
            self.att_cmd = desired;
            self.att_cmd_init = true;
            return desired;
        }
        // Angle of the relative quaternion q_err = conj(prev) * desired.
        let q_err = quat_mul(quat_conj(self.att_cmd), desired);
        let (axis, angle) = quat_to_axis_angle(q_err);
        let max_step = ATT_SLEW_RADPS * dt.max(1e-3);
        if angle <= max_step {
            self.att_cmd = desired;
        } else {
            self.att_cmd = quat_mul(self.att_cmd, quat_from_axis_angle(axis, max_step));
            self.att_cmd = quat_normalize(self.att_cmd);
        }
        self.att_cmd
    }

    fn set_phase(&mut self, p: Phase, t: f64) {
        if p != self.phase {
            eprintln!("[fsw] t={t:8.2}s  {:?} -> {p:?}", self.phase);
            self.phase = p;
            self.phase_t0 = t;
        }
    }

    fn cutoff_with_purge(&mut self, cmd: &mut Command, t: f64) {
        cmd.engines = [0.0; N_ENGINES];
        self.purge_until = t + 5.0;
    }

    fn step(&mut self, s: &SensorPacket) -> Command {
        self.nav.step(s);
        let mut cmd = Command {
            attitude: self.nav.att,
            phase: self.phase as i32 as f64,
            ..Default::default()
        };
        // Helium pressurization runs the whole flight; purge per schedule.
        cmd.valves[V_HE_LOX] = 1.0;
        cmd.valves[V_HE_RP1] = 1.0;
        cmd.valves[V_PURGE] = if s.t < self.purge_until { 1.0 } else { 0.0 };

        if !self.nav.initialized {
            return cmd;
        }
        let p = &s.params;
        let t = s.t;
        let alt = self.nav.altitude();
        let h_agl = self.nav.height_agl();
        let speed = self.nav.speed();

        // Common frames.
        if self.phase == Phase::PadPress && self.pad_pos == [0.0; 3] {
            self.pad_pos = self.nav.pos;
            let (lat, lon, _) = ecef_to_geodetic(self.pad_pos);
            let ned = ned_basis(lat, lon);
            self.up_pad = scale(ned[2], -1.0);
            let az = p.azimuth_deg.to_radians();
            self.track_dir = normalize(add(scale(ned[0], az.cos()), scale(ned[1], az.sin())));
        }
        let up_here = {
            let (lat, lon, _) = ecef_to_geodetic(self.nav.pos);
            scale(ned_basis(lat, lon)[2], -1.0)
        };
        // Liftoff mark for the profile clock: first sustained climb.
        if self.t_liftoff < 0.0 && dot(self.nav.vel, up_here) > 1.0 {
            self.t_liftoff = t;
        }

        match self.phase {
            Phase::PadPress => {
                // Open feed + igniter isolation, then light the nine.
                cmd.valves[V_MAIN_LOX] = 1.0;
                cmd.valves[V_MAIN_RP1] = 1.0;
                cmd.valves[V_TEATEB] = 1.0;
                cmd.attitude = quat_between([1.0, 0.0, 0.0], self.up_pad);
                cmd.tvc_enable = 1.0;
                if t >= 0.2 {
                    cmd.engines = [p.ascent_throttle; N_ENGINES];
                    self.set_phase(Phase::VerticalRise, t);
                }
            }
            Phase::VerticalRise => {
                cmd.valves[V_MAIN_LOX] = 1.0;
                cmd.valves[V_MAIN_RP1] = 1.0;
                cmd.valves[V_TEATEB] = 1.0;
                cmd.engines = [p.ascent_throttle; N_ENGINES];
                cmd.attitude = quat_between([1.0, 0.0, 0.0], self.up_pad);
                cmd.tvc_enable = 1.0;
                // No RCS during powered ascent: TVC owns pitch/yaw and the
                // symmetric stack has no roll disturbance — cold gas is
                // reserved for the flip/coast/descent (budget realism).
                if t >= p.kick_start_s {
                    self.set_phase(Phase::PitchKick, t);
                }
            }
            Phase::PitchKick => {
                cmd.valves[V_MAIN_LOX] = 1.0;
                cmd.valves[V_MAIN_RP1] = 1.0;
                cmd.valves[V_TEATEB] = 1.0;
                cmd.engines = [p.ascent_throttle; N_ENGINES];
                cmd.tvc_enable = 1.0;
                // Ramp the nose from vertical toward the track azimuth.
                let f = ((t - self.phase_t0) / p.kick_ramp_s).clamp(0.0, 1.0);
                let angle = f * p.kick_deg.to_radians();
                let dir = normalize(add(
                    scale(self.up_pad, angle.cos()),
                    scale(self.track_dir, angle.sin()),
                ));
                cmd.attitude = quat_between([1.0, 0.0, 0.0], dir);
                if f >= 1.0 && speed > 80.0 {
                    self.set_phase(Phase::GravityTurn, t);
                }
            }
            Phase::GravityTurn => {
                cmd.valves[V_MAIN_LOX] = 1.0;
                cmd.valves[V_MAIN_RP1] = 1.0;
                cmd.valves[V_TEATEB] = 1.0;
                cmd.tvc_enable = 1.0;
                // Lofted pitch program (WHITEPAPER 11.1): command the
                // flight-path angle as a function of speed, from vertical at
                // the kick down to `meco_fpa_deg` at MECO speed. RTLS flights
                // fly deliberately lofted; the exponent shapes the loft and
                // the campaign calibrates both.
                let v0 = 90.0;
                let f = ((speed - v0) / (p.meco_speed_mps - v0)).clamp(0.0, 1.0);
                let gamma = (90.0 - (90.0 - p.meco_fpa_deg) * f.powf(p.pitch_exp)).to_radians();
                let dir = normalize(add(
                    scale(up_here, gamma.sin()),
                    scale(self.track_dir, gamma.cos()),
                ));
                // Profile-following ascent (WHITEPAPER 11.1): the recorded
                // flight IS the designed reference. Feed-forward its
                // flight-path angle; close speed with bounded throttle
                // feedback (the monte-carlo skill's tracking pattern). The
                // parametric program is the fallback without a profile.
                let mut dir_cmd = dir;
                let mut u = p.ascent_throttle;
                if let (Some(profile), true) = (&self.ascent_profile, self.t_liftoff >= 0.0) {
                    let t_ref = t - self.t_liftoff;
                    let v_ref = profile.speed(t_ref);
                    let gamma_ref = (profile.vspeed(t_ref) / v_ref.max(30.0))
                        .clamp(-1.0, 1.0)
                        .asin();
                    // Altitude-error trim rotates the commanded path angle.
                    let alt_err = profile.altitude(t_ref) - alt;
                    let gamma_cmd =
                        (gamma_ref + (alt_err * 2.0e-4).clamp(-0.12, 0.12)).clamp(0.0, 1.55);
                    dir_cmd = normalize(add(
                        scale(up_here, gamma_cmd.sin()),
                        scale(self.track_dir, gamma_cmd.cos()),
                    ));
                    u = (p.ascent_throttle + (v_ref - speed) * 2.0e-3).clamp(0.62, 1.0);
                }
                cmd.attitude = quat_between([1.0, 0.0, 0.0], dir_cmd);
                // Throttle bucket floor still applies through Max-Q.
                let qbar = 0.5 * density(alt) * speed * speed;
                if qbar > p.bucket_q_on_pa && speed < 500.0 {
                    u = u.min(p.bucket_throttle);
                }
                // Acceleration limiting toward MECO (the real vehicle
                // throttles down to hold ~3.6 g as the stack lightens).
                let a_meas = norm(s.imu_accel);
                if a_meas > 34.0 {
                    u = (u * 34.0 / a_meas).max(THROTTLE_MIN);
                }
                cmd.engines = [u; N_ENGINES];
                if speed >= p.meco_speed_mps {
                    self.meco_speed_reached = true;
                    self.cutoff_with_purge(&mut cmd, t);
                    self.set_phase(Phase::Meco, t);
                }
            }
            Phase::Meco => {
                // Keep mains open a beat for shutdown, then close; stage sep
                // happens plant-side on the separation command timing.
                cmd.rcs_enable = 1.0;
                cmd.attitude = quat_between([1.0, 0.0, 0.0], normalize(self.nav.vel));
                if t - self.phase_t0 > 3.0 {
                    self.set_phase(Phase::Flip, t);
                }
            }
            Phase::Flip => {
                cmd.rcs_enable = 1.0;
                // Flip toward the boostback burn direction: where the IIP
                // needs to move, horizontally (the dog-leg heading).
                let (burn_dir, _) = self.boostback_solution(s, up_here, p);
                cmd.attitude = quat_between([1.0, 0.0, 0.0], burn_dir);
                let x_body = quat_rotate(self.nav.att, [1.0, 0.0, 0.0]);
                if dot(x_body, burn_dir) > 0.95 {
                    self.set_phase(Phase::Boostback, t);
                }
            }
            Phase::Boostback => {
                cmd.valves[V_MAIN_LOX] = 1.0;
                cmd.valves[V_MAIN_RP1] = 1.0;
                cmd.valves[V_TEATEB] = 1.0;
                cmd.tvc_enable = 1.0;
                cmd.rcs_enable = 1.0;
                // Full-vector IIP targeting (WHITEPAPER 11.3): LZ-1 is NOT on
                // the ascent track line (it sits ~14 km cross-track of it),
                // so the boostback flies a dog-leg — thrust along the
                // horizontal direction that walks the predicted impact point
                // onto the target. The entry burn will pull the IIP back
                // along the course by roughly `boostback_overshoot_m`, so
                // aim that far beyond LZ-1 along the current course.
                let (burn_dir, miss_h_mag) = self.boostback_solution(s, up_here, p);
                cmd.attitude = quat_between([1.0, 0.0, 0.0], burn_dir);
                let u = p.boostback_throttle.clamp(THROTTLE_MIN, 1.0);
                let mut engines = [0.0; N_ENGINES];
                engines[0] = u;
                engines[1] = u;
                engines[2] = u;
                cmd.engines = engines;
                if miss_h_mag < 1_000.0 {
                    eprintln!("[fsw] boostback cutoff: horizontal iip-to-target {miss_h_mag:.0} m");
                    self.cutoff_with_purge(&mut cmd, t);
                    self.set_phase(Phase::Coast, t);
                }
            }
            Phase::Coast => {
                cmd.rcs_enable = 1.0;
                // Engines-first for entry: point +X against the velocity.
                cmd.attitude = quat_between([1.0, 0.0, 0.0], scale(normalize(self.nav.vel), -1.0));
                let descending = dot(self.nav.vel, up_here) < 0.0;
                if descending && (speed >= p.entry_ignite_speed_mps || alt <= p.entry_ignite_alt_m)
                {
                    self.entry_v0 = speed;
                    self.set_phase(Phase::EntryBurn, t);
                }
            }
            Phase::EntryBurn => {
                cmd.valves[V_MAIN_LOX] = 1.0;
                cmd.valves[V_MAIN_RP1] = 1.0;
                cmd.valves[V_TEATEB] = 1.0;
                cmd.tvc_enable = 1.0;
                cmd.rcs_enable = 1.0;
                cmd.attitude = quat_between([1.0, 0.0, 0.0], scale(normalize(self.nav.vel), -1.0));
                let mut engines = [0.0; N_ENGINES];
                engines[0] = p.entry_throttle;
                engines[1] = p.entry_throttle;
                engines[2] = p.entry_throttle;
                cmd.engines = engines;
                // Fixed delta-v cutoff (the recorded 367 m/s): keeping the
                // entry burn repeatable makes its IIP pullback repeatable,
                // which the boostback overshoot bias absorbs (WHITEPAPER 11.4).
                if self.entry_v0 - speed >= p.entry_dv_mps {
                    let (along, cross) = self.iip_miss_components(s, up_here);
                    eprintln!("[fsw] entry cutoff: iip miss along={along:.0} cross={cross:.0} m");
                    self.cutoff_with_purge(&mut cmd, t);
                    self.set_phase(Phase::AeroDescent, t);
                }
            }
            Phase::AeroDescent => {
                // Grid fins own attitude in thick air (holding an AoA against
                // the static margin continuously would drain the cold-gas
                // budget); the RCS backstops in thin air.
                let qbar = 0.5 * density(alt) * speed * speed;
                cmd.rcs_enable = if qbar > 2_000.0 { 0.0 } else { 1.0 };
                // Retrograde, tilted to steer the drag vector toward LZ-1.
                let steer = self.descent_steer(s, up_here);
                let retro = normalize(add(scale(normalize(self.nav.vel), -1.0), steer));
                let desired = quat_between([1.0, 0.0, 0.0], retro);
                cmd.attitude = self.slew_attitude(desired, 0.01);
                cmd.fins = self.fin_attitude_pd(retro, s, false);
                // Hoverslam ignition (WHITEPAPER 11.5): ignite when the
                // descent rate reaches the three-engine opening profile,
                // charging ~2.5 s of spool-up distance against the altitude.
                let vdown = -dot(self.nav.vel, up_here);
                let a_land = 0.70 * self.landing_accel_net(s, 3.0);
                let h_eff = (h_agl - 2.5 * vdown.max(0.0)).max(1.0);
                let v_profile = (2.0 * a_land * h_eff).sqrt();
                if t - self.phase_t0 > 5.0
                    && ((t - self.phase_t0) as u64).is_multiple_of(10)
                    && dt_edge(t)
                {
                    let (along, cross) = self.iip_miss_components(s, up_here);
                    eprintln!(
                        "[fsw] t={t:6.1} descent: iip miss along={along:.0} cross={cross:.0} m, alt {alt:.0}"
                    );
                }
                if alt <= p.landing_arm_alt_m && vdown * p.landing_accel_margin >= v_profile {
                    let (along, cross) = self.iip_miss_components(s, up_here);
                    // Local state too, so a landing-burn change can be
                    // reproduced from the same starting point offline.
                    let (lat, lon, _) = ecef_to_geodetic(self.lz1_pos);
                    let [north, east, _] = ned_basis(lat, lon);
                    let d = sub(self.nav.pos, self.lz1_pos);
                    eprintln!(
                        "[fsw] landing ignition: iip miss along={along:.0} cross={cross:.0} m; \
                         h={h_agl:.0} m vdown={vdown:.1} m/s offset E={:.0} N={:.0} m vel E={:.1} N={:.1} m/s",
                        dot(d, east),
                        dot(d, north),
                        dot(self.nav.vel, east),
                        dot(self.nav.vel, north),
                    );
                    self.landing_escalated = true; // 3-engine start (hot RTLS)
                    self.landing_thrust_est_after = t + THRUST_EST_SETTLE_IGNITION_S;
                    self.set_phase(Phase::LandingBurn, t);
                }
            }
            Phase::LandingBurn => {
                cmd.valves[V_MAIN_LOX] = 1.0;
                cmd.valves[V_MAIN_RP1] = 1.0;
                cmd.valves[V_TEATEB] = 1.0;
                cmd.tvc_enable = 1.0;
                cmd.rcs_enable = 1.0;
                let mass = DRY_MASS_KG + s.lox_kg + s.rp1_kg;
                let vdown = -dot(self.nav.vel, up_here);
                let body_x = quat_rotate(self.nav.att, [1.0, 0.0, 0.0]);
                let cos_tilt = dot(body_x, up_here).max(0.6);
                // A leaning booster touches down on its low pad first.
                let h_pad = h_agl - LEG_PAD_RADIUS_M * (1.0 - cos_tilt * cos_tilt).sqrt();
                let lit = if self.landing_escalated && !self.landing_deescalated {
                    3.0
                } else {
                    1.0
                };
                // Thrust estimate: once the current engine set has settled,
                // compare the measured axial specific force with what the
                // engine model predicts for where the throttle actually is.
                self.landing_spool += (self.landing_throttle - self.landing_spool)
                    * (1.0 - (-GUIDANCE_DT_S / MERLIN_THROTTLE_TAU_S).exp());
                let predicted = lit * engine_thrust(self.landing_spool) / mass;
                if t >= self.landing_thrust_est_after && predicted > 1.0 {
                    let ratio = (s.imu_accel[0] / predicted).clamp(0.8, 1.25);
                    self.landing_thrust_ratio +=
                        (ratio - self.landing_thrust_ratio) * GUIDANCE_DT_S / THRUST_EST_TAU_S;
                }
                let k_thrust = self.landing_thrust_ratio;

                // 3 -> 1 handover (TEA-TEB): open on three and hand over to the
                // center engine once the state the outboard pair leaves behind,
                // tail-off impulse included, sits in the middle of the
                // single-engine band. Handing over on the band's edge strands
                // the booster below what one engine can fly at min throttle.
                if lit > 1.0 {
                    let tail_dv = 2.0
                        * k_thrust
                        * engine_thrust(self.landing_throttle)
                        * MERLIN_SHUTDOWN_TAU_S
                        / mass;
                    let h_after = h_pad - vdown * MERLIN_SHUTDOWN_TAU_S;
                    let (lo, hi) = decel_band(mass, 1.0, k_thrust);
                    let aim = lo + SINGLE_ENGINE_BAND_AIM * (hi - lo);
                    if required_decel(h_after, vdown - tail_dv) <= aim {
                        self.landing_deescalated = true;
                        self.landing_thrust_est_after = t + THRUST_EST_SETTLE_HANDOVER_S;
                    }
                }
                let three = self.landing_escalated && !self.landing_deescalated;
                let n = if three { 3.0 } else { 1.0 };

                // Vertical channel (WHITEPAPER 11.5): fly the constant
                // deceleration that reaches the touchdown speed exactly at the
                // pad, recomputed from the navigated state every step, so a
                // drag or mass error changes the next solution instead of
                // accumulating against a fixed profile. T_min/W > 1, so the
                // request is clamped to what the lit engines can deliver.
                let (lo, hi) = decel_band(mass, n, k_thrust);
                let a_net = required_decel(h_pad, vdown).clamp(lo, hi);
                let a_up = 9.81 + a_net;

                // Lateral channel: a stoppable approach onto LZ-1, then
                // velocity nulling inside the terminal gate.
                let a_lat = self.landing_lateral_accel(up_here, h_pad, vdown, a_net, a_up);
                let dir = normalize(add(scale(up_here, a_up), a_lat));
                let desired = quat_between([1.0, 0.0, 0.0], dir);
                cmd.attitude = self.slew_attitude(desired, 0.01);
                let cmd_x = quat_rotate(cmd.attitude, [1.0, 0.0, 0.0]);
                cmd.fins = self.fin_attitude_pd(cmd_x, s, true);

                // Size thrust for the attitude the booster actually has; the
                // engines push along that axis, not along the setpoint.
                let per_engine = mass * a_up / cos_tilt / n / k_thrust;
                let u =
                    ((per_engine + P_SL_PA * A_E_M2) / T_VAC_PER_ENGINE).clamp(THROTTLE_MIN, 1.0);
                self.landing_throttle = u;
                // The center engine has used its last TEA-TEB charge, so it
                // stays lit until contact. The one exception: if the booster
                // has stopped short and started to rise, min throttle will
                // only carry it higher (T_min/W > 1), so shut down while the
                // drop is still centimetres rather than climb until the
                // propellant runs out.
                let rising_short = h_agl < 3.0 && vdown < -0.2;
                let mut engines = [0.0; N_ENGINES];
                if !rising_short {
                    engines[0] = u;
                    if three {
                        engines[1] = u;
                        engines[2] = u;
                    }
                }
                cmd.engines = engines;
                if s.landed > 0.5 || (h_agl < 0.25 && vdown > 0.0 && vdown < 1.0 && speed < 1.5) {
                    self.cutoff_with_purge(&mut cmd, t);
                    self.set_phase(Phase::Touchdown, t);
                }
            }
            Phase::Touchdown => {
                cmd.engines = [0.0; N_ENGINES];
                cmd.attitude = quat_between([1.0, 0.0, 0.0], up_here);
            }
        }
        cmd.phase = self.phase as i32 as f64;
        cmd
    }

    /// Drag-aware ballistic impact point via forward integration of the same
    /// rotating-frame EOM plus an engines-first drag model (WHITEPAPER 11.3).
    /// The vacuum IIP would overshoot by tens of km — descent drag steepens
    /// the fall — so the predictor carries the FSW's own aero estimate.
    fn impact_point(&self, s: &SensorPacket) -> V3 {
        let mass = 25_600.0 + s.lox_kg + s.rp1_kg;
        let cd_s = if s.params.fsw_cd_s_m2 > 1.0 {
            s.params.fsw_cd_s_m2
        } else {
            22.0
        };
        let mut r = self.nav.pos;
        let mut v = self.nav.vel;
        let dt = 0.5;
        for _ in 0..2400 {
            let alt = ecef_to_geodetic(r).2;
            if alt <= self.nav.ground.landing_m {
                return r;
            }
            let speed = norm(v);
            let a_drag = if speed > 1.0 {
                scale(
                    normalize(v),
                    -0.5 * density(alt) * speed * speed * cd_s / mass,
                )
            } else {
                [0.0; 3]
            };
            let a = add(add(gravity(r), frame_accel(r, v)), a_drag);
            v = add(v, scale(a, dt));
            r = add(r, scale(v, dt));
        }
        r
    }

    /// Boostback burn solution: aim the predicted impact point at a target
    /// `boostback_overshoot_m` beyond LZ-1 along the approach course (the
    /// entry burn pulls the IIP back by about that much). Returns the burn
    /// direction and the horizontal IIP-to-target distance.
    fn boostback_solution(&self, s: &SensorPacket, up: V3, p: &GuidanceParams) -> (V3, f64) {
        let iip = self.impact_point(s);
        // Approach course: from LZ-1 back toward the booster's ground point,
        // extended beyond the pad — i.e. the direction the vehicle will be
        // flying when it arrives. Overshoot lies past LZ-1 along that course.
        let here = sub(
            self.nav.pos,
            scale(up, dot(sub(self.nav.pos, self.lz1_pos), up)),
        );
        let course = normalize(sub(self.lz1_pos, here)); // horizontal-ish
        let course_h = normalize(sub(course, scale(up, dot(course, up))));
        let target = add(self.lz1_pos, scale(course_h, p.boostback_overshoot_m));
        let miss = sub(target, iip);
        let miss_h = sub(miss, scale(up, dot(miss, up)));
        let mag = norm(miss_h);
        let dir = if mag > 1e-6 {
            normalize(sub(normalize(miss_h), scale(up, 0.10)))
        } else {
            scale(normalize(self.nav.vel), -1.0)
        };
        (dir, mag)
    }

    /// Signed IIP miss components: along the approach course (positive =
    /// undershoot, impact before LZ-1) and cross-course.
    fn iip_miss_components(&self, s: &SensorPacket, up: V3) -> (f64, f64) {
        let iip = self.impact_point(s);
        let v_h = sub(self.nav.vel, scale(up, dot(self.nav.vel, up)));
        let course = normalize(v_h);
        let miss = sub(self.lz1_pos, iip);
        let miss_h = sub(miss, scale(up, dot(miss, up)));
        let along = dot(miss_h, course);
        let cross_vec = sub(miss_h, scale(course, along));
        let cross_sign = if dot(cross(up, course), cross_vec) >= 0.0 {
            1.0
        } else {
            -1.0
        };
        (along, cross_sign * norm(cross_vec))
    }

    /// Drag-vector steering during aero descent (WHITEPAPER 11.4).
    /// Cross-track PD plus along-track AoA modulation. Engines-first CA > CN
    /// so any AoA lowers total deceleration (stretches the IIP). Tilt cap is
    /// scheduled down at high q̄ so Max-Q passes near zero AoA.
    fn descent_steer(&self, s: &SensorPacket, up: V3) -> V3 {
        let iip = self.impact_point(s);
        let miss = sub(self.lz1_pos, iip);
        let miss_h = sub(miss, scale(up, dot(miss, up)));
        let v_h = sub(self.nav.vel, scale(up, dot(self.nav.vel, up)));
        let course = normalize(v_h);
        let cross_err = sub(miss_h, scale(course, dot(miss_h, course)));
        let cross_vel = sub(v_h, scale(course, dot(v_h, course)));
        let pd = sub(
            scale(cross_err, 1.0 / 2_500.0),
            scale(cross_vel, 1.0 / 50.0),
        );
        let mag = norm(pd);
        let speed = self.nav.speed();
        let qbar = 0.5 * density(self.nav.altitude()) * speed * speed;
        // Full steer_tilt_cap below ~25 kPa; fade toward ~0.03 rad by 60 kPa+.
        let base_cap = s.params.steer_tilt_cap;
        // Full authority below ~30 kPa; fade toward 0.04 rad by 70 kPa+.
        let tilt_cap = (base_cap * (30_000.0 / qbar.max(5_000.0))).clamp(0.04, base_cap);
        let (along, _) = self.iip_miss_components(s, up);
        let cross_quiet = norm(cross_err) < 80.0 && norm(cross_vel) < 2.0;
        let tilt = if cross_quiet { 0.0 } else { mag.min(tilt_cap) };
        // Positive tilt along the PD direction deflects the drag vector toward
        // the target (fins have the correct plant sign on pitch).
        let cross_dir = if tilt > 1e-6 {
            scale(normalize(pd), tilt)
        } else {
            [0.0; 3]
        };
        // Along-track drag modulation: only add AoA when undershooting
        // (stretch range). Overshoot → 0 (max CA drag).
        let aoa_along = (along / 8_000.0).clamp(0.0, 0.10);
        let retro = scale(normalize(self.nav.vel), -1.0);
        let lift_dir = normalize(sub(course, scale(retro, dot(course, retro))));
        let along_tilt = scale(lift_dir, aoa_along);
        add(cross_dir, along_tilt)
    }

    /// Torque-based fin PD: desired angular acceleration → deflection via
    /// q̄·effectiveness, so closed-loop bandwidth stays near `fin_wn` across
    /// the descent q̄ range. `rate_only` drops the attitude term (LandingBurn
    /// under TVC — fins damp residual rates only).
    fn fin_attitude_pd(&self, desired_dir: V3, s: &SensorPacket, rate_only: bool) -> [f64; 3] {
        let x_body = quat_rotate(self.nav.att, [1.0, 0.0, 0.0]);
        let err_world = cross(x_body, desired_dir);
        let err_body = quat_rotate_inv(self.nav.att, err_world);
        let wn = s.params.fin_wn;
        let zeta = 0.85;
        let err = if rate_only { [0.0, 0.0, 0.0] } else { err_body };
        let alpha = [
            wn * wn * err[0] - 2.0 * zeta * wn * s.imu_gyro[0],
            wn * wn * err[1] - 2.0 * zeta * wn * s.imu_gyro[1],
            wn * wn * err[2] - 2.0 * zeta * wn * s.imu_gyro[2],
        ];
        let speed = self.nav.speed().max(1.0);
        let qbar = 0.5 * density(self.nav.altitude()) * speed * speed;
        // Floor keeps authority in thin air; pair of fins × lever.
        let q_eff = qbar.max(2_000.0);
        // Plant mapping (aero.fin_mix / fin_wrench): +pitch → −My, +yaw → +Mz.
        let k_defl = 2.0 * q_eff * FIN_S_M2 * FIN_CN_DELTA * FIN_LEVER_M;
        let pitch = (-FIN_I_TRANS * alpha[1] / k_defl).clamp(-0.35, 0.35);
        let yaw = (FIN_I_TRANS * alpha[2] / k_defl).clamp(-0.35, 0.35);
        let roll = (FIN_I_TRANS * alpha[0] / k_defl).clamp(-0.35, 0.35);
        [pitch, yaw, roll]
    }

    fn landing_accel_net(&self, s: &SensorPacket, n_engines: f64) -> f64 {
        // Near-full throttle at the estimated landing mass (propellant
        // knowledge comes from the FSW's own flow accounting; the bridge
        // forwards it as config).
        let mass = DRY_MASS_KG + s.lox_kg + s.rp1_kg;
        let thrust = n_engines * 0.85 * (T_VAC_PER_ENGINE - P_SL_PA * A_E_M2);
        (thrust / mass - 9.81).max(1.0)
    }

    /// Horizontal thrust acceleration for the landing burn (WHITEPAPER 11.5).
    ///
    /// Above the terminal gate the lateral channel mirrors the vertical one:
    /// it steers toward the aim point at the fastest speed it could still
    /// brake to zero in the distance left, using a fixed share of the tilt
    /// allowance averaged over the heights the stop will take, and leaving
    /// time for the lean to reverse. A divert too large for the burn therefore ends short of the
    /// pad and slow, never fast and past it. Near the aim point the profile
    /// turns into a plain proportional approach.
    ///
    /// Terminal, below the gate: null horizontal velocity only (Apollo
    /// P66-style). A metre of miss this close to the pad is not worth a lean.
    ///
    /// Both lead the state by the drift the current lean adds before the
    /// attitude can respond.
    fn landing_lateral_accel(&self, up: V3, h_pad: f64, vdown: f64, a_net: f64, a_up: f64) -> V3 {
        let horizontal = |a: V3| sub(a, scale(up, dot(a, up)));
        // Plan from where the booster will be once a new attitude can take
        // effect: the current lean keeps accelerating it sideways for about
        // one attitude lag.
        let x_body = quat_rotate(self.nav.att, [1.0, 0.0, 0.0]);
        let lean = scale(horizontal(x_body), a_up / dot(x_body, up).max(0.6));
        let lag = ATTITUDE_LAG_S;
        let v = add(horizontal(self.nav.vel), scale(lean, lag));
        let a_lat = if h_pad > TERMINAL_GATE_M {
            // Wind-bias the aim point upwind by the expected terminal drift.
            let t_land = (vdown - ZEM_V_TD_MPS).max(0.0) / a_net;
            let wind = horizontal(self.nav.wind_est);
            let aim = sub(self.lz1_pos, scale(wind, t_land.min(25.0)));
            let r_now = horizontal(sub(self.nav.pos, aim));
            let r = add(
                add(r_now, scale(horizontal(self.nav.vel), lag)),
                scale(lean, 0.5 * lag * lag),
            );
            let dist = norm(r);
            // Stopping-distance speed with a reaction time: the lean has to
            // swing through vertical before it brakes.
            let t_react = 2.0 * lag;
            // The tilt allowance shrinks as the booster descends, so brake
            // with what is left on average between here and where the stop
            // would finish, not with what is available right now.
            let speed_now = norm(v);
            let mut a_brake = DIVERT_BRAKE_MARGIN * a_up * landing_tilt_cap(h_pad).tan();
            for _ in 0..2 {
                let t_stop = t_react + speed_now / a_brake;
                let h_end = (h_pad - 0.5 * vdown * t_stop).max(0.0);
                a_brake = DIVERT_BRAKE_MARGIN * a_up * mean_tilt_cap(h_end, h_pad).tan();
            }
            let speed =
                -a_brake * t_react + ((a_brake * t_react).powi(2) + 2.0 * a_brake * dist).sqrt();
            let v_cmd = if dist > 1e-6 {
                scale(r, -speed / dist)
            } else {
                [0.0; 3]
            };
            scale(sub(v_cmd, v), 1.0 / LATERAL_VEL_TAU_S)
        } else {
            scale(v, -1.0 / LATERAL_VEL_TAU_S)
        };
        let max_lat = a_up * landing_tilt_cap(h_pad).tan();
        let mag = norm(a_lat);
        if mag > max_lat && mag > 1e-6 {
            scale(a_lat, max_lat / mag)
        } else {
            a_lat
        }
    }
}

/// Sea-level thrust of one Merlin at throttle `u` (FSW engine model).
fn engine_thrust(u: f64) -> f64 {
    (u * T_VAC_PER_ENGINE - P_SL_PA * A_E_M2).max(0.0)
}

/// Net upward acceleration (thrust/m − g) that `n_engines` can produce at
/// `mass`, as (min throttle, full throttle), with the model thrust scaled by
/// the in-flight estimate `k_thrust`.
fn decel_band(mass: f64, n_engines: f64, k_thrust: f64) -> (f64, f64) {
    let net = |u: f64| n_engines * k_thrust * engine_thrust(u) / mass - 9.81;
    (net(THROTTLE_MIN), net(1.0))
}

/// Constant deceleration that brings the descent rate to the touchdown
/// speed exactly at the pad: the hoverslam solution from the current state.
fn required_decel(h_agl: f64, vdown: f64) -> f64 {
    (vdown * vdown - ZEM_V_TD_MPS * ZEM_V_TD_MPS).max(0.0) / (2.0 * h_agl.max(0.3))
}

/// Average of `landing_tilt_cap` over the heights from `lo` to `hi`.
fn mean_tilt_cap(lo: f64, hi: f64) -> f64 {
    const N: usize = 8;
    (0..N)
        .map(|i| landing_tilt_cap(lo + (hi - lo) * (i as f64 + 0.5) / N as f64))
        .sum::<f64>()
        / N as f64
}

/// Thrust tilt allowed during the landing burn: the full divert allowance
/// above `TILT_TAPER_START_M`, a small trim allowance at the terminal gate,
/// and near-vertical at the pad. The TVC loop needs seconds to stand a
/// leaning booster back up, so the taper starts well above the gate.
fn landing_tilt_cap(h_pad: f64) -> f64 {
    if h_pad > TERMINAL_GATE_M {
        let f = ((h_pad - TERMINAL_GATE_M) / (TILT_TAPER_START_M - TERMINAL_GATE_M)).min(1.0);
        TERMINAL_TILT_CAP_RAD + f * (ZEM_TILT_CAP_RAD - TERMINAL_TILT_CAP_RAD)
    } else {
        let f = (h_pad / TERMINAL_GATE_M).max(0.0);
        TOUCHDOWN_TILT_CAP_RAD + f * (TERMINAL_TILT_CAP_RAD - TOUCHDOWN_TILT_CAP_RAD)
    }
}

/// True on ~one guidance tick per second (for rate-limited diagnostics).
fn dt_edge(t: f64) -> bool {
    (t - t.floor()) < 0.011
}

fn port_from_env(name: &str, default: u16) -> u16 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn fsw_from_args(mut args: impl Iterator<Item = String>) -> std::io::Result<Fsw> {
    let mode = match (args.next(), args.next(), args.next()) {
        (None, None, None) => "photo3d".to_string(),
        (Some(flag), Some(mode), None) if flag == "--terrain-mode" => mode,
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "usage: falcon9-fsw [--terrain-mode legacy|aerial|photo3d]",
            ));
        }
    };
    Fsw::new(&mode)
}

fn main() -> std::io::Result<()> {
    let mut fsw = fsw_from_args(std::env::args().skip(1))?;
    let state_port = port_from_env("ELODIN_MC_PORT_STATE", 9114);
    let command_port = port_from_env("ELODIN_MC_PORT_COMMAND", 9115);
    let sock = UdpSocket::bind(("127.0.0.1", state_port))?;
    eprintln!("[fsw] listening on 127.0.0.1:{state_port}, replying to :{command_port}");

    let mut buf = [0u8; STATE_FLOATS * 8];
    loop {
        let (n, _src) = sock.recv_from(&mut buf)?;
        if n < STATE_FLOATS * 8 {
            continue;
        }
        let mut vals = [0.0f64; STATE_FLOATS];
        for (i, v) in vals.iter_mut().enumerate() {
            *v = f64::from_le_bytes(buf[i * 8..i * 8 + 8].try_into().unwrap());
        }
        let packet = SensorPacket::parse(&vals);
        let cmd = fsw.step(&packet).pack();
        let mut out = [0u8; CMD_FLOATS * 8];
        for (i, v) in cmd.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        sock.send_to(&out, ("127.0.0.1", command_port))?;
    }
}

#[cfg(test)]
mod terminal_guidance_tests {
    use super::*;

    #[test]
    fn fsw_pad_heights_and_ecef_target_match_the_plant_dataset() {
        let source: serde_json::Value =
            serde_json::from_str(include_str!("../../data/sites.json")).unwrap();
        for mode in ["legacy", "aerial", "photo3d"] {
            let launch = source[mode]["launch_m"].as_f64().unwrap();
            let landing = source[mode]["landing_m"].as_f64().unwrap();
            let mut fsw =
                fsw_from_args(["--terrain-mode", mode].map(String::from).into_iter()).unwrap();
            let (_, _, target_altitude) = ecef_to_geodetic(fsw.lz1_pos);
            assert!((target_altitude - landing).abs() < 1e-6);
            fsw.nav.pos = fsw.lz1_pos;
            assert_eq!(fsw.nav.site_alt(), landing);
            let (lat, lon, _) = ecef_to_geodetic(fsw.lz1_pos);
            fsw.nav.pos = geodetic_to_ecef(lat, lon, landing + 50.0);
            fsw.nav.radar_agl_m = 30.0;
            fsw.nav.station_m = 20.0;
            assert!((fsw.nav.altitude() - (landing + 50.0)).abs() < 1e-6);
            assert!((fsw.nav.height_agl() - 30.0).abs() < 1e-6);
            fsw.nav.pos = add(fsw.lz1_pos, [10_000.0, 0.0, 0.0]);
            assert_eq!(fsw.nav.site_alt(), launch);
        }
    }

    #[test]
    fn standalone_default_is_photo3d_and_invalid_modes_are_rejected() {
        let fsw = fsw_from_args(std::iter::empty()).unwrap();
        let (_, _, altitude) = ecef_to_geodetic(fsw.lz1_pos);
        assert!((altitude - Sites::load().photo3d.landing_m).abs() < 1e-6);
        for args in [
            vec!["--terrain-mode", "typo"],
            vec!["--terrain-mode"],
            vec!["--unknown", "aerial"],
            vec!["--terrain-mode", "aerial", "extra"],
        ] {
            assert!(fsw_from_args(args.into_iter().map(String::from)).is_err());
        }
    }

    #[test]
    fn radar_clearance_uses_the_selected_ellipsoid_ground_height() {
        let ground = GroundHeights {
            launch_m: -7.0,
            landing_m: -25.0,
        };
        let target = geodetic_to_ecef(0.0, 0.0, ground.landing_m);
        let mut nav = Navigator::new(ground, target);
        nav.pos = geodetic_to_ecef(0.0, 0.0, ground.landing_m + 50.0);
        nav.radar_agl_m = 30.0;
        nav.station_m = 20.0;
        assert!((nav.altitude() - 25.0).abs() < 1e-6);
        assert!((nav.height_agl() - 30.0).abs() < 1e-6);
    }

    #[test]
    fn terminal_guidance_brakes_residual_lateral_velocity() {
        let (lat, lon, _) = ecef_to_geodetic(Fsw::new("photo3d").unwrap().lz1_pos);
        let basis = ned_basis(lat, lon);
        let up = scale(basis[2], -1.0);
        let east = basis[1];
        for lateral_speed in [-5.5, 5.5] {
            let mut fsw = Fsw::new("photo3d").unwrap();
            fsw.phase = Phase::LandingBurn;
            fsw.nav.initialized = true;
            fsw.nav.last_t = 1.0;
            // Above the target, but still moving sideways: proximity alone
            // must not disable lateral braking before ground contact.
            fsw.nav.pos = add(fsw.lz1_pos, scale(up, 35.0));
            fsw.nav.vel = add(scale(east, lateral_speed), scale(up, -8.0));
            fsw.nav.att = quat_between([1.0, 0.0, 0.0], up);
            let mut values = [0.0; STATE_FLOATS];
            values[0] = 1.0;
            values[14] = -1.0;
            values[20] = 3000.0;
            values[21] = 1300.0;
            values[33] = 1.273;
            let command = fsw.step(&SensorPacket::parse(&values));
            let thrust_direction = quat_rotate(command.attitude, [1.0, 0.0, 0.0]);
            assert!(
                dot(thrust_direction, east) * lateral_speed < -0.01,
                "terminal thrust must oppose lateral velocity: speed={lateral_speed}, direction={thrust_direction:?}",
            );
        }
    }
}

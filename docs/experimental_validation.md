# Physical modal validation protocol

This protocol is for the **assembled, unpowered** reference frame. It cannot be
completed until the carbon parts, motors or equal-mass dummies, payload, and an
accelerometer are physically available. Keep the test data separate from
simulated results.

1. Measure and record each part's mass, arm dimensions, hub plate dimensions,
   motor and payload positions, laminate supplier data, fasteners, and sensor mass.
   Update the baseline design file with the measured values before comparing
   frequencies. Photograph and identify the assembly revision.
2. Suspend the frame on long, soft cords so it can move freely in three
   dimensions. Keep suspension frequencies well below the first flexible mode.
   Switch motors off and remove propellers. Do not clamp the hub.
3. Attach a calibrated accelerometer at the hub, then at each arm near its motor
   position. Record the sensing axis, sample rate, units, sensor mass, and mount.
   Use a sample rate at least five times the highest frequency you intend to
   inspect. A light sensor is preferable because mass at an arm tip can shift a
   mode.
4. Tap each arm in the vertical and in-plane directions using a light,
   repeatable impulse. Record at least five clean decays per position and axis,
   with no clipping. Export each run as CSV with `time_s` and
   `acceleration_m_s2` columns. Keep a separate metadata CSV for run ID, tap
   point, response point, axis, assembled revision, and sample rate.
5. Use `validation.impact_spectrum` on each run. The script checks sampling,
   applies a Hann window, and reports spectral peaks. Look for peaks repeated
   across runs and positions; one peak alone is not proof of a structural mode.
   Inspect whether arm-to-arm phase and amplitude match the FEM shape. Record
   ambiguous or missing modes as unidentified.
6. Compare identified frequencies with the **pre-test** FEM prediction. Aim for
   10% or less frequency discrepancy for the first three identifiable flexible
   modes. If the difference is larger, check measured mass, orthotropic material
   properties, joint preload/slip, and sensor loading in that order. Record
   before/after model versions; do not hide calibration in the test result.

The impact test characterizes the stationary frame. Powered rotors add forcing,
possible gyroscopic effects, and operating-condition changes. Use the Campbell
check as a screening tool, not proof of vibration amplitude or flight safety.

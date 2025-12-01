# Realistic simulation parameters — summary tables

## A — Orbital / propagation parameters

| Parameter                        |                    Symbol & units | Typical / physically meaningful range                                                  | Recommended simulation value(s)                                                                   | Why / justification (refs)                                                                                                               |
| -------------------------------- | --------------------------------: | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Altitude (LEO)                   |                          (h) (km) | 160 — 2,000 km (LEO region); most debris & operations concentrated 300–1200 km.        | **Sample LEO:** 300, 500, 800, 1200 km (uniform or clustered sampling)                            | LEO defined up to ~2000 km; debris concentration clusters 300–1200 km. ([Wikipedia][1])                                                  |
| Semi-major axis / orbital period |                (a) (km) / (T) (s) | Derived from altitude; include circular and low-ecc cases (ecc < 0.05)                 | Use circular (ecc ≤ 0.01) and moderate eccentric (0.01–0.2) samples depending on target class     | Typical debris often near-circular LEO; include eccentric cases for transfer/upper-stage fragments. ([NASA Technical Reports Server][2]) |
| Inclination                      |                         (i) (deg) | 0–180°; common LEO inclinations cluster around 53°, 57°, 98° (SSO)                     | Sample [10°, 53°, 57°, 98°, 137°] to reflect common launch inclinations                           | Real catalog distributions show preferred inclinations by launch sites / SSO. ([arXiv][3])                                               |
| J2 perturbation                  |                                 — | Always include for LEO; secular precession important                                   | Include J2 (minimum); optionally J3/J4 for longer horizons                                        | J2 dominates secular perturbations and affects RAAN/ω precession — necessary for realistic propagation. ([Carleton University][4])       |
| Drag model                       | density ρ (kg/m³) via NRLMSISE-00 | Atmospheric density significant below ~1200 km and highly variable with solar activity | Use NRLMSISE-00 (or equivalent) with F10.7 & Ap inputs; vary F10.7 in scenarios. ([Wikipedia][5]) |                                                                                                                                          |

---

## B — Object physical properties (affecting drag/SRP & observability)

| Parameter                      |                           Symbol & units | Typical range                                        | Recommended sampling / value                                                                                           | Why / justification (refs)                                                                                                                                              |
| ------------------------------ | ---------------------------------------: | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Area-to-mass ratio (AMR)       |                            (A/m) (m²/kg) | ~0.005 (dense) — up to 20–45 (HAMR) (observed)       | Sample classes: LAMR: 0.005–0.2; Typical: 0.2–1.0; HAMR: 1.0–20. Use log-uniform sampling within classes.              | Observational studies report AMR spanning orders of magnitude; HAMR objects exist and are strongly perturbed by SRP. ([ccd.aiub.unibe.ch][6])                           |
| Ballistic coefficient (B)      | (B = \frac{m}{C_D A}) or inverse (m²/kg) | Commonly expressed via (C_D A/m); values vary widely | Represent with either (C_D\in[2.0,2.5]) and (A/m) above; sample (B) consistent with AMR classes (eg. B ≈ 0.05–2 m²/kg) | Ballistic coefficient is the main parameter for drag acceleration; use heterogeneity to model uncertainties. ([conference.sdo.esoc.esa.int][7])                         |
| Physical size / diameter       |                                    d (m) | mm-scale fragments up to meters                      | Use size bins: <0.1 m (small fragments), 0.1–1 m (large fragments), >1 m (structures)                                  | Size distribution informs optical magnitude, radar RCS, and detection probability. (NASA/ESA population models used for sampling). ([NASA Technical Reports Server][2]) |
| Reflectivity / magnitude model |                         visual mag (mag) | Depends on size, albedo, phase angle                 | Use brightness models (Lambertian + standard albedo 0.1–0.3) to map size→magnitude; add ±1–2 mag randomness            | Needed for optical detectability modeling and SNR. (Standard photometric approximations)                                                                                |

---

## C — Sensor / measurement parameterization

| Parameter                               |        Units | Realistic range / values                                                                    | Recommended sim values                                                                                           | Why / justification (refs)                                                                                                                    |
| --------------------------------------- | -----------: | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Optical astrometric noise               |       arcsec | 0.03" (very precise telescopes) — few arcsec (small sensors)                                | **Baseline optical noise:** 0.3"–3.0" (use 0.3" for high-quality 1 m-class, 1–3" for survey telescopes).         | Modern survey systems can reach sub-arcsec WCS solutions but small sensors are a few arcsec; use range for domain randomization. ([A&A][8]) |
| Optical exposure time                   |            s | 0.1 — 300 s (depends on instrument & search strategy)                                       | Use 1–10 s for surveys; 30–150 s for deep obs (streaks appear).                                                  | Exposure controls motion blur / streak length and detection probability. ([arXiv][3])                                                         |
| Optical detection probability / SNR     |            — | Varies with mag & exposure; assume P_det drops below SNR threshold                          | Model detection probability with logistic over magnitude; tune for desired completeness (e.g., 90% at mag ≤ 10). | Photometric detection practices; include false positives and dropouts.                                                                        |
| Radar range noise                       |       meters | tens to hundreds of meters depending on radar                                               | **Baseline radar range noise:** 10–100 m; Doppler (range-rate) noise: 0.001–0.05 km/s                            | High-performance radars are tens of meters accurate; older/small radars larger. Use literature reviews. ([MDPI][9])                           |
| Radar revisit / scan cadence            |      s / min | scanning radars: minutes between passes; dedicated tracking: continuous for seconds/minutes | Simulate scan cadences: revisit 5–30 minutes; tracklets lasting 30–300 s when detected.                          | Radar schedules control fragmentation of tracklets and measurement density. ([NASA Technical Reports Server][2])                              |
| Angular measurement noise (radar az/el) |          deg | 0.01°–0.5° depending on system                                                              | Use 0.01°–0.2° for modern radars / phased arrays; larger for small radars.                                       | Angular accuracy impacts triangulation and initial orbit determination. ([MDPI][9])                                                           |
| Image PSF / pixel scale                 | arcsec/pixel | 0.1–2 arcsec/pix typical                                                                    | Use 0.3–1.0 arcsec/pix for realistic star-tracking cameras                                                       | Important for image simulations & streak modeling. ([Optica Publishing Group][10])                                                            |

---

## D — Environment & calendar inputs

| Parameter                 |                  Units | Typical values / sampling   | Recommended sim values                                                             | Why / justification (refs)                                                                                   |
| ------------------------- | ---------------------: | --------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Solar radio flux (F10.7)  | sfu (solar flux units) | Quiet: ≈ 70 — Active: > 200 | Sample F10.7 ∈ {70, 100, 150, 200} for different solar conditions                  | Atmospheric density/drag depends strongly on solar activity; sampling is essential. ([AGU Publications][11]) |
| Geomagnetic activity (Ap) |                  index | Quiet ~0–10; storms >30     | Sample Ap ∈ {5, 15, 50} to represent quiet, moderate, active                       | Affects atmospheric density and upper-atmosphere dynamics; include in drag model (NRLMSISE).                 |
| Atmospheric model         |                      — | NRLMSISE-00 commonly used   | Use NRLMSISE-00 with F10.7 & Ap inputs for drag density profiles. ([Wikipedia][5]) |                                                                                                              |

---

## E — Simulation / numerical parameters

| Parameter                        |        Units | Typical choice / guidance                                                  | Recommendation                                                                                                                      |
| -------------------------------- | -----------: | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Integration timestep             |            s | 0.1–60 s depending on required accuracy                                    | **Default:** 10 s for medium-fidelity LEO propagation; decrease to 1 s for high-accuracy short arcs or fragments.                   |
| Collocation / training times     |            s | Use observation times + extra collocation points                           | Use observation times for measurement loss + additional uniform collocation every 10–60 s to enforce dynamics.                      |
| Simulation horizon               | hours / days | Hours for short-term tracking; days–years for environment evolution        | For training trackers: 1–72 hours. For environment studies: months–years.                                                           |
| Number of objects per experiment |        count | tens → thousands, depending on compute                                     | Start small (50–200) for development; scale to 1k–10k for population-level experiments.                                             |
| Propagator fidelity              |            — | SGP4 for TLE-consistency; numerical integrator + J2+drag for high fidelity | Use SGP4 when matching TLE behavior; use numerical integrator + J2/drag/SRP for labeled truth. ([NASA Technical Reports Server][2]) |

# Scenario presets — realistic combinations you can plug in

Below are **ready-to-use scenario presets** (each is a full set of parameter choices). Use these as seeds in your sims and vary them via domain randomization.

### Scenario 1 — Baseline LEO optical+radar survey (nominal)

* **Population:** 200 objects at altitudes 300, 500, 800, 1200 km (evenly split). Eccentricity ≤ 0.01. Inclinations: sample {53°, 98°}.
* **Object properties:** AMR distribution: log-uniform in [0.01, 1.0] m²/kg. (C_D = 2.2) nominal. Size bins 0.1–1 m typical.
* **Propagation:** numerical integrator with 2-body + J2 + drag (NRLMSISE-00; F10.7=100, Ap=5). Timestep = 10 s.
* **Sensors:** optical noise = 0.5" rms (exposure 5 s), detection dropouts 10%; radar range noise = 30 m, range-rate noise = 0.01 km/s. Radar revisit = 15 min.
* **Horizon:** 24 h, collocation points every 20 s.
* **Why:** Represents multi-sensor ops with moderate solar activity; good for training hybrid trackers. (See refs on AMR, NRLMSISE-00, optical/radar accuracy). ([ccd.aiub.unibe.ch][6])

### Scenario 2 — GEO optical survey (long-term)

* **Population:** 100 objects at GEO altitude (~35,786 km), low eccentricity.
* **Object properties:** AMR small (0.01–0.2 m²/kg), large sizes (>1 m). SRP more relevant; include SRP coefficient 1.2.
* **Propagation:** 2-body + SRP + geopotential truncated (J2 negligible at GEO); timestep = 60–300 s.
* **Sensors:** optical noise 0.2" (long exposures 30–120 s). Radar typically not used for GEO (skip radar).
* **Horizon:** 7 days (long-term behaviour of GEO station keeping).
* **Why:** GEO dynamics dominated by SRP, third-body perturbations; optical is primary. (Use SRP and solar indices). ([AGU Publications][11])

### Scenario 3 — High-noise small-telescope optical (edge case)

* **Population:** 300 objects at 500–1000 km. AMR uniform 0.05–2.0.
* **Sensor:** small survey camera (0.5–0.3 m aperture): optical noise 2–3" arcsec, exposure 30 s, detection prob lower (include 20–30% missed obs). Add variable cloud-weather dropout model.
* **Purpose:** Test robustness to low-SNR, sparse, and biased optical measurements. Useful for training a model to handle weak observations.

### Scenario 4 — Fragmentation cloud (stress test)

* **Event:** simulate fragmentation at t0 producing 50–500 fragments near 700 km altitude; fragments sized 0.01–0.5 m; AMR broad (0.01–10 m²/kg).
* **Propagation:** short-term high-fidelity with small timestep (1 s) for first hours (highly divergent orbits).
* **Sensors:** dense optical sampling in first 4–12 h (simulate targeted follow-up) + radar sweeps. Add large variability in ballistic coefficients.
* **Purpose:** test data-association, track initiation, and NP-SNN ability to handle dense, chaotic populations.

# How to sample & randomize (practical guidelines)

1. **Use stratified sampling over physically meaningful classes** (e.g., LAMR vs HAMR) rather than naive uniform sampling. This preserves tail behavior (rare HAMR objects). ([conference.sdo.esoc.esa.int][12])
2. **Domain randomization**: randomize sensor noise, exposure times, F10.7 & Ap indices, and ballistic coefficients within the preset ranges to reduce sim→real gap.
3. **Curriculum scheduling**: start training on “easy” scenarios (dense observations, low noise) and gradually introduce sparse/noisy scenarios (Scenario 3 & 4) to stabilize learning.
4. **Record sim metadata**: save seeds + full parameter set per generated track (AMR, C_D, F10.7, Ap) so you can later analyze failure modes.
5. **Validate distribution coverage**: compare marginal distributions of simulated orbital elements and magnitudes to public catalog snapshots (TLE/SATCAT) and tune priors accordingly. ([NASA Technical Reports Server][2])

# Short justification paragraph you can paste into your report

> We generated synthetic debris trajectories and sensor measurements using parameter ranges grounded in the SSA literature and operational practice. Orbital sampling was focused on the LEO regime (300–1200 km) where catalog density is highest, and dynamics included J2 perturbations and atmospheric drag driven by the NRLMSISE-00 density model with variable F10.7 and Ap indices. Object physical properties (area-to-mass, ballistic coefficient, size) were sampled across classes from low-AMR, dense fragments to HAMR populations to represent the wide range observed in catalog studies. Sensor models used astrometric noise levels spanning sub-arcsecond precision to several arcseconds, and radar noise consistent with surveillance radars (tens of meters in range). These choices are supported by SSA literature and atmospheric/observation models and were combined into scenario presets (baseline LEO survey, GEO survey, small-telescope edge case, fragmentation cloud) to exercise the NP-SNN model under realistic operating conditions. ([Wikipedia][5])

# References (select — use in report)

* NRLMSISE-00 atmospheric model documentation and usage (Picone et al., JGR 2002). ([AGU Publications][13])
* Area-to-mass ratio studies and HAMR objects (Herzog et al.; Schildknecht & Früh). ([ccd.aiub.unibe.ch][6])
* Radar measurement and detection overview for space debris (Muntoni et al., MDPI Applied Sciences 2021). ([MDPI][9])
* Typical LEO altitudes and debris concentrations (LEO definition + detection clusters). ([Wikipedia][1])
* Optical astrometry and instrument precisions (survey WCS RMS, telescope instrument specs). ([A&A

  ][8])
* Solar flux (F10.7) as driver of atmospheric density and drag. ([AGU Publications][11])


[1]: https://en.wikipedia.org/wiki/Low_Earth_orbit?utm_source=chatgpt.com "Low Earth orbit"
[2]: https://ntrs.nasa.gov/api/citations/20100005592/downloads/20100005592.pdf?utm_source=chatgpt.com "Current and Near-term Future Measurements of the Orbital ..."
[3]: https://arxiv.org/html/2507.14994v1?utm_source=chatgpt.com "Impact of Low-Earth Orbit Satellites on the China Space ..."
[4]: https://carleton.ca/spacecraft/wp-content/uploads/AAS-16-495.pdf?utm_source=chatgpt.com "AAS 16-495 NONLINEAR ANALYTICAL EQUATIONS OF ..."
[5]: https://en.wikipedia.org/wiki/NRLMSISE-00?utm_source=chatgpt.com "NRLMSISE-00"
[6]: https://ccd.aiub.unibe.ch/publist/data/2012/artproc/JH_AMOS2012.pdf?utm_source=chatgpt.com "analysis of the long-term area-to-mass ratio variation"
[7]: https://conference.sdo.esoc.esa.int/proceedings/sdc3/paper/8/SDC3-paper8.pdf?utm_source=chatgpt.com "SP-473 - Space Debris Proceedings"
[8]: https://www.aanda.org/articles/aa/full_html/2019/11/aa33294-18/aa33294-18.html?utm_source=chatgpt.com "The OTELO survey - I. Description, data reduction, and ..."
[9]: https://www.mdpi.com/2076-3417/11/4/1364?utm_source=chatgpt.com "A Review on Radar Measurements for Space Debris ..."
[10]: https://opg.optica.org/ao/abstract.cfm?uri=ao-59-11-3508&utm_source=chatgpt.com "Recommended optical system design for the SSST"
[11]: https://agupubs.onlinelibrary.wiley.com/doi/10.1002/swe.20064?utm_source=chatgpt.com "The 10.7 cm solar radio flux (F10.7) - Tapping - AGU Journals"
[12]: https://conference.sdo.esoc.esa.int/proceedings/sdc6/paper/170/SDC6-paper170.pdf?utm_source=chatgpt.com "HIGH AREA-TO-MASS RATIO OBJECT POPULATION ..."
[13]: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2002JA009430?utm_source=chatgpt.com "NRLMSISE‐00 empirical model of the atmosphere: Statistical ..."

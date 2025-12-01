# NP-SNN Orbit Visualizations

## Overview
This directory contains visualization outputs from the NP-SNN space debris tracking project's orbit analysis tools.

## Files Generated

### 1. orbit_examples.png
- **Purpose**: Comparison of major orbital regimes
- **Content**: LEO (ISS-like), MEO (GPS-like), and GEO (Geostationary) orbit examples
- **Key Features**: 
  - LEO: 408 km altitude, 51.6° inclination (similar to ISS)
  - MEO: 20,200 km altitude, 55° inclination (similar to GPS)
  - GEO: 35,786 km altitude, nearly equatorial (geostationary)

### 2. leo_debris_field.png
- **Purpose**: Visualization of crowded LEO debris environment
- **Content**: 50 randomly generated debris objects in LEO regime
- **Key Features**:
  - Quarter-orbit segments for spatial clarity
  - Color-coded objects for distinction
  - Demonstrates complex multi-object scenarios for neural network training

### 3. trajectory_propagation.png
- **Purpose**: Validation of orbital mechanics and J2 perturbation effects
- **Content**: 3-hour trajectory propagation with altitude evolution
- **Key Features**:
  - 3D trajectory visualization with start/end markers
  - Altitude vs. time plots showing J2 perturbation effects
  - Multiple object types (debris and operational satellites)

### 4. orbital_element_distributions.png
- **Purpose**: Statistical validation of dataset generator
- **Content**: Parameter distributions for LEO debris vs operational satellites
- **Key Features**:
  - 1000 samples each of debris and operational satellites
  - Altitude, inclination, and eccentricity distributions
  - Statistical validation of realistic parameter sampling

## Dataset Generator Statistics

### LEO Debris (n=1000)
- **Altitude**: 1085.6 ± 526.4 km
- **Inclination**: 91.0 ± 52.2°
- **Eccentricity**: 0.0502 ± 0.0195

### Operational Satellites (n=1000)
- **Altitude**: 9540.9 ± 12291.2 km (LEO to GEO range)
- **Inclination**: 42.4 ± 31.7°
- **Eccentricity**: 0.0094 ± 0.0053

## Usage in Reports
These visualizations are suitable for:
- Technical presentations and papers
- Dataset validation documentation
- Model architecture explanations
- Orbital mechanics verification

All images are saved at 300 DPI for publication quality.

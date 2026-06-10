# Paper Dataset Analysis Notes

This document collects analysis points that should be considered when deciding
how much synthetic data to generate and what to report in the paper.

## Main Question

The dataset should not only be larger. It should be measurably richer:

- richer scene composition;
- richer building/tree/water/road/ground distributions;
- richer viewpoint variation through mild near-nadir views;
- richer appearance variation through random seeds;
- richer geographic variation through multiple cities and rural/suburban areas.

The final paper should connect these dataset properties to downstream
performance, not only report qualitative generated images.

## Segmentation Distribution as Dataset Richness

For each generated or real tile, compute semantic composition from segmentation:

| Metric | Meaning |
|---|---|
| `road_ratio` | fraction of road pixels |
| `water_ratio` | fraction of water pixels |
| `foliage_ratio` | tree/foliage pixels |
| `building_ratio` | building pixels |
| `grass_ratio` | grass/open vegetation pixels |
| `ground_ratio` | bare/other ground pixels |
| `green_total_ratio` | `foliage + grass` |
| `class_entropy` | semantic diversity within a tile |
| `dominant_class` | largest semantic class in a tile |
| `rare_class_coverage` | whether water/building/foliage tails are covered |

The Omaha pilot already shows why this matters:

- building is common enough for dense-urban sampling;
- foliage has a long tail, so tree-rich scenes can be selected;
- water is rare, with only 19 of 984 Omaha tiles having `water_ratio >= 10%`;
- therefore water-rich sampling needs explicit oversampling or additional
  cities/regions.

This should become a standard dataset table:

```text
dataset/city | n_tiles | mean/p50/p90 building | mean/p50/p90 foliage |
mean/p50/p90 water | class entropy | rare class counts
```

## Omaha, US3D, and Additional City Comparison

We should compare semantic richness between:

- Omaha OSM pipeline output;
- US3D-derived segmentation;
- Jacksonville, Florida (`JAX`), the other current US3D city in this repo;
- future added cities.

The comparison should use the same class mapping where possible. If US3D labels
do not exactly match the OSM class set, define a reduced common taxonomy before
comparing.

Proposed common taxonomy:

```text
building
vegetation/tree
water
road/impervious
ground/open
unknown/ignored
```

The goal is to show whether the synthetic OSM pipeline fills gaps in US3D or
real data, for example:

- more water-rich scenes;
- more low-density rural scenes;
- more dense commercial building scenes;
- more tree-heavy suburban scenes;
- more road-grid variation.

## Near-Nadir Analysis

Near-nadir views should be treated as viewpoint augmentation, not just extra
samples.

Main hypothesis:

- `root`, `near-nadir-1`, and `near-nadir-2` improve robustness to mild
  off-nadir satellite imagery;
- `near-nadir-3` is uncertain and should be evaluated;
- `near-nadir-4` is likely too slanted and should be excluded from the main
  recipe unless downstream ablation proves otherwise.

Report:

- artifact/failure rate by view;
- downstream performance by view set;
- qualitative examples showing when slant helps or hurts;
- label-image alignment concerns for aggressive off-nadir views.

Recommended ablation:

```text
root only
root + nn1
root + nn1 + nn2
root + nn1 + nn2 + nn3
root + nn1 + nn2 + nn3 + nn4 as stress test
```

## Random Seed Analysis

Random seeds should be treated as appearance augmentation.

Confirmed screening seeds:

```text
1,2,4,8,16,32,64,128
```

Main hypothesis:

- multiple seeds improve texture/color/style robustness;
- some seeds may have higher artifact rates;
- seed quality should be evaluated statistically, not from one attractive
  example.

Report:

- artifact/failure rate by seed;
- artifact/failure rate by `view x seed`;
- downstream performance for 1 seed vs multiple seeds;
- whether seed diversity helps all scene types equally.

Recommended ablation:

```text
1 seed
2 seeds
4 seeds
8 seeds
```

## Artifact Screening as Pre-Training Quality Control

Artifact screening should remove obviously broken images, not decide final
scientific value.

Use reproducible metrics:

- brightness and clipping;
- contrast;
- saturation;
- blur/sharpness proxy;
- entropy/detail proxy;
- weak edge alignment with segmentation/depth boundaries.

Then report failure rates by:

```text
view
seed
city
scene group
view x seed
view x scene group
```

Downstream task performance remains the final evaluation.

## Multi-City Dataset Plan

The final dataset should include 5 to 10 cities/regions, not just Omaha.

Desired coverage:

- dense urban downtown;
- suburban residential;
- industrial/commercial areas;
- tree-heavy suburbs;
- water/river/lake/coastal scenes;
- rural or small-town scenes;
- agricultural/open-ground scenes.

Candidate expansion strategy:

1. Keep Omaha as the first controlled pilot.
2. Add Jacksonville, Florida (`JAX`), the other current US3D city in this repo.
3. Add several rural or small-town areas where OSM coverage is good and scene
   composition differs from Omaha.
4. For every city, compute the same segmentation richness table before deciding
   how many tiles to generate.

The generation budget per city should be based on distribution gaps, not a
fixed equal number of tiles. For example, if Omaha lacks water, another city can
contribute more water-rich scenes.


## Near-Term Additional City Candidates

Existing US3D city pair:

- Omaha, Nebraska (`OMA`)
- Jacksonville, Florida (`JAX`)

Recommended three additional North-American candidates for the next OSM scan:

| Candidate | Why it is useful | Expected role |
|---|---|---|
| Wichita, Kansas | Great Plains/Midwest urban form; population and land area are in the Omaha scale range; likely low-rise/suburban/industrial with open land. | Omaha-like city with slightly different road/building/green distribution. |
| Des Moines, Iowa | Midwest capital/metro with suburbs, rivers, tree cover, and medium-density urban core. | Omaha-like but smaller; good for validating whether Omaha results transfer within the Midwest. |
| Tampa, Florida | Florida city with water/coastal/vegetation patterns closer to Jacksonville; also appears in CORE3D public data context. | Jacksonville-like water/vegetation-heavy counterweight to Midwest cities. |

Backup candidates:

- Lincoln, Nebraska: very Omaha-like, useful as a conservative control, but may
  be too close geographically and visually if we want broader diversity.
- Kansas City, Missouri: larger than Omaha but useful if we want a bigger
  Midwest metro with river/water and dense-suburban variation.
- Richmond, Virginia: also appears in CORE3D public data context and provides a
  mid-Atlantic urban form, but it may differ more from Omaha/Jacksonville.

The final choice should not be based only on city-level population or area. For
paper-grade evidence, run the same OSM tile planning and segmentation statistics
on each candidate, then compare them by the richness metrics above.

Recommended pre-scan per candidate:

```text
city -> generate OSM tile plan -> render/collect seg only if possible -> compute
building/foliage/water/road/ground ratios -> compare against Omaha/JAX/US3D
```

Selection target:

- keep cities whose building-ratio distribution overlaps Omaha/JAX enough to be
  comparable;
- prefer cities that add missing tails, especially water-rich or rural/open
  scenes;
- avoid cities whose OSM building/road coverage is visibly sparse or noisy.

## Paper-Level Figures and Tables

Potential paper artifacts:

1. Segmentation distribution histograms by dataset/city.
2. Ternary or scatter plots:
   - building vs foliage vs water;
   - building vs road;
   - foliage vs grass.
3. Tile preview grids for selected scene groups:
   - building-rich;
   - tree-rich;
   - water-rich;
   - rural/open.
4. Viewpoint comparison:
   - root, nn1, nn2, nn3, nn4 for the same tile.
5. Seed comparison:
   - same tile/view across selected seeds.
6. Downstream ablation table:
   - real only;
   - real + synthetic root;
   - real + synthetic root multi-seed;
   - real + synthetic mild near-nadir multi-seed;
   - optional nn3/nn4 stress test.

## Open Items

- Define the common taxonomy for US3D-vs-OSM segmentation comparison.
- Choose 5 to 10 final cities/regions after scanning OSM and available real
  imagery coverage.
- Decide whether rural scenes should be a separate sampling group, alongside
  building/tree/water.
- Decide the final downstream task and split protocol before claiming robustness
  gains.

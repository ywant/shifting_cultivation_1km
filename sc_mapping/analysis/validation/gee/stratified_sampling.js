/*
 * Stratified validation sampling for the shifting cultivation classifier
 * ======================================================================
 * Generates a spatially stratified random sample for accuracy assessment
 * of the pantropical 1 km shifting cultivation map.
 *
 * Stratification design
 * ---------------------
 * Four strata are defined within the region of interest to ensure
 * validation samples capture both commission and omission errors:
 *
 *   Stratum 1 — Confirmed SC      : model predicts SC AND WRI GDM
 *                                    agree (high-confidence positive)
 *   Stratum 2 — Commission check  : model predicts SC but WRI GDM does not
 *                                    (possible false positive)
 *   Stratum 3 — Omission check    : model does NOT predict SC, but WRI GDM
 *                                    labels it as SC AND model probability is
 *                                    intermediate (possible false negative)
 *   Stratum 4 — Background        : all remaining pixels within the ROI
 *
 * Reference datasets
 * ------------------
 * - Model prediction  : EfficientNet-B1 SC classifier output (this paper)
 * - Model probability : Maximum softmax probability for the SC class
 * - WRI GDM (v1.2)    : Global forest-loss driver classification
 *                       (class 3 = shifting cultivation)
 *                       https://www.globalforestwatch.org/
 *
 * Output
 * ------
 * A CSV of stratified sample points exported to Google Drive
 * (folder: GEE_exports, file: stratified_validation_samples_centroids.csv).
 * Each point carries its stratum label; field validation is performed
 * offline using high-resolution imagery in Google Earth Pro.
 *
 * How to run
 * ----------
 * 1. Open https://code.earthengine.google.com
 * 2. Paste this script into a new file
 * 3. The asset paths are pre-filled; assets are publicly accessible
 * 4. Click Run, then submit the export task from the Tasks tab
 *
 * Seed: 42 (fixed for reproducibility)
 */


// =============================================================================
// CONFIGURATION  —  edit these values before running
// =============================================================================

var CONFIG = {
  // Probability thresholds defining the "intermediate" zone for Stratum 3
  intermediateLow  : 0.33,
  intermediateHigh : 0.67,

  // Number of random sample points per stratum
  // [stratum1, stratum2, stratum3, stratum4]
  samplesPerStratum: [100, 100, 100, 100],

  // Sampling scale in metres (should match the map resolution: 1 km)
  scale: 1000,

  // Buffer radius (pixels at CONFIG.scale) around predicted SC pixels
  // used to define the neighbourhood for stratum 3
  bufferRadiusPx: 250,

  // Random seed — do not change after sampling to ensure reproducibility
  seed: 42,

  // Google Drive export settings
  exportFolder     : 'GEE_exports',
  exportDescription: 'stratified_validation_samples_centroids'
};


// =============================================================================
// ASSET IMPORTS
// =============================================================================

// Region of interest: 1-degree grid cells covering tropical forest extent,
// produced by intersecting the pantropical prediction footprint with the
// tropical forest biome boundary.
var intersect_area = ee.FeatureCollection(
  'projects/ee-wanting/assets/intersect_area'
);

// SC model prediction raster: argmax class index (0-4) at 1 km resolution.
// Class legend: 0=Forest, 1=SC, 2=SecVeg/Agr, 3=Agriculture, 4=Mosaic.
// Two-date fusion of December 2019 + June 2020 NICFI basemaps.
var modelPred = ee.Image(
  'projects/ee-wanting/assets/pred_ori_oct_all_Dec20'
);

// SC model probability raster: maximum softmax probability for class 1 (SC).
var modelProb = ee.Image(
  'projects/ee-wanting/assets/conf_ori_oct_all_Dec20'
);

// WRI Global Deforestation Monitor (GDM) forest-loss driver classification
// at 1 km resolution (v1.2, 2001-2024). Class 3 = shifting cultivation.
// Source: https://www.globalforestwatch.org/
var auxSims = ee.Image(
  'projects/landandcarbon/assets/wri_gdm_drivers_forest_loss_1km/v1_2_2001_2024'
);

// Additional assets used in related GEE scripts (not used in this script):
//   SC prediction points (patch-level accuracy assessment):
//     ee.FeatureCollection("projects/ee-wanting/assets/shiftingcultivation/droplast_sc_points")
//   Country boundaries (FAO GAUL, for regional breakdowns):
//     ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0")


// =============================================================================
// PREPROCESSING
// =============================================================================

var regionOfInterest = intersect_area.geometry();

// Clip model outputs to ROI; assign class 5 (outside) to unmasked pixels
var pred_clipped = modelPred.clip(intersect_area).unmask(5);
var prob_clipped = modelProb.clip(intersect_area).unmask(5);

// Extract the Sims driver classification band
var forestDriver = auxSims.select('classification');


// =============================================================================
// BINARY MASKS
// =============================================================================

// Model predicts shifting cultivation (class 1)
var isShifting = pred_clipped.unmask(0).eq(1);

// Sims et al. labels pixel as shifting cultivation (class 3)
var simsIsShifting = forestDriver.unmask(0).eq(3);

// Intermediate model probability (uncertain predictions)
var intermediateProb = prob_clipped
  .gte(CONFIG.intermediateLow)
  .and(prob_clipped.lt(CONFIG.intermediateHigh));

// 1 km neighbourhood buffer around predicted SC pixels
// (used to widen the omission-check stratum)
var shiftingBuffer = isShifting.focal_max({
  radius: CONFIG.bufferRadiusPx,
  units : 'pixels'
});


// =============================================================================
// STRATIFICATION
// =============================================================================

// Stratum 1: Confirmed SC — model and Sims agree
var stratum1 = isShifting.and(simsIsShifting);

// Stratum 2: Commission check — model says SC, Sims does not
var stratum2 = isShifting.and(simsIsShifting.not());

// Stratum 3: Omission check — intermediate model probability,
//            not predicted as SC, but Sims labels it SC
var stratum3 = intermediateProb
  .and(isShifting.not())
  .and(simsIsShifting);

// Stratum 4: Background — all remaining pixels within the ROI
var stratum4 = ee.Image(1)
  .clip(regionOfInterest)
  .and(stratum1.not())
  .and(stratum2.not())
  .and(stratum3.not());

// Combine into a single integer band (values 1–4)
var strataMap = ee.Image(0)
  .where(stratum1, 1)
  .where(stratum2, 2)
  .where(stratum3, 3)
  .where(stratum4, 4)
  .rename('stratum')
  .toInt()
  .clip(intersect_area);


// =============================================================================
// DIAGNOSTICS  —  pixel counts per stratum (print to Console)
// =============================================================================

var strataPixelCounts = strataMap.reduceRegion({
  reducer   : ee.Reducer.frequencyHistogram(),
  geometry  : regionOfInterest,
  scale     : CONFIG.scale,
  maxPixels : 1e13
});
print('Pixel counts per stratum:', strataPixelCounts);


// =============================================================================
// STRATIFIED RANDOM SAMPLING
// =============================================================================

var samples = strataMap.stratifiedSample({
  numPoints  : CONFIG.samplesPerStratum[0],  // default per class (overridden below)
  classBand  : 'stratum',
  classValues: [1, 2, 3, 4],
  classPoints: CONFIG.samplesPerStratum,     // per-class override [100,100,100,100]
  region     : regionOfInterest,
  scale      : CONFIG.scale,
  geometries : true,
  seed       : CONFIG.seed
});

print('Total sample points:', samples.size());


// =============================================================================
// MAP VISUALISATION  (optional — for inspection only)
// =============================================================================

var predPalette = ['ForestGreen', 'Firebrick', 'LightSteelBlue', 'Gold', 'HotPink'];
var simsPalette = ['#E39D29', '#E58074', '#E9D700', '#51A44E',
                   '#895128', '#A354A0', '#3A209A'];
var strataPalette = ['red', 'yellow', 'pink', 'gray'];

Map.addLayer(pred_clipped,
  {min: 0, max: 4, palette: predPalette}, 'SC prediction (clipped)');
Map.addLayer(forestDriver,
  {min: 1, max: 7, palette: simsPalette}, 'Sims et al. forest driver');
Map.addLayer(strataMap,
  {min: 1, max: 4, palette: strataPalette}, 'Validation strata');
Map.addLayer(samples, {color: 'red'}, 'Sample points');
Map.centerObject(regionOfInterest, 4);


// =============================================================================
// EXPORT TO GOOGLE DRIVE
// =============================================================================

Export.table.toDrive({
  collection : samples,
  description: CONFIG.exportDescription,
  folder     : CONFIG.exportFolder,
  fileFormat : 'CSV'
});

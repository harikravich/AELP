CREATE TABLE IF NOT EXISTS `${PROJECT_ID}.${DATASET}.creative_visual_features` (
  ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  file STRING,
  motion_mean FLOAT64,
  motion_std FLOAT64,
  lap_var FLOAT64,
  tenengrad FLOAT64,
  colorfulness FLOAT64,
  legibility FLOAT64,
  persons INT64,
  phones INT64,
  lufs FLOAT64
);


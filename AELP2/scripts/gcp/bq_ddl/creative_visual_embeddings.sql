CREATE TABLE IF NOT EXISTS `${PROJECT_ID}.${DATASET}.creative_visual_embeddings` (
  ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  stem STRING,
  model STRING,
  dim INT64,
  embedding BYTES
);


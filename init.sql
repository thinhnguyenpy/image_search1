CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE leaf_metadata (
    id UUID PRIMARY KEY,
    species_name TEXT,
    scientific_name TEXT,
    image_path TEXT,
    detailed_features JSONB, 
    fused_vector vector(128) 
);

CREATE INDEX ON leaf_metadata USING hnsw (fused_vector vector_cosine_ops);
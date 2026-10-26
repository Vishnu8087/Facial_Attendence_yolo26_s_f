-- init/01-extensions.sql
-- Loaded automatically by pgvector image on first boot.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- fuzzy name search
CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; -- uuid_generate_v4()

-- Optional: tune for analytics workloads
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';

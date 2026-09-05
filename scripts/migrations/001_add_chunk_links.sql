-- Migration: Add chunk linking columns for section reconstruction
-- Run this before re-ingesting documents

-- Add navigation columns
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS prev_chunk_id UUID REFERENCES chunks(id);
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS next_chunk_id UUID REFERENCES chunks(id);

-- Add cached boundary detection columns (1=yes, 0=no, null=unknown)
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS is_section_start INTEGER;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS is_section_end INTEGER;

-- Add indexes for efficient navigation
CREATE INDEX IF NOT EXISTS idx_chunks_prev ON chunks(prev_chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunks_next ON chunks(next_chunk_id);

-- Verify the changes
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'chunks'
ORDER BY ordinal_position;

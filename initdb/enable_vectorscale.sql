
-- Połącz się z nową bazą i aktywuj rozszerzenia
\connect similarity_search_service_db;

CREATE EXTENSION IF NOT EXISTS vector CASCADE;
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;

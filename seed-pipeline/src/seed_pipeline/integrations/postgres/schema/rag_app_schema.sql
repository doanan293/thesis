CREATE TABLE IF NOT EXISTS users (
    user_id text PRIMARY KEY,
    email text UNIQUE,
    display_name text NOT NULL DEFAULT '',
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id text PRIMARY KEY,
    user_id text NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    title text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    message_id text PRIMARY KEY,
    session_id text NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role text NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content text NOT NULL,
    model_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS retrieval_runs (
    run_id text PRIMARY KEY,
    session_id text NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    message_id text NOT NULL REFERENCES chat_messages(message_id) ON DELETE CASCADE,
    corpus_version text NOT NULL,
    retriever_config jsonb NOT NULL DEFAULT '{}'::jsonb,
    query_text text NOT NULL,
    query_embedding_model text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS retrieval_hits (
    run_id text NOT NULL REFERENCES retrieval_runs(run_id) ON DELETE CASCADE,
    rank integer NOT NULL,
    chunk_id text NOT NULL,
    section_id text NOT NULL,
    table_id text NOT NULL DEFAULT '',
    qdrant_score double precision NOT NULL DEFAULT 0,
    rerank_score double precision,
    hydrate_strategy text NOT NULL,
    cited boolean NOT NULL DEFAULT false,
    snippet text NOT NULL DEFAULT '',
    PRIMARY KEY (run_id, rank)
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id text PRIMARY KEY,
    user_id text NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    session_id text NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    message_id text NOT NULL REFERENCES chat_messages(message_id) ON DELETE CASCADE,
    rating text NOT NULL DEFAULT '',
    note text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now()
);

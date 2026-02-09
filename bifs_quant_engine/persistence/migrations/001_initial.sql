-- Initial database schema for BIFS Quant Engine
-- Version: 001
-- Description: Core tables for orders, fills, positions, and audit

-- ─────────────────────────────────────────────────────────────────────────────
-- ORDERS TABLE
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    client_order_id TEXT UNIQUE,
    broker_order_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,  -- buy, sell, short, cover
    order_type TEXT NOT NULL DEFAULT 'market',
    quantity INTEGER NOT NULL,
    limit_price REAL,
    stop_price REAL,
    status TEXT NOT NULL DEFAULT 'pending',
    filled_quantity INTEGER DEFAULT 0,
    avg_fill_price REAL,
    strategy_id TEXT,
    parent_order_id TEXT,
    reject_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    submitted_at TIMESTAMP,
    filled_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_strategy ON orders(strategy_id);
CREATE INDEX IF NOT EXISTS idx_orders_created ON orders(created_at);

-- ─────────────────────────────────────────────────────────────────────────────
-- FILLS TABLE
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    broker_fill_id TEXT,
    broker_order_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price REAL NOT NULL,
    commission REAL DEFAULT 0,
    slippage REAL DEFAULT 0,
    filled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

CREATE INDEX IF NOT EXISTS idx_fills_order ON fills(order_id);
CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills(symbol);
CREATE INDEX IF NOT EXISTS idx_fills_filled_at ON fills(filled_at);

-- ─────────────────────────────────────────────────────────────────────────────
-- POSITIONS TABLE (Current positions snapshot)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    avg_cost REAL NOT NULL,
    realized_pnl REAL DEFAULT 0,
    borrow_rate REAL DEFAULT 0,
    is_hard_to_borrow INTEGER DEFAULT 0,
    last_fill_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol)
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);

-- ─────────────────────────────────────────────────────────────────────────────
-- PORTFOLIO SNAPSHOTS TABLE
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cash REAL NOT NULL,
    long_market_value REAL DEFAULT 0,
    short_market_value REAL DEFAULT 0,
    gross_exposure REAL DEFAULT 0,
    net_exposure REAL DEFAULT 0,
    total_equity REAL NOT NULL,
    daily_pnl REAL DEFAULT 0,
    realized_pnl REAL DEFAULT 0,
    unrealized_pnl REAL DEFAULT 0,
    leverage REAL DEFAULT 1,
    position_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON portfolio_snapshots(timestamp);

-- ─────────────────────────────────────────────────────────────────────────────
-- RISK STATES TABLE
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS risk_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    starting_equity REAL NOT NULL,
    current_equity REAL NOT NULL,
    high_water_mark REAL NOT NULL,
    daily_pnl REAL DEFAULT 0,
    daily_pnl_pct REAL DEFAULT 0,
    current_drawdown REAL DEFAULT 0,
    max_drawdown REAL DEFAULT 0,
    gross_exposure REAL DEFAULT 0,
    net_exposure REAL DEFAULT 0,
    long_exposure REAL DEFAULT 0,
    short_exposure REAL DEFAULT 0,
    largest_position_pct REAL DEFAULT 0,
    position_count INTEGER DEFAULT 0,
    daily_loss_breaker INTEGER DEFAULT 0,
    drawdown_breaker INTEGER DEFAULT 0,
    volatility_breaker INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_risk_states_timestamp ON risk_states(timestamp);

-- ─────────────────────────────────────────────────────────────────────────────
-- EQUITY CURVE TABLE
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS equity_curve (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL UNIQUE,
    starting_equity REAL NOT NULL,
    ending_equity REAL NOT NULL,
    daily_return REAL DEFAULT 0,
    high_water_mark REAL NOT NULL,
    drawdown REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_equity_curve_date ON equity_curve(date);

-- ─────────────────────────────────────────────────────────────────────────────
-- AUDIT LOG TABLE
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_type TEXT NOT NULL,
    entity_type TEXT,
    entity_id TEXT,
    action TEXT NOT NULL,
    details TEXT,  -- JSON
    user_id TEXT,
    ip_address TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_type, entity_id);

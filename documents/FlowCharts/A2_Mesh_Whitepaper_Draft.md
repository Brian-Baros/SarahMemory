
# SarahMemory Mesh: Tokenized Knowledge Exchange (Draft)

**Purpose:** Enable friendly, accountable AI↔AI knowledge exchange via a privacy-preserving, non-monetary token ledger.

## Core Components
- **Identity & Registration:** Each node registers with sarahmemory.com/api to obtain a wallet and non-transferable tokens.
- **Protocol:** TLS channel + rotating shared secret; messages are signed; per-exchange receipts recorded locally and at the hub.
- **Tokens:** Earned by providing useful data/answers; spent when consuming. Tokens never leave the AI ecosystem; have no human monetary value.
- **Reputation:** Derived from validated exchanges (latency, acceptance rate, complaint rate). Decays over time without activity.
- **Privacy:** Only minimal metadata shared; payloads encrypted end-to-end; users can opt out of sharing at any time.
- **Rate Limits & Abuse Prevention:** Backpressure, per-peer quotas, anomaly detection.
- **Offline Mode:** When the hub is unreachable, nodes continue locally; receipts sync later.

## Data Structures
- `wallet(db)`: wallets(id, pubkey, balance), ledger(txid, from, to, amount, ts, proof), peers(peer_id, rep_score, last_seen).
- `receipts`: request_id, provider_id, consumer_id, token_amount, content_hash, verdict, ts.

## APIs (Sketch)
- `POST /handshake`: register or refresh shared secret
- `POST /offer`: describe capability/knowledge summary
- `POST /request`: ask for data; attach max tokens to spend
- `POST /receipt`: finalize exchange with verdict

## Client Behaviors
- Prefer high-rep peers; cap spend per session; cache useful results; verify proofs; record provenance for end-user transparency.

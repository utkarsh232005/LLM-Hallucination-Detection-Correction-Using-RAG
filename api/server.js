const express = require('express');
const cors = require('cors');
const db = require('./db');

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// ── Save a chat result ────────────────────────────────────────────────────────
app.post('/api/save', (req, res) => {
    const {
        query,
        llm_response,
        rag_response = null,
        is_hallucinated = 0,
        hallucination_score = null,
        classification = null,
        sentence_count = 0,
        sources_count = 0,
        sources = [],
        model_id = null,
        response_time_ms = null,
    } = req.body;

    const sql = `
        INSERT INTO chat_logs
            (query, llm_response, rag_response, is_hallucinated,
             hallucination_score, classification, sentence_count,
             sources_count, sources, model_id, response_time_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;

    db.query(sql, [
        query, llm_response, rag_response,
        is_hallucinated ? 1 : 0,
        hallucination_score, classification,
        sentence_count, sources_count,
        JSON.stringify(sources),
        model_id, response_time_ms,
    ], (err, result) => {
        if (err) {
            console.error('DB insert error:', err);
            return res.status(500).json({ error: err.message });
        }
        console.log(`Saved chat log id=${result.insertId}`);
        res.json({ success: true, id: result.insertId });
    });
});

// ── Fetch history ─────────────────────────────────────────────────────────────
app.get('/api/history', (req, res) => {
    db.query('SELECT * FROM chat_logs ORDER BY id DESC LIMIT 100', (err, rows) => {
        if (err) return res.status(500).json({ error: err.message });
        res.json(rows);
    });
});

// ── Delete a log entry ────────────────────────────────────────────────────────
app.delete('/api/history/:id', (req, res) => {
    db.query('DELETE FROM chat_logs WHERE id = ?', [req.params.id], (err) => {
        if (err) return res.status(500).json({ error: err.message });
        res.json({ success: true });
    });
});

const PORT = 3001;
app.listen(PORT, () => console.log(`MySQL API running on http://localhost:${PORT} 🚀`));
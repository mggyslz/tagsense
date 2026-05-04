import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'history.db')


def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_conn() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS analysis_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
                text        TEXT NOT NULL,
                model       TEXT NOT NULL,
                label       TEXT NOT NULL,
                confidence  REAL NOT NULL,
                density     REAL NOT NULL,
                band        TEXT NOT NULL,
                lang_type   TEXT NOT NULL,
                emoji_count INTEGER NOT NULL
            )
        ''')
        conn.commit()


def log_result(text, model, label, confidence, density, band, lang_type, emoji_count):
    with get_conn() as conn:
        conn.execute('''
            INSERT INTO analysis_log
                (text, model, label, confidence, density, band, lang_type, emoji_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (text, model, label, confidence, density, band, lang_type, emoji_count))
        conn.commit()


def get_history(model_filter=None, sentiment_filter=None):
    query = 'SELECT * FROM analysis_log'
    params = []
    conditions = []
    if model_filter:
        conditions.append('model = ?')
        params.append(model_filter)
    if sentiment_filter:
        conditions.append('label = ?')
        params.append(sentiment_filter)
    if conditions:
        query += ' WHERE ' + ' AND '.join(conditions)
    query += ' ORDER BY timestamp DESC'
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]

def get_history_paged(model_filter=None, sentiment_filter=None, page=1, per_page=20):
    conditions = []
    params = []
    if model_filter:
        conditions.append('model = ?')
        params.append(model_filter)
    if sentiment_filter:
        conditions.append('label = ?')
        params.append(sentiment_filter)

    where = (' WHERE ' + ' AND '.join(conditions)) if conditions else ''

    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        total = conn.execute(
            f'SELECT COUNT(*) FROM analysis_log{where}', params
        ).fetchone()[0]

        offset = (page - 1) * per_page
        rows = conn.execute(
            f'SELECT * FROM analysis_log{where} ORDER BY timestamp DESC LIMIT ? OFFSET ?',
            params + [per_page, offset]
        ).fetchall()

    return [dict(r) for r in rows], total


def clear_history():
    with get_conn() as conn:
        conn.execute('DELETE FROM analysis_log')
        conn.commit()


def export_history_csv():
    rows = get_history()
    if not rows:
        return ''
    headers = ['id', 'timestamp', 'text', 'model', 'label', 'confidence',
               'density', 'band', 'lang_type', 'emoji_count']
    lines = [','.join(headers)]
    for r in rows:
        row_vals = []
        for h in headers:
            val = str(r.get(h, ''))
            if ',' in val or '"' in val or '\n' in val:
                val = '"' + val.replace('"', '""') + '"'
            row_vals.append(val)
        lines.append(','.join(row_vals))
    return '\n'.join(lines)


def export_history_json():
    import json
    rows = get_history()
    output = []
    for r in rows:
        band_raw = r.get('band', '')
        density_range = band_raw.split(':', 1)[1].strip() if ':' in band_raw else band_raw

        output.append({
            'id':                   r['id'],
            'timestamp':            r['timestamp'],
            'text':                 r['text'],
            'model':                r['model'],
            'source':               None,
            'language_type':        r['lang_type'].lower(),
            'code_switch_density':  round(r['density'], 4),
            'density_range':        density_range,
            'band':                 band_raw,
            'emoji_count':          r['emoji_count'],
            'has_emoji':            r['emoji_count'] > 0,
            'sentiment':            r['label'].lower(),
            'confidence':           round(r['confidence'], 4),
        })
    return json.dumps(output, indent=4, ensure_ascii=False)


init_db()
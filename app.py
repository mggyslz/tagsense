from flask import Flask, render_template, request, jsonify, Response
from predict import predict, predict_all, MODELS
from utils import compute_density, detect_emoji
from logger import log_result, get_history, get_history_paged, clear_history, export_history_csv
from enhancer import enhance_pre, enhance_post, enhance_post_compare

app = Flask(__name__)



@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/analyze')
def index():
    return render_template('index.html', models=list(MODELS.keys()))


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = (data.get('text') or '').strip()
    model_name = data.get('model', 'mbert')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Pre-process
    enhanced_text = enhance_pre(text)

    result = predict(enhanced_text, model_name)
    csd, band, lang_type, token_count = compute_density(text)   # use original for CSD
    has_emoji, emoji_count = detect_emoji(text)

    # Post-process (pass original text for sarcasm/contradiction pattern matching)
    result = enhance_post(result, text, predict_fn=lambda t: predict(t, model_name))

    true_label = result['label']
    true_conf  = result['confidence'].get(true_label.lower(), max(result['confidence'].values()))

    log_result(
        text=text,
        model=model_name,
        label=true_label,
        confidence=true_conf,
        density=csd,
        band=band,
        lang_type=lang_type,
        emoji_count=emoji_count,
    )

    return jsonify({
        'label':                result['label'],
        'label_id':             result['label_id'],
        'confidence':           result['confidence'],
        'csd':                  round(csd * 100, 2),
        'band':                 band,
        'lang_type':            lang_type,
        'token_count':          token_count,
        'has_emoji':            has_emoji,
        'emoji_count':          emoji_count,
        'is_placeholder':       result.get('is_placeholder', False),
        # Enhancement metadata
        'low_confidence':       result.get('low_confidence', False),
        'override_triggered':   result.get('override_triggered', False),
        'override_keyword':     result.get('override_keyword'),
        'contradiction_detected': result.get('contradiction_detected', False),
        'sarcasm_detected':     result.get('sarcasm_detected', False),
    })



@app.route('/batch')
def batch():
    return render_template('batch.html', models=list(MODELS.keys()))


@app.route('/batch/run', methods=['POST'])
def batch_run():
    data = request.get_json()
    raw_text = (data.get('text') or '').strip()
    model_name = data.get('model', 'mbert')

    if not raw_text:
        return jsonify({'error': 'No text provided'}), 400

    posts = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not posts:
        return jsonify({'error': 'No valid posts found'}), 400

    results = []
    for post in posts:
        enhanced_post = enhance_pre(post)
        r = predict(enhanced_post, model_name)
        csd, band, lang_type, token_count = compute_density(post)
        has_emoji, emoji_count = detect_emoji(post)
        r = enhance_post(r, post, predict_fn=lambda t: predict(t, model_name))

        true_label = r['label']
        true_conf  = r['confidence'].get(true_label.lower(), max(r['confidence'].values()))

        log_result(
            text=post,
            model=model_name,
            label=true_label,
            confidence=true_conf,
            density=csd,
            band=band,
            lang_type=lang_type,
            emoji_count=emoji_count,
        )
        results.append({
            'text':               post[:120] + ('…' if len(post) > 120 else ''),
            'full_text':          post,
            'label':              true_label,
            'confidence':         r['confidence'],
            'top_conf':           round(true_conf * 100, 1),
            'csd':                round(csd * 100, 2),
            'band':               band,
            'lang_type':          lang_type,
            'has_emoji':          has_emoji,
            'emoji_count':        emoji_count,
            # Enhancement metadata
            'low_confidence':     r.get('low_confidence', False),
            'override_triggered': r.get('override_triggered', False),
            'contradiction_detected': r.get('contradiction_detected', False),
            'sarcasm_detected':   r.get('sarcasm_detected', False),
        })

    dist = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    for r in results:
        dist[r['label']] = dist.get(r['label'], 0) + 1

    return jsonify({'results': results, 'distribution': dist})



@app.route('/compare')
def compare():
    return render_template('compare.html', models=list(MODELS.keys()))


@app.route('/compare/run', methods=['POST'])
def compare_run():
    data = request.get_json()
    text = (data.get('text') or '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    enhanced_text = enhance_pre(text)
    all_results = predict_all(enhanced_text)
    csd, band, lang_type, token_count = compute_density(text)
    has_emoji, emoji_count = detect_emoji(text)

    # Post-process all models + majority vote
    enhanced_all = enhance_post_compare(all_results, text, predict_fn=lambda t: predict(t, 'mbert'))
    vote_meta = enhanced_all.pop("__vote__")

    formatted = {}
    for model_name, r in enhanced_all.items():
        true_label = r['label']
        true_conf  = r['confidence'].get(true_label.lower(), max(r['confidence'].values()))

        log_result(
            text=text,
            model=model_name,
            label=true_label,
            confidence=true_conf,
            density=csd,
            band=band,
            lang_type=lang_type,
            emoji_count=emoji_count,
        )
        formatted[model_name] = {
            'label':                true_label,
            'label_id':             r['label_id'],
            'confidence':           r['confidence'],
            'top_conf':             round(true_conf * 100, 1),
            'low_confidence':       r.get('low_confidence', False),
            'override_triggered':   r.get('override_triggered', False),
            'contradiction_detected': r.get('contradiction_detected', False),
            'sarcasm_detected':     r.get('sarcasm_detected', False),
        }

    return jsonify({
        'models':       formatted,
        'csd':          round(csd * 100, 2),
        'band':         band,
        'lang_type':    lang_type,
        'token_count':  token_count,
        'has_emoji':    has_emoji,
        'emoji_count':  emoji_count,
        # Majority vote meta
        'vote':         vote_meta,
    })



@app.route('/history')
def history():
    from datetime import timezone, timedelta

    PHT = timezone(timedelta(hours=8))
    model_filter = request.args.get('model', '')
    sentiment_filter = request.args.get('sentiment', '')
    page = int(request.args.get('page', 1))
    per_page = 20

    rows, total_count = get_history_paged(
        model_filter or None,
        sentiment_filter or None,
        page=page,
        per_page=per_page
    )
    total_pages = max(1, (total_count + per_page - 1) // per_page)

    from datetime import datetime
    for row in rows:
        try:
            dt = datetime.fromisoformat(str(row['timestamp']).replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            row['timestamp'] = dt.astimezone(PHT).strftime('%Y-%m-%d %H:%M')
        except Exception:
            pass

    return render_template('history.html',
                           rows=rows,
                           models=list(MODELS.keys()),
                           model_filter=model_filter,
                           sentiment_filter=sentiment_filter,
                           page=page,
                           total_pages=total_pages,
                           total_count=total_count)


@app.route('/history/clear', methods=['POST'])
def history_clear():
    clear_history()
    return jsonify({'status': 'ok'})


@app.route('/history/export')
def history_export():
    csv_data = export_history_csv()
    return Response(
        csv_data,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=tagsense_history.csv'}
    )


@app.route('/history/export/json')
def history_export_json():
    from logger import export_history_json
    json_data = export_history_json()
    return Response(
        json_data,
        mimetype='application/json',
        headers={'Content-Disposition': 'attachment; filename=tagsense_history.json'}
    )



@app.route('/about')
def about():
    return render_template('about.html')



if __name__ == '__main__':
    app.run(debug=True, port=5000)
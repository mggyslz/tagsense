"""
Pre-processing:
  1. Text Normalization     — lowercase, repeated chars, abbreviations, symbols
  2. Slang & Taglish Mapping — informal terms → clearer sentiment words
  3. Emoji Handling         — inject sentiment cue words from emojis
  4. Negation Handling      — detect "not/hindi/di" and flip nearby words
  5. Amplifier Injection    — sobrang/napaka-/grabe + sentiment → boosted cues

Post-processing:
  6. Majority Voting        — used by compare mode (3 models)
  7. Confidence Flagging    — flag predictions below threshold (no override)
  8. Keyword Override       — strong unambiguous terms force a label
  9. Contradiction Handling — "good but bad" → Neutral
 10. Sarcasm Detection      — expanded: emojis, repetition, ironic praise, daw patterns
 11. Charot/Char Softener   — split at joker word, re-process after-part to determine final label
"""
import re

ABBREVIATIONS = {
    "idk":    "i don't know",
    "imo":    "in my opinion",
    "imho":   "in my humble opinion",
    "lol":    "laughing",
    "lmao":   "laughing",
    "rofl":   "laughing",
    "omg":    "oh my god",
    "omfg":   "oh my god",
    "wtf":    "what the",
    "wth":    "what the",
    "ngl":    "not going to lie",
    "tbh":    "to be honest",
    "irl":    "in real life",
    "smh":    "shaking my head",
    "brb":    "be right back",
    "btw":    "by the way",
    "rn":     "right now",
    "atm":    "at the moment",
    "nvm":    "never mind",
    "ikr":    "i know right",
    "iirc":   "if i recall correctly",
    "afaik":  "as far as i know",
    "fyi":    "for your information",
    "asap":   "as soon as possible",
    "eta":    "estimated time",
    "thx":    "thanks",
    "ty":     "thank you",
    "tysm":   "thank you so much",
    "np":     "no problem",
    "ok":     "okay",
    "k":      "okay",
    "bc":     "because",
    "b4":     "before",
    "ur":     "you are",
    "u":      "you",
    "r":      "are",
    "n":      "and",
    "2":      "to",
    "4":      "for",
    
    "dba":    "di ba",
    "diba":   "di ba",
    "kasi":   "kasi",
    "cguro":  "siguro",
    "cgro":   "siguro",
    "tlga":   "talaga",
    "tlaga":  "talaga",
    "nmn":    "naman",
    "nman":   "naman",
    "wlang":  "walang",
    "wla":    "wala",
    "pls":    "please",
    "tnx":    "thanks",
    "thnx":   "thanks",
    "hnd":    "hindi",
    "hndi":   "hindi",
    "ndi":    "hindi",
    "pede":   "pwede",
    "di":     "di",
    "gng":    "ganong",
    "khit":   "kahit",
    "lng":    "lang",
    "nlng":   "na lang",
    "nlang":  "na lang",
    "aq":     "ako",
    "nxt":    "next",
    "msg":    "message",
    "pic":    "picture",
    "pics":   "pictures",
    "vid":    "video",
    "vids":   "videos",
    "gr8":    "great",
    "l8r":    "later",
    "str8":   "straight",
    "fav":    "favorite",
    "bf":     "boyfriend",
    "gf":     "girlfriend",
    "bff":    "best friend",
    "ldr":    "long distance relationship",
}

def normalize_text(text: str) -> str:
    """Lowercase, collapse repeated chars, expand abbreviations, strip junk."""
    text = text.lower()
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[!?]{2,}', '!', text)

    tokens = text.split()
    expanded = []
    for tok in tokens:
        clean_tok = re.sub(r"[^\w]", "", tok)
        if clean_tok in ABBREVIATIONS:
            expanded.append(ABBREVIATIONS[clean_tok])
        else:
            expanded.append(tok)
    text = ' '.join(expanded)
    return text.strip()


SLANG_MAP = {
    
    "angas":        "impressive",
    "angst":        "impressive",
    "petmalu":      "amazing",
    "lupet":        "amazing",
    "werpa":        "great",
    "lodi":         "idol great",
    "lods":         "idol great",
    "idol":         "great admiration",
    "solid":        "very good",
    "legit":        "genuine good",
    "bet":          "i like it",
    "sana all":     "i wish lucky",
    "sanaol":       "i wish lucky",
    "gigil":        "overwhelmingly cute positive",
    "kilig":        "happy excited romantic",
    
    
    
    "naks":         "impressed",
    "galing":       "talented good",
    "husay":        "skilled good",
    "ganda":        "beautiful positive",
    "maganda":      "beautiful positive",
    "astig":        "cool impressive",
    "sosyal":       "classy impressive",
    "chill":        "relaxed okay",
    "swabe":        "smooth good",
    "swak":         "fits perfectly good",
    "pogi":         "handsome positive",
    "cute":         "cute positive",
    "gwap":         "handsome positive",
    "gwapo":        "handsome positive",
    "maporma":      "stylish positive",
    "bongga":       "fabulous amazing",
    "slay":         "excellent",
    "grabe":        "intense very",
    "sobra":        "very much",
    "super":        "very",
    "todo":         "full effort",
    "sulit":        "worth it good value",
    "tipid":        "saves money good",

    
    "basura":       "garbage bad",
    "bwisit":       "annoying bad",
    "gago":         "stupid bad",
    "tangina":      "curse bad",
    "putangina":    "curse bad",
    "leche":        "curse bad",
    "bobo":         "stupid bad",
    "tanga":        "stupid bad",
    "panget":       "ugly bad",
    "pangit":       "ugly bad",
    "pangget":      "ugly bad",
    "nakakainis":   "annoying negative",
    "nakakaiyak":   "sad negative",
    "nakakagalit":  "anger negative",
    "sakit":        "pain negative",
    "masakit":      "painful negative",
    "malas":        "unlucky bad",
    "baliw":        "crazy negative",
    "loka":         "crazy negative",
    "loko":         "crazy negative",
    "asar":         "annoying bad",
    "nakaka-asar":  "annoying bad",
    "nakakaasar":   "annoying bad",
    "inis":         "annoyed bad",
    "galit":        "angry bad",
    "malungkot":    "sad negative",
    "lungkot":      "sadness negative",
    "iyak":         "cry sad",
    "umiyak":       "crying sad",
    "pagod":        "tired negative",
    "hirap":        "difficult hard negative",
    "mahirap":      "difficult hard negative",
    "kawawa":       "pitiful sad",
    "badtrip":      "annoying bad experience",
    "bad trip":     "annoying bad experience",
    "nakaka-badtrip": "annoying bad",
    "nakakabadtrip": "annoying bad",
    "bitin":        "disappointing not enough",
    "sayang":       "waste disappointing",
    "tsk":          "disapproval negative",
    "huhu":         "sad crying",
    "nako":         "dismay negative",
    "jusko":        "dismay negative",
    "sus":          "dismay doubt negative",
    "demonyo":      "devil bad",

    
    "downgrade":        "worse regression bad",
    "naglalag":         "lagging slow frustrating bad",
    "lag":              "slow frustrating bad",
    "lagging":          "slow frustrating bad",
    "nag-crash":        "crashed broken bad",
    "crash":            "broken failed bad",
    "sira":             "broken bad",
    "nasira":           "broken damaged bad",
    "hindi naayos":     "not fixed unresolved bad",
    "hindi pa naayos":  "still not fixed bad",
    "bugs":             "errors problems bad",
    "glitch":           "error broken bad",
    "glitchy":          "broken unreliable bad",
    "not working":      "broken failed bad",
    "ayaw gumana":      "not working broken bad",
    "hindi gumagana":   "not working broken bad",
    "di gumagana":      "not working broken bad",
    "di na-update":     "not updated bad",
    "outdated":         "old outdated bad",
    "mabagal":          "slow bad",
    "bagal":            "slow frustrating bad",
    "ang bagal":        "very slow frustrating bad",
    "nag-hang":         "frozen broken bad",
    "hang":             "frozen unresponsive bad",
    "error":            "problem broken bad",
    "mali":             "wrong bad",
    "hindi nag-aayos":  "not fixing negligent bad",
    "pabaya":           "negligent irresponsible bad",
    "palpak":           "failed botched bad",
    "sablay":           "failed disappointing bad",
    "bulok":            "rotten terrible bad",
    "walang kwenta":    "worthless useless bad",
    "wala talagang kwenta": "completely worthless bad",
    "scam":             "fraud bad",
    "ripoff":           "overpriced bad unfair",
    "nanloko":          "deceived scammed bad",
    "lokohin":          "deceive cheat bad",

    
    "naayos na":        "fixed resolved good",
    "gumana na":        "working now good",
    "updated na":       "updated improved good",
    "nag-improve":      "improved better good",
    "mas mabilis":      "faster better good",
    "mabilis":          "fast efficient good",
    "maayos":           "proper good well-done",
    "ayos":             "good fine okay",
    "ayos na":          "fixed good resolved",
    "sulit na sulit":   "very worth it excellent value",
    "worth it":         "worth it good value",
    "maginhawa":        "convenient comfortable good",
    "madaling gamitin": "easy to use good",
    "user-friendly":    "easy good convenient",
    "seamless":         "smooth effortless good",
    "highly recommend": "highly recommend excellent",

    
    "daw":              "allegedly supposedly",
    "raw":              "allegedly supposedly",
    "kunwari":          "pretend fake",
    "parang hindi":     "seems not doubtful",
    "actually":         "actually contrasting",
    "talaga naman":     "really exasperated",
    "naman talaga":     "really exasperated",
    "ano ba yan":       "what the frustrated bad",
    "ano ba":           "frustrated annoyed",
    "grabe naman talaga": "excessively much intense negative",
    "grabe naman":      "too much intense",

    
    "basta":        "just",
    "ganoon":       "like that",
    "ganun":        "like that",
    "ganito":       "like this",
    "siguro":       "maybe",
    "parang":       "like seems",
    "mukha":        "looks like",
    "feeling":      "feels like",
    "ewan":         "i don't know uncertain",
    "aywan":        "i don't know uncertain",
    "haha":         "laughing",
    "hehe":         "laughing",
    "hihi":         "laughing",
    "ampota":       "curse frustration",
}

def map_slang(text: str) -> str:
    """Replace known slang/Taglish terms with sentiment-clearer equivalents."""
    for slang, replacement in sorted(SLANG_MAP.items(), key=lambda x: -len(x[0])):
        pattern = r'\b' + re.escape(slang) + r'\b'
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text






EMOJI_SENTIMENT = {
    
    "😍": "very positive love",
    "🥰": "very positive love",
    "😘": "positive love",
    "❤️": "love positive",
    "🧡": "love positive",
    "💛": "love positive",
    "💚": "love positive",
    "💙": "love positive",
    "💜": "love positive",
    "🖤": "love",
    "🤍": "love",
    "💕": "love positive",
    "💞": "love positive",
    "💓": "love positive",
    "💗": "love positive",
    "💖": "love positive",
    "💝": "love positive",
    "😊": "happy positive",
    "😁": "very happy positive",
    "😄": "very happy positive",
    "😀": "happy positive",
    "🥳": "celebrating very happy",
    "🎉": "celebration happy",
    "🎊": "celebration happy",
    "🏆": "winner great success",
    "⭐": "great positive",
    "🌟": "amazing positive",
    "✨": "positive sparkle",
    "👍": "good positive approve",
    "👌": "perfect okay positive",
    "🙌": "praise very good",
    "👏": "applause great",
    "🤩": "amazing positive star-struck",
    "😎": "cool positive confident",
    "🥹": "touched positive emotional",
    "😂": "very funny laughing",
    "🤣": "very funny laughing",
    "😆": "funny laughing",
    "😋": "yummy positive",
    "😇": "pure good positive",

    
    "😢": "sad crying negative",
    "😭": "very sad crying negative",
    "😤": "frustrated angry negative",
    "😠": "angry negative",
    "😡": "very angry negative",
    "🤬": "furious very angry negative",
    "😞": "disappointed negative",
    "😔": "sad disappointed negative",
    "😟": "worried negative",
    "😣": "struggling negative",
    "😩": "exhausted negative",
    "😫": "very tired negative",
    "🥺": "sad pleading negative",
    "😰": "anxious scared negative",
    "😨": "scared negative",
    "😱": "shocked scared negative",
    "😖": "frustrated negative",
    "💔": "broken heart sad negative",
    "👎": "bad disapprove negative",
    "🤢": "disgusted negative",
    "🤮": "disgusted very bad negative",
    "🙄": "eye-roll sarcastic doubt negative",
    "😒": "unamused negative",
    "🤦": "facepalm negative disbelief",
    "🤦‍♂️": "facepalm negative disbelief",
    "🤦‍♀️": "facepalm negative disbelief",
    "😑": "expressionless annoyed negative",
    "😐": "neutral flat",
    "🙃": "ironic sarcastic negative",     
    "☹️": "sad negative",
    "😓": "disappointed sweating negative",
    "🥲": "sad trying to smile bittersweet",
    "🫠": "melting overwhelmed negative",  
    "😮‍💨": "exhausted sigh negative",
    "🫡": "salute sarcastic",
    "💀": "dead done finished sarcastic negative",  
    "🗿": "stone face expressionless sarcastic",

    
    "🤔": "thinking uncertain neutral",
    "😶": "speechless neutral",
    "🤷": "uncertain don't know neutral",
    "🤷‍♂️": "uncertain don't know neutral",
    "🤷‍♀️": "uncertain don't know neutral",
    "😅": "relieved awkward nervous",
    "😬": "awkward uncertain uncomfortable",
    "🫤": "uncertain neutral",
}

def inject_emoji_sentiment(text: str) -> str:
    """Append sentiment cue words from emojis instead of removing them."""
    injected = []
    for emoji_char, sentiment_words in EMOJI_SENTIMENT.items():
        if emoji_char in text:
            injected.append(sentiment_words)
    if injected:
        text = text + ' ' + ' '.join(injected)
    return text






def enhance_pre(text: str) -> str:
    """
    Run all pre-processing steps on raw input text.
    Returns a cleaned, enriched string ready for model inference.
    """
    text = inject_emoji_sentiment(text)   
    text = normalize_text(text)           
    text = map_slang(text)                
    text = handle_negation(text)          
    text = inject_amplifiers(text)        
    return text






POSITIVE_WORDS = {
    "good", "great", "amazing", "wonderful", "fantastic", "excellent",
    "awesome", "beautiful", "perfect", "nice", "love", "happy", "best",
    "positive", "brilliant", "outstanding", "superb", "incredible",
    "pleasant", "enjoy", "enjoyed", "impressive", "glowing", "proud",
    "thrilled", "delighted", "glad", "grateful", "satisfied", "helpful",
    "recommend", "fast", "smooth", "seamless", "convenient", "easy",
    
    "maganda", "ganda", "galing", "masaya", "mabuti", "husay",
    "maayos", "ayos", "mabilis", "maginhawa", "sulit", "bet",
    "solid", "astig", "bongga", "slay", "lupet", "petmalu",
}

NEGATIVE_WORDS = {
    "bad", "terrible", "horrible", "awful", "disgusting", "hate",
    "worst", "ugly", "poor", "disappointing", "negative", "failed",
    "broken", "useless", "wrong", "garbage", "trash", "pathetic",
    "sad", "angry", "annoying", "frustrating", "painful", "depressing",
    "slow", "crash", "lag", "lagging", "error", "bug", "bugs",
    "downgrade", "worse", "scam", "ripoff", "glitch",
    
    "panget", "pangit", "basura", "bwisit", "masamang", "masama",
    "sira", "palpak", "sablay", "bulok", "badtrip", "asar",
    "mabagal", "pabaya", "mali",
    
    "annoying", "stupid", "curse", "unlucky", "pitiful",
    "worthless", "fraud", "overpriced", "rotten", "negligent",
    "crashed", "lagging", "unreliable", "regression", "frozen",
}

NEGATION_WORDS = {
    "not", "never", "no", "neither", "nor", "nobody", "nothing",
    "nowhere", "hardly", "scarcely", "barely", "without", "lack",
    
    "hindi", "hinde", "di", "wala", "walang", "huwag", "wag",
    "hnd", "hndi", "ndi", "ayaw", "ayawan",
}


NEGATION_PHRASES = [
    (r'\bnever again\b',                "absolutely refuse terrible"),
    (r'\bhindi (na\s+)?gumagana\b',     "not working broken bad"),
    (r'\bdi (na\s+)?gumagana\b',        "not working broken bad"),
    (r'\bayaw (na\s+)?gumana\b',        "refuses to work broken bad"),
    (r'\bhindi (pa\s+)?naayos\b',       "still not fixed bad unresolved"),
    (r'\bdi (pa\s+)?naayos\b',          "still not fixed bad unresolved"),
    (r'\bwalang kwenta\b',              "worthless useless bad"),
    (r'\bhindi maganda\b',              "not good bad"),
    (r'\bdi maganda\b',                 "not good bad"),
    (r'\bhindi okay\b',                 "not okay bad"),
    (r'\bdi okay\b',                    "not okay bad"),
    
    (r'\bhindi (naman\s+)?masama\b',    "actually good not bad"),
    (r'\bdi (naman\s+)?masama\b',       "actually good not bad"),
    (r'\bnot (that\s+)?bad\b',          "acceptable okay good"),
    (r'\bnot bad at all\b',             "good positive acceptable"),
    
    (r'\bhindi (naman\s+)?pangit\b',    "not ugly okay good"),
    (r'\bdi (naman\s+)?pangit\b',       "not ugly okay good"),
    (r'\bnot (so\s+)?terrible\b',       "tolerable okay"),
    (r'\bnot (so\s+)?bad\b',            "acceptable okay"),
]

def handle_negation(text: str) -> str:
    """
    1. Collapse double-negation tokens FIRST ("hindi hindi masama" → "masama")
       so phrase-level rules don't fire on a partial match like "hindi masama"
       inside "hindi hindi masama".
    2. Replace known multi-word negation phrases (most specific).
    3. Token-level: annotate sentiment words within a 6-word window after
       any negation word with a 'negated_' prefix so the model sees them
       as semantically flipped.
    """
    
    text = re.sub(r'\b(hindi|di|not)\s+\1\b', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'\s{2,}', ' ', text)

    
    for pattern, replacement in NEGATION_PHRASES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    
    tokens = text.split()
    result = list(tokens)
    i = 0
    while i < len(tokens):
        if tokens[i].lower().rstrip('.,!?') in NEGATION_WORDS:
            window_end = min(i + 7, len(tokens))
            for j in range(i + 1, window_end):
                word = tokens[j].lower().rstrip('.,!?')
                if word in POSITIVE_WORDS or word in NEGATIVE_WORDS:
                    result[j] = "negated_" + tokens[j]
                    break
            i += 1
        else:
            i += 1
    return ' '.join(result)


AMPLIFIERS = {
    
    "sobrang":      "very extremely",
    "napaka":       "very extremely",
    "napaka-":      "very extremely",
    "talagang":     "truly really",
    "talaga":       "truly really",
    "super":        "very",
    "grabe":        "extremely intense",
    "todo":         "completely fully",
    "todo-todo":    "completely absolutely",
    "puro":         "purely only",
    "lubos":        "completely utterly",
    "lubos na":     "completely utterly",
    "tunay":        "truly genuinely",
    "tunay na":     "truly genuinely",
    "totoong":      "truly genuinely",
    "very":         "very",
    "extremely":    "extremely",
    "absolutely":   "absolutely",
    "utterly":      "utterly",
    "totally":      "totally",
    "completely":   "completely",
}



AMPLIFIER_POSITIVE_CUES = {
    
    "sobrang", "napaka", "grabe", "napaka-", "super", "todo", "lubos",
}
AMPLIFIER_NEGATIVE_CUES = {
    "sobrang", "napaka", "grabe", "napaka-", "super", "todo", "lubos",
}


AMPLIFIER_PATTERNS = [
    
    (r'\b(sobrang|napaka-?|grabe ang|super)\s+(ganda|maganda|galing|husay|sarap|sulit|ayos|bilis|gwapo|cute|masaya|saya)\b',
     "extremely_beautiful very_positive excellent"),
    (r'\b(sobrang|napaka-?|grabe ang|super)\s+(galing|husay|slay|bongga|astig|lupet|petmalu)\b',
     "extremely_skilled very_positive amazing"),
    (r'\b(grabe\s+ang\s+)?galing\s+galing\b',
     "extremely_talented very_positive amazing"),
    (r'\b(ang\s+)?(ganda|galing)\s+(ganda|galing)\b',
     "extremely_positive very_good amazing"),
    (r'\bnapaka\s*sulit\b',
     "extremely_worth_it very_positive great_value"),
    (r'\bsobrang\s+sulit\b',
     "extremely_worth_it very_positive great_value"),

    
    (r'\b(sobrang|napaka-?|grabe ang|super)\s+(pangit|panget|basura|palpak|sira|bwisit|masamang|malas|bagal|mabagal)\b',
     "extremely_bad very_negative terrible"),
    (r'\b(sobrang|napaka-?|grabe ang|super)\s+(masakit|hirap|mahirap|lungkot|malungkot|galit|inis|asar)\b',
     "extremely_painful very_negative terrible"),
    (r'\bsobrang\s+basura\b',
     "extremely_garbage very_negative worthless"),
    (r'\bgrabe\s+ang\s+(palpak|downgrade|sira|crash|lag|basura)\b',
     "extremely_bad very_negative terrible_failure"),
    (r'\bnakaka[-]?(inis|badtrip|asar|galit|lungkot|iyak)\b',
     "very_annoying negative causing_negative_emotion"),
]

def inject_amplifiers(text: str) -> str:
    """
    Detect amplifier + sentiment word combinations and inject boosted cue phrases.
    This runs after negation so negated amplifiers don't get mis-boosted.
    """
    for pattern, cue in AMPLIFIER_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            text = text + ' ' + cue
    return text






def majority_vote(model_results: dict) -> dict:
    """
    Given a dict of {model_name: predict() result}, return the majority
    sentiment label and a meta dict for transparency.
    """
    vote_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for model_name, result in model_results.items():
        label = result.get("label", "Neutral")
        vote_counts[label] = vote_counts.get(label, 0) + 1

    max_votes = max(vote_counts.values())
    winners = [label for label, count in vote_counts.items() if count == max_votes]

    is_tie = len(winners) > 1
    is_unanimous = max_votes == len(model_results)

    if is_tie:
        avg_conf = {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}
        for result in model_results.values():
            for lbl in avg_conf:
                avg_conf[lbl] += result["confidence"].get(lbl.lower(), 0.0)
        n = len(model_results)
        for lbl in avg_conf:
            avg_conf[lbl] /= n
        final_label = max(avg_conf, key=avg_conf.get)
    else:
        final_label = winners[0]

    return {
        "label":        final_label,
        "vote_counts":  vote_counts,
        "is_unanimous": is_unanimous,
        "is_tie":       is_tie,
    }






CONFIDENCE_THRESHOLD = 0.72

def flag_low_confidence(result: dict) -> dict:
    """
    If the top confidence is below CONFIDENCE_THRESHOLD, add a flag.
    Does NOT change the label — just marks it for UI display.

    Skips flagging if override_triggered is True — an override means we
    already have a strong signal, so showing Low Confidence alongside
    Override would be contradictory and misleading.
    """
    if result.get("override_triggered"):
        result["low_confidence"] = False
        result["confidence_threshold"] = CONFIDENCE_THRESHOLD
        return result

    label = result.get("label", "Neutral")
    conf = result.get("confidence", {}).get(label.lower(), 1.0)
    result["low_confidence"] = conf < CONFIDENCE_THRESHOLD
    result["confidence_threshold"] = CONFIDENCE_THRESHOLD
    return result







STRONG_POSITIVE_PHRASES = {
    "best experience ever", "absolutely love", "incredibly amazing",
    "so happy", "beyond happy", "highly recommend", "10/10",
    "perfect experience", "exceeded expectations",
    "sobrang ganda", "sobrang sarap", "sobrang galing",
    "the best", "greatest ever", "love it so much",
    "pinakamahusay", "napakagaling", "napakaganda", "napakasarap",
    "sobrang sulit", "worth every", "totally worth it",
    "best decision", "no regrets", "zero regrets",
    
    "grabe ang ganda", "grabe ang galing", "grabe ang sarap",
    "ang galing galing", "ang ganda ganda",
    "napaka-sulit", "napakasulit", "napaka sulit",
    "sobrang worth it", "super worth it",
    "di na mahahanap pa", "wala nang iba",
    "5 stars", "five stars", "five star",
    "best talaga", "legit the best",
}

STRONG_NEGATIVE_PHRASES = {
    "worst experience ever", "absolutely terrible", "complete waste",
    "total disaster", "never again", "deeply disappointed",
    "worst ever", "absolutely hate", "disgusting experience",
    "pinakamasama", "pinakapangit", "napakasama",
    "worst nightmare", "absolute garbage", "completely broken",
    "stay away", "do not buy", "do not recommend",
    "walang kwenta talaga", "sobrang palpak", "grabe ang palpak",
    "hindi na ako babalik", "hindi na ko babalik",
    "never ever again", "complete disappointment",
    "total waste of", "parang downgrade", "grabe ang downgrade",
    "ayaw na ayaw", "pinaka-worst", "pinaka worst",
    
    "grabe ang sira", "grabe ang basura", "grabe ang sablay",
    "sobrang basura", "sobrang pangit", "sobrang palpak",
    "napaka-palpak", "napakapalpak",
    "worst thing ever", "absolute worst",
    "scam talaga", "solid na scam",
    "di na babalik", "hinding hindi na babalik",
    "1 star", "one star", "zero stars", "0 stars",
    "sayang pera", "sayang ang pera",
    "piliin nyo na lang iba", "pumunta na lang kayo sa iba",
}




_DYNAMIC_STRONG_POSITIVE = re.compile(
    r'\b(sobrang|napaka-?|grabe ang|super|tunay na)\s+'
    r'(maganda|ganda|galing|husay|sarap|sulit|masaya|ayos|bongga|lupet|petmalu|astig|slay)\b',
    re.IGNORECASE
)
_DYNAMIC_STRONG_NEGATIVE = re.compile(
    r'\b(sobrang|napaka-?|grabe ang|super|tunay na)\s+'
    r'(pangit|panget|basura|palpak|sira|bwisit|masama|masamang|malas|bulok|sablay|downgrade)\b',
    re.IGNORECASE
)

def keyword_override(text: str, result: dict) -> dict:
    """
    Override model label when a strong keyword phrase OR a dynamic
    amplifier + strong-word pattern is detected.

    Priority:
      0. Sarcasm already detected — skip override entirely to avoid
         contradictory badges (sarcasm wins; keyword is the ironic part)
      1. Double negation ("hindi hindi masama") → Neutral (pre-processing
         strips it but post still sees the original text)
      2. Hard-coded phrases (most precise)
      3. Dynamic amplifier regex (catches sobrang/napaka- + sentiment word)
      4. nakaka- constructions
    """
    
    if result.get('sarcasm_detected'):
        result['override_triggered'] = False
        result['override_keyword'] = None
        return result

    text_lower = text.lower()

    
    
    _DOUBLE_NEG = re.compile(
        r'\b(hindi|di|not)\s+(hindi|di|not)\s+\w+\b',
        re.IGNORECASE
    )
    if _DOUBLE_NEG.search(text):
        result["label"] = "Neutral"
        result["override_triggered"] = True
        result["override_keyword"] = "double_negation"
        return result

    for phrase in STRONG_POSITIVE_PHRASES:
        if phrase in text_lower:
            result["label"] = "Positive"
            result["override_triggered"] = True
            result["override_keyword"] = phrase
            return result

    for phrase in STRONG_NEGATIVE_PHRASES:
        if phrase in text_lower:
            result["label"] = "Negative"
            result["override_triggered"] = True
            result["override_keyword"] = phrase
            return result

    
    m = _DYNAMIC_STRONG_POSITIVE.search(text)
    if m:
        result["label"] = "Positive"
        result["override_triggered"] = True
        result["override_keyword"] = m.group(0)
        return result

    m = _DYNAMIC_STRONG_NEGATIVE.search(text)
    if m:
        result["label"] = "Negative"
        result["override_triggered"] = True
        result["override_keyword"] = m.group(0)
        return result

    
    _NAKAKA_NEG = re.compile(
        r'\bnakaka[-]?(inis|badtrip|asar|galit|lungkot|iyak|takot|hilo|pagod|transfer|sakit|dismaya|bitin)\b',
        re.IGNORECASE
    )
    if _NAKAKA_NEG.search(text):
        result["label"] = "Negative"
        result["override_triggered"] = True
        result["override_keyword"] = _NAKAKA_NEG.search(text).group(0)
        return result

    result["override_triggered"] = False
    result["override_keyword"] = None
    return result






CONTRAST_CONJUNCTIONS = {
    "but", "however", "although", "though", "yet", "despite",
    "nevertheless", "nonetheless", "even though", "even if",
    "except", "while", "whereas",
    
    "pero", "subalit", "ngunit", "kahit", "kahit na",
    "bagama't", "gayunpaman",
}

def has_contradiction(text: str) -> bool:
    """
    Returns True if text contains a contrast conjunction AND both
    a positive and a negative sentiment word on either side of it.
    """
    text_lower = text.lower()
    tokens = set(text_lower.split())
    has_contrast = any(conj in text_lower for conj in CONTRAST_CONJUNCTIONS)
    if not has_contrast:
        return False
    has_positive = bool(tokens & POSITIVE_WORDS)
    has_negative = bool(tokens & NEGATIVE_WORDS)
    return has_positive and has_negative

def apply_contradiction(result: dict, original_text: str) -> dict:
    """
    Downgrade to Neutral only when:
      1. A contrast conjunction AND mixed pos/neg words are present, AND
      2. The model's top confidence is below 80%, AND
      3. The model did NOT already predict Neutral.
    """
    label = result.get("label", "Neutral")
    top_conf = result.get("confidence", {}).get(label.lower(), 1.0)

    if top_conf >= 0.80:
        result["contradiction_detected"] = False
        return result

    if label != "Neutral" and has_contradiction(original_text):
        result["label"] = "Neutral"
        result["contradiction_detected"] = True
    else:
        result["contradiction_detected"] = False
    return result






SARCASM_PATTERNS = [
    

    
    (r'\b(wow|great|amazing|nice|brilliant|perfect|excellent|galing|maganda|helpful|sulit|ang galing|ang ganda|okay|fine|ayos|good|sige|oo|naman)\b.*🙄', "Negative"),
    (r'🙄.*\b(wow|great|amazing|nice|brilliant|perfect|excellent|galing|maganda|helpful|sulit|okay|fine|ayos|good|sige|oo|naman)\b', "Negative"),
    
    (r'.{5,}\s*🙄\s*$', "Negative"),

    
    (r'\b(great|amazing|perfect|galing|helpful|nice|ang galing|ang ganda)\b.*(🤦|🤦‍♂️|🤦‍♀️|😑|😒)', "Negative"),
    (r'(🤦|🤦‍♂️|🤦‍♀️|😑|😒).*\b(great|amazing|perfect|galing|helpful|nice)\b', "Negative"),

    
    (r'\b(good|great|nice|amazing|perfect|fine|okay|okay naman|ayos|galing|maganda)\b.*🙃', "Negative"),
    (r'🙃.*\b(good|great|nice|amazing|perfect|fine|okay|ayos|galing|maganda)\b', "Negative"),
    
    (r'(good|great|fine|okay|sure|oo|oo naman|oo nga)\s*[.,!]*\s*🙃', "Negative"),

    
    (r'\b(nakakatawa|nakakainis|nakakagalit|ang galing|ang ganda|sana all)\b.*💀', "Negative"),
    (r'💀.*\b(nakakainis|nakakagalit|palpak|sira|basura|pangit)\b', "Negative"),

    
    (r'\b(amazing|great|galing|maganda|perfect|helpful)\b.*🗿', "Negative"),
    (r'🗿.*\b(amazing|great|galing|maganda|perfect|helpful)\b', "Negative"),

    
    (r'\btalaga\b.*🙄', "Negative"),
    (r'🙄.*\btalaga\b', "Negative"),

    
    (r'\bso (helpful|useful|smart|clever|galing|maganda)\b.*🙄', "Negative"),

    

    
    (r'\byeah right\b', "Negative"),
    (r'\bsure sure\b', "Negative"),
    (r'\boo oo\b', "Negative"),                      
    (r'\boo talaga\b', "Negative"),                  
    (r'\boo nga\s*(naman|talaga)?\b.*\b(pero|kaso|sira|palpak|hindi|di)\b', "Negative"),

    
    (r'\bas if\b.*\b(good|great|nice|ganda|maganda|gumana|maayos|helpful)\b', "Negative"),

    
    (r'\boo naman\b.*\b(panget|basura|pangit|masama|bwisit|sira|palpak|downgrade)\b', "Negative"),

    
    (r'\bof course\b.*\b(again|always|lagi|ulit|palagi|pa|na naman)\b', "Negative"),

    
    (r'\bwow\b.*\b(terrible|horrible|awful|panget|basura|bwisit|sira|palpak|downgrade|naglalag|mabagal|crash|error)\b', "Negative"),

    
    (r'\b(nice job|great job|galing|ang galing)\b.*\b(bug|bugs|error|sira|palpak|hindi naayos|di naayos|crash|lag)\b', "Negative"),

    
    (r'\b(galing|maganda|okay|ayos|perfect|working|fixed|naayos)\b\s+daw\b.*\b(sira|crash|hindi|di|broken|palpak|lag|mabagal)\b', "Negative"),

    
    (r'\bsobrang (helpful|galing|maganda|ayos)\b.*\b(ayaw|hindi|di|crash|sira|lag|mabagal|broken|error)\b', "Negative"),

    
    (r'\bang saya\b.*\b(naglalag|lag|crash|sira|ayaw|hindi|di|mabagal|broken|error|palpak)\b', "Negative"),

    
    (r'\b(upgrade|improvement|update|fix|galing|maganda)\b.*\bactually\b.*\b(downgrade|worse|broken|sira|palpak|basura|naglalag|mabagal)\b', "Negative"),
    (r'\b(upgrade|improvement|update|fix)\b.*\b(downgrade|worse|broken|sira|palpak)\b', "Negative"),

    
    (r'\bgrabe naman\b.*\b(sira|palpak|pangit|panget|basura|crash|lag|mabagal|downgrade)\b', "Negative"),

    
    (r'\b(ang saya|ang ganda|ang galing|ang ayos)\b.*(🙄|🙃|😑|😒|🤦)', "Negative"),

    
    
    (r'\b(basura|palpak|sira|pangit|panget|terrible|awful|hate)\b.*\b(char|charot|joke lang|joke)\b', "Negative"),

    
    (r'\bfeeling\b.*\b(world class|five star|5 star|premium|luxury)\b.*\b(sira|palpak|pangit|basura|downgrade|lag|mabagal)\b', "Negative"),

    
    (r'\b(WOW|AMAZING|GREAT|PERFECT|EXCELLENT)\b.*\b(terrible|horrible|broken|sira|palpak|basura|pangit|crash|lag)\b', "Negative"),

    
    (r'\b(natuwa|nasaya|nagustuhan)\s+ako\b.*\b(pero|kaso|sira|palpak|hindi|di|ayaw)\b', "Negative"),
]

def detect_sarcasm(text: str, result: dict) -> dict:
    """
    If a sarcasm pattern matches and the current label is Positive,
    flip it to Negative.

    NEW BEHAVIOR:
    - Also flips Neutral → Negative when a sarcasm pattern has high pattern
      confidence (emoji-anchored patterns are very reliable).
    - Adds 'sarcasm_detected' and 'sarcasm_pattern' fields.
    """
    text_combined = text  
    text_lower = text.lower()

    
    HIGH_CONFIDENCE_EMOJI_PATTERNS = {
        "🙄", "🤦", "🤦‍♂️", "🤦‍♀️", "😑", "😒", "🙃",
    }
    has_sarcasm_emoji = any(e in text for e in HIGH_CONFIDENCE_EMOJI_PATTERNS)

    for pattern, forced_label in SARCASM_PATTERNS:
        if re.search(pattern, text_combined, re.IGNORECASE | re.DOTALL):
            result["sarcasm_detected"] = True
            result["sarcasm_pattern"] = pattern

            current_label = result.get("label", "Neutral")
            if current_label == "Positive":
                result["label"] = forced_label
            elif current_label == "Neutral" and has_sarcasm_emoji:
                
                result["label"] = forced_label
            return result

    result["sarcasm_detected"] = False
    result["sarcasm_pattern"] = None
    return result

_CHAROT_SPLIT = re.compile(
    r'\b(char|charot|joke lang|joke|haha|hehe|jk|kidding)\b',
    re.IGNORECASE
)

def apply_charot_softener(text: str, result: dict, predict_fn) -> dict:
    """
    Detect char/charot/joke lang and re-process the text AFTER the joker word.

    Rules:
      - Split text at first char/charot/joke lang occurrence.
      - If after-part is meaningful (>3 words): run predict_fn on it,
        enhance it, and use that label as the final result.
      - If after-part is empty or too short: keep original label but
        lower confidence and flag low_confidence=True.
      - Neutral + short/empty after-part: flag as uncertain.
    """
    match = _CHAROT_SPLIT.search(text)
    if not match:
        result["charot_detected"] = False
        return result

    result["charot_detected"] = True
    after_part  = text[match.end():].strip()
    after_words = [w for w in after_part.split() if w]

    original_label = result.get("label", "Neutral")
    conf           = result.get("confidence", {})

    
    if len(after_words) < 4:
        if original_label == "Neutral":
            result["low_confidence"] = True
            result["charot_effect"]  = "flagged_uncertain_short_after"
        else:
            neg_conf  = conf.get(original_label.lower(), 0.6)
            reduction = min(neg_conf * 0.35, 0.25)
            conf[original_label.lower()] = round(neg_conf - reduction, 4)
            conf["neutral"]              = round(conf.get("neutral", 0.2) + reduction, 4)
            result["confidence"]         = conf
            result["low_confidence"]     = True
            result["charot_effect"]      = "softened_short_after"
        return result

    
    after_enhanced = enhance_pre(after_part)
    after_result   = predict_fn(after_enhanced)
    after_result   = enhance_post(after_result, after_part)   

    after_label = after_result.get("label", "Neutral")

    result["label"]          = after_label
    result["confidence"]     = after_result.get("confidence", conf)
    result["low_confidence"] = after_result.get("low_confidence", False)
    result["charot_effect"]  = f"reprocessed_after_charot → {after_label}"

    return result

def enhance_post(result: dict, original_text: str, predict_fn=None) -> dict:
    """
    Run all post-processing steps on a single model's predict() result.

    Steps (order matters):
      1. Sarcasm detection    — flip Positive→Negative (or Neutral→Negative for emoji)
      2. Charot softener      — split at joker word, re-process after-part
      3. Contradiction        — downgrade to Neutral if mixed signals
      4. Keyword override     — force label on very strong terms (runs after
                                contradiction so contradiction doesn't undo
                                a strong explicit keyword)
      5. Confidence flagging  — mark low-confidence predictions
    """
    result = detect_sarcasm(original_text, result)
    if predict_fn is not None:
        result = apply_charot_softener(original_text, result, predict_fn)
    result = apply_contradiction(result, original_text)
    result = keyword_override(original_text, result)
    result = flag_low_confidence(result)
    return result

def enhance_post_compare(all_results: dict, original_text: str, predict_fn=None) -> dict:
    """
    Run post-processing on all models' results (compare mode) and append
    a majority-vote summary under the '__vote__' key.

    Args:
        all_results:  dict of {model_name: result_dict} from predict_all()
        original_text: the original (pre-enhanced) user text
        predict_fn:   optional callable for charot re-processing

    Returns:
        The same dict with each result enhanced in-place, plus a '__vote__'
        entry containing:
            label       — majority label (tie → 'Neutral')
            counts      — {'Positive': n, 'Neutral': n, 'Negative': n}
            agreement   — 'full' | 'majority' | 'split'
            confidence  — mean confidence of the winning label across voters
    """
    enhanced = {}
    for model_name, result in all_results.items():
        enhanced[model_name] = enhance_post(result, original_text, predict_fn)

    
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    conf_sum = {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}

    for model_name, r in enhanced.items():
        label = r.get("label", "Neutral")
        counts[label] = counts.get(label, 0) + 1
        conf_sum[label] = conf_sum.get(label, 0.0) + r.get("confidence", {}).get(label.lower(), 0.0)

    total_models = len(enhanced)
    winner = max(counts, key=lambda lbl: (counts[lbl], lbl == "Neutral"))

    
    winning_votes = counts[winner]
    if winning_votes == total_models:
        agreement = "full"
    elif winning_votes > total_models / 2:
        agreement = "majority"
    else:
        
        winner = "Neutral"
        agreement = "split"

    mean_conf = round(conf_sum[winner] / max(winning_votes, 1), 4)

    enhanced["__vote__"] = {
        "label":      winner,
        "counts":     counts,
        "agreement":  agreement,
        "confidence": mean_conf,
    }

    return enhanced
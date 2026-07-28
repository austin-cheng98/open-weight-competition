"""Pull model-level usage rows out of an archived OpenRouter rankings page."""
import json, re, sys

PUSH = re.compile(r'self\.__next_f\.push\(\[1,')
OBJ = re.compile(r'\{"[^{}]*?total_prompt_tokens":\s*\d+[^{}]*?\}')
STATS = re.compile(r'\{"endpoint_id":"[^"]+"[^{}]*?\}')


def flight(s):
    """Concatenate the streamed RSC payload chunks."""
    out = []
    for m in PUSH.finditer(s):
        i = s.find('"', m.end())
        if i < 0:
            continue
        j, esc = i + 1, False
        while j < len(s):
            c = s[j]
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                break
            j += 1
        try:
            out.append(json.loads(s[i:j + 1]))
        except Exception:
            pass
    return "".join(out)


def usage_rows(s):
    body = flight(s) or s
    rows = []
    for m in OBJ.finditer(body):
        try:
            rows.append(json.loads(m.group(0)))
        except Exception:
            pass
    return rows


def endpoint_stats(s):
    body = flight(s) or s
    out = []
    for m in STATS.finditer(body):
        try:
            out.append(json.loads(m.group(0)))
        except Exception:
            pass
    return out


if __name__ == '__main__':
    import collections
    for f in sys.argv[1:]:
        r = usage_rows(open(f, encoding='utf-8', errors='ignore').read())
        d = collections.Counter(x.get('date', '?')[:10] for x in r)
        print(f, len(r), 'rows', d.most_common(2), sorted(r[0])[:8] if r else '')

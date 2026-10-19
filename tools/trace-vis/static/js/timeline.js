/* Renders a normalized event list as a vertical timeline.
   Event kinds (see parsers.py):
   system | user | assistant_text | thinking | tool_call | log | run_end */
(function () {
  'use strict';

  var esc = function (s) { return window.TV.esc(s); };
  var md = function (s) { return window.TV.md(s); };

  /* This chip counts assistant events, which is not the round's model-call
     count: a turn that fires a tool with no commentary emits no assistant_text
     event at all, only its tool_call events. So it reads low against the
     header, which spells the gap out as "N model calls (M with text)". */
  var LABEL = {
    system: 'System',
    user: 'User',
    assistant_text: 'Assistant',
    thinking: 'Thinking',
    tool_call: 'Tool',
    log: 'Log',
    run_end: 'Run ended'
  };

  /* Pull the most meaningful single line out of a tool's arguments, so the
     timeline reads like a transcript without expanding anything. */
  var PREFERRED_ARGS = ['command', 'cmd', 'path', 'file_path', 'filePath',
                        'pattern', 'query', 'url', 'content', 'description'];

  function callSummary(input) {
    if (input == null) return '';
    input = parseMaybeJson(input);
    if (typeof input === 'string') return input;
    if (typeof input !== 'object') return String(input);
    for (var i = 0; i < PREFERRED_ARGS.length; i++) {
      var k = PREFERRED_ARGS[i];
      if (input[k] != null && typeof input[k] !== 'object') {
        var v = String(input[k]);
        return v;
      }
    }
    var keys = Object.keys(input);
    if (!keys.length) return '';
    return keys.map(function (k) {
      var v = input[k];
      if (v && typeof v === 'object') v = JSON.stringify(v);
      return k + '=' + String(v);
    }).join('  ');
  }

  function clip(s, n) {
    s = String(s == null ? '' : s);
    var nl = s.indexOf('\n');
    var oneline = nl >= 0 ? s.slice(0, nl) : s;
    var more = nl >= 0 || oneline.length > n;
    if (oneline.length > n) oneline = oneline.slice(0, n);
    return { text: oneline, clipped: more };
  }

  /* ---------------------------------------------------------- payloads ---
     Tool inputs and outputs are JSON: a shell result arrives as
     {"returncode": 0, "output": "/problem\ntotal 16\n..."}, so a raw dump shows
     the escape "\n" where the line should break. Decode the JSON and print the
     string values as real text.

     Only text we could actually decode as JSON is unescaped. A payload that is
     already plain text is passed through untouched — a "\n" inside source code,
     a diff or a log line is two characters there, not a line break, and
     "fixing" it would corrupt what the agent really saw. */

  var ESCAPES = { b: '\b', f: '\f', n: '\n', r: '\r', t: '\t' };

  function unescapeJsonish(s) {
    /* one pass, so a literal backslash can't be re-read as an escape */
    return s.replace(/\\(u[0-9a-fA-F]{4}|.)/g, function (m, g) {
      if (g.charAt(0) === 'u') return String.fromCharCode(parseInt(g.slice(1), 16));
      return ESCAPES[g] !== undefined ? ESCAPES[g] : g;
    });
  }

  function parseMaybeJson(v) {
    if (typeof v !== 'string') return v;
    var s = v.trim();
    if (s.charAt(0) !== '{' && s.charAt(0) !== '[') return v;
    try {
      var p = JSON.parse(s);
      if (p && typeof p === 'object') return p;
    } catch (e) {
      /* Truncated by MAX_TEXT (parsers.py) or otherwise unparseable. It still
         opened as a JSON document, so the escapes in it are JSON escapes. */
      if (s.indexOf('\\n') >= 0 || s.indexOf('\\"') >= 0) return unescapeJsonish(s);
    }
    return v;
  }

  /* JSON-ish dump for nested values: same shape as JSON.stringify(v, null, 2),
     except a multi-line string becomes an indented block (YAML's `|`) instead
     of one unreadable line. Display only — not valid JSON, and it says so. */
  function dumpPretty(v, pad) {
    pad = pad || '';
    var inner = pad + '  ';
    if (v === null || v === undefined) return 'null';
    var t = typeof v;
    if (t === 'number' || t === 'boolean') return String(v);
    if (t === 'string') {
      if (v.indexOf('\n') < 0) return JSON.stringify(v);
      return '|\n' + v.split('\n').map(function (l) { return inner + l; }).join('\n');
    }
    if (t !== 'object') return String(v);
    var keys = Object.keys(v);
    if (Array.isArray(v)) {
      if (!v.length) return '[]';
      return '[\n' + v.map(function (x) {
        return inner + dumpPretty(x, inner);
      }).join('\n') + '\n' + pad + ']';
    }
    if (!keys.length) return '{}';
    return '{\n' + keys.map(function (k) {
      return inner + k + ': ' + dumpPretty(v[k], inner);
    }).join('\n') + '\n' + pad + '}';
  }

  /* A payload as blocks to print: {label, text}. A multi-line string field gets
     its own block so its text stays flush left and copies out verbatim; the
     scalar fields around it collapse into one `key: value` block. */
  function payloadBlocks(v) {
    v = parseMaybeJson(v);
    if (v == null || v === '') return [];
    if (typeof v === 'string') {
      /* kimi's meta rows read "[rate_limit] {...}" — keep the tag, open the rest */
      var m = /^(\[[^\]\n]*\]\s*)([[{][\s\S]*)$/.exec(v);
      if (m) {
        var rest = parseMaybeJson(m[2]);
        if (typeof rest !== 'string') {
          return [{ label: null, text: m[1].trim() }].concat(payloadBlocks(rest));
        }
      }
      return [{ label: null, text: v }];
    }
    if (typeof v !== 'object') return [{ label: null, text: String(v) }];
    if (Array.isArray(v)) return [{ label: null, text: dumpPretty(v) }];

    var blocks = [];
    var inline = [];
    var flush = function () {
      if (inline.length) blocks.push({ label: null, text: inline.join('\n') });
      inline = [];
    };
    Object.keys(v).forEach(function (k) {
      var val = v[k];
      if (typeof val === 'string' && val.indexOf('\n') >= 0) {
        flush();
        blocks.push({ label: k, text: val });
      } else if (val === null || typeof val !== 'object') {
        inline.push(k + ': ' + (typeof val === 'string' ? val : String(val)));
      } else {
        flush();
        blocks.push({ label: k, text: dumpPretty(val) });
      }
    });
    flush();
    if (!blocks.length) blocks.push({ label: null, text: '{}' });
    return blocks;
  }

  function renderBlocks(blocks, cls) {
    return blocks.map(function (b) {
      return (b.label ? '<div class="pkey">' + esc(b.label) + '</div>' : '') +
             '<pre class="' + cls + '">' + esc(b.text) + '</pre>';
    }).join('');
  }

  function renderPayload(v, cls) {
    return renderBlocks(payloadBlocks(v), cls);
  }

  function truncNote(what) {
    return '<div class="trunc">' + esc(what) + ' truncated for display — open the file on disk for the full text.</div>';
  }

  function timeStr(ts) {
    if (!ts) return '';
    try {
      var d = new Date(ts * 1000);
      if (isNaN(d.getTime())) return '';
      return d.toISOString().replace('T', ' ').slice(0, 19) + 'Z';
    } catch (e) { return ''; }
  }

  function renderEvent(ev) {
    var kind = ev.kind || 'log';
    var cls = 'ev ' + kind + (ev.is_error ? ' err' : '') +
              (kind === 'run_end' ? ' sev-' + (ev.end_severity || 'error') : '');
    var h = [];
    h.push('<div class="' + cls + '" data-kind="' + esc(kind) + '">');
    h.push('<div class="ev-box">');

    /* header */
    h.push('<div class="ev-head">');
    h.push('<span class="who">' + esc(LABEL[kind] || kind) + '</span>');
    if (kind === 'tool_call') {
      h.push('<span class="tool">' + esc(ev.tool_name || '?') + '</span>');
      if (ev.is_error) h.push('<span class="chip fail">error</span>');
      else if (ev.tool_output == null) h.push('<span class="chip none">no output</span>');
    } else if (kind === 'run_end') {
      var sev = ev.end_severity || 'error';
      h.push('<span class="chip ' + (sev === 'ok' ? 'pass' : sev === 'warn' ? 'soft' : 'fail') +
             '">' + esc(ev.end_reason || ev.end_stop_reason || 'ended') + '</span>');
    }
    if (ev.finish_reason) {
      h.push('<span class="chip warn" title="' + esc(ev.finish_reason_detail || '') +
             '">finish: ' + esc(ev.finish_reason) + '</span>');
    }
    var t = timeStr(ev.timestamp);
    if (t) h.push('<span class="chip soft">' + esc(t) + '</span>');
    h.push('<span class="idx">#' + (ev.idx != null ? ev.idx : '') + '</span>');
    h.push('</div>');

    if (kind === 'tool_call') {
      /* one-line call summary, full input + output behind disclosures */
      var sum = clip(callSummary(ev.tool_input), 220);
      if (sum.text) {
        h.push('<div class="callline">' + esc(sum.text) +
               (sum.clipped ? '<span class="dim"> …</span>' : '') + '</div>');
      }
      var blocks = payloadBlocks(ev.tool_input);
      /* skip the disclosure only when the summary line already showed all of it */
      var same = !sum.clipped && blocks.length === 1 && !blocks[0].label &&
                 blocks[0].text === sum.text;
      if (blocks.length && !same) {
        h.push('<details class="sub"><summary>Input</summary>' +
               renderBlocks(blocks, 'out') + '</details>');
      }
      if (ev.tool_output != null && ev.tool_output !== '') {
        h.push('<details class="sub"><summary>Output' +
               (ev.is_error ? ' (error)' : '') + '</summary>' +
               renderPayload(ev.tool_output, ev.is_error ? 'out err' : 'out') +
               (ev.tool_output_truncated ? truncNote('Output') : '') +
               '</details>');
      }
    } else if (kind === 'thinking') {
      h.push('<details class="sub" open><summary>Reasoning</summary>' +
             '<div class="ev-body"><div class="md">' + md(ev.text) + '</div>' +
             (ev.text_truncated ? truncNote('Text') : '') + '</div></details>');
    } else if (kind === 'system') {
      /* the system prompt is long scaffolding — collapsed unless asked for */
      h.push('<details class="sub"><summary>System prompt (' +
             String(ev.text || '').length + ' chars)</summary>' +
             '<div class="ev-body"><div class="md">' + md(ev.text) + '</div>' +
             (ev.text_truncated ? truncNote('Text') : '') + '</div></details>');
    } else if (kind === 'run_end') {
      /* Last node of the trace: why it stopped. Without it a run killed by a
         context overflow (pi, swe-mini), a step-cap/provider error (kimi) or
         the turn budget (claude) just ends mid-conversation, looking like the
         model quit on its own. parsers.py::_pi_run_end / _kimi_run_end /
         _claude_run_end / _swemini_run_end. */
      h.push('<div class="endline">' + esc(ev.text || '') + '</div>');
      if (ev.end_tokens_used && ev.end_tokens_limit) {
        var within = Math.max(2, Math.min(98,
          Math.round(ev.end_tokens_limit / ev.end_tokens_used * 100)));
        h.push('<div class="overbar" title="' + ev.end_tokens_limit +
               ' token limit, ' + ev.end_tokens_used + ' sent">' +
               '<span class="fill" style="width:' + within + '%"></span></div>');
      }
      if (ev.tool_output) {
        h.push('<details class="sub"><summary>Provider message</summary>' +
               renderPayload(ev.tool_output, 'out err') + '</details>');
      }
    } else if (kind === 'log') {
      h.push(renderPayload(ev.text || ev.tool_output || '', 'out') ||
             '<pre class="out"></pre>');
    } else {
      h.push('<div class="ev-body"><div class="md">' + md(ev.text) + '</div></div>');
      if (ev.text_truncated) h.push(truncNote('Text'));
    }

    h.push('</div></div>');
    return h.join('');
  }

  /* Kinds the filter bar offers, in timeline order of usefulness. */
  var FILTER_KINDS = ['system', 'user', 'assistant_text', 'thinking', 'tool_call',
                      'log', 'run_end'];

  function renderTimeline(events) {
    if (!events.length) {
      return '<div class="empty">No events matched the current filters.</div>';
    }
    return '<div class="timeline">' + events.map(renderEvent).join('') + '</div>';
  }

  window.TV = window.TV || {};
  window.TV.renderTimeline = renderTimeline;
  window.TV.payloadBlocks = payloadBlocks;
  window.TV.renderEvent = renderEvent;
  window.TV.FILTER_KINDS = FILTER_KINDS;
  window.TV.LABEL = LABEL;
})();

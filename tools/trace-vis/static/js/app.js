/* trace-vis SPA: hash routing, no build step.

   Routes
     #/                          overview
     #/problems                  problem list        (pick a problem first)
     #/problems/<p>              agents for problem  -> campaigns -> runs
     #/agents                    agent list          (pick an agent first)
     #/agents/<a>                campaigns for agent -> problems    -> runs
     #/models                    model list          (pick a model first)
     #/models/<m>                agents for model    -> campaigns   -> runs
     #/run/<problem>/<agent>/<model>/<campaign>/round-<n>
*/
(function () {
  'use strict';

  var esc = function (s) { return window.TV.esc(s); };
  var view = document.getElementById('view');

  /* Which runs the lists show. "curated" (the default) is the set the pass
     matrix in slides/figures/05_pass_matrix.png is built from: latest campaign
     per problem x agent x model, A40 only, known harness/config bugs dropped.
     "all" is every row in results/runs.jsonl. Server side: curation.py. */
  var SCOPE = 'curated';
  try { if (localStorage.getItem('tv-scope') === 'all') SCOPE = 'all'; } catch (e) {}

  // ------------------------------------------------------------ helpers

  function api(path, params) {
    var qs = params ? '?' + Object.keys(params)
      .filter(function (k) { return params[k] != null && params[k] !== ''; })
      .map(function (k) { return encodeURIComponent(k) + '=' + encodeURIComponent(params[k]); })
      .join('&') : '';
    return fetch(path + qs).then(function (r) {
      return r.json().then(function (body) {
        if (!r.ok || body.error) throw new Error(body.error || ('HTTP ' + r.status));
        return body;
      });
    });
  }

  function pct(x) { return x == null ? '—' : Math.round(x * 100) + '%'; }
  function num(x) { return x == null ? '—' : String(x); }

  /* The failure name is r.fail_class: one fine-grained slug per failing round
     (context_overflow, format_error, compile_error, mse, ...), computed once in
     checks/failure_class.py and carried through runs.jsonl, so the chip here,
     the timeline's last node and results/summary.json all say the same word.
     It replaced the failing STAGE's name, which read "build" on any round whose
     agent died before a compiler ever ran — most of them. fail_label is the
     sentence-long version, shown on hover. */
  function passChip(passed, r) {
    if (passed === null || passed === undefined) return '<span class="chip none">ungraded</span>';
    if (passed) return '<span class="chip pass">pass</span>';
    var why = (r && (r.fail_class || r.failed_stage)) || '';
    var tip = (r && r.fail_label) || '';
    /* the detail is EVIDENCE (a compiler line, an exception). A run-end reason
       has none and repeats the label, so don't print the same words twice. */
    if (r && r.fail_detail && r.fail_detail !== tip) {
      tip = tip ? tip + ' \u2014 ' + r.fail_detail : r.fail_detail;
    }
    return '<span class="chip fail"' + (tip ? ' title="' + esc(tip) + '"' : '') +
           '>fail' + (why ? ': ' + esc(why) : '') + '</span>';
  }

  function metricStr(m) {
    if (!m || m.value == null) return '—';
    var v = typeof m.value === 'number'
      ? (Math.abs(m.value) >= 1000 || (Math.abs(m.value) < 0.001 && m.value !== 0)
          ? m.value.toExponential(3) : String(Number(m.value.toFixed(6))))
      : String(m.value);
    return esc(v) + (m.unit ? ' <span class="chip soft">' + esc(m.unit) + '</span>' : '');
  }

  function passBar(n_passed, n_graded) {
    if (!n_graded) return '';
    return '<div class="bar"><i style="width:' + (100 * n_passed / n_graded) + '%"></i></div>';
  }

  function crumbs(parts) {
    return '<div class="crumb">' + parts.map(function (p, i) {
      var sep = i ? '<span class="sep">/</span>' : '';
      return sep + (p.href ? '<a href="' + p.href + '">' + esc(p.label) + '</a>'
                           : '<span>' + esc(p.label) + '</span>');
    }).join('') + '</div>';
  }

  function setTab(name) {
    document.querySelectorAll('nav.tabs a').forEach(function (a) {
      a.classList.toggle('active', a.dataset.tab === name);
    });
  }

  function showError(e) {
    view.innerHTML = '<div class="err-banner">' + esc(e.message || String(e)) + '</div>' +
                     '<p><a href="#/">← back to overview</a></p>';
  }

  function warnBanner(warnings) {
    if (!warnings || !warnings.length) return '';
    return '<div class="warn-banner">' + warnings.map(esc).join('<br>') + '</div>';
  }

  /* One line under the heading saying what the current scope hides, with the
     switch to the other one. `info` is the `scope` block the API returns. */
  function scopeNote(info) {
    if (!info) return '';
    var hidden = info.n_all - info.n_curated;
    if (info.served === 'all') {
      return '<div class="scope-note">Showing <b>all ' + info.n_all + ' runs</b>, ' +
             'including superseded campaigns and known harness/config bugs. ' +
             '<a href="#" data-set-scope="curated">Show only the important ones</a></div>';
    }
    return '<div class="scope-note">Showing the <b>' + info.n_curated + ' important runs</b> — ' +
           'latest campaign per problem × agent × model, A40 only, known harness/config bugs ' +
           'excluded (the set behind <code>slides/figures/05_pass_matrix.png</code>). ' +
           hidden + ' hidden. <a href="#" data-set-scope="all">Show all runs</a></div>';
  }

  function wireScopeLinks() {
    view.querySelectorAll('a[data-set-scope]').forEach(function (a) {
      a.addEventListener('click', function (ev) {
        ev.preventDefault();
        setScope(a.dataset.setScope);
      });
    });
  }

  function setScope(s) {
    SCOPE = (s === 'all') ? 'all' : 'curated';
    try { localStorage.setItem('tv-scope', SCOPE); } catch (e) {}
    paintScopeSwitch();
    route();
  }

  function paintScopeSwitch() {
    document.querySelectorAll('#scope-sw button').forEach(function (b) {
      b.classList.toggle('on', b.dataset.scope === SCOPE);
    });
  }

  /* run_id is "problem/agent/model/campaign/round-N": keep the slashes as route
     separators, encode only what's inside each segment. */
  function runHref(runId) {
    return '#/run/' + runId.split('/').map(encodeURIComponent).join('/');
  }

  // ------------------------------------------------------- run list table

  function runsTable(runs) {
    if (!runs.length) return '<div class="empty">No runs.</div>';
    var anyMetric = runs.some(function (r) { return r.metric && r.metric.value != null; });
    var h = ['<table class="runs"><thead><tr>',
             '<th>Round</th><th>Status</th>'];
    if (anyMetric) h.push('<th>Metric</th>');
    h.push('<th class="num">Turns</th><th class="num">Steps</th><th class="num">Tokens</th>',
           '</tr></thead><tbody>');
    runs.forEach(function (r) {
      /* in the "all" scope, dim the rounds the curated scope leaves out */
      h.push(r.curated === false
        ? '<tr class="dim" title="not in the important-runs scope">'
        : '<tr>');
      /* new tab: a rounds table is a list you work through, and the timeline
         is a long read — going back would lose your place in the list. */
      h.push('<td><a target="_blank" rel="noopener" href="' + runHref(r.run_id) +
             '">round ' + num(r.round) + '</a></td>');
      h.push('<td>' + passChip(r.passed, r) + '</td>');
      if (anyMetric) h.push('<td>' + metricStr(r.metric) + '</td>');
      h.push('<td class="num">' + num(r.turns) + '</td>');
      h.push('<td class="num">' + num(r.steps) + '</td>');
      h.push('<td class="num">' + (r.total_tokens != null ? r.total_tokens.toLocaleString() : '—') + '</td>');
      h.push('</tr>');
    });
    h.push('</tbody></table>');
    return h.join('');
  }

  function campaignsBlock(campaigns) {
    if (!campaigns || !campaigns.length) return '<div class="empty">No campaigns.</div>';
    return campaigns.map(function (c) {
      var partial = (c.n_runs_all != null && c.n_runs_all > c.n_runs)
        ? '<span class="chip soft" title="rounds dropped as infra failures">' +
          c.n_runs + ' of ' + c.n_runs_all + ' rounds</span>' : '';
      return '<div class="campaign-head">' +
               '<span class="model">' + esc(c.model) + '</span>' +
               '<span class="ts">' + esc(c.campaign) + '</span>' +
               '<span class="chip ' + (c.n_passed ? 'pass' : 'none') + '">' +
                 c.n_passed + '/' + c.n_graded + ' pass</span>' + partial +
             '</div>' + runsTable(c.runs);
    }).join('');
  }

  /* The agent view groups by campaign, not by problem: one submission runs the
     same model against every problem under one timestamp, so a "wave" is what
     you actually compare. Problems become sub-sections inside it, newest wave
     first. (The problem view keeps the opposite pivot.) */
  function groupWaves(problems) {
    var waves = {};
    (problems || []).forEach(function (p) {
      (p.campaigns || []).forEach(function (c) {
        var key = c.model + ' ' + c.campaign;
        if (!waves[key]) {
          waves[key] = { model: c.model, campaign: c.campaign, parts: [],
                         n_runs: 0, n_runs_all: 0, n_graded: 0, n_passed: 0 };
        }
        var w = waves[key];
        w.parts.push({ problem: p.name, campaign: c });
        w.n_runs += c.n_runs || 0;
        w.n_runs_all += (c.n_runs_all != null ? c.n_runs_all : c.n_runs || 0);
        w.n_graded += c.n_graded || 0;
        w.n_passed += c.n_passed || 0;
      });
    });
    return Object.keys(waves).map(function (k) { return waves[k]; })
      .sort(function (a, b) {
        if (a.campaign !== b.campaign) return a.campaign < b.campaign ? 1 : -1;
        return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
      });
  }

  function wavesBlock(waves) {
    if (!waves.length) return '<div class="empty">No campaigns.</div>';
    return waves.map(function (w) {
      var partial = w.n_runs_all > w.n_runs
        ? '<span class="chip soft" title="rounds dropped as infra failures">' +
          w.n_runs + ' of ' + w.n_runs_all + ' rounds</span>' : '';
      var h = ['<section class="wave">'];
      h.push('<div class="wave-head">' +
               '<span class="model">' + esc(w.model) + '</span>' +
               '<span class="ts">' + esc(w.campaign) + '</span>' +
               '<span class="chip ' + (w.n_passed ? 'pass' : 'none') + '">' +
                 w.n_passed + '/' + w.n_graded + ' pass</span>' + partial +
               '<span class="count">' + w.parts.length +
                 (w.parts.length === 1 ? ' problem' : ' problems') + '</span>' +
             '</div>');
      w.parts.forEach(function (part) {
        var c = part.campaign;
        h.push('<div class="wave-prob">' +
                 '<a href="#/problems/' + encodeURIComponent(part.problem) + '">' +
                   esc(part.problem) + '</a>' +
                 '<span class="chip ' + (c.n_passed ? 'pass' : 'none') + '">' +
                   c.n_passed + '/' + c.n_graded + '</span>' +
               '</div>');
        h.push(runsTable(c.runs));
      });
      h.push('</section>');
      return h.join('');
    }).join('');
  }

  // ------------------------------------------------------------- views

  function viewHome() {
    setTab('home');
    return api('/api/summary', { scope: SCOPE }).then(function (s) {
      var t = s.totals;
      var h = [];
      h.push(warnBanner(s.warnings));
      h.push('<h1>Agent trace explorer</h1>');
      h.push('<p class="sub">Browse runs by problem or by agent, then open one to read its trajectory.</p>');
      h.push(scopeNote(s.scope));

      h.push('<div class="stats">');
      [['Runs', t.n_runs], ['Campaigns', t.n_campaigns], ['Problems', t.n_problems],
       ['Agents', t.n_agents], ['Models', t.n_models], ['Pass rate', pct(t.pass_rate)]
      ].forEach(function (p) {
        h.push('<div class="stat"><div class="v">' + esc(String(p[1])) + '</div><div class="k">' + esc(p[0]) + '</div></div>');
      });
      h.push('</div>');

      h.push('<h2>Problems</h2><div class="grid">');
      s.problems.forEach(function (p) {
        h.push('<a class="card" href="#/problems/' + encodeURIComponent(p.name) + '">' +
               '<div class="name">' + esc(p.name) + '</div>' +
               '<div class="meta">' + p.n_runs + ' runs · ' + p.agents.length + ' agents · ' + pct(p.pass_rate) + ' pass</div>' +
               passBar(p.n_passed, p.n_graded) + '</a>');
      });
      h.push('</div>');

      h.push('<h2>Agents</h2><div class="grid">');
      s.agents.forEach(function (a) {
        h.push('<a class="card" href="#/agents/' + encodeURIComponent(a.name) + '">' +
               '<div class="name">' + esc(a.name) + '</div>' +
               '<div class="meta">' + a.n_runs + ' runs · ' + a.problems.length + ' problems · ' + pct(a.pass_rate) + ' pass</div>' +
               passBar(a.n_passed, a.n_graded) + '</a>');
      });
      h.push('</div>');

      h.push('<h2>Models</h2><div class="grid">');
      s.models.forEach(function (m) {
        h.push('<a class="card" href="#/models/' + encodeURIComponent(m.name) + '">' +
               '<div class="name">' + esc(m.name) + '</div>' +
               '<div class="meta">' + m.n_runs + ' runs \u00b7 ' + m.agents.length + ' agents \u00b7 ' + pct(m.pass_rate) + ' pass</div>' +
               passBar(m.n_passed, m.n_graded) + '</a>');
      });
      h.push('</div>');
      view.innerHTML = h.join('');
      wireScopeLinks();
    });
  }

  function viewProblemList() {
    setTab('problems');
    return api('/api/summary', { scope: SCOPE }).then(function (s) {
      var h = [crumbs([{ label: 'Problems' }]),
               '<h1>Problems</h1>',
               '<p class="sub">Pick a problem, then an agent, then a run.</p>',
               scopeNote(s.scope),
               '<div class="grid">'];
      s.problems.forEach(function (p) {
        h.push('<a class="card" href="#/problems/' + encodeURIComponent(p.name) + '">' +
               '<div class="name">' + esc(p.name) + '</div>' +
               '<div class="meta">' + p.n_runs + ' runs · ' + p.agents.length + ' agents · ' + pct(p.pass_rate) + ' pass</div>' +
               passBar(p.n_passed, p.n_graded) + '</a>');
      });
      h.push('</div>');
      view.innerHTML = h.join('');
      wireScopeLinks();
    });
  }

  function viewAgentList() {
    setTab('agents');
    return api('/api/summary', { scope: SCOPE }).then(function (s) {
      var h = [crumbs([{ label: 'Agents' }]),
               '<h1>Agents</h1>',
               '<p class="sub">Pick an agent, then a problem, then a run.</p>',
               scopeNote(s.scope),
               '<div class="grid">'];
      s.agents.forEach(function (a) {
        h.push('<a class="card" href="#/agents/' + encodeURIComponent(a.name) + '">' +
               '<div class="name">' + esc(a.name) + '</div>' +
               '<div class="meta">' + a.n_runs + ' runs · ' + a.problems.length + ' problems · ' + pct(a.pass_rate) + ' pass</div>' +
               passBar(a.n_passed, a.n_graded) + '</a>');
      });
      h.push('</div>');
      view.innerHTML = h.join('');
      wireScopeLinks();
    });
  }

  function viewModelList() {
    setTab('models');
    return api('/api/summary', { scope: SCOPE }).then(function (s) {
      var h = [crumbs([{ label: 'Models' }]),
               '<h1>Models</h1>',
               '<p class="sub">Pick a model, then an agent, then a run.</p>',
               scopeNote(s.scope),
               '<div class="grid">'];
      s.models.forEach(function (m) {
        h.push('<a class="card" href="#/models/' + encodeURIComponent(m.name) + '">' +
               '<div class="name">' + esc(m.name) + '</div>' +
               '<div class="meta">' + m.n_runs + ' runs \u00b7 ' + m.agents.length +
                 ' agents \u00b7 ' + m.problems.length + ' problems \u00b7 ' + pct(m.pass_rate) + ' pass</div>' +
               passBar(m.n_passed, m.n_graded) + '</a>');
      });
      h.push('</div>');
      view.innerHTML = h.join('');
      wireScopeLinks();
    });
  }

  /* problem -> agents -> campaigns -> runs */
  function viewProblem(name) {
    setTab('problems');
    return api('/api/problem', { name: name, scope: SCOPE }).then(function (d) {
      var h = [crumbs([{ label: 'Problems', href: '#/problems' }, { label: d.problem }])];
      h.push('<h1>' + esc(d.problem) + '</h1>');
      h.push('<p class="sub">' + d.n_runs + ' runs across ' + d.agents.length +
             ' agents · ' + pct(d.pass_rate) + ' pass</p>');
      h.push(scopeNote(d.scope));
      if (!d.agents.length) h.push('<div class="empty">No runs in this scope.</div>');
      d.agents.forEach(function (a) {
        h.push('<h2>' + esc(a.name) +
               ' <span class="chip ' + (a.n_passed ? 'pass' : 'none') + '">' +
               a.n_passed + '/' + a.n_graded + '</span></h2>');
        h.push(campaignsBlock(a.campaigns));
      });
      view.innerHTML = h.join('');
      wireScopeLinks();
    });
  }

  /* agent -> campaigns (waves) -> problems -> runs */
  function viewAgent(name) {
    setTab('agents');
    return api('/api/agent', { name: name, scope: SCOPE }).then(function (d) {
      var h = [crumbs([{ label: 'Agents', href: '#/agents' }, { label: d.agent }])];
      var waves = groupWaves(d.problems);
      h.push('<h1>' + esc(d.agent) + '</h1>');
      h.push('<p class="sub">' + d.n_runs + ' runs · ' + waves.length +
             ' campaigns across ' + d.problems.length +
             ' problems · ' + pct(d.pass_rate) + ' pass</p>');
      h.push(scopeNote(d.scope));
      if (!waves.length) h.push('<div class="empty">No runs in this scope.</div>');
      else h.push(wavesBlock(waves));
      view.innerHTML = h.join('');
      wireScopeLinks();
    });
  }

  /* model -> agents -> campaigns (waves) -> problems -> runs

     Same weights, one harness per section: the point of this view is reading
     down the agents to see which harness got more out of the model. Inside a
     section it is the agent view's layout, so a wave still means one campaign. */
  function viewModel(name) {
    setTab('models');
    return api('/api/model', { name: name, scope: SCOPE }).then(function (d) {
      var h = [crumbs([{ label: 'Models', href: '#/models' }, { label: d.model }])];
      h.push('<h1>' + esc(d.model) + '</h1>');
      h.push('<p class="sub">' + d.n_runs + ' runs across ' + d.agents.length +
             (d.agents.length === 1 ? ' agent' : ' agents') +
             ' \u00b7 ' + pct(d.pass_rate) + ' pass</p>');
      h.push(scopeNote(d.scope));
      if (!d.agents.length) h.push('<div class="empty">No runs in this scope.</div>');
      d.agents.forEach(function (a) {
        h.push('<h2><a href="#/agents/' + encodeURIComponent(a.name) + '">' + esc(a.name) + '</a>' +
               ' <span class="chip ' + (a.n_passed ? 'pass' : 'none') + '">' +
               a.n_passed + '/' + a.n_graded + '</span></h2>');
        h.push(wavesBlock(groupWaves(a.problems)));
      });
      view.innerHTML = h.join('');
      wireScopeLinks();
    });
  }

  // ---------------------------------------------------------- run view

  var runState = null;

  /* Derived from the trace's own kind counts, so a kind the parsers grow later
     still shows up instead of being silently filtered away. System prompts are
     the one kind off by default — they're scaffolding, not conversation. */
  function defaultFilters(byKind) {
    var active = {};
    Object.keys(byKind || {}).forEach(function (k) { active[k] = (k !== 'system'); });
    window.TV.FILTER_KINDS.forEach(function (k) {
      if (!(k in active)) active[k] = (k !== 'system');
    });
    return active;
  }

  function viewRun(runId) {
    setTab(null);
    document.querySelector('main').classList.add('wide');
    view.innerHTML = '<div class="loading">Loading trace…</div>';
    return api('/api/run', { run: runId }).then(function (meta) {
      return api('/api/trace', { run: runId, limit: 400 }).then(function (page) {
        runState = {
          runId: runId, meta: meta, page: page,
          events: page.events.slice(),
          active: defaultFilters(page.stats.by_kind)
        };
        renderRun();
      });
    });
  }

  function renderRun() {
    var meta = runState.meta, page = runState.page, r = meta.run;
    var h = [];

    h.push(crumbs([
      { label: 'Problems', href: '#/problems' },
      { label: r.problem, href: '#/problems/' + encodeURIComponent(r.problem) },
      { label: r.agent, href: '#/agents/' + encodeURIComponent(r.agent) },
      { label: r.model },
      { label: 'round ' + r.round }
    ]));

    /* run header */
    h.push('<div class="runhead">');
    h.push('<div class="title"><h1>' + esc(r.problem) + ' · ' + esc(r.agent) + ' · round ' + num(r.round) + '</h1>' +
           passChip(r.passed, r) + '</div>');
    h.push('<div class="kv">');
    h.push('<span><b>model</b> <code>' + esc(r.model_raw || r.model) + '</code></span>');
    h.push('<span><b>campaign</b> <code>' + esc(r.campaign) + '</code></span>');
    /* spell the failure out here too: the chip has room for the slug only */
    if (r.passed === false && (r.fail_label || r.fail_class)) {
      h.push('<span><b>why</b> ' + esc(r.fail_label || r.fail_class) +
             (r.fail_detail && r.fail_detail !== (r.fail_label || r.fail_class)
                ? ' <span class="chip soft">' + esc(r.fail_detail) + '</span>' : '') +
             '</span>');
    }
    h.push('<span><b>schema</b> <code>' + esc(meta.schema || 'unknown') + '</code></span>');
    h.push('<span><b>trace</b> <code>' + esc(meta.trace_file || '—') + '</code></span>');
    if (r.metric && r.metric.value != null) h.push('<span><b>' + esc(r.metric.name || 'metric') + '</b> ' + metricStr(r.metric) + '</span>');
    if (r.turns != null) {
      /* `turns` is model calls. The timeline's "Assistant" chip counts only the
         turns that said something -- a turn that fires a tool with no prose
         emits no assistant_text event -- so the two disagree on purpose. Spell
         the difference out here rather than leave the chip looking wrong. */
      var withText = (page.stats && page.stats.by_kind) ? page.stats.by_kind.assistant_text : null;
      h.push('<span><b>turns</b> ' + num(r.turns) + ' model calls' +
             (withText != null && withText !== r.turns
                ? ' <span class="dim">(' + num(withText) + ' with text)</span>' : '') +
             '</span>');
    }
    if (r.total_tokens != null) h.push('<span><b>tokens</b> ' + r.total_tokens.toLocaleString() + '</span>');
    if (r.peak_context != null) h.push('<span><b>peak ctx</b> ' + r.peak_context.toLocaleString() + '</span>');
    if (r.gpu) h.push('<span><b>gpu</b> <code>' + esc(r.gpu) + '</code></span>');
    h.push('</div>');
    h.push('<div class="kv" style="margin-top:6px"><span><b>path</b> <code>' + esc(meta.round_rel) + '</code></span></div>');
    h.push('</div>');

    /* a run reached by link can be one the curated scope leaves out */
    if (meta.curated === false) {
      h.push('<div class="scope-note">Outside the important-runs scope: a superseded ' +
             'campaign, a known harness/config bug, a &lt; 3-round campaign, or not measured ' +
             'on an A40. It still reads normally.</div>');
    }

    /* side panels */
    if (meta.solution) {
      h.push('<details class="panel"><summary>solution.json</summary><div class="body">' +
             '<pre class="out">' + esc(JSON.stringify(meta.solution, null, 2)) + '</pre></div></details>');
    }
    if (r.stages && r.stages.length) {
      var rows = r.stages.map(function (st) {
        var status = st.skipped ? '<span class="chip none">skipped</span>'
                   : (st.passed ? '<span class="chip pass">pass</span>' : '<span class="chip fail">fail</span>');
        return '<tr><td>' + esc(st.name) + '</td><td>' + status + '</td><td>' + esc(st.detail || '') + '</td></tr>';
      }).join('');
      h.push('<details class="panel"><summary>Check stages</summary><div class="body">' +
             '<table class="runs"><thead><tr><th>Stage</th><th>Status</th><th>Detail</th></tr></thead><tbody>' +
             rows + '</tbody></table></div></details>');
    }
    meta.files.forEach(function (f) {
      if (f.name === 'solution.json' || f.name === 'pass_result.json') return;
      h.push('<details class="panel" data-file="' + esc(f.name) + '"><summary>' + esc(f.name) +
             ' <span class="chip soft">' + (f.bytes > 1024 ? Math.round(f.bytes / 1024) + ' KB' : f.bytes + ' B') +
             '</span></summary><div class="body"><div class="loading">Loading…</div></div></details>');
    });

    /* warnings from the parser */
    h.push(warnBanner(page.warnings));

    /* filters */
    var counts = page.stats.by_kind || {};
    var kinds = window.TV.FILTER_KINDS.concat(
      Object.keys(counts).filter(function (k) { return window.TV.FILTER_KINDS.indexOf(k) < 0; }));
    h.push('<div class="filters">');
    kinds.forEach(function (k) {
      if (!counts[k]) return;
      h.push('<button data-kind="' + k + '" class="' + (runState.active[k] ? 'on' : '') + '">' +
             esc(window.TV.LABEL[k] || k) + ' <span class="chip soft">' + counts[k] + '</span></button>');
    });
    h.push('<span class="count" id="ev-count"></span>');
    h.push('</div>');

    h.push('<div id="tl"></div>');

    if (page.has_more) {
      h.push('<div class="more-wrap"><button id="more">Load more (' +
             (page.total_filtered - runState.events.length) + ' remaining)</button></div>');
    }

    view.innerHTML = h.join('');
    paintTimeline();
    wireRun();
  }

  function paintTimeline() {
    var evs = runState.events.filter(function (e) { return runState.active[e.kind]; });
    document.getElementById('tl').innerHTML = window.TV.renderTimeline(evs);
    var c = document.getElementById('ev-count');
    if (c) {
      c.textContent = evs.length + ' of ' + runState.page.stats.total_events + ' events' +
        (runState.events.length < runState.page.stats.total_events
          ? ' (' + runState.events.length + ' loaded)' : '');
    }
  }

  function wireRun() {
    view.querySelectorAll('.filters button[data-kind]').forEach(function (b) {
      b.addEventListener('click', function () {
        var k = b.dataset.kind;
        runState.active[k] = !runState.active[k];
        b.classList.toggle('on', runState.active[k]);
        paintTimeline();
      });
    });

    var more = document.getElementById('more');
    if (more) {
      more.addEventListener('click', function () {
        more.disabled = true;
        more.textContent = 'Loading…';
        api('/api/trace', { run: runState.runId, offset: runState.events.length, limit: 400 })
          .then(function (p) {
            runState.events = runState.events.concat(p.events);
            runState.page = p;
            renderRun();
          })
          .catch(showError);
      });
    }

    /* lazy-load side file contents on first expand */
    view.querySelectorAll('details.panel[data-file]').forEach(function (d) {
      d.addEventListener('toggle', function () {
        if (!d.open || d.dataset.loaded) return;
        d.dataset.loaded = '1';
        var body = d.querySelector('.body');
        api('/api/file', { run: runState.runId, name: d.dataset.file })
          .then(function (f) {
            body.innerHTML = '<pre class="out">' + esc(f.content) + '</pre>' +
              (f.truncated ? '<div class="trunc">File truncated at 512 KB.</div>' : '');
          })
          .catch(function (e) {
            body.innerHTML = '<div class="err-banner">' + esc(e.message) + '</div>';
          });
      });
    });
  }

  // ------------------------------------------------------------- router

  function route() {
    var hash = location.hash.replace(/^#/, '') || '/';
    var parts = hash.split('/').filter(Boolean).map(decodeURIComponent);
    document.querySelector('main').classList.remove('wide');
    window.scrollTo(0, 0);

    var p;
    if (!parts.length) p = viewHome();
    else if (parts[0] === 'problems') p = parts[1] ? viewProblem(parts[1]) : viewProblemList();
    else if (parts[0] === 'agents') p = parts[1] ? viewAgent(parts[1]) : viewAgentList();
    else if (parts[0] === 'models') p = parts[1] ? viewModel(parts[1]) : viewModelList();
    else if (parts[0] === 'run') p = viewRun(parts.slice(1).join('/'));
    else p = Promise.reject(new Error('Unknown route: ' + hash));

    Promise.resolve(p).catch(showError);
  }

  window.addEventListener('hashchange', route);

  /* scope switch in the header, persisted; re-renders the current view */
  paintScopeSwitch();
  document.querySelectorAll('#scope-sw button').forEach(function (b) {
    b.addEventListener('click', function () { setScope(b.dataset.scope); });
  });

  /* theme toggle, persisted */
  var themeBtn = document.getElementById('theme-btn');
  var saved = null;
  try { saved = localStorage.getItem('tv-theme'); } catch (e) {}
  if (saved) document.documentElement.dataset.theme = saved;
  themeBtn.addEventListener('click', function () {
    var next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('tv-theme', next); } catch (e) {}
  });

  /* back-to-top */
  var totop = document.getElementById('totop');
  window.addEventListener('scroll', function () {
    totop.classList.toggle('show', window.scrollY > 600);
  });
  totop.addEventListener('click', function () { window.scrollTo({ top: 0, behavior: 'smooth' }); });

  route();
})();

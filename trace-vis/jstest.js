/* Behavioural test for the payload rendering in static/js/timeline.js — the
   part that turns a tool result like {"returncode":0,"output":"a\nb"} into
   readable text instead of a line full of "\n".
 *
 * There is no node on the login nodes and none in the image, so this is NOT a
 * build gate; `selftest.py --js` runs it when a runtime can be found:
 *
 *     python3 selftest.py --js --workspace /path/to/nhr-eval-workspace
 *     node jstest.js [cases.json]
 *
 * `cases.json` is optional and holds real parsed events
 * ([{src, ev}, ...] — selftest.py --js --workspace writes it); without it only
 * the synthetic cases run.
 */
'use strict';
const fs = require('fs');
const path = require('path');

const TL = path.join(__dirname, 'static', 'js', 'timeline.js');
const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
global.window = {
  TV: {
    esc: s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ESC[c]),
    md: s => '<md>' + String(s || '') + '</md>'
  }
};
eval(fs.readFileSync(TL, 'utf8'));
const TV = global.window.TV;

let fail = 0, pass = 0;
function check(ok, what, extra) {
  if (ok) { pass++; console.log('  ok   ' + what); }
  else { fail++; console.log('  FAIL ' + what + (extra ? '  -- ' + extra : '')); }
}

console.log('payloads (synthetic)');

const shell = JSON.stringify({ returncode: 0, output: '/problem\ntotal 16\nsrc\n' }, null, 2);
let b = TV.payloadBlocks(shell);
check(b.length === 2, 'a shell result splits into scalars + an output block', JSON.stringify(b));
check(b[0].text === 'returncode: 0', 'scalar fields collapse into one block', b[0].text);
check(b[1].label === 'output', 'the multi-line field keeps its name', b[1].label);
check(b[1].text === '/problem\ntotal 16\nsrc\n', 'its newlines are real newlines',
      JSON.stringify(b[1].text));

const plain = 'total 16\ndrwxr-xr-x 3 x\n';
check(TV.payloadBlocks(plain).length === 1 && TV.payloadBlocks(plain)[0].text === plain,
      'plain text output passes through untouched');

/* the reason only decoded JSON is unescaped: in source code "\n" is two
   characters and rewriting it would misreport what the agent wrote */
const code = 'printf("hello\\n");\nreturn 0;';
check(TV.payloadBlocks(code)[0].text === code,
      'a literal \\n in source code is NOT turned into a newline',
      JSON.stringify(TV.payloadBlocks(code)[0].text));

const nested = JSON.stringify({ a: 1, sub: { file: 'x.cu', body: 'l1\nl2' }, arr: [1, 'two'] });
b = TV.payloadBlocks(nested);
check(b.map(x => x.label).join(',') === ',sub,arr', 'nested values get their own blocks',
      JSON.stringify(b.map(x => x.label)));
check(b[1].text.indexOf('\n    l1\n    l2') >= 0,
      'a nested multi-line string prints as an indented block', JSON.stringify(b[1].text));
check(b[1].text.indexOf('\\n') < 0, 'no escape sequence survives in a nested string', b[1].text);

const trunc = shell.slice(0, shell.indexOf('total 16') + 8);   // as MAX_TEXT leaves it
b = TV.payloadBlocks(trunc);
check(b.length === 1 && b[0].text.indexOf('\\n') < 0 && b[0].text.indexOf('/problem\ntotal 16') > 0,
      'JSON truncated by MAX_TEXT still gets its escapes decoded', JSON.stringify(b[0].text));

b = TV.payloadBlocks('[rate_limit] {"attempt": 2, "wait_s": 30}');
check(b.length === 2 && b[0].text === '[rate_limit]' && b[1].text.indexOf('attempt: 2') === 0,
      "kimi's tagged meta line keeps the tag and opens the json", JSON.stringify(b));

check(TV.payloadBlocks(null).length === 0 && TV.payloadBlocks('').length === 0,
      'empty payloads render nothing');
check(TV.payloadBlocks({}).length === 1, 'an empty object still renders');
check(TV.payloadBlocks('{"nope": ')[0].text === '{"nope": ',
      'text that is neither parseable nor escaped is left alone');

/* --- real events, if the caller supplied them --------------------------- */
const casesFile = process.argv[2];
if (casesFile) {
  console.log('payloads (real traces)');
  const cases = JSON.parse(fs.readFileSync(casesFile, 'utf8'));
  let jsonOuts = 0, longIns = 0;
  cases.forEach((c, i) => {
    const ev = c.ev;
    ev.idx = i;
    const html = TV.renderEvent(ev);
    check(html.indexOf('<div class="ev ') === 0 && html.endsWith('</div></div>'),
          'renders: ' + (ev.tool_name || ev.kind) + ' ' + String(c.src).slice(0, 46));
    if ((ev.tool_output || '').trim().startsWith('{')) {
      jsonOuts++;
      const bs = TV.payloadBlocks(ev.tool_output);
      check(bs.map(x => x.text).join('\n').indexOf('\\n') < 0,
            'no literal \\n left in the output of ' + ev.tool_name);
      let real = null;
      try { real = JSON.parse(ev.tool_output); } catch (e) { /* truncated */ }
      if (real && typeof real.output === 'string' && real.output.indexOf('\n') >= 0) {
        const blk = bs.filter(x => x.label === 'output')[0];
        check(!!blk && blk.text === real.output,
              'the output block is byte-identical to the json value');
      }
    }
    if (ev.tool_input && typeof ev.tool_input === 'object') {
      const long = Object.keys(ev.tool_input).filter(
        k => typeof ev.tool_input[k] === 'string' && ev.tool_input[k].indexOf('\n') >= 0);
      if (long.length) {
        longIns++;
        const blk = TV.payloadBlocks(ev.tool_input).filter(x => x.label === long[0])[0];
        check(!!blk && blk.text === ev.tool_input[long[0]],
              'input field ' + long[0] + ' keeps its text verbatim');
      }
    }
  });
  check(jsonOuts >= 1, 'exercised a real json tool output', 'n=' + jsonOuts);
  check(longIns >= 1, 'exercised a real multi-line tool input', 'n=' + longIns);
}

console.log(fail ? `\njstest: FAILED — ${fail}/${pass + fail} checks failed`
                 : `\njstest: all ${pass} checks passed`);
process.exit(fail ? 1 : 0);

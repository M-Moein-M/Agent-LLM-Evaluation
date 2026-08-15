/* Markdown rendering: marked -> DOMPurify. Both vendored, so this works with
   no network inside the container. */
(function () {
  'use strict';

  if (window.marked && window.marked.setOptions) {
    window.marked.setOptions({
      gfm: true,
      breaks: true,
      headerIds: false,
      mangle: false
    });
  }

  function escapeHtml(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  /* Agent text is model output — always sanitize before inserting. */
  function renderMarkdown(text) {
    if (text == null || text === '') return '';
    var html;
    try {
      html = window.marked ? window.marked.parse(String(text)) : '<p>' + escapeHtml(text) + '</p>';
    } catch (e) {
      return '<p>' + escapeHtml(text) + '</p>';
    }
    if (window.DOMPurify) {
      html = window.DOMPurify.sanitize(html, { ADD_ATTR: ['target'] });
    }
    return html;
  }

  window.TV = window.TV || {};
  window.TV.md = renderMarkdown;
  window.TV.esc = escapeHtml;
})();

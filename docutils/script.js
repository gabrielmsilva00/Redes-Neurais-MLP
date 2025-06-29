Prism.plugins.NormalizeWhitespace.setDefaults({
            'remove-trailing': true,
            'remove-indent': true,
            'left-trim': true,
            'right-trim': true,
            'break-lines': 69,
});
function loadSection(templateId) {
  const tpl = document.getElementById(templateId);
  const clone = tpl.content.cloneNode(true);
  const target = document.getElementById('main-content');
  target.innerHTML = '';
  target.appendChild(clone);
  Prism.highlightAll();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
document.addEventListener('DOMContentLoaded', () => {
  loadSection('overview-template');
  document.querySelectorAll('nav a[data-template]').forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      const tplId = link.getAttribute('data-template');
      loadSection(tplId);
    });
  });
});

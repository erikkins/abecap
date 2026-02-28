/**
 * Format a date string for user-facing display.
 *
 * @param {string|Date} dateStr - ISO date string or Date object
 * @param {object} opts
 * @param {boolean} opts.compact - Use short form "2/27" for tight spaces (charts, tables)
 * @param {boolean} opts.includeYear - Force include year even for current year
 * @returns {string} e.g. "Feb 27", "Feb 27, 2025", "2/27"
 */
export function formatDate(dateStr, { compact = false, includeYear = false } = {}) {
  if (!dateStr) return '';
  const d = typeof dateStr === 'string' ? new Date(dateStr + (dateStr.length === 10 ? 'T12:00:00' : '')) : dateStr;
  if (isNaN(d.getTime())) return '';

  const now = new Date();
  const sameYear = d.getFullYear() === now.getFullYear();

  if (compact) {
    return `${d.getMonth() + 1}/${d.getDate()}`;
  }

  const opts = { month: 'short', day: 'numeric' };
  if (!sameYear || includeYear) {
    opts.year = 'numeric';
  }
  return d.toLocaleDateString('en-US', opts);
}

/**
 * Format for chart tick marks — very compact.
 * e.g. "Feb 27" or "Feb '25" for cross-year
 */
export function formatChartDate(dateStr) {
  if (!dateStr) return '';
  const d = new Date(dateStr + (typeof dateStr === 'string' && dateStr.length === 10 ? 'T12:00:00' : ''));
  if (isNaN(d.getTime())) return '';
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

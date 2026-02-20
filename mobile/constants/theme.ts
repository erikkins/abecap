/**
 * RigaCap brand colors and theme constants.
 * Navy + gold palette matching the web dashboard.
 */

export const Colors = {
  // Brand
  navy: '#0A1628',
  navyLight: '#1A2A4A',
  navyMid: '#132038',
  gold: '#C9A54E',
  goldLight: '#D4B96A',

  // Backgrounds
  background: '#0A1628',
  card: '#132038',
  cardBorder: '#1E3455',

  // Text
  textPrimary: '#FFFFFF',
  textSecondary: '#8899B0',
  textMuted: '#5A6B80',

  // Status
  green: '#22C55E',
  red: '#EF4444',
  yellow: '#F59E0B',
  blue: '#3B82F6',

  // Regime colors
  regime: {
    strong_bull: '#22C55E',
    weak_bull: '#86EFAC',
    rotating_bull: '#BBF7D0',
    range_bound: '#F59E0B',
    weak_bear: '#FCA5A5',
    panic_crash: '#EF4444',
    recovery: '#3B82F6',
  } as Record<string, string>,
};

export const Spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
};

export const FontSize = {
  xs: 11,
  sm: 13,
  md: 15,
  lg: 18,
  xl: 22,
  xxl: 28,
};

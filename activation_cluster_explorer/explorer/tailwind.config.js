const isProduction = !process.env.NODE_ENV === 'production';

module.exports = {
  theme: {
    colors: {
      transparent: 'transparent',
      current: 'currentColor',

      /* Greyscale */
      white: '#fff',
      lightgrey: '#F7F7F7',
      midgrey: '#cfcfcf',
      darkgrey: '#7f7f7f',
      grey: '#1e1e1e',
      black: '#0f0f0f',

      highlight: '#0071e3',
      highlight_hover: '#0077ed',

      button_outline: '#cfcfcf',

      error: '#ff453a',

      /* ColorAliases */
      text: {
        light: '#0f0f0f',
        dimmed: '#cfcfcf',
        dark: '#fff',
      },
      body: {
        light: '#fff',
        dark: '#1e1e1e',
      },
      button: '#0071e3',
      button_hover: '#0077ed',
    },
    minWidth: {
      'compare': '600px',
    },
  },
  fontFamily: {
    sans: ['-apple-system', 'BlinkMacSystemFont', 'Helvetica', 'sans-serif'],
  },
  darkMode: 'media',
  plugins: [require('@tailwindcss/forms')],
  purge: {
    content: ['./src/**/*.svelte'],
    // this is for extracting Svelte `class:` syntax but is not perfect yet
    defaultExtractor: (content) => {
      const broadMatches = content.match(/[^<>"'`\s]*[^<>"'`\s:]/g) || [];
      const broadMatchesWithoutTrailingSlash = broadMatches.map((match) =>
        _.trimEnd(match, '\\')
      );
      const matches = broadMatches.concat(broadMatchesWithoutTrailingSlash);
      return matches;
    },
    enabled: isProduction,
  },
  future: {
    purgeLayersByDefault: true,
    removeDeprecatedGapUtilities: true,
  },
};

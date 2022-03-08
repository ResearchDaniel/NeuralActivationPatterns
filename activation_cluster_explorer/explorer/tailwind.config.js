const isProduction = !process.env.NODE_ENV === 'production';

module.exports = {
  theme: {
    colors: {
      transparent: 'transparent',
      current: 'currentColor',

      /* Greyscale */
      white: '#fff',
      grey: '#cfcfcf',
      black: '#0f0f0f',

      /* Highlight */
      highlight: '#0071e3',
      highlight_hover: '#0077ed',

      /* Text */
      text: {
        light: '#0f0f0f',
        dimmed: '#cfcfcf',
        dark: '#fff',
      },

      /* Buttons */
      button: '#0071e3',
      button_hover: '#0077ed',
    },
    minWidth: {
      'compare': '800px',
    },
    extend: {
      animation: {
        "ping-slow": "ping 2s cubic-bezier(0, 0, 0.2, 1) infinite"

      }
    }
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

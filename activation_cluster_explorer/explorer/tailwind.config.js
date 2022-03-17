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
      black_semi: '#0f0f0f77',

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
    extend: {
      animation: {
        "ping-slow": "ping 2s cubic-bezier(0, 0, 0.2, 1) infinite"
      },
      boxShadow: {
        "top": "0 -5px 10px 0px rgba(0, 0, 0, 0.25)"
      },
      minWidth: {
        'compare': '800px',
      },
    }
  },
  fontFamily: {
    sans: ['-apple-system', 'BlinkMacSystemFont', 'Helvetica', 'sans-serif'],
  },
  darkMode: 'media',
  plugins: [require('@tailwindcss/forms')],
  content: {
    files: ['./src/**/*.svelte'],
    extract: {
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
  },
  future: {
    purgeLayersByDefault: true,
    removeDeprecatedGapUtilities: true,
  },
};

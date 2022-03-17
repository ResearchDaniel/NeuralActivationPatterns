const purgecss = require('@fullhuman/postcss-purgecss');
const purgeSvelte = require('purgecss-from-svelte');
const env = process.env.NODE_ENV;

module.exports = {
  plugins: [
    require('postcss-import')(),
    require('postcss-preset-env')({
      browsers: 'last 2 versions',
    }),
    require('tailwindcss')('./tailwind.config.js'),
    require('postcss-extend')(),
    require('autoprefixer')(),
    ...(env !== 'production'
      ? []
      : [
        purgecss({
          content: ['**/*.html'],
          css: ['**/*.css'],
          extractors: [
            {
              extractor: purgeSvelte,
              extensions: ['svelte'],
            },
          ],
        }),
        require('cssnano')({ preset: 'default' }),
      ]),
  ],
};

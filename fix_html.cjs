const fs = require('fs');
let html = fs.readFileSync('project/index.html', 'utf8');

// Remove data-start, data-duration, data-track-index from video and img tags ONLY
html = html.replace(/<(video|img)([^>]*?)>/g, (match, tag, attrs) => {
  attrs = attrs
    .replace(/\s*data-start="[^"]*"/g, '')
    .replace(/\s*data-duration="[^"]*"/g, '')
    .replace(/\s*data-track-index="[^"]*"/g, '');
  return '<' + tag + attrs + '>';
});

// Remove infinite GSAP repeats
html = html.replace(/repeat:\s*-1/g, 'repeat: 0');

fs.writeFileSync('project/index.html', html);
console.log('Fixed project/index.html - removed bad nesting attributes');

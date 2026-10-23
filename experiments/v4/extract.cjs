const fs = require('node:fs');
const zlib = require('node:zlib');
const path = require('node:path');
const nb = JSON.parse(fs.readFileSync(path.join(__dirname, 'baseline.ipynb'), 'utf8'));
const text = nb.cells[10].source.join('');
const encoded = text.match(/_b64\.b64decode\('([^']+)'\)/)[1];
const sources = JSON.parse(zlib.inflateSync(Buffer.from(encoded, 'base64')));
fs.mkdirSync(path.join(__dirname, 'embedded'), {recursive:true});
for (const [name, code] of Object.entries(sources)) {
  fs.writeFileSync(path.join(__dirname, 'embedded', name), code);
  console.log(name, code.length);
}

import fs from 'fs';

const testBuffer = Buffer.from('test videotest video test video', 'utf-8');
const mimeType = 'video/mp4';
const filename = 'test.mp4';

async function testUguu() {
  const form = new FormData();
  form.append('files[]', new Blob([testBuffer], { type: mimeType }), filename);
  try {
    const res = await fetch('https://uguu.se/upload.php', { method: 'POST', body: form });
    console.log('uguu:', res.status, await res.text());
  } catch (e: any) { console.log('uguu error:', e.message); }
}

async function testLitterbox() {
  const form = new FormData();
  form.append('reqtype', 'fileupload');
  form.append('time', '1h');
  form.append('fileToUpload', new Blob([testBuffer], { type: mimeType }), filename);
  try {
    const res = await fetch('https://litterbox.catbox.moe/user/api.php', { method: 'POST', body: form });
    console.log('litterbox:', res.status, await res.text());
  } catch (e: any) { console.log('litterbox error:', e.message); }
}

async function testCatbox() {
  const form = new FormData();
  form.append('reqtype', 'fileupload');
  form.append('fileToUpload', new Blob([testBuffer], { type: mimeType }), filename);
  try {
    const res = await fetch('https://catbox.moe/user/api.php', { 
        method: 'POST', 
        body: form,
        headers: { 'User-Agent': 'curl/7.68.0' }
    });
    console.log('catbox curl:', res.status, await res.text());
  } catch (e: any) { console.log('catbox error:', e.message); }
}

async function run() {
  await testUguu();
  await testLitterbox();
  await testCatbox();
}

run();

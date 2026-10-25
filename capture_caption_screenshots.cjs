const path = require('path');
const fs = require('fs');
const puppeteer = require('D:/Engineering/Return Guard/frontend/node_modules/puppeteer-core');

const chromePath = fs.existsSync('C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe')
  ? 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
  : 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';

const outDir = path.join(__dirname, 'screenshots');
if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

async function run() {
  console.log('Launching browser with:', chromePath);
  const browser = await puppeteer.launch({
    executablePath: chromePath,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--window-size=1440,920']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 920, deviceScaleFactor: 2 });

  console.log('Navigating to http://127.0.0.1:8002/...');
  await page.goto('http://127.0.0.1:8002/', { waitUntil: 'networkidle0', timeout: 30000 });
  await new Promise(r => setTimeout(r, 2000));

  // 1. Select the first preset (Dog playing frisbee or child blocks)
  console.log('Selecting first preset card...');
  const presetCard = await page.$('.preset-card');
  if (presetCard) {
    await presetCard.click();
    await new Promise(r => setTimeout(r, 1000));
  }

  // 2. Click Generate Caption (Beam Search K=3 is default)
  console.log('Clicking Generate Caption (Beam Search)...');
  const genBtn = await page.$('#btn-generate');
  if (genBtn) {
    await genBtn.click();
    // Wait for caption and stats to finish rendering
    await page.waitForFunction(() => {
      const txt = document.getElementById('txt-generated-caption');
      return txt && txt.textContent && !txt.textContent.includes('Select an image');
    }, { timeout: 15000 });
    await new Promise(r => setTimeout(r, 2000));
    await page.screenshot({ path: path.join(outDir, '01_neural_caption_studio_beam_search.png') });
    console.log('Saved 01_neural_caption_studio_beam_search.png');
  }

  // 3. Click one of the generated token chips to trigger Softmax Inspector
  console.log('Inspecting token distribution...');
  const tokenChips = await page.$$('.token-chip');
  if (tokenChips.length > 2) {
    await tokenChips[2].click();
    await new Promise(r => setTimeout(r, 1500));
    await page.screenshot({ path: path.join(outDir, '02_token_probability_inspector.png') });
    console.log('Saved 02_token_probability_inspector.png');
  }

  // 4. Switch to BLEU Evaluation Lab Tab
  console.log('Navigating to BLEU Evaluation Lab...');
  const evalTab = await page.$('#tab-eval-btn');
  if (evalTab) {
    await evalTab.click();
    await new Promise(r => setTimeout(r, 1500));
    await page.screenshot({ path: path.join(outDir, '03_bleu_evaluation_lab.png') });
    console.log('Saved 03_bleu_evaluation_lab.png');
  }

  // 5. Switch to Architecture & Flow Tab
  console.log('Navigating to Architecture & Flow...');
  const archTab = await page.$('#tab-arch-btn');
  if (archTab) {
    await archTab.click();
    await new Promise(r => setTimeout(r, 1500));
    await page.screenshot({ path: path.join(outDir, '04_model_architecture_spec.png') });
    console.log('Saved 04_model_architecture_spec.png');
  }

  // 6. Return to Studio, select Mountain Hiker, switch to Greedy Search, and generate
  console.log('Switching to Mountain Hiker with Greedy Search...');
  const studioTab = await page.$('#tab-studio-btn');
  if (studioTab) {
    await studioTab.click();
    await new Promise(r => setTimeout(r, 1000));
  }

  const allPresets = await page.$$('.preset-card');
  if (allPresets.length > 1) {
    await allPresets[1].click(); // Mountain hiker
    await new Promise(r => setTimeout(r, 1000));
  }

  const greedyBtn = await page.$('#btn-strategy-greedy');
  if (greedyBtn) {
    await greedyBtn.click();
    await new Promise(r => setTimeout(r, 500));
  }

  if (genBtn) {
    await genBtn.click();
    await new Promise(r => setTimeout(r, 3000));
    await page.screenshot({ path: path.join(outDir, '05_greedy_search_hiker.png') });
    console.log('Saved 05_greedy_search_hiker.png');
  }

  await browser.close();
  console.log('All Caption Generator screenshots captured successfully!');
}

run().catch(err => {
  console.error('Screenshot error:', err);
  process.exit(1);
});

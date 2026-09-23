// Requires Playwright. Reads local public fixture data only; no AI calls.
const { chromium } = require('playwright');
const { execFileSync } = require('child_process');
const fs = require('fs');
const assert = require('assert');
const cases = JSON.parse(execFileSync(process.env.PYTHON || '.venv/bin/python', ['-c', `
import pandas as pd,json
n=pd.read_csv('out/nodes_roles.csv');e=pd.read_parquet('data/edges.parquet')
c=n[n.role.eq('consolidator')].sort_values(['priority_score','gid'],ascending=[False,True]).iloc[0]
g=int(c.gid)
i=e[e.dst.eq(g)&e.src.ne(g)].sort_values(['sum_kzt','src'],ascending=[False,True]).to_dict('records')[0]
print(json.dumps({'collector':str(g),'sender':str(int(i['src'])),'collectors':int(n.role.eq('consolidator').sum()),'boundary':str(int(n[n.truncated_by_depth].gid.iloc[0]))}))
`], {encoding:'utf8'}));
(async () => {
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
 try {
  const page=await browser.newPage({viewport:{width:1600,height:1100}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/*',r=>{const host=new URL(r.request().url()).hostname;return ['localhost','127.0.0.1'].includes(host)?r.continue():r.abort();});
  await page.goto(process.env.APP_URL || 'http://127.0.0.1:8514');
  await page.getByRole('button',{name:'Открыть очередь сборщиков',exact:true}).click();
  console.log('collector expected',cases.collector);
  await page.waitForFunction(g=>document.querySelector('.money-id')?.innerText.trim()===g,cases.collector);
  const search=page.getByRole('textbox',{name:'Поиск по полному номеру клиента',exact:true});
  assert((await search.inputValue())==='');
  await page.getByRole('tab',{name:'Понятная схема потоков',exact:true}).click();
  await page.getByRole('heading',{name:'От кого получил → выбранный клиент → кому отправил',exact:true}).waitFor();
  await page.getByRole('button',{name:'Открыть клиента',exact:true}).first().click();
  console.log('sender expected',cases.sender);
  await page.waitForFunction(g=>document.querySelector('.money-id')?.innerText.trim()===g,cases.sender);
  await search.fill(cases.boundary);await search.press('Enter');
  await page.waitForFunction(g=>document.querySelector('.money-id')?.innerText.trim()===g,cases.boundary);
  assert((await page.getByText('Граница данных: на четвёртом шаге обход остановлен.',{exact:false}).count())>0);
  await search.fill('999');await search.press('Enter');
  await page.getByText('Клиент 999 отсутствует в данных.',{exact:false}).waitFor();
  await search.fill(cases.collector);await search.press('Enter');
  await page.waitForFunction(g=>document.querySelector('.money-id')?.innerText.trim()===g,cases.collector);
  await page.getByRole('tab',{name:'Дополнительный анализ',exact:true}).click();
  await page.locator('svg[aria-label="Поступления и отправления по дням, тенге"]').waitFor();
  await page.getByRole('tab',{name:'Схема связей',exact:true}).click();
  await page.getByRole('tab',{name:'Полная сеть',exact:true}).click();
  const frame=page.frameLocator('iframe').first();await frame.locator('canvas').waitFor();
  await frame.getByRole('button',{name:'К выбранному клиенту',exact:true}).click();
  await frame.getByRole('button',{name:'Приблизить',exact:true}).click();
  await frame.getByRole('button',{name:'Вся схема',exact:true}).click();
  assert.strictEqual(await frame.locator('script[src^="http"],link[href^="http"]').count(),0);
  assert.strictEqual(await page.locator('[data-testid="stException"]').count(),0);
  assert.deepStrictEqual(errors,[]);
  await page.getByRole('tab',{name:'Понятная схема потоков',exact:true}).click();
  await page.getByRole('heading',{name:'От кого получил → выбранный клиент → кому отправил',exact:true}).scrollIntoViewIfNeeded();
  await page.screenshot({path:'out/browser-review.png'});
  console.log(JSON.stringify({status:'passed',cases,externalNetwork:'blocked',pageErrors:errors,checks:['collector queue','largest sender navigation','boundary warning','unknown gid','offline graph','graph controls','local daily chart']}));
 } finally {await browser.close();}
})().catch(e=>{console.error(e.stack);process.exitCode=1});

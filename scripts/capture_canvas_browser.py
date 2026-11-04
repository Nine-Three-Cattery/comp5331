#!/usr/bin/env python3
"""Capture remaining course routes and directly linked external pages."""
import json
from pathlib import Path
import re
import subprocess
import sys

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "output/playwright/canvas-72132").resolve()
CLI = "/Users/nagato/.codex/skills/playwright/scripts/playwright_cli.sh"
manifest = json.loads((OUT / "manifest.json").read_text())
targets = [r for r in manifest["records"] if r["status"] == "browser_pending" or
           (r["kind"] == "external" and "rvcmedia.ust.hk" not in r["url"])]
if len(sys.argv) > 2:
    targets = json.loads(Path(sys.argv[2]).read_text())
folder = OUT / "raw/browser"
folder.mkdir(parents=True, exist_ok=True)
results_path = OUT / "raw/browser-results.json"
results = json.loads(results_path.read_text()) if results_path.exists() else []
for index, item in enumerate(targets, 1):
    code = """async page => {
      let status=null, navigationError=null;
      try {const r=await page.goto(URL,{waitUntil:'domcontentloaded',timeout:30000});status=r?.status();}
      catch(e){navigationError=String(e).slice(0,350);}
      await page.waitForLoadState('networkidle',{timeout:6000}).catch(()=>{});
      const extract=()=>{
        const node=(document.querySelector('#content') || document.querySelector('main') || document.body);
        if(!node) return {url:location.href,title:document.title,html:'',text:'',links:[],media:[]};
        const copy=node.cloneNode(true);
        copy.querySelectorAll('script,noscript,style').forEach(x=>x.remove());
        return {url:location.href,title:document.title,html:copy.innerHTML,text:node.innerText,
          links:Array.from(node.querySelectorAll('a[href]')).map(x=>({title:x.textContent.trim(),url:x.href})),
          media:Array.from(node.querySelectorAll('video,audio,source')).map(x=>x.src||x.currentSrc).filter(Boolean)};
      };
      const content=await page.evaluate(extract);
      const frames=[];
      for(const frame of page.frames().slice(1)){
        if(frame.url().includes('post_message_forwarding')) continue;
        if(frame.url()==='about:blank') continue;
        try {
          await frame.waitForFunction(()=>document.body?.innerText?.trim().length>0,null,{timeout:10000}).catch(()=>{});
          frames.push(await frame.evaluate(extract));
        }
        catch(e){frames.push({url:frame.url(),error:String(e).slice(0,250)});}
      }
      return {status,navigationError,...content,frames};
    }""".replace("URL", json.dumps(item.get("navigate_url", item["url"])))
    try:
        run = subprocess.run([CLI, "-s=comp5331-canvas", "--raw", "run-code", code],
                             capture_output=True, text=True, timeout=100)
        if run.returncode:
            raise RuntimeError(run.stdout[-1000:] + run.stderr[-500:])
        data = json.loads(run.stdout)
    except (subprocess.TimeoutExpired, RuntimeError, ValueError) as error:
        data = {"url": item["url"], "error": str(error), "status": "error"}
    data["requested_url"] = item["url"]
    data["kind"] = "tool" if item["kind"] == "tool" else item["kind"]
    (folder / (Path(sys.argv[2]).stem + f"--{index:02d}.json" if len(sys.argv) > 2 else f"{index:02d}.json")).write_text(json.dumps(data, ensure_ascii=False, indent=2))
    results = [r for r in results if r.get("requested_url") != item["url"]]
    results.append(data)
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f'{index}/{len(targets)} HTTP {data.get("status")}: {data.get("title", item["url"])}', flush=True)
    print(re.sub(r"\s+", " ", data.get("text", ""))[:300], flush=True)
    for frame in data.get("frames", []):
        print("FRAME:", frame.get("url", "").split("?")[0],
              re.sub(r"\s+", " ", frame.get("text", frame.get("error", "")))[:350], flush=True)

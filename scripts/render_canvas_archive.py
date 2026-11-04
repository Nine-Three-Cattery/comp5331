#!/usr/bin/env python3
"""Build an offline HTML index and Chinese README from a Canvas manifest."""

import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path
from urllib.parse import quote


GROUPS = (
    ("files", "资料与附件"),
    ("pages", "课程页面"),
    ("activities", "作业与活动"),
    ("external", "外部页面与工具"),
    ("api", "原始数据"),
    ("unavailable", "未保存与受限内容"),
)


def is_saved(record):
    status = str(record.get("status", "")).lower()
    success = status in {"ok", "success", "saved", "downloaded"}
    if status.isdigit():
        success = 200 <= int(status) < 300
    return bool(record.get("local_path")) and success


def group_for(record):
    if not is_saved(record):
        return "unavailable"
    kind = record.get("kind")
    if kind == "file":
        return "files"
    if kind == "api":
        return "api"
    if kind in {"discussion", "assignment", "quiz", "calendar", "group"}:
        return "activities"
    if kind in {"external", "tool"}:
        return "external"
    return "pages"


def detail_text(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False) if value is not None else ""


def local_href(path):
    return escape(quote(str(path), safe="/"), quote=True)


def record_html(record):
    title = str(record.get("title") or record.get("url") or "未命名条目")
    url = str(record.get("url") or "")
    local = record.get("local_path")
    if is_saved(record):
        heading = f'<a class="title" href="{local_href(local)}">{escape(title)}</a>'
    else:
        heading = f'<span class="title">{escape(title)}</span>'
    status = escape(str(record.get("status", "未记录")))
    kind = escape(str(record.get("kind", "")))
    detail = detail_text(record.get("detail"))
    detail_html = f'<p class="detail">{escape(detail)}</p>' if detail else ""
    source_html = ""
    sources = [s for s in record.get("source_urls", []) if s != url]
    if sources:
        links = "".join(
            f'<li><a href="{escape(str(s), quote=True)}">{escape(str(s))}</a></li>'
            for s in sources
        )
        source_html = f'<details><summary>来源页面</summary><ul>{links}</ul></details>'
    remote = (
        f'<a class="remote" href="{escape(url, quote=True)}">在线原址</a>'
        if url else ""
    )
    return (
        f'<li class="record"><div class="record-top">{heading}'
        f'<span class="pill">{kind} · {status}</span></div>'
        f'{detail_html}<div class="record-links">{remote}'
        f'<span class="path">{escape(str(local or url))}</span></div>'
        f'{source_html}</li>'
    )


def render(archive_dir):
    manifest = json.loads((archive_dir / "manifest.json").read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    title = str(manifest.get("course_title") or "Canvas 课程归档")
    captured = str(manifest.get("captured_at") or "未记录")
    root = str(manifest.get("root_url") or "")
    notes = [detail_text(n) for n in manifest.get("notes", [])]
    counts = Counter(group_for(r) for r in records)
    saved = sum(is_saved(r) for r in records)
    sections = []
    navigation = []
    for key, label in GROUPS:
        items = [r for r in records if group_for(r) == key]
        if not items:
            continue
        navigation.append(f'<a href="#{key}">{label}<span>{len(items)}</span></a>')
        cards = "".join(record_html(r) for r in items)
        sections.append(
            f'<section id="{key}"><h2>{label} <span>{len(items)}</span></h2>'
            f'<ul class="records">{cards}</ul></section>'
        )
    notes_html = "".join(f"<li>{escape(n)}</li>" for n in notes)
    if notes_html:
        notes_html = f'<aside><h2>范围与限制</h2><ul>{notes_html}</ul></aside>'
    document = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ · 离线归档</title>
<style>
:root{color-scheme:light;--ink:#172838;--muted:#587083;--line:#dce5ec;--accent:#096078}
*{box-sizing:border-box}body{margin:0;background:#f4f7f9;color:var(--ink);font:16px/1.65 system-ui,-apple-system,sans-serif}
main{max-width:1080px;margin:auto;padding:42px 24px 60px}a{color:var(--accent);text-underline-offset:3px}
header{margin-bottom:28px}.eyebrow{font-size:13px;letter-spacing:.12em;color:var(--accent);margin:0 0 8px}
h1{font-size:clamp(25px,4vw,38px);line-height:1.25;margin:0 0 18px;max-width:900px}
.meta{color:var(--muted);font-size:14px;margin:7px 0}.stats{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0}
.stat{background:#fff;border:1px solid var(--line);padding:9px 15px;border-radius:9px}.stat strong{font-size:20px;margin-right:7px}
nav{display:flex;flex-wrap:wrap;gap:8px;margin:20px 0}nav a{font-size:14px;background:#e8eff3;border-radius:6px;padding:5px 10px;text-decoration:none}nav span{margin-left:7px;color:var(--muted)}
.search{position:sticky;top:0;z-index:1;background:#f4f7f9ef;padding:12px 0;backdrop-filter:blur(8px)}
label{display:block;font-weight:600;font-size:14px;margin-bottom:5px}input{width:100%;border:1px solid #bccdd8;border-radius:9px;padding:12px 14px;font:inherit;background:#fff;color:var(--ink)}
#search-count{font-size:13px;color:var(--muted);display:block;min-height:22px;margin-top:5px}
section{scroll-margin-top:125px;margin:24px 0 32px}h2{font-size:20px;margin:0 0 12px}h2 span{font-weight:400;color:var(--muted);font-size:14px}
.records{list-style:none;padding:0;margin:0;display:grid;gap:9px}.record{background:#fff;border:1px solid var(--line);border-radius:9px;padding:14px 17px;overflow-wrap:anywhere}
.record-top{display:flex;gap:12px;justify-content:space-between;align-items:baseline}.title{font-weight:600}.pill{white-space:nowrap;font-size:11px;color:var(--muted);background:#eef3f6;padding:2px 7px;border-radius:5px}
.record-links{display:flex;align-items:baseline;gap:12px;font-size:12px;margin-top:4px}.remote{white-space:nowrap}.path{color:var(--muted)}.detail{font-size:13px;margin:8px 0;color:#68542a;white-space:pre-wrap}
details{font-size:12px;color:var(--muted);margin-top:7px}details ul{margin:4px 0;padding-left:20px}aside{padding:18px 20px;background:#edf2f5;border-radius:9px;font-size:14px}aside h2{font-size:16px}aside ul{margin:0;padding-left:20px}
footer{border-top:1px solid var(--line);padding-top:18px;color:var(--muted);font-size:13px}[hidden]{display:none!important}
@media(max-width:600px){main{padding:26px 16px}.record-top{display:block}.pill{display:inline-block;margin-top:5px}.record-links{display:block}.path{display:block;margin-top:3px}}
</style></head><body><main>
<header><p class="eyebrow">CANVAS · LOCAL ARCHIVE</p><h1>__TITLE__</h1>
<p class="meta">抓取时间：__CAPTURED__</p><p class="meta">根页面：<a href="__ROOT__">__ROOT_TEXT__</a></p>
<div class="stats"><div class="stat"><strong>__SAVED__</strong>已保存条目</div><div class="stat"><strong>__FILES__</strong>资料与附件</div><div class="stat"><strong>__UNAVAILABLE__</strong>未保存或受限</div></div>
<p class="meta">点击标题打开本地内容；“在线原址”仍需网络及相应权限。数量按清单条目计算。</p>
<nav aria-label="归档分类">__NAV__</nav></header>
__NOTES__
<div class="search"><label for="search">搜索归档目录</label><input id="search" type="search" placeholder="输入标题、文件名或网址" autocomplete="off"><output id="search-count" aria-live="polite"></output></div>
__SECTIONS__
<footer><a href="README.md">抓取说明</a> · <a href="manifest.json">完整清单 manifest.json</a></footer>
</main><script>
const search=document.querySelector('#search');
const rows=[...document.querySelectorAll('.record')];
const sections=[...document.querySelectorAll('section')];
search.addEventListener('input',()=>{
  const query=search.value.trim().toLocaleLowerCase();
  let count=0;
  for(const row of rows){row.hidden=!row.textContent.toLocaleLowerCase().includes(query);if(!row.hidden)count++;}
  for(const section of sections)section.hidden=![...section.querySelectorAll('.record')].some(row=>!row.hidden);
  document.querySelector('#search-count').textContent=query?`找到 ${count} 个条目`:`共 ${rows.length} 个条目`;
});
search.dispatchEvent(new Event('input'));
</script></body></html>
"""
    replacements = {
        "__TITLE__": escape(title),
        "__CAPTURED__": escape(captured),
        "__ROOT__": escape(root, quote=True),
        "__ROOT_TEXT__": escape(root),
        "__SAVED__": str(saved),
        "__FILES__": str(counts["files"]),
        "__UNAVAILABLE__": str(counts["unavailable"]),
        "__NAV__": "".join(navigation),
        "__NOTES__": notes_html,
        "__SECTIONS__": "".join(sections),
    }
    for marker, value in replacements.items():
        document = document.replace(marker, value)
    (archive_dir / "index.html").write_text(document, encoding="utf-8")

    lines = [
        f"# {title}：本地归档", "", f"抓取时间：{captured}", "",
        f"根页面：[课程主页]({root})", "",
        "打开 [index.html](index.html) 浏览和搜索本地目录。点击条目标题打开已保存文件；在线原址可能需要重新登录。", "",
        "本次归档收录抓取程序实际保存的课程页面、资料和活动数据；完整来源与请求状态见 [manifest.json](manifest.json)。", "",
        f"清单共 {len(records)} 个条目，其中 {saved} 个已保存，{counts['unavailable']} 个未保存或受限。以下数量按清单条目计算；页面和 API 数据可能对应同一课程资源。", "",
        "| 内容 | 已保存条目数 |", "| --- | ---: |",
    ]
    lines.extend(f"| {label} | {counts[key]} |" for key, label in GROUPS if key != "unavailable")
    lines.extend(["", "归档仅代表抓取时当前账号可访问的内容。未保存条目保留原网址和实际状态，不计入已保存数量；第三方工具的登录、交互和受限内容以清单及下方说明为准。", ""])
    if notes:
        lines.append("抓取说明：")
        lines.append("")
        lines.extend(f"- {note}" for note in notes)
        lines.append("")
    unavailable = [r for r in records if not is_saved(r)]
    if unavailable:
        lines.extend(["未保存或受限条目：", ""])
        for record in unavailable:
            label = str(record.get("title") or record.get("url") or "未命名条目")
            label = label.replace("[", r"\[").replace("]", r"\]")
            url = str(record.get("url") or "")
            target = f"[{label}](<{url}>)" if url else label
            detail = detail_text(record.get("detail"))
            suffix = f"；{detail}" if detail else ""
            lines.append(f"- {target}：{record.get('status', '未记录')}{suffix}")
        lines.append("")
    (archive_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"已生成 {archive_dir / 'index.html'}")
    print(f"清单 {len(records)} 条，已保存 {saved} 条，未保存或受限 {counts['unavailable']} 条。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_dir", type=Path, help="包含 manifest.json 的归档目录")
    args = parser.parse_args()
    render(args.archive_dir)

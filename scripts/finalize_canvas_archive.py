#!/usr/bin/env python3
"""Merge captured browser pages and render a Canvas archive for offline reading."""

import argparse
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

from render_canvas_archive import is_saved, render


FILE_ID = re.compile(r"/(?:courses/72132/)?files/(\d+)(?:/|$)")
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
SKIP_CONTENT = {"script", "style", "template", "head", "iframe", "object"}
SKIP_TAGS = {"html", "body", "form", "input", "button", "base", "link", "meta", "embed"}


def without_fragment(url):
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, p.path.rstrip("/"), p.query, ""))


def canvas_file_id(url):
    p = urlsplit(url)
    match = FILE_ID.search(p.path) if p.hostname == "canvas.ust.hk" else None
    return match.group(1) if match else None


class Text(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skipping = []
        self.feed(html or "")

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_CONTENT:
            self.skipping.append(tag)

    def handle_endtag(self, tag):
        if self.skipping and tag == self.skipping[-1]:
            self.skipping.pop()

    def handle_data(self, data):
        if not self.skipping:
            self.parts.append(data)

    @property
    def text(self):
        return " ".join(" ".join(self.parts).split())


class References(HTMLParser):
    def __init__(self, html, base):
        super().__init__(convert_charrefs=True)
        self.urls = []
        self.base = base
        self.feed(html or "")

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in {"href", "src"} and value:
                self.urls.append(urljoin(self.base, value))


class StaticHTML(HTMLParser):
    def __init__(self, html, base, page_path, url_map, file_map):
        super().__init__(convert_charrefs=True)
        self.base, self.page_path = base, page_path
        self.url_map, self.file_map = url_map, file_map
        self.output, self.skipping = [], []
        self.feed(html or "")

    def rewrite(self, value):
        absolute = urljoin(self.base, value)
        p = urlsplit(absolute)
        if p.scheme not in {"http", "https"}:
            return value
        target = self.url_map.get(without_fragment(absolute))
        fid = canvas_file_id(absolute)
        if fid:
            target = self.file_map.get(fid, target)
        if not target:
            return absolute
        relative = os.path.relpath(target, Path(self.page_path).parent).replace(os.sep, "/")
        return quote(relative, safe="/") + ("#" + p.fragment if p.fragment else "")

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_CONTENT:
            if tag == "iframe" and not self.skipping:
                self.output.append('<p class="frame-note">嵌入内容见本页后附的实际框架快照。</p>')
            self.skipping.append(tag)
            return
        if self.skipping or tag in SKIP_TAGS:
            return
        static_attrs = []
        for name, value in attrs:
            if name.startswith("on") or name in {"hidden", "aria-hidden", "srcdoc", "contenteditable", "formaction"}:
                continue
            if name in {"href", "src"} and value:
                if value.strip().lower().startswith("javascript:"):
                    continue
                value = self.rewrite(value)
            if name == "style" and value:
                value = re.sub(r"(?:display\s*:\s*none|visibility\s*:\s*hidden)\s*!?[^;]*(?:;|$)", "", value, flags=re.I)
            static_attrs.append(name if value is None else f'{name}="{escape(value, quote=True)}"')
        suffix = " " + " ".join(static_attrs) if static_attrs else ""
        self.output.append(f"<{tag}{suffix}>")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if self.skipping:
            if tag == self.skipping[-1]:
                self.skipping.pop()
            return
        if tag not in SKIP_TAGS and tag not in VOID_TAGS:
            self.output.append(f"</{tag}>")

    def handle_data(self, data):
        if not self.skipping:
            self.output.append(escape(data))

    @property
    def html(self):
        return "".join(self.output)


def has_content(item):
    html = item.get("html", "") or ""
    return bool(Text(html).text or (item.get("text") or "").strip() or re.search(r"<(?:img|video|audio|table)\b", html, re.I))


def auth_prompt(item):
    p = urlsplit(item.get("url") or "")
    title = (item.get("title") or "").strip().lower()
    text = (item.get("text") or Text(item.get("html", "")).text).lower()
    if p.hostname in {"login.microsoftonline.com", "shib.ust.hk", "login.ust.hk", "accounts.google.com"} or (p.hostname or "").endswith("duosecurity.com"):
        return True
    if re.search(r'<input\b[^>]*type\s*=\s*["\']?password', item.get("html", ""), re.I):
        return True
    if title in {"login", "log in", "sign in", "sign in to your account", "consent", "authorization required"}:
        return True
    prompts = ("please sign in", "please log in", "sign in to continue", "consent required", "authorization required", "grant access to")
    return len(text) < 1500 and any(prompt in text for prompt in prompts)


def usable_frames(item):
    return [frame for frame in item.get("frames", [])
            if frame.get("url") != "about:blank"
            and "post_message_forwarding" not in frame.get("url", "")]


def browser_record(item, path):
    requested = item.get("requested_url") or item.get("url")
    status = item.get("status", "unknown")
    details = [f"浏览器 HTTP 状态：{status}。"]
    kind = item.get("kind", "page")
    frames = usable_frames(item)
    failed = (isinstance(status, int) and status >= 400) or status in {"error", "unavailable", "blocked"} or bool(item.get("error"))
    if item.get("error"):
        details.append(str(item["error"]))
    if item.get("navigationError"):
        details.append("导航提示：" + str(item["navigationError"]))
    if not failed and auth_prompt(item):
        status, failed = "auth_required", True
        details.append("当前快照为登录或授权提示，未获取目标内容。")
    if kind == "tool":
        actual = [frame for frame in frames if has_content(frame) and not frame.get("error") and not auth_prompt(frame)]
        if not failed and not actual:
            status, failed = "unavailable", True
            details.append("工具只有启动外壳、空框架或登录/授权提示，未获取工具内部内容。")
        elif not failed:
            details.append("仅保存实际 iframe 可见内容的静态快照，不表示第三方工具内部内容已全部抓全。")
    elif not failed and not has_content(item) and not any(has_content(f) and not auth_prompt(f) for f in frames):
        status, failed = "empty", True
        details.append("页面与框架正文为空，未计为已保存内容。")
    if not failed and not isinstance(status, int):
        status = "saved"
    return {"url": requested, "title": item.get("title") or requested, "kind": kind,
            "status": status, "local_path": None if failed else path, "detail": " ".join(details), "source_urls": []}


def document(title, source, captured_at, content, page_path):
    back = quote(os.path.relpath("index.html", Path(page_path).parent).replace(os.sep, "/"), safe="/")
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title>
<style>body{{max-width:1000px;margin:32px auto;padding:0 22px;color:#192c3b;font:16px/1.65 system-ui,sans-serif}}a{{color:#08637c}}header{{border-bottom:1px solid #dce5ec;padding-bottom:15px;margin-bottom:24px}}header p,.frame-note{{font-size:13px;color:#5a6d79}}h1{{line-height:1.3}}img,video{{max-width:100%;height:auto}}table{{border-collapse:collapse;max-width:100%}}td,th{{padding:6px 10px;border:1px solid #cbd7de}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f7;padding:14px}}.frame-snapshot{{border-top:2px solid #dce5ec;margin-top:35px;padding-top:20px}}.frame-note{{padding:9px 12px;background:#f2f5f7}}</style></head>
<body><header><a href="{back}">← 返回归档目录</a> · <a href="{escape(source, quote=True)}">在线原址</a><h1>{escape(title)}</h1><p>抓取时间：{escape(captured_at)}。本页为已抓取内容的静态副本。</p></header>{content}</body></html>'''


def finalize(archive_dir):
    manifest_path = archive_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bodies = json.loads((archive_dir / "raw/page-bodies.json").read_text(encoding="utf-8"))
    additional_pages = archive_dir / "raw/supplemental-pages.json"
    if additional_pages.exists():
        bodies.update(json.loads(additional_pages.read_text(encoding="utf-8")))
    browser_path = archive_dir / "raw/browser-results.json"
    browser_items = json.loads(browser_path.read_text(encoding="utf-8")) if browser_path.exists() else []
    browser = {r.get("requested_url") or r.get("url"): r for r in browser_items}
    records = {r["url"]: r for r in manifest["records"]}

    def upsert(record):
        previous = records.get(record["url"], {})
        record["source_urls"] = list(dict.fromkeys(previous.get("source_urls", []) + record.get("source_urls", [])))
        previous.update(record)
        records[record["url"]] = previous

    recording_path = archive_dir / "raw/recording-result.json"
    if recording_path.exists():
        upsert(json.loads(recording_path.read_text(encoding="utf-8")))
    for number, item in enumerate(browser.values(), 1):
        upsert(browser_record(item, f"pages/browser--{number:02d}.html"))
    spreadsheet_path = archive_dir / "raw/spreadsheet-result.json"
    if spreadsheet_path.exists():
        upsert(json.loads(spreadsheet_path.read_text(encoding="utf-8")))
    additional_records = archive_dir / "raw/supplemental-records.json"
    if additional_records.exists():
        for item in json.loads(additional_records.read_text(encoding="utf-8")):
            upsert(item)

    url_map = {without_fragment(r["url"]): r["local_path"] for r in records.values() if is_saved(r)}
    file_map = {canvas_file_id(r["url"]): r["local_path"] for r in records.values()
                if r.get("kind") == "file" and is_saved(r) and canvas_file_id(r["url"])}
    file_records = {canvas_file_id(r["url"]): r for r in records.values() if r.get("kind") == "file" and canvas_file_id(r["url"])}
    for requested, item in browser.items():
        record = records[requested]
        if is_saved(record) and item.get("url"):
            url_map[without_fragment(item["url"])] = record["local_path"]

    def references(html, source):
        for url in References(html, source).urls:
            file_record = file_records.get(canvas_file_id(url))
            if file_record is not None and source not in file_record.setdefault("source_urls", []):
                file_record["source_urls"].append(source)

    def static(html, source, path):
        references(html, source)
        return StaticHTML(html, source, path, url_map, file_map).html

    def save_page(path, title, source, content):
        target = archive_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(document(title, source, manifest.get("captured_at", ""), content, path), encoding="utf-8")

    for url, page in bodies.items():
        path = page["path"]
        content = static(page.get("body", ""), url, path) + static(page.get("extra", ""), url, path)
        save_page(path, page.get("title") or url, url, content)
    for requested, item in browser.items():
        record = records[requested]
        source = item.get("url") or requested
        for frame in usable_frames(item):
            references(frame.get("html", ""), frame.get("url") or source)
        references(item.get("html", ""), source)
        if not is_saved(record):
            continue
        path = record["local_path"]
        if not path.lower().endswith((".html", ".htm")):
            continue
        content = static(item.get("html", ""), source, path)
        if not Text(content).text and item.get("text"):
            content += "<pre>" + escape(item["text"]) + "</pre>"
        for frame in usable_frames(item):
            frame_url = frame.get("url") or source
            frame_content = static(frame.get("html", ""), frame_url, path)
            if not Text(frame_content).text and frame.get("text"):
                frame_content += "<pre>" + escape(frame["text"]) + "</pre>"
            if not frame_content.strip() and not frame.get("error"):
                continue
            heading = escape(frame.get("title") or "嵌入框架")
            content += f'<section class="frame-snapshot"><h2>{heading}</h2><p><a href="{escape(frame_url, quote=True)}">框架原址</a></p>{frame_content}'
            if frame.get("error"):
                content += "<pre>" + escape(str(frame["error"])) + "</pre>"
            content += "</section>"
        save_page(path, record["title"], requested, content)

    manifest["records"] = list(records.values())
    note = "离线 HTML 已移除脚本与交互表单，已下载课程页面和文件链接改为本地相对路径；第三方工具仅保存实际获取的框架静态快照。"
    notes = [note,
             "范围为课程 72132 的实际链接及当前账号可读取的课程、小组内容；外部教师网站、校园地图和图书馆保存落地页或可见内容，不递归整个外部网站。",
             "课程页面目录和整课文件列表接口受限，因此沿课程正文中的实际链接收集；无法枚举未链接或被权限隐藏的内容。",
             "测验仅保存当前可见说明，题目或结果按 Canvas 原有状态锁定；一个讨论主题的回复接口返回 403。",
             "Lucid、Chat、Zoom 的浏览器框架未返回可见正文，图书馆导航超时；这些入口保留实际失败状态。",
             "录像按用户后续要求只保留播放链接，未下载完整 MP4。"]
    manifest["notes"] = list(dict.fromkeys(manifest.get("notes", []) + notes))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    render(archive_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_dir", nargs="?", type=Path, default=Path("output/playwright/canvas-72132"))
    finalize(parser.parse_args().archive_dir)

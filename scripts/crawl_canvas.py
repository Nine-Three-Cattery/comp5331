#!/usr/bin/env python3
"""Archive readable course content using an existing Playwright login."""
import http.cookiejar
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from html import escape
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPCookieProcessor, Request, build_opener

BASE = "https://canvas.ust.hk"
ROOT = BASE + "/courses/72132"
API = BASE + "/api/v1/courses/72132"
CLI = "/Users/nagato/.codex/skills/playwright/scripts/playwright_cli.sh"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "output/playwright/canvas-72132").resolve()
OUT.mkdir(parents=True, exist_ok=True)


def browser(code):
    run = subprocess.run([CLI, "-s=comp5331-canvas", "--raw", "run-code", code],
                         capture_output=True, text=True, check=True)
    return json.loads(run.stdout)


class Links(HTMLParser):
    def __init__(self, body, base):
        super().__init__()
        self.urls = []
        self.base = base
        self.feed(body or "")

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for attr in (["href"] if tag == "a" else ["src", "poster"]):
            value = attrs.get(attr)
            if value and not value.startswith(("data:", "mailto:", "javascript:", "tel:", "#")):
                self.urls.append(urljoin(self.base, value))


def name(value):
    return re.sub(r"[/\\\x00-\x1f]", "_", str(value)).strip()[:180] or "untitled"


manifest = {"root_url": ROOT, "course_title": "COMP5331", "captured_at":
            datetime.now().astimezone().isoformat(), "records": [], "notes": []}
if (OUT / "manifest.json").exists():
    manifest = json.loads((OUT / "manifest.json").read_text())
records = {r["url"]: r for r in manifest["records"]}
pages = {}
pending = []
queued = set()
file_ids = set()
api_data = {}


def save_manifest():
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


def record(url, title, kind, status, local_path=None, source=None, detail=None):
    item = records.get(url)
    if item is None:
        item = {"url": url, "title": title, "kind": kind, "status": status,
                "local_path": local_path, "source_urls": []}
        records[url] = item
        manifest["records"].append(item)
    else:
        item.update(title=title or item["title"], kind=kind, status=status, local_path=local_path)
    if source and source not in item["source_urls"]:
        item["source_urls"].append(source)
    if detail:
        item["detail"] = detail
    save_manifest()
    return item


def write_json(rel, value):
    target = OUT / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def fetch(url, headers=None):
    try:
        return opener.open(Request(url, headers={"User-Agent": user_agent,
                                                 "Accept": "application/json,text/html,*/*", **(headers or {})}), timeout=40)
    except HTTPError as error:
        return error


def api(url, label):
    cached = OUT / ("api/" + name(label) + ".json")
    if cached.exists():
        data = json.loads(cached.read_text())
        if records.get(url, {}).get("status") != 200:
            return None
        if isinstance(data, list):
            for part in sorted(cached.parent.glob(name(label) + "--*.json")):
                data.extend(json.loads(part.read_text()))
        return data
    result = []
    cursor = url
    count = 0
    while cursor:
        count += 1
        rel = "api/" + name(label) + (f"--{count}" if count > 1 else "") + ".json"
        try:
            response = fetch(cursor)
            status = response.status
            raw = response.read()
            try:
                value = json.loads(raw.decode("utf-8-sig").removeprefix("while(1);"))
            except (ValueError, UnicodeError):
                record(cursor, label, "api", status, detail="Response was not JSON; possibly a login redirect.")
                return None
            write_json(rel, value)
            record(cursor, label, "api", status, rel)
            if status != 200:
                print(f"API {label}: HTTP {status}", flush=True)
                return None
            if isinstance(value, list):
                result.extend(value)
            else:
                api_data[url] = value
                return value
            match = re.search(r'<([^>]+)>;\s*rel="next"', response.headers.get("Link", ""))
            cursor = match.group(1) if match else None
        except (URLError, TimeoutError, OSError) as error:
            record(cursor, label, "api", "error", detail=str(error))
            print(f"API {label}: {error}", flush=True)
            return None
    api_data[url] = result
    print(f"API {label}: {len(result)} items", flush=True)
    return result


def discover(body, source):
    for url in Links(body, source).urls:
        enqueue(url, source)


def enqueue(url, source=ROOT):
    parts = urlsplit(urljoin(source, url))
    if parts.scheme not in ("http", "https"):
        return
    url = urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), parts.query, ""))
    match = re.search(r"/(?:courses/72132/)?files/(\d+)", parts.path)
    if parts.hostname == "canvas.ust.hk" and match:
        file_ids.add(int(match.group(1)))
        return
    if url not in queued:
        queued.add(url)
        pending.append((url, source))


def page_record(url, title, body, kind="page", extra="", source=None):
    rel = "pages/" + name(urlsplit(url).path.strip("/").replace("/", "--"))
    if urlsplit(url).query:
        rel += "--" + name(urlsplit(url).query.replace("&", "_"))
    rel += ".html"
    pages[url] = {"title": title, "body": body or "", "extra": extra, "path": rel}
    record(url, title, kind, 200, rel, source)
    discover(body, url)


def crawl_page(url, source):
    path = urlsplit(url).path
    match = re.fullmatch(r"/courses/72132/pages/([^/]+)", path)
    if match:
        data = api(API + "/pages/" + match.group(1), "page-" + match.group(1))
        if data and isinstance(data, dict):
            page_record(url, data.get("title", path), data.get("body", ""), source=source)
        else:
            record(url, unquote(match.group(1)), "page", "unavailable", source=source,
                   detail="Canvas page API did not return readable page content.")
        return
    if url in records:
        if source not in records[url]["source_urls"]:
            records[url]["source_urls"].append(source)
        return
    if urlsplit(url).hostname == "canvas.ust.hk":
        record(url, unquote(path.rsplit("/", 1)[-1]) or "Home", "tool" if "/external_tools/" in path else "page",
               "browser_pending", source=source, detail="Requires rendered browser capture.")
        return
    record(url, url, "external", "pending", source=source)


def download_file(metadata, folder_paths):
    fid = metadata["id"]
    url = ROOT + "/files/" + str(fid)
    filename = metadata.get("display_name") or metadata.get("filename") or str(fid)
    folder = folder_paths.get(metadata.get("folder_id"), "")
    rel = "files/" + ((folder + "/") if folder else "") + str(fid) + "--" + name(filename)
    dest = OUT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    download_url = metadata.get("url") or BASE + "/files/" + str(fid) + "/download"
    expected = metadata.get("size")
    result = {"url": url, "title": filename, "kind": "file", "status": "error"}
    try:
        for attempt in range(3):
            existing = dest.stat().st_size if dest.exists() else 0
            if expected is not None and existing == expected:
                break
            headers = {"Range": f"bytes={existing}-"} if expected and 0 < existing < expected else {}
            response = fetch(download_url, headers)
            if response.status not in (200, 206):
                return {**result, "status": response.status, "detail": "File download was denied."}
            ctype = response.headers.get_content_type()
            if "html" in ctype and not filename.lower().endswith((".html", ".htm")):
                return {**result, "status": "unavailable", "detail": "Received HTML instead of the file."}
            append = response.status == 206 and response.headers.get("Content-Range", "").startswith(f"bytes {existing}-")
            with dest.open("ab" if append else "wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if expected is None or dest.stat().st_size == expected:
                break
        actual = dest.stat().st_size
        if expected is not None and actual != expected:
            result.update(status="incomplete", local_path=rel,
                          detail=f"Expected {expected} bytes, received {actual} bytes.")
        else:
            result.update(status=200, local_path=rel, detail=f"{actual} bytes")
        print(f"FILE {fid}: {filename} ({actual} bytes)", flush=True)
    except (URLError, TimeoutError, OSError) as error:
        result.update(detail=str(error))
        print(f"FILE {fid}: {error}", flush=True)
    return result


with tempfile.TemporaryDirectory(prefix="comp5331-canvas-") as temporary:
    auth_path = Path(temporary) / "browser-state.json"
    subprocess.run([CLI, "-s=comp5331-canvas", "state-save", str(auth_path)],
                   capture_output=True, text=True, check=True)
    auth_path.chmod(0o600)
    state = json.loads(auth_path.read_text())
    jar = http.cookiejar.CookieJar()
    for cookie in state["cookies"]:
        jar.set_cookie(http.cookiejar.Cookie(0, cookie["name"], cookie["value"], None, False,
                       cookie["domain"], cookie["domain"].startswith("."), cookie["domain"].startswith("."),
                       cookie["path"], True, cookie["secure"],
                       int(cookie["expires"]) if cookie["expires"] > 0 else None,
                       cookie["expires"] <= 0, None, None, {}, False))
    opener = build_opener(HTTPCookieProcessor(jar))
    user_agent = browser("async page => await page.evaluate(() => navigator.userAgent)")
    home_path = OUT / "raw/home-browser.json"
    home = json.loads(home_path.read_text()) if home_path.exists() else browser('async page => ({url:page.url(),title:await page.title(),html:await page.locator("#content").innerHTML(),links:await page.locator("#content a[href], #right-side a[href], #section-tabs a[href]").evaluateAll(xs=>xs.map(a=>({title:a.textContent.trim(),url:a.href})))})')
    write_json("raw/home-browser.json", home)
    page_record(ROOT, home["title"], home["html"])
    for link in home["links"]:
        enqueue(link["url"])
    course = api(API + "?include[]=syllabus_body", "course")
    if not course or not course.get("id"):
        raise SystemExit("Existing login could not read course metadata.")
    manifest["course_title"] = course["name"]
    categories = {}
    for key, suffix in [
        ("tabs", "/tabs"), ("pages", "/pages?per_page=100"),
        ("modules", "/modules?per_page=100&include[]=items"),
        ("assignments", "/assignments?per_page=100&include[]=submission"),
        ("discussions", "/discussion_topics?per_page=100"),
        ("quizzes", "/quizzes?per_page=100"), ("files", "/files?per_page=100"),
        ("folders", "/folders?per_page=100"), ("groups", "/groups?per_page=100")]:
        categories[key] = api(API + suffix, key) or []
    announcements = api(BASE + "/api/v1/announcements?context_codes[]=course_72132&per_page=100", "announcements") or []
    calendar = api(BASE + "/api/v1/calendar_events?context_codes[]=course_72132&all_events=true&per_page=100", "calendar") or []
    for item in categories["pages"]:
        enqueue(item.get("html_url") or ROOT + "/pages/" + item["url"])
    for module in categories["modules"]:
        items = api(API + f'/modules/{module["id"]}/items?per_page=100', f'module-{module["id"]}') or []
        for item in items:
            if item.get("html_url"):
                enqueue(item["html_url"])
            if item.get("external_url"):
                enqueue(item["external_url"])
    for key, kind in [("assignments", "assignment"), ("discussions", "discussion"), ("quizzes", "quiz")]:
        for item in categories[key]:
            url = item.get("html_url") or ROOT + f'/{key}/{item["id"]}'
            extra = "<h2>Metadata</h2><pre>" + escape(json.dumps(item, ensure_ascii=False, indent=2)) + "</pre>"
            page_record(url, item.get("name") or item.get("title") or str(item["id"]),
                        item.get("description") or item.get("message") or "", kind, extra)
            if kind == "discussion":
                view = api(API + f'/discussion_topics/{item["id"]}/view', f'discussion-{item["id"]}-view')
                if view:
                    pages[url]["extra"] += "<h2>Discussion replies (raw)</h2><pre>" + escape(json.dumps(view, ensure_ascii=False, indent=2)) + "</pre>"
                    discover(json.dumps(view, ensure_ascii=False), url)
            if kind == "assignment":
                sub = api(API + f'/assignments/{item["id"]}/submissions/self?include[]=submission_comments&include[]=submission_history&include[]=rubric_assessment', f'assignment-{item["id"]}-my-submission')
                if sub:
                    for attachment in sub.get("attachments", []):
                        file_ids.add(attachment["id"])
                    pages[url]["extra"] += "<h2>My submission and feedback</h2><pre>" + escape(json.dumps(sub, ensure_ascii=False, indent=2)) + "</pre>"
    for item in announcements:
        page_record(item.get("html_url") or ROOT + f'/discussion_topics/{item["id"]}',
                    item.get("title", "Announcement"), item.get("message", ""), "discussion")
    for item in calendar:
        if item.get("html_url"):
            page_record(item["html_url"], item.get("title", "Calendar event"), item.get("description", ""), "calendar",
                        "<pre>" + escape(json.dumps(item, ensure_ascii=False, indent=2)) + "</pre>")
    i = 0
    while i < len(pending):
        url, source = pending[i]
        i += 1
        crawl_page(url, source)
    folder_paths = {}
    for folder in categories["folders"]:
        components = folder.get("full_name", "").split("/")[1:]
        folder_paths[folder["id"]] = "/".join(name(x) for x in components)
    file_map = {item["id"]: item for item in categories["files"]}
    for fid in sorted(file_ids - file_map.keys()):
        data = api(BASE + "/api/v1/files/" + str(fid), "file-" + str(fid))
        if data and data.get("id"):
            file_map[fid] = data
        else:
            record(ROOT + "/files/" + str(fid), str(fid), "file", "unavailable", detail="Linked file metadata was not accessible.")
    write_json("raw/page-bodies.json", pages)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download_file, metadata, folder_paths) for metadata in file_map.values()]
        for future in as_completed(futures):
            record(**future.result())
    write_json("raw/browser-pending.json", [r for r in manifest["records"] if r["status"] == "browser_pending"])
    write_json("raw/external-pending.json", [r for r in manifest["records"] if r["kind"] == "external"])
    manifest["notes"].append("课程页面与附件使用本人已登录会话读取；未修改课程内容、未开始测验、未提交作业或发送消息。")
    manifest["notes"].append("原始 API 响应保存在 api/；正文保存在 pages/；附件保存在 files/。未成功获取的入口保留在清单中。")
    save_manifest()
    print(json.dumps({"records": len(manifest["records"]), "pages": len(pages), "files": len(file_map),
                      "browser_pending": len([r for r in manifest["records"] if r["status"] == "browser_pending"]),
                      "external_pending": len([r for r in manifest["records"] if r["kind"] == "external"])}, ensure_ascii=False), flush=True)

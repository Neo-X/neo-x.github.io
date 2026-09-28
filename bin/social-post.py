#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "requests-oauthlib", "pyyaml"]
# ///
"""Post a blog post's social media drafts to X, Bluesky, and LinkedIn.

The drafts live in a Liquid comment block at the bottom of the post:

    {% comment %}
    --- twitter ---
    Text for X. {{URL}}
    --- bluesky ---
    Text for Bluesky. {{URL}}
    --- linkedin ---
    Longer text for LinkedIn. {{URL}}
    {% endcomment %}

Usage:
    bin/social-post _i18n/en/_posts/2026-09-28-knowledge-vs-information.md [--dry-run] [--only twitter,bluesky]
    bin/social-post auth linkedin

Credentials are read from ~/.config/social-post/credentials.json (setup steps: ~/.claude/skills/social-draft/SKILL.md).
"""

import argparse
import datetime as dt
import http.server
import json
import re
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.parse
import webbrowser
from pathlib import Path

import requests
import yaml
from requests_oauthlib import OAuth1

REPO = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path.home() / ".config" / "social-post"
CREDS_FILE = CONFIG_DIR / "credentials.json"
PLATFORMS = ["twitter", "bluesky", "linkedin"]
LIMITS = {"twitter": 280, "bluesky": 300, "linkedin": 3000}
X_URL_LENGTH = 23  # X counts every link as 23 characters
LINKEDIN_REDIRECT = "http://localhost:8765/callback"


# ---------------------------------------------------------------- post parsing

def load_post(path):
    text = path.read_text()
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        sys.exit(f"{path}: no front matter found")
    front = yaml.safe_load(m.group(1)) or {}
    return text, front


def site_config():
    return yaml.safe_load((REPO / "_config.yml").read_text())


def post_url(path, cfg):
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})-(.+)\.md$", path.name)
    if not m:
        sys.exit(f"{path.name}: expected YYYY-MM-DD-slug.md")
    year, month, day, slug = m.groups()
    permalink = cfg.get("permalink", "/:year/:month/:day/:title.html")
    for key, val in {"year": year, "month": month, "day": day, "title": slug}.items():
        permalink = permalink.replace(f":{key}", val)
    return cfg["url"].rstrip("/") + cfg.get("baseurl", "") + permalink


def parse_drafts(text, url):
    blocks = re.findall(r"\{%\s*comment\s*%\}(.*?)\{%\s*endcomment\s*%\}", text, re.S)
    drafts = {}
    for block in blocks:
        parts = re.split(r"^---\s*(\w+)\s*---\s*$", block, flags=re.M)
        for name, body in zip(parts[1::2], parts[2::2]):
            if name.lower() in PLATFORMS:
                drafts[name.lower()] = body.strip().replace("{{URL}}", url)
    return drafts


def cover_path(front):
    cover = front.get("cover") or ""
    m = re.search(r'src="([^"]+)"', cover) or re.match(r"\s*(\S+\.(?:png|jpe?g|svg|gif|webp))\s*$", cover)
    if not m:
        return None
    return REPO / m.group(1).lstrip("/")


def cover_image(front):
    """Return (bytes, mime) for the post's cover, rasterizing SVG to PNG."""
    path = cover_path(front)
    if path is None or not path.exists():
        print("  ! no cover image found; posting without an image")
        return None
    if path.suffix.lower() == ".svg":
        out = Path(tempfile.mkdtemp()) / (path.stem + ".png")
        subprocess.run(["inkscape", str(path), "-o", str(out), "-w", "1200",
                        "-b", "#ffffff"], check=True, capture_output=True)
        path = out
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp"}[path.suffix.lower()]
    return path.read_bytes(), mime


def record_social(path, platform, link):
    """Add `social: {platform: link}` to the post's front matter without reformatting it."""
    text = path.read_text()
    end = text.index("\n---\n", 4)
    front = text[:end]
    if re.search(r"^social:\s*$", front, re.M):
        front = re.sub(r"^social:\s*$", f"social:\n   {platform}: {link}", front, count=1, flags=re.M)
    else:
        front += f"\nsocial:\n   {platform}: {link}"
    path.write_text(front + text[end:])


# ---------------------------------------------------------------- validation

def x_length(text):
    return len(re.sub(r"https?://\S+", "x" * X_URL_LENGTH, text))


def length(platform, text):
    return x_length(text) if platform == "twitter" else len(text)


# ---------------------------------------------------------------- credentials

def load_creds():
    if not CREDS_FILE.exists():
        sys.exit(f"Missing {CREDS_FILE}. See ~/.claude/skills/social-draft/SKILL.md for setup.")
    return json.loads(CREDS_FILE.read_text())


def save_creds(creds):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CREDS_FILE.write_text(json.dumps(creds, indent=2))
    CREDS_FILE.chmod(0o600)


def check(resp, what):
    if not resp.ok:
        sys.exit(f"{what} failed ({resp.status_code}): {resp.text}")
    return resp.json() if resp.content else {}


# ---------------------------------------------------------------- X

def post_twitter(creds, text, image, meta):
    c = creds["twitter"]
    auth = OAuth1(c["api_key"], c["api_secret"], c["access_token"], c["access_token_secret"])
    body = {"text": text}
    if image:
        data, mime = image
        r = requests.post("https://api.x.com/2/media/upload", auth=auth,
                          files={"media": ("cover", data, mime)},
                          data={"media_category": "tweet_image"})
        media = check(r, "X media upload")
        media_id = media.get("data", {}).get("id") or media.get("media_id_string")
        body["media"] = {"media_ids": [media_id]}
    r = requests.post("https://api.x.com/2/tweets", auth=auth, json=body)
    tweet_id = check(r, "X post")["data"]["id"]
    return f"https://x.com/{creds['twitter'].get('username', 'i')}/status/{tweet_id}"


# ---------------------------------------------------------------- Bluesky

def post_bluesky(creds, text, image, meta):
    c = creds["bluesky"]
    pds = c.get("pds", "https://bsky.social")
    session = check(requests.post(f"{pds}/xrpc/com.atproto.server.createSession",
                                  json={"identifier": c["handle"], "password": c["app_password"]}),
                    "Bluesky login")
    headers = {"Authorization": f"Bearer {session['accessJwt']}"}

    # Links must be marked up as facets with UTF-8 byte offsets to be clickable.
    facets = []
    for m in re.finditer(r"https?://\S+", text):
        start = len(text[:m.start()].encode())
        facets.append({"index": {"byteStart": start, "byteEnd": start + len(m.group().encode())},
                       "features": [{"$type": "app.bsky.richtext.facet#link", "uri": m.group()}]})

    external = {"uri": meta["url"], "title": meta["title"], "description": meta["description"]}
    if image:
        data, mime = image
        blob = check(requests.post(f"{pds}/xrpc/com.atproto.repo.uploadBlob", data=data,
                                   headers={**headers, "Content-Type": mime}),
                     "Bluesky image upload")
        external["thumb"] = blob["blob"]

    record = {"$type": "app.bsky.feed.post", "text": text, "facets": facets,
              "createdAt": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
              "langs": ["en"],
              "embed": {"$type": "app.bsky.embed.external", "external": external}}
    r = check(requests.post(f"{pds}/xrpc/com.atproto.repo.createRecord", headers=headers,
                            json={"repo": session["did"], "collection": "app.bsky.feed.post",
                                  "record": record}),
              "Bluesky post")
    rkey = r["uri"].rsplit("/", 1)[-1]
    return f"https://bsky.app/profile/{c['handle']}/post/{rkey}"


# ---------------------------------------------------------------- LinkedIn

def linkedin_text(text):
    """Escape LinkedIn 'little text' reserved characters and turn #Tags into hashtags."""
    placeholders = {}

    def keep_tag(m):
        key = f"\x00{len(placeholders)}\x00"
        placeholders[key] = "{hashtag|\\#|" + m.group(1) + "}"
        return key

    def keep_url(m):
        key = f"\x00{len(placeholders)}\x00"
        placeholders[key] = m.group()
        return key

    text = re.sub(r"https?://\S+", keep_url, text)
    text = re.sub(r"(?<!\S)#(\w+)", keep_tag, text)
    text = re.sub(r"([\\|{}@\[\]()<>#*_~])", r"\\\1", text)
    for key, val in placeholders.items():
        text = text.replace(key, val)
    return text


def linkedin_headers(c):
    return {"Authorization": f"Bearer {c['access_token']}",
            "LinkedIn-Version": c.get("version", "202608"),
            "X-Restli-Protocol-Version": "2.0.0"}


def post_linkedin(creds, text, image, meta):
    c = creds["linkedin"]
    if c.get("expires_at", 0) < time.time():
        sys.exit("LinkedIn token expired. Run: bin/social-post auth linkedin")
    headers = linkedin_headers(c)
    article = {"source": meta["url"], "title": meta["title"], "description": meta["description"]}
    if image:
        data, mime = image
        init = check(requests.post("https://api.linkedin.com/rest/images?action=initializeUpload",
                                   headers=headers,
                                   json={"initializeUploadRequest": {"owner": c["person_urn"]}}),
                     "LinkedIn image init")["value"]
        up = requests.put(init["uploadUrl"], data=data,
                          headers={"Authorization": headers["Authorization"], "Content-Type": mime})
        if not up.ok:
            sys.exit(f"LinkedIn image upload failed ({up.status_code}): {up.text}")
        article["thumbnail"] = init["image"]

    body = {"author": c["person_urn"], "commentary": linkedin_text(text), "visibility": "PUBLIC",
            "distribution": {"feedDistribution": "MAIN_FEED", "targetEntities": [],
                             "thirdPartyDistributionChannels": []},
            "content": {"article": article},
            "lifecycleState": "PUBLISHED", "isReshareDisabledByAuthor": False}
    r = requests.post("https://api.linkedin.com/rest/posts", headers=headers, json=body)
    if r.status_code != 201:
        sys.exit(f"LinkedIn post failed ({r.status_code}): {r.text}")
    return f"https://www.linkedin.com/feed/update/{r.headers['x-restli-id']}/"


def auth_linkedin():
    creds = load_creds()
    c = creds.setdefault("linkedin", {})
    if not c.get("client_id") or not c.get("client_secret"):
        sys.exit(f"Add linkedin.client_id and linkedin.client_secret to {CREDS_FILE} first.")
    state = secrets.token_urlsafe(16)
    params = {"response_type": "code", "client_id": c["client_id"], "redirect_uri": LINKEDIN_REDIRECT,
              "state": state, "scope": "openid profile w_member_social"}
    result = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            result.update({k: v[0] for k, v in q.items()})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"LinkedIn authorization received. You can close this tab.")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("localhost", 8765), Handler)
    webbrowser.open("https://www.linkedin.com/oauth/v2/authorization?" + urllib.parse.urlencode(params))
    print("Waiting for LinkedIn authorization in the browser...")
    server.handle_request()
    if result.get("state") != state or "code" not in result:
        sys.exit(f"Authorization failed: {result}")

    token = check(requests.post("https://www.linkedin.com/oauth/v2/accessToken", data={
        "grant_type": "authorization_code", "code": result["code"], "redirect_uri": LINKEDIN_REDIRECT,
        "client_id": c["client_id"], "client_secret": c["client_secret"]}), "LinkedIn token exchange")
    c["access_token"] = token["access_token"]
    c["expires_at"] = int(time.time()) + int(token["expires_in"])
    me = check(requests.get("https://api.linkedin.com/v2/userinfo",
                            headers={"Authorization": f"Bearer {c['access_token']}"}), "LinkedIn userinfo")
    c["person_urn"] = f"urn:li:person:{me['sub']}"
    save_creds(creds)
    expires = dt.datetime.fromtimestamp(c["expires_at"]).date()
    print(f"LinkedIn authorized as {me.get('name', me['sub'])}; token valid until {expires}.")


# ---------------------------------------------------------------- main

POSTERS = {"twitter": post_twitter, "bluesky": post_bluesky, "linkedin": post_linkedin}


def main():
    if sys.argv[1:3] == ["auth", "linkedin"]:
        return auth_linkedin()

    ap = argparse.ArgumentParser(description="Post a blog post's social drafts.")
    ap.add_argument("post", type=Path)
    ap.add_argument("--only", help="comma-separated subset of: " + ",".join(PLATFORMS))
    ap.add_argument("--dry-run", action="store_true", help="validate and preview without posting")
    ap.add_argument("--yes", action="store_true", help="skip the per-platform confirmation")
    args = ap.parse_args()

    path = args.post.resolve()
    text, front = load_post(path)
    url = post_url(path, site_config())
    drafts = parse_drafts(text, url)
    wanted = args.only.split(",") if args.only else PLATFORMS
    already = front.get("social") or {}
    meta = {"url": url, "title": front.get("title", ""),
            "description": front.get("description") or front.get("summary", "")}

    ok = True
    for p in wanted:
        if p not in drafts:
            print(f"[{p}] no draft found")
            continue
        n = length(p, drafts[p])
        flag = "OK" if n <= LIMITS[p] else "TOO LONG"
        print(f"\n[{p}] {n}/{LIMITS[p]} {flag}" + (f"  (already posted: {already[p]})" if p in already else ""))
        print("  " + drafts[p].replace("\n", "\n  "))
        ok &= n <= LIMITS[p]
    if not ok:
        sys.exit("\nFix the drafts that are too long before posting.")
    if args.dry_run:
        return

    live = requests.head(url, allow_redirects=True, timeout=15)
    if live.status_code != 200:
        sys.exit(f"\n{url} returned {live.status_code}. Deploy the post before sharing it.")

    creds = load_creds()
    image = cover_image(front)
    for p in wanted:
        if p not in drafts or p in already:
            continue
        if p not in creds:
            print(f"[{p}] skipped: no credentials in {CREDS_FILE}")
            continue
        if not args.yes and input(f"\nPost to {p}? [y/N] ").strip().lower() != "y":
            continue
        link = POSTERS[p](creds, drafts[p], image, meta)
        record_social(path, p, link)
        print(f"[{p}] posted: {link}")


if __name__ == "__main__":
    main()

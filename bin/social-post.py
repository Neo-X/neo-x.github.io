#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "requests-oauthlib", "pyyaml", "pillow"]
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

A single post per platform shares the blog post as a link card. A platform with `--- twitter reply ---` /
`--- bluesky reply ---` sections is posted as a thread instead: the cover image is attached to the first post
(alt text from the cover's `alt` attribute) and each reply is threaded under it.

Usage:
    bin/social-post _i18n/en/_posts/2026-09-28-knowledge-vs-information.md [--dry-run] [--only twitter,bluesky]
    bin/social-post standalone path/to/drafts.md [--dry-run] [--only twitter,bluesky]
    bin/social-post auth linkedin
    bin/social-post auth linkedin-token   # finish setup from a token made with LinkedIn's token generator
    bin/social-post check                 # test each platform's login without posting

`standalone` posts drafts that are not tied to a blog post (e.g. an event announcement), with an
image attached to the post itself and optional replies threaded under it on X and Bluesky:

    ---
    image: diagram.png        # relative to the drafts file
    alt: Alt text for the image.
    ---
    Notes before the first marker are ignored.
    --- twitter ---
    Main post.
    --- twitter reply ---
    First reply, posted under the main post.
    --- bluesky ---
    ...
    --- linkedin ---
    ...

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


def split_threads(text, path):
    """ `--- platform ---` / `--- platform reply ---` sections; returns {platform: [post, reply, ...]}. """
    threads = {}
    parts = re.split(r"^---\s*(\w+)(\s+reply)?\s*---\s*$", text, flags=re.M)
    for name, reply, body in zip(parts[1::3], parts[2::3], parts[3::3]):
        name = name.lower()
        if name not in PLATFORMS:
            sys.exit(f"{path}: unknown section '{name}'")
        if reply:
            if name not in threads:
                sys.exit(f"{path}: '{name} reply' comes before the {name} post")
            if name == "linkedin":
                sys.exit(f"{path}: LinkedIn posts can't have replies")
            threads[name].append(body.strip())
        else:
            threads[name] = [body.strip()]
    return threads


def parse_drafts(text, url, path):
    blocks = re.findall(r"\{%\s*comment\s*%\}(.*?)\{%\s*endcomment\s*%\}", text, re.S)
    drafts = {}
    for block in blocks:
        if re.search(r"^---\s*(twitter|bluesky|linkedin)(\s+reply)?\s*---\s*$", block, re.M):
            block = block[re.search(r"^---\s*\w+", block, re.M).start():]
            for name, posts in split_threads(block, path).items():
                drafts[name] = [t.replace("{{URL}}", url) for t in posts]
    return drafts


def cover_path(front):
    cover = front.get("cover") or ""
    m = re.search(r'src="([^"]+)"', cover) or re.match(r"\s*(\S+\.(?:png|jpe?g|svg|gif|webp))\s*$", cover)
    if not m:
        return None
    return REPO / m.group(1).lstrip("/")


def cover_alt(front):
    m = re.search(r'alt="([^"]*)"', front.get("cover") or "")
    return m.group(1) if m else None


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


CRED_KEYS = {"twitter": ["api_key", "api_secret", "access_token", "access_token_secret"],
             "bluesky": ["handle", "app_password"],
             "linkedin": ["access_token", "person_urn"]}


def has_creds(creds, platform):
    c = creds.get(platform) or {}
    return all(c.get(k) for k in CRED_KEYS[platform])


def save_creds(creds):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CREDS_FILE.write_text(json.dumps(creds, indent=2))
    CREDS_FILE.chmod(0o600)


def check(resp, what):
    if not resp.ok:
        sys.exit(f"{what} failed ({resp.status_code}): {resp.text}")
    return resp.json() if resp.content else {}


# ---------------------------------------------------------------- X

def x_auth(c):
    return OAuth1(c["api_key"], c["api_secret"], c["access_token"], c["access_token_secret"])


def x_upload(auth, image, alt=None):
    data, mime = image
    r = requests.post("https://api.x.com/2/media/upload", auth=auth,
                      files={"media": ("cover", data, mime)},
                      data={"media_category": "tweet_image"})
    media = check(r, "X media upload")
    media_id = media.get("data", {}).get("id") or media.get("media_id_string")
    if alt:
        r = requests.post("https://api.x.com/2/media/metadata", auth=auth,
                          json={"id": media_id, "metadata": {"alt_text": {"text": alt[:1000]}}})
        if not r.ok:
            print(f"  ! X alt text not set ({r.status_code}): {r.text}")
    return media_id


def x_tweet(auth, text, media_id=None, reply_to=None):
    body = {"text": text}
    if media_id:
        body["media"] = {"media_ids": [media_id]}
    if reply_to:
        body["reply"] = {"in_reply_to_tweet_id": reply_to}
    r = requests.post("https://api.x.com/2/tweets", auth=auth, json=body)
    return check(r, "X post")["data"]["id"]


def x_link(creds, tweet_id):
    return f"https://x.com/{creds['twitter'].get('username', 'i')}/status/{tweet_id}"


def post_twitter(creds, text, image, meta):
    auth = x_auth(creds["twitter"])
    media_id = x_upload(auth, image) if image else None
    return x_link(creds, x_tweet(auth, text, media_id))


# ---------------------------------------------------------------- Bluesky

def bsky_session(c):
    pds = c.get("pds", "https://bsky.social")
    session = check(requests.post(f"{pds}/xrpc/com.atproto.server.createSession",
                                  json={"identifier": c["handle"].lstrip("@"), "password": c["app_password"]}),
                    "Bluesky login")
    return {"pds": pds, "did": session["did"], "headers": {"Authorization": f"Bearer {session['accessJwt']}"}}


def bsky_facets(text):
    """Links and #tags must be marked up as facets with UTF-8 byte offsets to be clickable."""
    facets = []
    for m in re.finditer(r"https?://\S+", text):
        start = len(text[:m.start()].encode())
        facets.append({"index": {"byteStart": start, "byteEnd": start + len(m.group().encode())},
                       "features": [{"$type": "app.bsky.richtext.facet#link", "uri": m.group()}]})
    for m in re.finditer(r"(?<!\S)#(\w+)", text):
        start = len(text[:m.start()].encode())
        facets.append({"index": {"byteStart": start, "byteEnd": start + len(m.group().encode())},
                       "features": [{"$type": "app.bsky.richtext.facet#tag", "tag": m.group(1)}]})
    return facets


def bsky_upload(s, image):
    data, mime = image
    return check(requests.post(f"{s['pds']}/xrpc/com.atproto.repo.uploadBlob", data=data,
                               headers={**s["headers"], "Content-Type": mime}),
                 "Bluesky image upload")["blob"]


def bsky_create(s, text, embed=None, reply=None):
    """ Post a record; returns its {"uri", "cid"} ref (what replies point to). """
    record = {"$type": "app.bsky.feed.post", "text": text, "facets": bsky_facets(text),
              "createdAt": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
              "langs": ["en"]}
    if embed:
        record["embed"] = embed
    if reply:
        record["reply"] = reply
    r = check(requests.post(f"{s['pds']}/xrpc/com.atproto.repo.createRecord", headers=s["headers"],
                            json={"repo": s["did"], "collection": "app.bsky.feed.post", "record": record}),
              "Bluesky post")
    return {"uri": r["uri"], "cid": r["cid"]}


def bsky_link(creds, ref):
    return f"https://bsky.app/profile/{creds['bluesky']['handle'].lstrip('@')}/post/{ref['uri'].rsplit('/', 1)[-1]}"


def post_bluesky(creds, text, image, meta):
    s = bsky_session(creds["bluesky"])
    external = {"uri": meta["url"], "title": meta["title"], "description": meta["description"]}
    if image:
        external["thumb"] = bsky_upload(s, image)
    ref = bsky_create(s, text, embed={"$type": "app.bsky.embed.external", "external": external})
    return bsky_link(creds, ref)


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


def linkedin_creds(creds):
    c = creds["linkedin"]
    if c.get("expires_at", 0) < time.time():
        sys.exit("LinkedIn token expired. Run: bin/social-post auth linkedin")
    return c, linkedin_headers(c)


def linkedin_upload(c, headers, image):
    """ Upload an image; returns its urn:li:image. """
    data, mime = image
    init = check(requests.post("https://api.linkedin.com/rest/images?action=initializeUpload",
                               headers=headers,
                               json={"initializeUploadRequest": {"owner": c["person_urn"]}}),
                 "LinkedIn image init")["value"]
    up = requests.put(init["uploadUrl"], data=data,
                      headers={"Authorization": headers["Authorization"], "Content-Type": mime})
    if not up.ok:
        sys.exit(f"LinkedIn image upload failed ({up.status_code}): {up.text}")
    return init["image"]


def linkedin_post(c, headers, text, content):
    body = {"author": c["person_urn"], "commentary": linkedin_text(text), "visibility": "PUBLIC",
            "distribution": {"feedDistribution": "MAIN_FEED", "targetEntities": [],
                             "thirdPartyDistributionChannels": []},
            "content": content,
            "lifecycleState": "PUBLISHED", "isReshareDisabledByAuthor": False}
    r = requests.post("https://api.linkedin.com/rest/posts", headers=headers, json=body)
    if r.status_code != 201:
        sys.exit(f"LinkedIn post failed ({r.status_code}): {r.text}")
    return f"https://www.linkedin.com/feed/update/{r.headers['x-restli-id']}/"


def post_linkedin(creds, text, image, meta):
    c, headers = linkedin_creds(creds)
    article = {"source": meta["url"], "title": meta["title"], "description": meta["description"]}
    if image:
        article["thumbnail"] = linkedin_upload(c, headers, image)
    return linkedin_post(c, headers, text, {"article": article})




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


# ---------------------------------------------------------------- standalone posts

BSKY_MAX_IMAGE = 950_000  # Bluesky rejects image blobs over ~1 MB


def parse_standalone(path):
    """ Front matter plus `--- platform ---` / `--- platform reply ---` sections; returns (front, {platform: [posts]}). """
    text = path.read_text()
    front = {}
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if m:
        front = yaml.safe_load(m.group(1)) or {}
        text = text[m.end():]
    return front, split_threads(text, path)


def load_image(path, max_bytes=None):
    """ (bytes, mime, (width, height)) for an image file; SVGs are rasterized, and with max_bytes the image is
    re-encoded as JPEG, shrinking it until it fits. """
    import io
    from PIL import Image
    if path.suffix.lower() == ".svg":
        out = Path(tempfile.mkdtemp()) / (path.stem + ".png")
        subprocess.run(["inkscape", str(path), "-o", str(out), "-w", "2000", "-b", "#ffffff"],
                       check=True, capture_output=True)
        path = out
    data = path.read_bytes()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp"}[path.suffix.lower()]
    img = Image.open(io.BytesIO(data))
    size = img.size
    if max_bytes and len(data) > max_bytes:
        img = img.convert("RGB")
        while True:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=88)
            if buf.tell() <= max_bytes:
                break
            img = img.resize((int(img.width * 0.85), int(img.height * 0.85)), Image.LANCZOS)
        data, mime, size = buf.getvalue(), "image/jpeg", img.size
    return data, mime, size


def thread_twitter(creds, posts, image, alt):
    auth = x_auth(creds["twitter"])
    media_id = x_upload(auth, image[:2], alt) if image else None
    first = prev = x_tweet(auth, posts[0], media_id)
    for text in posts[1:]:
        prev = x_tweet(auth, text, reply_to=prev)
    return x_link(creds, first)


def thread_bluesky(creds, posts, image, alt):
    s = bsky_session(creds["bluesky"])
    embed = None
    if image:
        w, h = image[2]
        embed = {"$type": "app.bsky.embed.images",
                 "images": [{"alt": alt or "", "image": bsky_upload(s, image[:2]),
                             "aspectRatio": {"width": w, "height": h}}]}
    root = prev = bsky_create(s, posts[0], embed=embed)
    for text in posts[1:]:
        prev = bsky_create(s, text, reply={"root": root, "parent": prev})
    return bsky_link(creds, root)


def thread_linkedin(creds, posts, image, alt):
    c, headers = linkedin_creds(creds)
    content = None
    if image:
        content = {"media": {"id": linkedin_upload(c, headers, image[:2]), "altText": alt or ""}}
    return linkedin_post(c, headers, posts[0], content)


THREADERS = {"twitter": thread_twitter, "bluesky": thread_bluesky, "linkedin": thread_linkedin}


def main_standalone(argv):
    ap = argparse.ArgumentParser(prog="social-post standalone",
                                 description="Post standalone drafts (image + optional reply thread).")
    ap.add_argument("drafts", type=Path)
    ap.add_argument("--only", help="comma-separated subset of: " + ",".join(PLATFORMS))
    ap.add_argument("--dry-run", action="store_true", help="validate and preview without posting")
    ap.add_argument("--yes", action="store_true", help="skip the per-platform confirmation")
    args = ap.parse_args(argv)

    path = args.drafts.resolve()
    front, threads = parse_standalone(path)
    wanted = args.only.split(",") if args.only else PLATFORMS
    already = front.get("social") or {}
    image_path = (path.parent / front["image"]) if front.get("image") else None
    if image_path and not image_path.exists():
        sys.exit(f"{image_path} not found")
    alt = front.get("alt")
    if image_path and not alt:
        print("  ! no `alt:` text in the front matter; the image will have no alt text")

    ok = True
    for p in wanted:
        if p not in threads:
            print(f"[{p}] no draft found")
            continue
        print(f"\n[{p}]" + (f"  (already posted: {already[p]})" if p in already else "")
              + (f"  image: {image_path.name}" if image_path else ""))
        for i, text in enumerate(threads[p]):
            n = length(p, text)
            flag = "OK" if n <= LIMITS[p] else "TOO LONG"
            print(f"  {'post' if i == 0 else f'reply {i}'}: {n}/{LIMITS[p]} {flag}")
            print("    " + text.replace("\n", "\n    "))
            ok &= n <= LIMITS[p]
    if not ok:
        sys.exit("\nFix the drafts that are too long before posting.")
    if args.dry_run:
        if image_path:
            data, mime, size = load_image(image_path, BSKY_MAX_IMAGE)
            print(f"\nimage for Bluesky: {size[0]}x{size[1]} {mime}, {len(data) / 1e3:.0f} KB")
        return

    creds = load_creds()
    for p in wanted:
        if p not in threads or p in already:
            continue
        if not has_creds(creds, p):
            print(f"[{p}] skipped: credentials missing or blank in {CREDS_FILE}")
            continue
        if not args.yes and input(f"\nPost to {p}? [y/N] ").strip().lower() != "y":
            continue
        image = load_image(image_path, BSKY_MAX_IMAGE if p == "bluesky" else None) if image_path else None
        link = THREADERS[p](creds, threads[p], image, alt)
        record_social(path, p, link)
        print(f"[{p}] posted: {link}")


def auth_linkedin_token():
    """ Set up LinkedIn from an access token pasted into linkedin.access_token (e.g. from the developer portal's
    token generator, with the openid, profile and w_member_social scopes): looks up the member URN and expiry. """
    creds = load_creds()
    c = creds.setdefault("linkedin", {})
    if not c.get("access_token"):
        sys.exit(f"Paste the token into linkedin.access_token in {CREDS_FILE} first.")
    me = check(requests.get("https://api.linkedin.com/v2/userinfo",
                            headers={"Authorization": f"Bearer {c['access_token']}"}),
               "LinkedIn userinfo (the token needs the openid and profile scopes)")
    c["person_urn"] = f"urn:li:person:{me['sub']}"
    c["expires_at"] = int(time.time()) + 60 * 24 * 3600  # generator tokens last 60 days
    if c.get("client_id") and c.get("client_secret"):
        info = requests.post("https://www.linkedin.com/oauth/v2/introspectToken",
                             data={"client_id": c["client_id"], "client_secret": c["client_secret"],
                                   "token": c["access_token"]})
        if info.ok and info.json().get("expires_at"):
            c["expires_at"] = int(info.json()["expires_at"])
    save_creds(creds)
    expires = dt.datetime.fromtimestamp(c["expires_at"]).date()
    print(f"LinkedIn set up for {me.get('name', me['sub'])}; token valid until about {expires}.")


def check_connections():
    """ Log in to each platform without posting, and say what to fix for any that fail. """
    creds = load_creds()
    for p in PLATFORMS:
        if not has_creds(creds, p):
            print(f"[{p}] not set up: blank or missing {', '.join(CRED_KEYS[p])}")
            continue
        c = creds[p]
        try:
            if p == "bluesky":
                pds = c.get("pds", "https://bsky.social")
                r = requests.post(f"{pds}/xrpc/com.atproto.server.createSession", timeout=20,
                                  json={"identifier": c["handle"].lstrip("@"), "password": c["app_password"]})
                ok, who = r.ok, (r.json().get("handle") if r.ok else None)
            elif p == "linkedin":
                r = requests.get("https://api.linkedin.com/v2/userinfo", timeout=20,
                                 headers={"Authorization": f"Bearer {c['access_token']}"})
                ok, who = r.ok, (r.json().get("name") if r.ok else None)
                if ok:
                    who += f", token valid until about {dt.datetime.fromtimestamp(c.get('expires_at', 0)).date()}"
            else:
                r = requests.get("https://api.x.com/2/users/me", auth=x_auth(c), timeout=20)
                ok, who = r.ok, (r.json()["data"]["username"] if r.ok else None)
        except requests.RequestException as e:
            print(f"[{p}] network error: {e}")
            continue
        print(f"[{p}] OK ({who})" if ok else f"[{p}] FAILED ({r.status_code}): {r.text[:200]}")


# ---------------------------------------------------------------- main

POSTERS = {"twitter": post_twitter, "bluesky": post_bluesky, "linkedin": post_linkedin}


def main():
    if sys.argv[1:3] == ["auth", "linkedin"]:
        return auth_linkedin()
    if sys.argv[1:3] == ["auth", "linkedin-token"]:
        return auth_linkedin_token()
    if sys.argv[1:2] == ["check"]:
        return check_connections()
    if sys.argv[1:2] == ["standalone"]:
        return main_standalone(sys.argv[2:])

    ap = argparse.ArgumentParser(description="Post a blog post's social drafts.")
    ap.add_argument("post", type=Path)
    ap.add_argument("--only", help="comma-separated subset of: " + ",".join(PLATFORMS))
    ap.add_argument("--dry-run", action="store_true", help="validate and preview without posting")
    ap.add_argument("--yes", action="store_true", help="skip the per-platform confirmation")
    args = ap.parse_args()

    path = args.post.resolve()
    text, front = load_post(path)
    url = post_url(path, site_config())
    drafts = parse_drafts(text, url, path)
    wanted = args.only.split(",") if args.only else PLATFORMS
    already = front.get("social") or {}
    meta = {"url": url, "title": front.get("title", ""),
            "description": front.get("description") or front.get("summary", "")}

    ok = True
    for p in wanted:
        if p not in drafts:
            print(f"[{p}] no draft found")
            continue
        print(f"\n[{p}]" + ("  thread, cover attached" if len(drafts[p]) > 1 else "  link card")
              + (f"  (already posted: {already[p]})" if p in already else ""))
        for i, post in enumerate(drafts[p]):
            n = length(p, post)
            flag = "OK" if n <= LIMITS[p] else "TOO LONG"
            print(f"  {'post' if i == 0 else f'reply {i}'}: {n}/{LIMITS[p]} {flag}")
            print("    " + post.replace("\n", "\n    "))
            ok &= n <= LIMITS[p]
    if not ok:
        sys.exit("\nFix the drafts that are too long before posting.")
    if args.dry_run:
        return

    # the web host rejects the default python-requests user agent with a 400
    live = requests.head(url, allow_redirects=True, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0 (compatible; social-post)"})
    if live.status_code != 200:
        sys.exit(f"\n{url} returned {live.status_code}. Deploy the post before sharing it.")

    creds = load_creds()
    image = cover_image(front)
    for p in wanted:
        if p not in drafts or p in already:
            continue
        if not has_creds(creds, p):
            print(f"[{p}] skipped: credentials missing or blank in {CREDS_FILE}")
            continue
        if not args.yes and input(f"\nPost to {p}? [y/N] ").strip().lower() != "y":
            continue
        if len(drafts[p]) > 1:
            cover = cover_path(front)
            if cover is None or not cover.exists():
                sys.exit(f"[{p}] a thread needs the post's cover image to attach")
            thread_image = load_image(cover, BSKY_MAX_IMAGE if p == "bluesky" else None)
            link = THREADERS[p](creds, drafts[p], thread_image, cover_alt(front))
        else:
            link = POSTERS[p](creds, drafts[p][0], image, meta)
        record_social(path, p, link)
        print(f"[{p}] posted: {link}")


if __name__ == "__main__":
    main()

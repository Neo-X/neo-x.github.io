# social-post

`social-post.py` posts drafts to Bluesky, LinkedIn and X from the command line. It has two modes:

- **Blog posts** read the drafts from a Liquid comment block at the bottom of a post and share the post as a link card with its cover image.
- **Standalone posts** read the drafts from their own file, such as an event announcement. The image goes on the post itself, and on X and Bluesky the replies are threaded under it.

Every command checks each draft's length against the platform's limit. Without `--yes`, it asks before posting to each platform. After posting, it saves the link under `social:` in the file's front matter. Running it again skips the platforms already recorded there, so nothing gets posted twice.

## Platform status

| Platform | Status | Notes |
|---|---|---|
| Bluesky | Works | App password; nothing expires |
| LinkedIn | Works | Access token, valid for 60 days (current one until about 2026-12-18) |
| X | Not used | Since Feb 2026, new X API apps can only post on pay-per-use credits. Post to X by hand instead. |

## Blog posts

```
bin/social-post.py _i18n/en/_posts/2026-09-28-knowledge-vs-information.md --dry-run
bin/social-post.py _i18n/en/_posts/2026-09-28-knowledge-vs-information.md --only bluesky,linkedin
```

The drafts go in a comment block at the bottom of the post. `{{URL}}` is replaced with the post's URL.

```
{% comment %}
--- twitter ---
Text for X. {{URL}}
--- bluesky ---
Text for Bluesky. {{URL}}
--- linkedin ---
Longer text for LinkedIn. {{URL}}
{% endcomment %}
```

A single post per platform is shared as a link card. To post a thread instead, add `--- twitter reply ---` or `--- bluesky reply ---` sections after that platform's post. The post's `cover` image is then attached to the first post, with the cover's `alt` attribute as alt text, and each reply is threaded under it. Use this on X to keep the link out of the main post. LinkedIn posts can't have replies.

Before posting, the script checks that the post is live on the website. It uses the post's `cover` as the image and rasterizes SVG covers to PNG with Inkscape. The `social-draft` Claude skill writes these blocks.

## Standalone posts

```
bin/social-post.py standalone path/to/drafts.md --dry-run
bin/social-post.py standalone path/to/drafts.md --only bluesky,linkedin
```

A drafts file looks like this:

```
---
image: diagram.png        # relative to the drafts file; PNG, JPEG, GIF, WebP or SVG
alt: Alt text for the image.
---
Notes before the first marker are ignored.
--- twitter ---
Main post.
--- twitter reply ---
First reply, threaded under the main post.
--- twitter reply ---
Second reply, under the first.
--- bluesky ---
Main post. #Hashtags and links become clickable.
--- bluesky reply ---
...
--- linkedin ---
One post. LinkedIn posts can't have replies.
```

For Bluesky, an image over about 1 MB is re-encoded as JPEG and shrunk until it fits.

**On X, put links in a reply, not the main post.** X shows posts with external links to fewer people, so keep the main post link-free. Put the URL in the last `twitter reply` (e.g. `Register: https://...`). Links in X API posts also cost more than 10× as much as plain posts. Bluesky and LinkedIn don't penalize links, so their main posts can keep them.

Example: `~/playground/mini-grp/mini-wm/docs/policy_vs_world_models_posts.md`, the MAIN Educational 2026 tutorial announcement.

## Options

- `--dry-run` shows each post with its length and doesn't post anything.
- `--only bluesky,linkedin` posts to only the listed platforms.
- `--yes` skips the confirmation before each platform.

## Setting up and fixing connections

### Check the connections first

```
bin/social-post.py check
```

This logs in to each platform without posting. It prints `OK` with the account name, `not set up`, or `FAILED` with the platform's error. LinkedIn also shows when its token expires. Run it whenever a post fails, then use the section for the failing platform below.

### Where the credentials live

The credentials live in `~/.config/social-post/credentials.json`, outside the repo and readable only by you (mode 600). Edit them with any text editor. Any platform with blank fields is skipped.

```json
{
  "bluesky":  {"handle": "glenberseth.bsky.social", "app_password": "..."},
  "linkedin": {"access_token": "...", "client_id": "...", "client_secret": "...", "version": "202608",
               "person_urn": "(filled in by the auth commands)", "expires_at": 0},
  "twitter":  {"api_key": "...", "api_secret": "...", "access_token": "...", "access_token_secret": "...", "username": "GlenBerseth"}
}
```

If the file is lost, recreate it with `mkdir -p ~/.config/social-post`, then `chmod 600 ~/.config/social-post/credentials.json` after writing it. Then redo the setup below for each platform. Paste secrets into the file, never into a chat or a commit.

### Bluesky

**Setup.**
1. In Bluesky (app or bsky.app), go to Settings → Privacy and security → App passwords → Add App Password.
2. Name it `social-post` and copy the password (format `xxxx-xxxx-xxxx-xxxx`). Bluesky shows it only once.
3. Put it in `bluesky.app_password`, and your full handle in `bluesky.handle`. A leading `@` is fine.

The app password never expires on its own.

**Fixing it.** If `check` fails with `401` or `AuthenticationRequired`, the password was revoked or mistyped. Delete the old one in the same settings page, make a new one, paste it in, and run `check` again. Your normal Bluesky login password does not work here; it has to be an app password.

### LinkedIn

**The app** (already set up). It lives at [linkedin.com/developers](https://www.linkedin.com/developers) → My apps:
- **Products:** *Sign In with LinkedIn using OpenID Connect* and *Share on LinkedIn*.
- **Auth tab:** the redirect URL `http://localhost:8765/callback`. The Client ID and Client Secret shown there go in `linkedin.client_id` and `linkedin.client_secret`.
- **App logo:** `assets/images/logo/logo-512.png`.

**Renewing the token (every 60 days).** The current token is valid until about 2026-12-18; `check` shows the date. When it expires, posting stops with "LinkedIn token expired". To renew, log in again through the browser:

```
bin/social-post.py auth linkedin
```

This opens LinkedIn in your browser. Approve it, and the tab says "authorization received". The new token, member ID and expiry are saved automatically. Run it from a terminal on this machine, because the browser has to reach `localhost:8765`. Then run `bin/social-post.py check`.

**Alternative: the token generator.** Use this if the browser login fails.
1. In the developer portal, open Docs and tools → OAuth Token Tools → Create token.
2. Pick the app and tick the scopes `openid`, `profile` and `w_member_social`.
3. Paste the token into `linkedin.access_token`, then run:
   ```
   bin/social-post.py auth linkedin-token
   ```

This looks up your member ID, and also the exact expiry if `client_id` and `client_secret` are set.

**Fixing it.**

| Symptom | Fix |
|---|---|
| "LinkedIn token expired" | Run `bin/social-post.py auth linkedin` |
| `check` fails with `401` | The token was revoked or is wrong; renew it as above |
| `403` when posting | The token lacks `w_member_social`; renew it with that scope |
| "Authorization failed" or the browser shows a `redirect_uri` error | Add `http://localhost:8765/callback` under the app's Auth tab, exactly as written |
| "Address already in use" during `auth linkedin` | Something is using port 8765; close it, or use the token generator |
| "version ... not supported" (`426` or `400` naming `LinkedIn-Version`) | Set `linkedin.version` to a recent month, e.g. `202610` (format YYYYMM) |
| Client secret leaked or reset | Copy the new secret from the Auth tab into `client_secret`, then renew the token |

### X

X is posted by hand, because since Feb 2026 the X API only posts with pay-per-use credits. The code still supports X, if that ever changes:

- **Setup.**
  1. At developer.x.com, put the app inside a Project, with pay-per-use credits.
  2. Under the app's user authentication settings, choose *Read and write* and *Web App, Automated App or Bot*, with callback `http://localhost:8765/callback` and website `https://www.fracturedplane.com`.
  3. Under Keys and tokens, copy the API Key and Secret into `api_key` and `api_secret`.
  4. Generate the Access Token and Secret *after* switching to Read and write, and copy them into `access_token` and `access_token_secret`.
- **Unused keys.** The Bearer token is read-only and isn't used. The OAuth 2.0 Client ID and Secret aren't used either.
- **Errors.**
  - `403 client-not-enrolled`: the app isn't in a Project, or the Project has no API access or credits.
  - `403` saying you're not permitted to post: the access token was made while the app was read-only, so regenerate it.
  - `401`: the keys are wrong, so regenerate all four.

When drafting for X, keep links out of the main post (see *Standalone posts*).

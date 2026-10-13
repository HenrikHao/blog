# Hugo PaperMod blog for LazyCat

This repository contains a Hugo site using the PaperMod theme and packages the
generated static files as a native LazyCat LPK. The public site is available at
<https://blog.henrikhao.com/> through Caddy. Its native LazyCat URL remains
protected by LazyCat login.

PaperMod is tracked as a Git submodule at `themes/PaperMod`. Hugo runs from a
pinned, multi-architecture container, so you do not need to install Hugo on the
Mac.

## One-time setup after cloning

Install and start Docker Desktop, install `lzc-cli`, and clone the repository
with its theme:

```sh
git clone --recurse-submodules YOUR_GIT_URL Blog
cd Blog
lzc-cli box default
```

If you cloned without `--recurse-submodules`, fetch PaperMod afterward:

```sh
git submodule update --init --recursive
```

This local repository uses branch `main` and has the Git author configured as
`HenrikHao <zy.henrikhao@gmail.com>`. It does not have an `origin` yet. After
creating an empty repository on your Git host, connect and push it with:

```sh
git remote add origin YOUR_GIT_URL
git push -u origin main
```

## Customize the blog

Edit `hugo.yaml` to change the title, description, author, menus, or production
URL. Edit `content/about.md` for the About page. Static images belong in
`static/images/` and are referenced in Markdown as `/images/filename.jpg`.

The site title is **Rik'Log**. The deployed LazyCat app and subdomain are
**Blog** and `blog`. Changing the public hostname requires updating `baseURL`
in `hugo.yaml` and the matching Caddy site block.

## Write a post

Create a draft whose filename is a lowercase URL slug:

```sh
./scripts/new-post.sh my-first-post
```

Then edit `content/posts/my-first-post.md`. Posts use Markdown with YAML front
matter:

```md
---
title: "My first post"
date: 2026-08-27T20:00:00+10:00
draft: true
description: "A short summary shown in lists and search."
tags:
  - notes
---

Write the post here.
```

Drafts appear in local preview but not in production. Change `draft: true` to
`draft: false` when the post is ready to publish.

### Math

Add `math: true` to the front matter. Write inline math as `\( ... \)` and
display math as `$$ ... $$`. Hugo renders it to KaTeX markup at build time; the
page only loads the KaTeX stylesheet, from jsDelivr.

Keep math out of headings, and give math-heavy posts an explicit `summary`,
because the plain-text version of rendered math repeats each symbol.

### Interactive figures

Add `viz: true` to the front matter and place a figure with the `viz`
shortcode. The caption is optional and may contain Markdown and math:

```md
{{< viz name="attention" >}}
One query attending to five tokens.
{{< /viz >}}
```

The sources for a figure live in `assets/viz/<post-file-name>/`:

- `<name>.html` is the markup, with any figure-specific `<style>`. It is
  inlined into the page.
- `<name>.js` is optional. It is minified, fingerprinted and loaded after the
  shared helpers, and starts with `Viz.mount("<name>", (root) => { ... })`.
- Any other `<name>.*` file (an image, a JSON file) is published as it is, and
  the script gets its URL from `Viz.assets(root)`, keyed by extension.

The same figure can be placed more than once in different modes with
`variant="..."`. The script reads it from `root.dataset.variant`, and its
styles should use the `.viz-<name>` class, because the element id then includes
the variant.

Shared pieces are `assets/js/viz-kit.js` (helpers) and
`assets/css/extended/viz.css` (colour roles for light and dark, controls,
legends). Figures should use the `--viz-*` custom properties, not raw colours,
so they follow the theme toggle.

### Series

Posts that belong together share a `series` entry in the front matter, which
gives them a listing page such as `/series/robot-learning/`:

```yaml
series:
  - Robot Learning
```

## Preview locally

```sh
./scripts/serve.sh
```

Open <http://localhost:1313/>. The preview refreshes when a source file changes;
stop it with `Ctrl-C`.

## Build

```sh
./scripts/build.sh
```

The production site is written to the ignored `public/` directory. The build
uses Hugo 0.165.0 from the official container image and pins its image digest.
To test another production URL without editing the config:

```sh
BLOG_BASE_URL=https://example.com/ ./scripts/build.sh
```

## Deploy to LazyCat

Make sure `lzc-cli box default` reports the intended NAS, then run:

```sh
./scripts/deploy.sh
```

The script lints the package, rebuilds the site, deploys it to the selected box,
and prints the live URL and runtime status. The equivalent manual commands are:

```sh
lzc-cli project lint
lzc-cli project deploy --release
lzc-cli project info --release
```

Every content deployment uses the version in `package.yml`. Increment that
version before deploying when LazyCat needs to distinguish an update, for
example `0.1.0` to `0.1.1`.

## Normal publishing workflow

```sh
./scripts/new-post.sh a-new-post
./scripts/serve.sh
# Edit the post and set draft: false, then stop the preview.
git add content/
git commit -m "Add a new post"
git push
./scripts/deploy.sh
```

The Git push backs up the source; the deploy command publishes the generated
site to LazyCat. Pushing alone does not deploy.

## Update PaperMod

PaperMod is pinned to a known commit, so theme changes do not arrive
unexpectedly. To update it deliberately:

```sh
git submodule update --remote --merge themes/PaperMod
./scripts/build.sh
git add themes/PaperMod
git commit -m "Update PaperMod theme"
```

Build and preview before deploying a theme update.

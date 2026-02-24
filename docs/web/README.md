# docs/web Folder

Local static documentation webpage for the framework.

## Files

- `index.html`
  - Main one-page guide with:
    - framework map
    - step-by-step new-channel workflow
    - custom input adapter template
    - links to key examples.

- `styles.css`
  - Page visual style (responsive layout, color system, typography, animation classes).

- `web.js`
  - Small scroll-reveal behavior for section cards/panels.

## Run Locally

From repository root:

```bash
python3 -m http.server 8080 --directory docs/web
```

Then open:

- `http://localhost:8080`

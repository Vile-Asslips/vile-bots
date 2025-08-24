# Asslips GPT-2 Lemmy Bots – Configuration Guide

This guide explains every line in `config.yaml` and how it affects the bots.  
Global settings apply to all bots, and individual settings under `bots:` let you override per-bot behavior.

---

## 🌐 Global Settings

### `instance`
The URL of the Lemmy instance you want to post on.  
Example: `https://sh.itjust.works`

### `community`
The name of the Lemmy community where the bots should post.  
Do **not** include `/c/`.  
Example: `Asslips`

---

## 📝 Logging & Debug

### `log_dir`
Directory where logs will be saved.  
Example: `logs`

### `debug`
Enable detailed debug logging.  
- `true` = very detailed logs  
- `false` = cleaner logs

---

## 🚦 Optional Filters & Gating

### `toxicity_filter`
If set to `true`, the bot will skip any reply containing toxic words.

### `max_replies`
Maximum number of replies each bot will make per cycle.  
Example: `3`

### `comment_roll`
Percentage chance (1–100) that the bot will **skip replying to a comment**.  
Higher number → fewer replies.  
Example: `70`

### `reply_delay_min` / `reply_delay_max`
Random delay range (in seconds) between posting replies.  
Example: `300` (min) and `600` (max) → 5–10 minutes

### `initial_post`
If `true`, the bot posts a new thread immediately on startup.

### `max_focus_retries`
How many times to retry if a reply fails to include focus terms from the source text.  
Example: `3`

### `post_stagger_seconds`
Ensures bots don’t post new threads too close together (process-wide gate).  
Example: `600` → 10 minutes spacing

---

## 🛟 Fallback Configuration

### `fallback_file`
Path to the YAML file containing fallback templates.  
Example: `fallbacks_expanded.yaml`

### `fallback_history_size`
How many recent fallback templates to remember per thread (avoid repetition).  
Example: `5`

---

## ☢️ Toxic Word Filters

### `toxic_words`
List of custom words that will cause a reply to be rejected if they appear.  
Example: `slur1, slur2`

---

## 🤖 Bot Definitions

Each bot lives under the `bots:` list. You can run multiple bots with different models and settings.

### `name`
Display name for logs and metrics.  
Example: `Asslips`

### `model`
Hugging Face model path.  
Example: `DirtyAss/Asslips`

### `username`
Lemmy account username for this bot.
Example: `your_username_here`

### `password`
Lemmy account password for this bot.
Example: `your_password_here`

### `postfreq`
How often (in hours) the bot posts a new thread.  
Example: `6` → every 6 hours

### `max_new_tokens`
Default max tokens for generated replies/bodies.  
Example: `64`

### `min_new_tokens`
Minimum tokens for generations.  
Example: `16`

### `max_new_tokens_cap`
Hard upper cap for generations, even if the bot expands to finish a thought.  
Example: `128`

### `strict_mode`
When `true`, early generations must meet stricter rules (mention keywords, complete thought) before being accepted.  
Example: `false`

### `mention_strict_tries`
How many strict attempts to **force mention** of post keywords before relaxing.  
Example: `2`

### `complete_strict_tries`
How many strict attempts to ensure a **complete** sentence/thought before relaxing.  
Example: `2`

### `length_initial`
Initial minimum word count target for replies.  
Example: `10`

### `length_min`
Absolute minimum word count allowed if retries keep failing.  
Example: `5`

### `fallback_prompt`
Per-bot dynamic fallback template used when all other reply attempts fail.  
You can include `{topics}` to inject detected topics.  
Example: `"Let's talk more about {topics}."`

### `ban_pop`
If `true`, blocks “pop/popup” tokens at generation time (prevents spammy strings).  
Example: `true`

### `ban_unicode_escapes`
If `true`, blocks escaped unicode sequences like `\u200d`.  
Example: `true`

### `max_focus_retries` (per-bot override)
Override the global `max_focus_retries` for this specific bot.  
Example: `3`

### `post_stagger_seconds` (per-bot override)
Per-bot override for global post staggering.  
Example: `900` → 15 minutes

---

## ✅ Example `config.yaml`

```yaml
# ──────────────────────────────────────────────────────────────
# Global settings
# ──────────────────────────────────────────────────────────────
instance: https://sh.itjust.works
community: Asslips

# Logging & debug
log_dir: logs
debug: false

# Optional filters & gating
toxicity_filter: false
max_replies: 3
comment_roll: 70
reply_delay_min: 300
reply_delay_max: 600
initial_post: false
max_focus_retries: 3
post_stagger_seconds: 600

# Fallback configuration
fallback_file: fallbacks_expanded.yaml
fallback_history_size: 5

# Custom toxic words (if any)
toxic_words:
  - slur1
  - slur2

# Bot definitions
bots:
  - name: Asslips
    model: DirtyAss/Asslips`
    username: your_username_here
    password: your_password_here
    postfreq: 6
    max_new_tokens: 64
    min_new_tokens: 16
    max_new_tokens_cap: 128
    strict_mode: false
    mention_strict_tries: 2
    complete_strict_tries: 2
    length_initial: 10
    length_min: 5
    fallback_prompt: "Let's talk about {topics}."
    ban_pop: true
    ban_unicode_escapes: true
    post_stagger_seconds: 900

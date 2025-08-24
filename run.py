#!/usr/bin/env python3
"""
Asslips17c.py – Lemmy GPT bots with:
  • adaptive token budgeting
  • externalized, diversified fallback templates (expanded)
  • direct grounding: title/body or comment in prompt
  • dynamic few-shot + keyword grounding
  • staged temperature/top_p fallback with best-candidate retention
  • refinement and focused pass before external fallback
  • aggressive completion and final finish-guard
  • prompt truncation to prevent overflow
  • suppression of control tokens & cleaning (including literal escape removal and control sequence stripping)
  • thread-history memory via BotDB
  • per-thread fallback deduplication
"""

import argparse
import logging
import random
import re
import threading
import time
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
import yaml
from collections import deque
from pythorhead import Lemmy
from bot_db.db import BotDB

# SortType compatibility shim
try:
    from pythorhead.enums import SortType
except ImportError:
    try:
        from pythorhead.const import SortType
    except ImportError:
        class _EnumPlaceholder(str):
            @property
            def value(self): return str(self)
        class SortType:
            New = _EnumPlaceholder("New")


# -------------------- fallback manager -------------------- #
class FallbackManager:
    def __init__(self, path: str, history_size: int = 5):
        raw = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        self.templates: list[str] = data.get("templates", []) or []
        if not self.templates:
            raise ValueError(f"No fallback templates found in {path}")
        self.history_size = history_size
        self.recent: dict[Any, deque] = {}  # post_id -> deque of recent template indices

    def _related(self, topics: list[str]) -> str:
        if len(topics) >= 2:
            return topics[1]
        elif topics:
            return topics[0]
        return ""

    def choose(self, topics: list[str], post_id: Any) -> str:
        topic = topics[0] if topics else ""
        related = self._related(topics) if topics else ""
        all_indices = list(range(len(self.templates)))
        used = self.recent.get(post_id, deque(maxlen=self.history_size))
        candidates = [i for i in all_indices if i not in used]
        if not candidates:
            candidates = all_indices
        idx = random.choice(candidates)
        template = self.templates[idx]
        # natural-language join: "a, b and c"
        if len(topics) > 1:
            topics_joined = ", ".join(topics[:-1]) + " and " + topics[-1]
        else:
            topics_joined = topics[0] if topics else ""
        filled = (template
                  .replace("{topic}", topic or "")
                  .replace("{related}", related or "")
                  .replace("{topics}", topics_joined))

        if post_id not in self.recent:
            self.recent[post_id] = deque(maxlen=self.history_size)
        self.recent[post_id].append(idx)
        return filled


# --- global inter-bot post staggering (across threads in this process) ---
_GLOBAL_STAGGER_LOCK = threading.Lock()
_GLOBAL_LAST_POST_AT = 0.0

def maybe_stagger_post(spacing_s: float, logger) -> bool:
    """Return True if this bot is allowed to post a new thread now, else False.
       Uses a global gate so bots don't post too close together.
    """
    global _GLOBAL_LAST_POST_AT
    if spacing_s <= 0:
        return True
    with _GLOBAL_STAGGER_LOCK:
        now = time.time()
        elapsed = now - _GLOBAL_LAST_POST_AT
        if elapsed >= spacing_s:
            _GLOBAL_LAST_POST_AT = now
            return True
        gap = spacing_s - elapsed
        logger.info("Global post spacing: skipping new thread, wait %.1fs more.", gap)
        return False

# -------------------- helpers -------------------- #
CLEAN_TOKENS = [
    "<|soss|>", "<|sot|>", "<|eot|>", "<|sost|>", "<|eost|>",
    "<|sols|>", "<|eols|>", "<|sor|>", "<|eor|>", "<|sol|>",
    "<|eol|>", "<|eoss|>", "<|pop|>", "<|pop-up|>", "<|pop!|>"
]
_ZWJ_RE = re.compile(r"[\u200d\uFE0F]")
_ZWSP = "\u200b"  # zero-width space

def clean(text: str) -> str:

    # strip known control tokens
    for tok in CLEAN_TOKENS:
        text = text.replace(tok, " ")
    # remove any <|...|> style control sequences (non-greedy)
    text = re.sub(r"<\|[^>]+\|>", "", text)
    # hard-kill any stray <|pop ... |> style tokens
    text = re.sub(r"(?is)<\|\s*pop[^|>]*\|>", "", text)
    # strip standalone 'pop/pops/popup/pop-up' but keep 'population'/'popular'
    text = re.sub(r"(?i)\bpop(?!ulation|ular)(?:-?up|s)?\b", "", text)
    # remove zero-width joiner / variation selectors
    # keep real ZWJ so emoji sequences stay intact
    # text = _ZWJ_RE.sub("", text)
    # strip literal escape sequences like \u200d, \U0001F600, \xA0, and even bare 'u200d'
    text = re.sub(r"(?i)\\[uU][0-9a-f]{2,8}", "", text)  # \uXXXX or \UXXXXXXXX
    text = re.sub(r"(?i)\\x[0-9a-f]{2}", "", text)       # \xHH
    text = re.sub(r"\\[0-7]{3}", "", text)               # octal \123
    text = re.sub(r"(?i)\\?u[0-9a-f]{2,6}", "", text)    # 'u200d' or '\u200d'

    # remove literal escape sequences and normalize newlines
    text = text.replace("\\u200b", "").replace("\\u200d", "").replace("\\200d", "")
    text = text.replace("\\n\\n", "\n\n").replace("\\n", "\n")
    text = text.replace(_ZWSP, "")
    # remove stray angle-bracket tokens like <esol>, <eso>, <sor>, etc.
    text = re.sub(r"<(?:esol|eso|eor|sol|sor)>", "", text, flags=re.IGNORECASE)
    # unescape common \" and \'
    text = text.replace("\\'", "'").replace('\\"', '"')

    # remove [removed]/[deleted]
    text = re.sub(r"\[(?:removed|deleted)\]", "", text, flags=re.IGNORECASE)

    # nuke incomplete or garbage URLs, but keep plain words and emojis
    # (kills things like "http://", "https://", "http://www.", "www.example", etc.)
    text = re.sub(r"(?:https?://\S+|www\.\S+)", "", text)

    # strip control-ish tokens that sneak through, including pipe-wrapped ones
    # e.g., <eoss>, <sost>, |eoss|, |sost|, or bare eoss/sost at boundaries
    text = re.sub(r"[|<]?\s*(?:eoss|eost|soss|sost|sols?|eols?|sor|eor)\s*[>|]?", "", text, flags=re.IGNORECASE)

    # punctuation throttling (keep emojis intact):
    # collapse !!! -> !, ??? -> ?, and long ..... -> ...
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\.{3,}", "...", text)

    # collapse whitespace
    return " ".join(text.strip().split())

    
# --- Sentence-start normalizer ---
_ZWS = "\u200b\u200c\u200d\u2060\ufeff\u00a0"

def force_sentence_start(s: str) -> str:
    """
    Normalize start-of-sentence:
    - strip zero-width/odd spaces and opening punctuation
    - drop dangling clause openers (and/or/but/so/because/which/that)
    - if it begins with a bare verb like "Is/Are/Was/Were", add a subject ("It/They")
    - capitalize first ASCII letter if lowercase
    """
    if not s:
        return s

    s = s.lstrip(_ZWS + " \t\r\n")
    s = s.lstrip('~`^*_+-–——=:;,.!?/\\|\"\'“”‘’«»()[]{}…')
    s = re.sub(r'^(?:and|or|but|so|because|which|that)\b[, ]*', '', s, flags=re.I)

    # Second pass if we stripped to empties/punct
    s = s.lstrip(_ZWS + " \t\r\n")
    s = s.lstrip('~`^*_+-–——=:;,.!?/\\|\"\'“”‘’«»()[]{}…')

    # Fix common missing-subject openings
    # "Is ..." -> "It is ...", "Are ..." -> "They are ...", etc.
    # Use flags=re.I instead of inline (?i) to avoid "global flags not at the start" errors
    s = re.sub(r'^(is)\b',   'It is',    s, flags=re.I)
    s = re.sub(r'^(was)\b',  'It was',   s, flags=re.I)
    s = re.sub(r'^(are)\b',  'They are', s, flags=re.I)
    s = re.sub(r'^(were)\b', 'They were', s, flags=re.I)

    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s

def split_title_body(raw: str) -> tuple[str, str]:
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        return "Untitled 🤔", ""
    title = re.sub(r"[<>|].*?$", "", lines[0][:200]).strip() or "Untitled 🤔"
    body = "\n".join(lines[1:]).strip()
    return title, body


def extract_keywords(text: str, max_words: int = 5) -> list[str]:
    stopset = {
        "the", "and", "a", "an", "of", "to", "in", "for", "on", "with",
        "that", "this", "it", "is", "was", "you", "have", "has", "but",
        "not", "what", "kind", "more", "than", "also", "they", "just",
        "i", "me", "my", "our", "your"
    }
    words = re.findall(r"\b\w+'\w+|\b\w+\b", text.lower())
    filtered = [w for w in words if w not in stopset and len(w) >= 3]
    freq: dict[str, int] = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1
    sorted_keywords = sorted(freq.keys(), key=lambda k: (-freq[k], len(k)))
    return sorted_keywords[:max_words]


def make_few_shot_example(topics: list[str]) -> str:
    primary = topics[0] if topics else "the topic"
    good = f"I hear you about {primary}. Here's a suggestion related to {primary}."
    bad = "I love pizza and cats."
    return f"Good example reply: \"{good}\"\nBad example reply: \"{bad}\"\n\n"


def is_relevant_enough(reply: str, topics: list[str]) -> bool:
    low = reply.lower()
    return any(re.search(rf"\b{re.escape(t.lower())}\b", low) for t in topics)

def mentions_any_keyword(reply: str, keywords: list[str]) -> bool:
    """Return True if reply mentions at least one keyword (case-insensitive, whole-word where possible)."""
    if not reply or not keywords:
        return False
    low = reply.lower()
    for kw in keywords:
        kw = (kw or "").strip()
        if not kw:
            continue
        # prefer whole-word match when alnum, else fallback to substring
        if re.search(rf"\b{re.escape(kw.lower())}\b", low) if re.search(r"[A-Za-z0-9]", kw) else (kw.lower() in low):
            return True
    return False



def iter_post_views(raw: Any):
    if isinstance(raw, dict):
        yield from raw.get("posts", [])
    else:
        for item in raw or []:
            if "post" in item and "creator" in item:
                yield item
            else:
                yield {"post": item, "creator": {"name": item.get("name", "")}}


def iter_comment_views(raw: Any):
    if isinstance(raw, dict):
        yield from raw.get("comments", [])
    else:
        for item in raw or []:
            if "comment" in item and "creator" in item:
                yield item
            else:
                yield {"comment": item, "creator": {"name": item.get("creator_name", "")}}


# -------------------- BotThread -------------------- #
class BotThread(threading.Thread):
    def __init__(self, bot_cfg: MappingProxyType, global_cfg: MappingProxyType, fallback_manager: FallbackManager):
        super().__init__(daemon=True, name=bot_cfg.get("name", "bot"))
        self.cfg = bot_cfg
        self.global_cfg = global_cfg
        self.fallback_manager = fallback_manager
        self.log = logging.getLogger(bot_cfg.get("name", "bot"))

        self.lemmy = Lemmy(global_cfg.get("instance"))
        self.lemmy.log_in(bot_cfg.get("username"), bot_cfg.get("password"))
        self.community_id = self.lemmy.discover_community(global_cfg.get("community"))

        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(bot_cfg.get("model"))
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(bot_cfg.get("model"))
        self.model.eval()

        bad_words_ids = []
        for tok in CLEAN_TOKENS:
            toks = self.tokenizer.encode(tok, add_special_tokens=False)
            if toks:
                bad_words_ids.append(toks)
        self.bad_words_ids = bad_words_ids

        # Optional: forbid unicode-escape starters so the model can't emit them
        if self.cfg.get("ban_unicode_escapes", True):
            for w in ["\\u", "\\U", "\\x"]:
                ids = self.tokenizer.encode(w, add_special_tokens=False)
                if ids:
                    self.bad_words_ids.append(ids)

        # Optional hard-ban of 'pop' spam during generation (toggle per-bot in config)
        if self.cfg.get("ban_pop", True):
            for w in ["pop", "pops", "popup", "pop-up", "Pop", "POP", "Pop!", "pop!"]:
                ids = self.tokenizer.encode(w, add_special_tokens=False)
                if ids:
                    self.bad_words_ids.append(ids)

        self.toxic_words = [w.casefold() for w in global_cfg.get("toxic_words", [])]

        self.freq_s = float(bot_cfg.get("postfreq", 1)) * 3600
        self.initial_post = bool(global_cfg.get("initial_post", False))
        self.roll_needed = int(global_cfg.get("comment_roll", 70))
        # inherit from global config if not provided per-bot
        self.max_replies = int(bot_cfg.get("max_replies", global_cfg.get("max_replies", 3)))
        self.delay_min   = float(bot_cfg.get("reply_delay_min", global_cfg.get("reply_delay_min", 300)))
        self.delay_max   = float(bot_cfg.get("reply_delay_max", global_cfg.get("reply_delay_max", 600)))
        # maximum regenerations if reply misses focus terms
        self.max_focus_retries = int(bot_cfg.get("max_focus_retries", global_cfg.get("max_focus_retries", 3)))
        # minimum gap between ANY bots posting new threads (seconds); 0 disables staggering
        self.global_post_spacing = float(bot_cfg.get("post_stagger_seconds", global_cfg.get("post_stagger_seconds", 0)))
        self.base_max_new_tokens = int(bot_cfg.get("max_new_tokens", 64))
        self.current_max_new_tokens = self.base_max_new_tokens
        self.min_new_tokens_allowed = int(bot_cfg.get("min_new_tokens", 16))
        self.max_new_tokens_cap = int(bot_cfg.get("max_new_tokens_cap", self.base_max_new_tokens))

        self.last_post_at = 0.0
        self.db = BotDB(bot_cfg.get("username"), self.community_id)
        self.stop_event = threading.Event()

    def _is_toxic(self, txt: str) -> bool:
        return any(w in txt.casefold() for w in self.toxic_words)

    def _is_incomplete_reply(self, text: str) -> bool:
        if not text or not text.strip():
            return True
        stripped = text.strip()
        if stripped.endswith(('.', '!', '?', '...')):
            return False
        last_word = stripped.split()[-1].lower()
        if last_word in {'and', 'or', 'but', 'with', 'to', 'for', 'because', 'so', 'if', 'that', 'which', 'when', 'while', 'as', 'than', 'then'}:
            return True
        if len(stripped.split()) < 8:
            return True
        return False

    def adjust_token_budget(self, last_reply: str, was_empty: bool):
        if was_empty:
            self.current_max_new_tokens = max(self.min_new_tokens_allowed, int(self.current_max_new_tokens * 0.75))
        else:
            incomplete = self._is_incomplete_reply(last_reply)
            if incomplete:
                self.current_max_new_tokens = min(self.max_new_tokens_cap, int(self.current_max_new_tokens * 1.25))
            else:
                if self.current_max_new_tokens > self.base_max_new_tokens:
                    self.current_max_new_tokens = max(self.base_max_new_tokens, int(self.current_max_new_tokens * 0.9))
                else:
                    self.current_max_new_tokens = max(self.min_new_tokens_allowed, int(self.current_max_new_tokens * 0.95))

    def _gen(self, prompt: str, skip_toxic: bool = False, temperature: float | None = None,
             top_p: float | None = None, override_new_tokens: int | None = None) -> str:
        if hasattr(self.model.config, 'n_ctx'):
            context_window = self.model.config.n_ctx
        elif hasattr(self.model.config, 'max_position_embeddings'):
            context_window = self.model.config.max_position_embeddings
        else:
            context_window = None

        if context_window is not None:
            if override_new_tokens is not None:
                max_prompt_len = max(1, context_window - override_new_tokens - 1)
                inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_prompt_len)
            else:
                max_prompt_len = max(1, context_window - 1)
                inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_prompt_len)
        else:
            inputs = self.tokenizer(prompt, return_tensors='pt')
        ids = inputs.input_ids
        attn = inputs.get('attention_mask', None)

        if not prompt.strip() or ids.size(1) == 0:
            bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            ids = torch.tensor([[bos]], device=ids.device)
            prompt_len = 1
            attn = torch.ones_like(ids)
        else:
            prompt_len = ids.size(1)

        if context_window is not None:
            if override_new_tokens is not None:
                allowed_new = min(override_new_tokens, context_window - prompt_len - 1)
            else:
                allowed_new = min(self.current_max_new_tokens, context_window - prompt_len - 1)
            allowed_new = max(1, allowed_new)
        else:
            allowed_new = override_new_tokens if override_new_tokens is not None else self.current_max_new_tokens

        min_new = 20 if allowed_new >= 20 else allowed_new
        temp = temperature if temperature is not None else 0.65
        tp = top_p if top_p is not None else 0.75

        for _ in range(4):
            with torch.no_grad():
                out = self.model.generate(
                    ids,
                    attention_mask=attn,
                    max_new_tokens=allowed_new,
                    min_new_tokens=min_new,
                    do_sample=True,
                    temperature=temp,
                    top_p=tp,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    bad_words_ids=self.bad_words_ids,
                )
            gen_ids = out[0, prompt_len:]
            txt = clean(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
            txt = force_sentence_start(txt)
            if txt and (skip_toxic or not self._is_toxic(txt)):
                return txt
        return ""

    def _try_complete(self, candidate: str, topic_str: str, temp: float, tp: float) -> str:
        """
        If candidate looks incomplete, attempt up to two refinements to finish the thought,
        giving more token budget.
        """
        for _ in range(2):
            if not self._is_incomplete_reply(candidate):
                break
            continuation_prompt = (
                f"The previous reply was cut off. Here is what it gave:\n\"{candidate}\"\n"
                f"Finish the thought in one or two sentences, referencing at least one of these topics: {topic_str}. "
                "Do not repeat the whole previous reply; just continue and conclude the idea. Reply:"
            )
            more = self._gen(continuation_prompt, skip_toxic=False, temperature=temp, top_p=tp, override_new_tokens=128).strip()
            if more.lower().startswith("reply:"):
                more = more[len("reply:"):].strip()
            if not more:
                break
            candidate = candidate.rstrip(" .,!?:;") + " " + more.lstrip()
        return candidate

    def _post(self, title: str, body: str) -> int | None:
        # belt-and-suspenders: ensure no links slip into the title here either
        title = re.sub(r'(https?://\S+|www\.\S+|\[[^\]]+\]\(https?://[^\s)]+\))', '', title)
        title = re.sub(r'!{9,}',  '!!!!!!!!', title)   # cap ! at 8
        title = re.sub(r'\?{9,}', '????????', title)   # cap ? at 8
        title = title.strip()
        # final sanitize for title/body
        title = force_sentence_start(clean(title))
        body  = force_sentence_start(clean(body)) 
        try:
            res = self.lemmy.post.create(self.community_id, title, body=body, nsfw=True)
            pid = res["post_view"]["post"]["id"]
            self.log.info("Posted: %s", title)
            try:
                self.db.record_post(source_id=pid,
                                    author=self.cfg.get("username"),
                                    text=f"{title}\n{body or ''}")
            except Exception:
                self.log.exception("Failed to record post in DB")
            return pid
        except Exception:
            self.log.exception("post failed")
            return None

    def _ensure_sentence_finish(self, text: str, is_title: bool = False) -> str:
        """
        If text looks incomplete (dangling conjunctions/clauses or no terminal punctuation),
        try a short, low-entropy continuation to finish the thought.
        Titles: no generation, just light tidy (and no forced period).
        """
        t = (text or "").strip()
        if not t:
            return t

        def _looks_incomplete(s: str) -> bool:
            if not s:
                return True
            # Ends with clause/conjunction or soft punctuation
            if re.search(r'(?:\b(?:and|or|but|because|when|which|that|if|though|although)\s*)$', s, flags=re.I):
                return True
            if s[-1:] in {",", ":", ";"}:
                return True
            # No terminal punctuation → likely cut off
            if s[-1:] not in {".", "!", "?"}:
                return True
            return False

        # Titles: do not generate; just trim any trailing junk and return
        if is_title:
            # If a title ends with connector words or soft punctuation, just strip them
            t = re.sub(r'[\s,:;]+$', '', t)
            t = re.sub(r'(?:\b(?:and|or|but|because|when|which|that|if|though|although)\s*)$', '', t, flags=re.I).strip()
            return t

        # Bodies/Comments: try up to two short continuations
        loops = 0
        while _looks_incomplete(t) and loops < 2:
            finish_prompt = (
                f"{t}\n\n"
                "Continue and finish the sentence naturally in the same tone. "
                "Keep it brief (a short clause). Output only the continuation."
            )
            extra = self._gen(
                finish_prompt,
                skip_toxic=False,
                temperature=0.4,
                top_p=0.9,
                override_new_tokens=48
            ).strip()

            # Strip leading "reply:" if the model emits it
            if extra.lower().startswith("reply:"):
                extra = extra[len("reply:"):].strip()

            if extra:
                t = (t.rstrip(" .,!?:;") + " " + extra.lstrip()).strip()
                t = clean(t)
                t = force_sentence_start(t)

            loops += 1

        # Ensure terminal punctuation for bodies/comments
        if t and t[-1] not in ".!?":
            t += "."

        return t


    def _looks_like_code_or_garbage(self, s: str, is_title: bool = False) -> bool:
        """
        Heuristics to block code-like or symbol-soup outputs.
        - High symbol ratio
        - Obvious code keywords/patterns
        - Lots of braces/brackets/backslashes or ``` fences
        Titles are allowed to be short, but still must not look like code.
        """
        if not s:
            return True

        import re
        t = s.strip()

        # obvious code keywords
        if re.search(r"\b(var|let|const|function|class|public|private|static|import|package|return|try|catch|finally|console\.log)\b", t, re.I):
            return True

        # code fences/backticks or XML/HTML tags
        if re.search(r"`{3,}", t) or re.search(r"</?[a-z][^>]*>", t, re.I):
            return True

        # heavy brace/paren/semicolon/backslash density
        if len(re.findall(r"[{}()\[\];\\]{2,}", t)) >= 2:
            return True

        # lots of escaped sequences like \n, \t, \r or path-like A\B
        if re.search(r"(?:\\[ntr])+|[A-Za-z]+\\[A-Za-z]+", t):
            return True

        # symbol ratio (ignore emojis)
        emoji_removed = re.sub(r"[\U00010000-\U0010FFFF]", "", t)
        alpha = sum(ch.isalpha() for ch in emoji_removed)
        non_alpha = sum((not ch.isalpha()) and (not ch.isspace()) for ch in emoji_removed)
        total = alpha + non_alpha
        if total > 0 and non_alpha / total > 0.55:
            return True

        # titles may be short but shouldn't be ALL caps + punctuation
        if is_title and re.fullmatch(r"[A-Z\s!\?\.\-:;,'/\\]+", t):
            return True

        # obvious URL or filename/querystring patterns (block as garbage)
        if re.search(r"(https?://|www\.|\.jpg|\.png|\.gif|\.webp|\.mp4|width=|height=|format=)", t, re.I):
            return True

        return False



    def _comment(self, post_id: int, content: str, parent_id: int | None = None) -> int | None:
        try:
            res = self.lemmy.comment.create(post_id, content, parent_id=parent_id)
            cid = (res or {}).get("comment_view", {}).get("comment", {}).get("id")
            self.log.info("Commented on %d (comment_id=%s)", post_id, str(cid))
            return cid
        except Exception:
            self.log.exception("comment failed")
            return None

    def _attempt_replies(self, sources: list[dict[str, Any]]) -> None:
        attempts = 0
        for src in sources:
            try:
                if attempts >= self.max_replies:
                    break
                if random.randint(1, 100) < self.roll_needed:
                    continue

                post_id    = src.get("post_id")
                is_comment = src.get("parent_id") is not None
                raw_text = src.get("text", "").strip()
                source_text = self.db.get_source(post_id, src.get("parent_id")) or raw_text
                history = self.db.get_thread_history(post_id, limit=3)
                history_str = ""
                if history:
                    reversed_hist = list(reversed(history))
                    history_lines = []
                    for author, gen in reversed_hist:
                        snippet = gen.replace("\n", " ").strip()
                        history_lines.append(f"{author}: {snippet}")
                    history_str = "Recent thread context:\n" + "\n".join(history_lines) + "\n\n"

                # build topics (and forced words) differently for posts vs. comments
                orig_topics = extract_keywords(raw_text, max_words=5)
                if not is_comment:
                    # for posts: inject 1–2 words from title & body
                    title, body = split_title_body(raw_text)
                    title_words = re.findall(r"\b[A-Za-z]{4,}\b", title)[:2]
                    body_words  = re.findall(r"\b[A-Za-z]{4,}\b", body)[:2]
                    topics      = orig_topics + title_words + body_words
                else:
                    # for comments: no forced words, but still define the vars
                    title_words = []
                    body_words  = []
                    topics      = orig_topics
                topic_str = ", ".join(topics) if topics else "general"
                few_shot  = make_few_shot_example(topics)

                # build prompt
                if is_comment:
                    focus_list = extract_keywords(source_text, max_words=8)
                    focus_line = ", ".join(focus_list) if focus_list else ""
                    prompt = (
                        f"{few_shot}"
                        f"{history_str}"
                        f"You are replying to a comment in the Asslips community. Original comment:\n\"{source_text}\"\n\n"
                        f"Main topics: {topic_str}\n"
                        + (f"Focus terms (use at least one verbatim): {focus_line}\n" if focus_line else "")
                        + "Instructions:\n"
                        "0. Begin with a complete sentence (subject + verb). Do not start with punctuation or a conjunction.\n"
                        "1. Briefly acknowledge something specific from that comment (reference at least one of the topics above) in one sentence.\n"
                        "2. Then provide a relevant, on-topic response or continuation.\n"
                        "3. Do not repeat more than 10 words verbatim; paraphrase in your own words.\n"
                        "4. Keep it to 2-3 sentences and finish the thought. Reply:"
                    )
                else:
                    # build prompt from the DB text, but make Title and Body explicit
                    t_db, b_db = split_title_body(source_text)
                    # keep a generous chunk of body but cap to avoid ctx overflow
                    body_to_use = b_db if len(b_db) <= 1200 else (b_db[:1200].rsplit(" ", 1)[0] + "...")
                    # focus terms for posts (title + visible body chunk)
                    focus_list = extract_keywords((t_db + " " + body_to_use), max_words=8)
                    focus_line = ", ".join(focus_list) if focus_list else ""
                    prompt = (
                        f"{few_shot}"
                        f"{history_str}"
                        f"You are participating in the Asslips community. Below is the original post:\n\n"
                        f"Title: \"{t_db}\"\n\n"
                        f"Body:\n{body_to_use}\n\n"
                        f"Main topics: {topic_str}\n"
                        + (f"Focus terms (use at least one verbatim): {focus_line}\n" if focus_line else "")
                        + "Instructions:\n"
                        "0. Begin with a complete sentence (subject + verb). Do not start with punctuation or a conjunction.\n"
                        "1. Briefly acknowledge something specific from the title or body (reference at least one of the topics above) in one sentence.\n"
                        "2. Then give an on-topic follow-up, advice, or commentary related to that.\n"
                        "3. Do not repeat more than 10 words verbatim from the original; use your own phrasing.\n"
                        "4. Keep it coherent, avoid unrelated rambling, and end with a complete thought. Limit to about 2-3 sentences. Reply:"
                    )

                reply = ""
                best_candidate = ""
                best_score = -1

                # staged attempts with dynamic strictness (per-bot settings)
                strict = self.cfg.get("strict_mode", False)
                mention_tries   = int(self.cfg.get("mention_strict_tries",   2))
                complete_tries  = int(self.cfg.get("complete_strict_tries",  2))
                len_init        = int(self.cfg.get("length_initial",       10))
                len_min         = int(self.cfg.get("length_min",            0))
                max_tries       = 5 if is_comment else 3

                for attempt in range(max_tries):
                    # pick temperature/top_p per stage
                    if attempt == 0:
                        temp, tp = 0.55, 0.75
                    elif attempt == 1:
                        temp, tp = 0.60, 0.80
                    else:
                        temp, tp = 0.75, 0.90

                    # generate
                    candidate = self._gen(prompt, skip_toxic=False, temperature=temp, top_p=tp).strip()
                    if candidate.lower().startswith("reply:"):
                        candidate = candidate[len("reply:"):].strip()
                    if not candidate or candidate.lower() == raw_text.lower():
                        continue

                    # finish cut-off replies
                    candidate = self._try_complete(candidate, topic_str, temp, tp)

                    # comment: one instruction-guided retry if off-topic/empty
                    if is_comment and (not candidate or not is_relevant_enough(candidate, topics)):
                        inst_prompt = (
                            f"You are replying to a comment in the Asslips community.\n"
                            f"Original comment: \"{source_text}\"\n\n"
                            "Instructions:\n"
                            "0. Begin with a complete sentence (subject + verb). Do not start with punctuation or a conjunction.\n"
                            "1. Briefly acknowledge something specific from that comment.\n"
                            "2. Provide a relevant, on-topic response or continuation.\n"
                            "3. Do not repeat more than 10 words verbatim; paraphrase.\n"
                            "4. Keep it coherent and finish with a complete thought in 2-3 sentences.\n"
                            "Reply:"
                        )
                        inst_candidate = self._gen(inst_prompt, skip_toxic=False, temperature=temp, top_p=tp).strip()
                        if inst_candidate:
                            candidate = inst_candidate

                    # decide which rules to enforce this round
                    enforce_mention  = strict and (attempt < mention_tries)  and not is_comment
                    enforce_complete = strict and (attempt < complete_tries)
                    length_threshold = max(
                        len_min,
                        len_init - int((len_init - len_min) * (attempt / max(1, max_tries - 1)))
                    )

                    # 1) forced-mention on posts
                    if enforce_mention:
                        if not any(re.search(rf"\b{re.escape(w)}\b", candidate, re.IGNORECASE) for w in title_words + body_words):
                            continue

                    # 2) complete-thought check
                    if enforce_complete and self._is_incomplete_reply(candidate):
                        continue

                    # 3) length requirement
                    if len(candidate.split()) < length_threshold:
                        continue

                    # simple scoring
                    score = 0
                    if is_relevant_enough(candidate, topics):
                        score += 2
                    if not self._is_incomplete_reply(candidate):
                        score += 1
                    if len(candidate.split()) >= length_threshold:
                        score += 1

                    # remember best
                    if score > best_score:
                        best_score     = score
                        best_candidate = candidate

                    # early accept if fully past enforced rules
                    if not enforce_mention and not (enforce_complete and self._is_incomplete_reply(candidate)) \
                       and len(candidate.split()) >= length_threshold:
                        reply = candidate
                        break

                # refinement pass
                if not reply and best_candidate:
                    refine_prompt = (
                        f"Improve the following reply to make it more on-topic, include at least one of the main topics: {topic_str}, "
                        f"and finish the thought. Previous reply: \"{best_candidate}\". Reply:"
                    )
                    refined = self._gen(refine_prompt, skip_toxic=False, temperature=0.7, top_p=0.85, override_new_tokens=128).strip()
                    if refined.lower().startswith("reply:"):
                        refined = refined[len("reply:"):].strip()
                    if refined and is_relevant_enough(refined, topics) and not self._is_incomplete_reply(refined):
                        reply = refined
                    else:
                        reply = best_candidate

                # focused low-entropy pass
                if not reply:
                    focused_prompt = f"{prompt}\n\nNow respond in one concise sentence that references at least one of the main topics and completes the thought."
                    focused = self._gen(focused_prompt, skip_toxic=False, temperature=0.3, top_p=0.9).strip()
                    if focused.lower().startswith("reply:"):
                        focused = focused[len("reply:"):].strip()
                    if focused and is_relevant_enough(focused, topics):
                        reply = focused

                # final finish guard if still incomplete
                if reply and self._is_incomplete_reply(reply):
                    finish_prompt = f"That last reply was cut off. Finish it cleanly: \"{reply}\". Reply:"
                    extra = self._gen(finish_prompt, skip_toxic=False, temperature=0.5, top_p=0.9, override_new_tokens=96).strip()
                    if extra.lower().startswith("reply:"):
                        extra = extra[len("reply:"):].strip()
                    if extra:
                        reply = reply.rstrip(" .,!?:;") + " " + extra.lstrip()

                # last-resort fallback
                if not reply:
                    bot_fp = self.cfg.get("fallback_prompt")
                    if bot_fp:
                        # gather candidate keywords (preserve order, dedupe)
                        if not is_comment:
                            pool = [w for w in (title_words + body_words) if w]
                        else:
                            pool = [w for w in (orig_topics) if w]
                        if not pool:
                            pool = [w for w in (topics) if w]
                        pool = list(dict.fromkeys(pool))  # dedupe, keep order

                        # build natural-language list safely
                        if not pool:
                            dynamic_topics = "this"
                        elif len(pool) == 1:
                            dynamic_topics = pool[0]
                        else:
                            dynamic_topics = ", ".join(pool[:-1]) + " and " + pool[-1]

                        # fill and generate
                        prompt = bot_fp.format(topics=dynamic_topics)
                        gen = self._gen(prompt, skip_toxic=False, temperature=0.75, top_p=0.9).strip()
                        reply = gen or prompt
                        self.log.debug("Dynamic multi-topic fallback for %s → %r", post_id, reply)
                    else:
                        if is_comment:
                            if best_candidate:
                                reply = best_candidate
                                self.log.debug("Best candidate used for comment %s → %r", post_id, reply)
                            else:
                                self.log.debug("No candidate for comment %s → skipping", post_id)
                                continue
                        else:
                            reply = self.fallback_manager.choose(topics, post_id)
                            self.log.debug("YAML fallback used for post %s → %s", post_id, reply)

                # dedupe: don't repost same
                last_hist = self.db.get_thread_history(post_id, limit=1)
                if last_hist:
                    last_author, last_gen = last_hist[-1]
                    if last_author == self.cfg.get("username") and last_gen.strip() == reply.strip():
                        continue

                # clean up + post (clean first, then enforce a proper sentence start)
                reply = clean(reply)
                reply = force_sentence_start(reply)
                reply = self._ensure_sentence_finish(reply, is_title=False)

                tries = 0
                while self._looks_like_code_or_garbage(reply, is_title=False) and tries < 2:
                    reply = self._gen(
                        f"Rewrite this reply in plain English (no code, no symbols, keep emojis):\n\n{reply}\n\n"
                        "Output one or two natural sentences.",
                        temperature=0.6,
                        top_p=0.9,
                    ).strip()
                    reply = clean(reply)
                    reply = force_sentence_start(reply)
                    reply = self._ensure_sentence_finish(reply, is_title=False)
                    tries += 1

                # No canned fallback: if it still looks like code/garbage, prefer our best real candidate.
                if self._looks_like_code_or_garbage(reply, is_title=False):
                    if best_candidate:
                        reply = clean(best_candidate)
                        reply = force_sentence_start(reply)
                        reply = self._ensure_sentence_finish(reply, is_title=False)
                    # else: keep the current reply as-is (no forced replacement)

                # --- Final relevance guard: must mention at least one focus term ---
                try:
                    if is_comment:
                        focus_terms = extract_keywords(source_text, max_words=8)
                    else:
                        # use full, untruncated text to get good terms (we still built body_to_use for the prompt)
                        t_full, b_full = split_title_body(source_text)
                        focus_terms = extract_keywords((t_full + " " + b_full), max_words=8)

                    if focus_terms and not mentions_any_keyword(reply, focus_terms):
                        focus_line = ", ".join(focus_terms)
                        self.log.info("Reply missed focus terms; retrying up to %d focused generations...", self.max_focus_retries)
                        success = False
                        for attempt in range(self.max_focus_retries):
                            focus_prompt = (
                                f"{prompt}\n\n"
                                f"Now respond in one concise sentence that references at least one of these focus terms: {focus_line}. "
                                f"Stay on-topic and finish the thought."
                            )
                            regen = self._gen(focus_prompt, skip_toxic=False, temperature=0.45, top_p=0.8).strip()
                            if regen.lower().startswith("reply:"):
                                regen = regen[len("reply:"):].strip()
                            if regen and mentions_any_keyword(regen, focus_terms):
                                reply = clean(regen)
                                reply = force_sentence_start(reply)
                                reply = self._ensure_sentence_finish(reply, is_title=False)
                                success = True
                                break

                        if not success:
                            self.log.info(
                                "All retries failed to reference focus terms. Skipping this reply on post_id=%s parent_id=%s.",
                                str(post_id), str(src.get("parent_id"))
                            )
                            continue
                except Exception:
                    # never block the whole bot loop if the guard itself crashes
                    pass

                cid = self._comment(post_id, reply, parent_id=src.get("parent_id"))
                self.log.info("Posted reply (comment_id=%s) on post_id=%s parent_id=%s",
                              str(cid), str(post_id), str(src.get("parent_id")))
                try:
                    # Persist reply so future turns get real thread memory
                    self.db.record_comment(comment_id=(cid or 0),
                                           post_id=post_id,
                                           author=self.cfg.get("username"),
                                           text=reply)
                except Exception:
                    # Don't let persistence failures crash the loop
                    self.log.exception("Failed to record comment in DB")
                attempts += 1
                time.sleep(random.uniform(self.delay_min, self.delay_max))

            except Exception:
                self.log.exception("Error handling item: %r", src)
                continue


    def _last_post_time(self) -> float | None:
        """
        Deprecated: community-feed based last post lookup.
        Posting gate now uses _last_user_topic_time() from the user profile.
        """
        return None

    def _last_user_topic_time(self) -> float | None:
        """
        Return the Unix timestamp (seconds) of the most recent THREAD (not comment)
        created by this bot in the target community. Uses the community feed and
        filters by creator == this bot's username.
        """
        try:
            username = self.cfg.get("username", "")
            if not username:
                return None

            newest = None
            # Scan a few pages to be robust on active communities
            for page in range(1, 5):  # up to ~200 newest posts (4 * 50)
                feed = self.lemmy.post.list(
                    page=page,
                    limit=50,
                    sort=SortType.New,
                    community_id=self.community_id,
                )
                for pv in iter_post_views(feed):
                    try:
                        creator = (pv.get("creator") or {}).get("name", "")
                        if not creator or creator != username:
                            continue
                        post = pv.get("post") or {}
                        published = post.get("published")
                        if not published:
                            continue
                        from datetime import datetime
                        ts = datetime.fromisoformat(published.replace("Z", "+00:00")).timestamp()
                        if newest is None or ts > newest:
                            newest = float(ts)
                    except Exception:
                        continue

                # If we found one on the first page, that's the newest—no need to keep paging
                if newest is not None and page == 1:
                    break

            return newest
        except Exception as e:
            self.log.warning(f"Could not fetch last user topic time: {e}")
            return None


    def run(self) -> None:

        while not self.stop_event.is_set():
            now = time.time()

            # On (re)boot, seed last_post_at from the user's profile (threads only)
            if not self.initial_post and self.last_post_at == 0:
                try:
                    last_ts = self._last_user_topic_time()
                    if last_ts:
                        self.last_post_at = last_ts
                        self.log.info(
                            "Boot seed (profile): last thread at %s UTC",
                            time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(last_ts)),
                        )
                except Exception as e:
                    self.log.warning(f"Could not seed last_post_at from profile: {e}")

            # Always check the user's profile to avoid reposting too soon after a reboot.
            effective_last = self.last_post_at
            try:
                srv_ts = self._last_user_topic_time()
                if srv_ts:
                    # Trust the profile time if it's newer than what we have locally
                    if effective_last == 0.0 or srv_ts > effective_last:
                        effective_last = srv_ts
                        self.last_post_at = srv_ts
                        self.log.info(
                            "Profile check: last thread at %s UTC",
                            time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(srv_ts)),
                        )
            except Exception as e:
                self.log.warning(f"Profile check failed: {e}")

            age = now - effective_last
            if not self.initial_post and age < self.freq_s:
                self.log.info(
                    "Skipping new thread — last user thread was %.2fh ago (< %.2fh).",
                    age / 3600, self.freq_s / 3600
                )
            elif self.initial_post or age >= self.freq_s:
                title = ""
                fallback_prompts = [
                    "Vile Asslips", "AgentGiga", "Krobix897",
                    "Buttholes", "Turds", "farts", "farting", "fart",
                    "dingle berry", "Lemmy", "Sour Asslips",
                ]

                # Count only real words (letters, 3+ chars). Emojis/punctuation don't count.
                def _real_word_count(s: str) -> int:
                    # strip URLs just for counting, keep emojis in original title
                    s_no_urls = re.sub(r'(https?://\S+|www\.\S+|\[[^\]]+\]\(https?://[^\s)]+\))', '', s)
                    return len(re.findall(r"\b[A-Za-z]{3,}\b", s_no_urls))

                # Try seed prompts first
                for p in [""] + fallback_prompts:
                    cand = self._gen(p).splitlines()[0][:200].strip() if p else ""
                    if cand:
                        cand = re.sub(r"[<>|].*?$", "", cand)
                        if (
                            cand != "Untitled 🤔"
                            and _real_word_count(cand) >= 4
                            and not self._looks_like_code_or_garbage(cand, is_title=True)
                            and not re.search(r"(https?://|www\.|\.jpg|\.png|\.gif|\.webp|width=|format=)", cand, re.I)
                        ):
                            title = cand
                            break

                # If still empty or too short, regenerate a few times until we have ≥4 real words
                attempts = 0
                while not title and attempts < 6:
                    cand = self._gen("").splitlines()[0][:200].strip()
                    if cand:
                        cand = re.sub(r"[<>|].*?$", "", cand)
                        if (
                            cand != "Untitled 🤔"
                            and _real_word_count(cand) >= 4
                            and not self._looks_like_code_or_garbage(cand, is_title=True)
                        ):
                            title = cand
                            break
                    attempts += 1

                if not title:
                    title = "Community of Asslips"

                body = ""
                for _ in range(3):
                    b = self._gen(title).strip()
                    if b and b.lower() != title.lower():
                        body = b
                        break
                if not body:
                    body = " "    
                    
                # normalize & ensure complete sentence for the post body
                body = clean(body)
                body = force_sentence_start(body)
                if self._is_incomplete_reply(body):
                    finish_prompt = f"That last post body was cut off. Finish it cleanly: \"{body}\". Reply:"
                    extra = self._gen(
                        finish_prompt,
                        skip_toxic=False,
                        temperature=0.5,
                        top_p=0.9,
                        override_new_tokens=128
                    ).strip()
                    if extra.lower().startswith("reply:"):
                        extra = extra[len("reply:"):].strip()
                    if extra:
                        body = body.rstrip(" .,!?:;") + " " + extra.lstrip()

                # final tidy + enforce sentence start again (in case continuation added junk)
                body = clean(body)
                body = force_sentence_start(body)
                body = self._ensure_sentence_finish(body, is_title=False)

                # Reject code/symbol soup; try a couple light regenerations
                tries = 0
                while self._looks_like_code_or_garbage(body, is_title=False) and tries < 2:
                    body = self._gen(
                        f"Rewrite this in plain English (no code, no symbols, keep emojis):\n\n{body}\n\n"
                        "Output a natural paragraph.",
                        temperature=0.6,
                        top_p=0.9,
                    ).strip()
                    body = clean(body)
                    body = force_sentence_start(body)
                    body = self._ensure_sentence_finish(body, is_title=False)
                    tries += 1

                # Final fallback if still messy
                if self._looks_like_code_or_garbage(body, is_title=False):
                    body = "I’m excited about this, and I’d love to hear what you think. 😊"

                # strip URLs from the title and move them into the body
                urls = re.findall(r'(https?://\S+|www\.\S+|\[[^\]]+\]\(https?://[^\s)]+\))', title)
                title = re.sub(r'(https?://\S+|www\.\S+|\[[^\]]+\]\(https?://[^\s)]+\))', '', title)

                # keep dramatic punctuation but cap at 8 if you want
                title = re.sub(r'\bsource\s*:?\s*$', '', title, flags=re.I)
                title = re.sub(r'!{9,}',  '!!!!!!!!', title)   # cap ! at 8
                title = re.sub(r'\?{9,}', '????????', title)   # cap ? at 8
                title = re.sub(r'\s{2,}', ' ', title).strip()
                title = self._ensure_sentence_finish(title, is_title=True)


                # if title emptied by scrubbing, fall back to a body snippet
                if not title:
                    title = ' '.join(body.split()[:8]) or 'Quick thought'

                # if we stripped links, append them to the body as sources
                if urls:
                    plain_urls = [re.sub(r'^\[.*?\]\((https?://[^\s)]+)\)$', r'\1', u) for u in urls]
                    body = (body.rstrip() + "\n\nSources: " + " ".join(plain_urls)).strip()

                # enforce global staggering so bots don't post back-to-back (non-blocking)
                if maybe_stagger_post(self.global_post_spacing, self.log):
                    self._post(title, body)
                    self.last_post_at = now
                    self.initial_post = False
                else:
                    self.log.info("Skipping new thread this cycle due to global spacing gate.")
                    
            feed = self.lemmy.post.list(
                page=1,
                limit=self.max_replies * 3,
                sort=SortType.New,
                community_id=self.community_id,
            )
            posts = []
            for pv in iter_post_views(feed):
                post_id = pv["post"]["id"]
                text = pv["post"]["name"] + "\n" + pv["post"].get("body", "")
                self.db.save_source(post_id, None, text)
                posts.append({
                    "post_id": post_id,
                    "text": text,
                    "parent_id": None,
                })
            self._attempt_replies(posts)

            cfeed = self.lemmy.comment.list(
                community_id=self.community_id,
                sort=SortType.New,
                page=1,
                limit=self.max_replies * 3,
            )
            comments = []
            for cv in iter_comment_views(cfeed):
                post_id = cv["comment"]["post_id"]
                parent_id = cv["comment"]["id"]
                text = cv["comment"]["content"]
                self.db.save_source(post_id, parent_id, text)
                comments.append({
                    "post_id": post_id,
                    "text": text,
                    "parent_id": parent_id,
                })
            self._attempt_replies(comments)

            time.sleep(5)


# -------------------- entrypoint -------------------- #
def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    fallback_path = cfg.get("fallback_file", "fallbacks_expanded.yaml")
    history_size = int(cfg.get("fallback_history_size", 5))
    fm = FallbackManager(fallback_path, history_size=history_size)

    log_dir = Path(cfg.get("log_dir", "logs"))
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if cfg.get("debug") else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_dir / f"asslips17c_{int(time.time())}.log", encoding="utf-8")],
    )
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(console)

    threads = [BotThread(MappingProxyType(b), MappingProxyType(cfg), fm) for b in cfg["bots"]]
    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[CTRL-C] Shutting down...")
        for t in threads:
            t.stop_event.set()
        for t in threads:
            t.join()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config.yaml")
    main(ap.parse_args().config)
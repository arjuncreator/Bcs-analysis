"""
BCS Prep Tracker v2 — প্রিলি-কেন্দ্রিক বিসিএস প্রস্তুতি ট্র্যাকার
Stack: Streamlit + SQLite (local) + Google Gemini API

Run:  streamlit run app.py
"""

import json
import os
import re
import sqlite3
from datetime import date, timedelta
from hashlib import md5
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ----------------------------------------------------------------------------
# কনফিগ
# ----------------------------------------------------------------------------
st.set_page_config(page_title="BCS Prep Tracker", page_icon="📚", layout="wide")

DB_PATH = os.environ.get(
    "BCS_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "bcs_tracker.db"),
)

# gemini-1.5-flash পুরনো; মডেল বদলাতে secrets/env এ GEMINI_MODEL সেট করুন।
DEFAULT_MODEL = "gemini-2.5-flash"

WEAK_THRESHOLD = 55.0       # এর নিচে গড় হলে Weakness Alert
DEFAULT_NEG = 0.5           # বিসিএস প্রিলিতে প্রতি ভুল উত্তরে কাটা নম্বর
ALL_SUBJECTS_LABEL = "সকল বিষয় (ফুল মডেল)"
SEP = "|"

EXAM_TYPES = {
    "topic_based": "টপিক বেজড (সাবজেক্ট + টপিক)",
    "revision": "সাপ্তাহিক / মিক্সড রিভিশন",
    "subject_test": "সাবজেক্ট টেস্ট",
    "full_model": "ফুল মডেল টেস্ট (২০০ মার্কস)",
    "important_topic": "গুরুত্বপূর্ণ টপিক এক্সাম",
}

SUBJECTS = [
    "বাংলা ভাষা ও সাহিত্য",
    "English Language & Literature",
    "বাংলাদেশ বিষয়াবলি",
    "আন্তর্জাতিক বিষয়াবলি",
    "ভূগোল, পরিবেশ ও দুর্যোগ ব্যবস্থাপনা",
    "সাধারণ বিজ্ঞান",
    "কম্পিউটার ও তথ্যপ্রযুক্তি",
    "গাণিতিক যুক্তি",
    "মানসিক দক্ষতা",
    "নৈতিকতা, মূল্যবোধ ও সুশাসন",
]

# পুরনো ডেটাবেসে এই কলামগুলো নিজে থেকে যোগ হবে (আগের ডেটা অক্ষত থাকবে)
NEW_COLUMNS = {
    "correct": "INTEGER",
    "wrong": "INTEGER",
    "skipped": "INTEGER",
    "time_minutes": "REAL",
    "neg_mark": "REAL",
    "m_unknown": "INTEGER",
    "m_careless": "INTEGER",
    "m_time": "INTEGER",
}


# ----------------------------------------------------------------------------
# ডেটাবেস
# ----------------------------------------------------------------------------
def db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with db_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS exams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                exam_date TEXT NOT NULL,
                exam_type TEXT NOT NULL,
                subjects TEXT NOT NULL,
                topics TEXT DEFAULT '',
                total_marks REAL NOT NULL,
                obtained_marks REAL NOT NULL,
                cutoff_marks REAL NOT NULL DEFAULT 0,
                gap REAL NOT NULL,
                percentage REAL NOT NULL,
                notes TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_exams_user ON exams(user_id, exam_date);
            """
        )
        existing = {r[1] for r in conn.execute("PRAGMA table_info(exams)")}
        for col, typ in NEW_COLUMNS.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE exams ADD COLUMN {col} {typ}")


def get_or_create_user(name: str) -> int:
    name = name.strip()
    with db_conn() as conn:
        row = conn.execute("SELECT id FROM users WHERE name = ?", (name,)).fetchone()
        if row:
            return row[0]
        cur = conn.execute("INSERT INTO users (name) VALUES (?)", (name,))
        return cur.lastrowid


def insert_exam(user_id, exam_date, exam_type, subjects, topics,
                total, obtained, cutoff, notes, extra=None):
    extra = extra or {}
    gap = round(obtained - cutoff, 2)
    pct = round(obtained / total * 100, 2)
    with db_conn() as conn:
        conn.execute(
            """INSERT INTO exams
               (user_id, exam_date, exam_type, subjects, topics, total_marks,
                obtained_marks, cutoff_marks, gap, percentage, notes,
                correct, wrong, skipped, time_minutes, neg_mark,
                m_unknown, m_careless, m_time)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (user_id, str(exam_date), exam_type, subjects, topics, total,
             obtained, cutoff, gap, pct, notes,
             extra.get("correct"), extra.get("wrong"), extra.get("skipped"),
             extra.get("time_minutes"), extra.get("neg_mark"),
             extra.get("m_unknown"), extra.get("m_careless"), extra.get("m_time")),
        )


def fetch_exams(user_id: int) -> pd.DataFrame:
    with db_conn() as conn:
        df = pd.read_sql_query(
            "SELECT id, exam_date, exam_type, subjects, topics, total_marks, "
            "obtained_marks, cutoff_marks, gap, percentage, notes, "
            "correct, wrong, skipped, time_minutes, neg_mark, "
            "m_unknown, m_careless, m_time "
            "FROM exams WHERE user_id = ? ORDER BY exam_date, id",
            conn,
            params=(user_id,),
        )
    if not df.empty:
        df["exam_date"] = pd.to_datetime(df["exam_date"])
        df["type_label"] = df["exam_type"].map(EXAM_TYPES).fillna(df["exam_type"])
    return df


def delete_exam(user_id: int, exam_id: int) -> bool:
    with db_conn() as conn:
        cur = conn.execute(
            "DELETE FROM exams WHERE id = ? AND user_id = ?", (exam_id, user_id)
        )
        return cur.rowcount > 0


# ----------------------------------------------------------------------------
# Gemini
# ----------------------------------------------------------------------------
def secret_or_env(name: str, default: str = "") -> str:
    try:
        val = st.secrets.get(name, "")
    except Exception:
        val = ""
    return val or os.environ.get(name, default)


def get_api_key() -> str:
    return st.session_state.get("api_key_input", "").strip() or secret_or_env(
        "GEMINI_API_KEY"
    )


def get_model_name() -> str:
    return secret_or_env("GEMINI_MODEL", DEFAULT_MODEL)


def gemini_client():
    from google import genai

    key = get_api_key()
    if not key:
        raise RuntimeError(
            "Gemini API Key পাওয়া যায়নি। সাইডবারে Key দিন অথবা GEMINI_API_KEY সেট করুন।"
        )
    return genai.Client(api_key=key)


def parse_json_loose(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


EXTRACT_PROMPT = """তুমি একটি পরীক্ষার রেজাল্ট স্ক্রিনশট বিশ্লেষণকারী। ছবিটি দেখে নিচের তথ্য বের করো এবং
শুধুমাত্র একটি JSON অবজেক্ট ফেরত দাও (কোনো অতিরিক্ত লেখা নয়):

{
  "exam_type": "topic_based | revision | subject_test | full_model | important_topic | null",
  "subjects": ["সাবজেক্টের নাম", "..."],
  "topics": ["টপিকের নাম", "..."],
  "total_marks": number | null,
  "obtained_marks": number | null,
  "cut_off_marks": number | null,
  "correct_answers": number | null,
  "wrong_answers": number | null,
  "skipped_answers": number | null,
  "time_minutes": number | null
}

নিয়ম:
- exam_type: একটি সাবজেক্ট + টপিক হলে topic_based; একাধিক সাবজেক্টের সাপ্তাহিক/মিক্সড হলে revision;
  শুধু একটি সাবজেক্টের পুরো পরীক্ষা হলে subject_test; ২০০ মার্কসের পূর্ণ মডেল টেস্ট হলে full_model;
  গুরুত্বপূর্ণ/ইম্পরট্যান্ট টপিকের পরীক্ষা হলে important_topic। নিশ্চিত না হলে null।
- ছবিতে যে তথ্য নেই বা অস্পষ্ট, সেটির মান null (সংখ্যার ক্ষেত্রে) বা খালি লিস্ট দাও। অনুমান করে বানিও না।
- সংখ্যা অবশ্যই ইংরেজি ডিজিটে (বাংলা সংখ্যা থাকলে রূপান্তর করো)।
- obtained_marks হলো পরীক্ষার্থীর প্রাপ্ত নম্বর (নেগেটিভ মার্কিং কাটার পরের মোট স্কোর)।
- correct/wrong/skipped হলো সঠিক, ভুল ও উত্তর না দেওয়া প্রশ্নের সংখ্যা।
"""


def extract_from_image(image: Image.Image) -> dict:
    from google.genai import types

    client = gemini_client()
    resp = client.models.generate_content(
        model=get_model_name(),
        contents=[EXTRACT_PROMPT, image],
        config=types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0.0
        ),
    )
    return parse_json_loose(resp.text)


def ai_guidance(summary: str) -> str:
    client = gemini_client()
    prompt = (
        "তুমি একজন অভিজ্ঞ বিসিএস মেন্টর। নিচে একজন প্রার্থীর মক টেস্ট পারফরম্যান্সের সারাংশ দেওয়া হলো। "
        "লক্ষ্য: বিসিএস প্রিলিমিনারি পাস করা। বাংলায়, সংক্ষেপে (সর্বোচ্চ ৩০০ শব্দ), বাস্তবসম্মত ও নির্দিষ্ট "
        "দিকনির্দেশনা দাও: ১) বর্তমান অবস্থার এক লাইনের মূল্যায়ন, ২) নেগেটিভ মার্কিং ও অ্যাটেম্পট কৌশল, "
        "৩) সবচেয়ে জরুরি ৩টি কাজ (নির্দিষ্ট সাবজেক্ট/টপিক উল্লেখ করে), ৪) আগামী ৭ দিনের সংক্ষিপ্ত রুটিন-ইঙ্গিত। "
        "অতিরঞ্জন বা মিথ্যা আশ্বাস দিও না; ডেটা কম হলে সেটাও উল্লেখ করো।\n\n" + summary
    )
    resp = client.models.generate_content(model=get_model_name(), contents=prompt)
    return resp.text.strip()


# ----------------------------------------------------------------------------
# অ্যানালিটিক্স হেল্পার
# ----------------------------------------------------------------------------
def split_items(raw: str) -> list:
    parts = re.split(r"[,\n|،;]+", raw or "")
    return [p.strip() for p in parts if p.strip()]


def explode_items(df: pd.DataFrame, col: str, exclude=()) -> pd.DataFrame:
    if df.empty:
        return df.assign(item=[])
    tmp = df.assign(item=df[col].fillna("").str.split(SEP)).explode("item")
    tmp["item"] = tmp["item"].fillna("").str.strip()
    tmp = tmp[(tmp["item"] != "") & (~tmp["item"].isin(exclude))]
    return tmp


def readiness_score(df: pd.DataFrame) -> float:
    d = df.sort_values(["exam_date", "id"])
    recent = d.tail(5)["percentage"].mean()
    overall = d["percentage"].mean()
    pass_rate = (d["gap"] >= 0).mean() * 100
    score = 0.5 * recent + 0.3 * overall + 0.2 * pass_rate
    return round(max(0.0, min(100.0, score)), 1)


def readiness_label(score: float) -> str:
    if score < 40:
        return "শুরুর ধাপ — ভিত্তি শক্ত করার সময়"
    if score < 55:
        return "উন্নতি দরকার — নিয়মিততা বাড়ান"
    if score < 70:
        return "ভালো পথে আছেন — দুর্বল জায়গাগুলো ঠিক করুন"
    return "শক্ত প্রস্তুতি — ধারাবাহিকতা ধরে রাখুন"


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("type_label").agg(
        পরীক্ষা=("id", "count"),
        গড়_শতাংশ=("percentage", "mean"),
        গড়_গ্যাপ=("gap", "mean"),
        সর্বোচ্চ_শতাংশ=("percentage", "max"),
    )
    last = df.sort_values(["exam_date", "id"]).groupby("type_label")["percentage"].last()
    g["সর্বশেষ_শতাংশ"] = last
    return g.round(2).reset_index().rename(columns={"type_label": "ক্যাটাগরি"})


def subject_stats(df: pd.DataFrame) -> pd.DataFrame:
    ex = explode_items(df, "subjects", exclude=(ALL_SUBJECTS_LABEL,))
    if ex.empty:
        return pd.DataFrame(columns=["সাবজেক্ট", "পরীক্ষা", "গড় %", "গড় Gap"])
    g = ex.groupby("item").agg(
        পরীক্ষা=("id", "nunique"), গড়=("percentage", "mean"), গ্যাপ=("gap", "mean")
    )
    g = g.round(2).reset_index().rename(
        columns={"item": "সাবজেক্ট", "গড়": "গড় %", "গ্যাপ": "গড় Gap"}
    )
    return g.sort_values("গড় %")


def topic_stats(df: pd.DataFrame) -> pd.DataFrame:
    ex = explode_items(df, "topics")
    if ex.empty:
        return pd.DataFrame(columns=["টপিক", "পরীক্ষা", "গড় %"])
    g = ex.groupby("item").agg(পরীক্ষা=("id", "nunique"), গড়=("percentage", "mean"))
    g = g.round(2).reset_index().rename(columns={"item": "টপিক", "গড়": "গড় %"})
    return g.sort_values("গড় %")


# ---- প্রিলি-ডিটেইল (সঠিক/ভুল/বাদ) ----
def detail_frame(df: pd.DataFrame) -> pd.DataFrame:
    """যেসব পরীক্ষায় সঠিক/ভুল/বাদ দেওয়া আছে, শুধু সেগুলো + হিসাব করা কলাম।"""
    if df.empty:
        return df
    d = df[df["correct"].notna() & df["wrong"].notna()].copy()
    if d.empty:
        return d
    d["skipped"] = d["skipped"].fillna(0)
    d["neg_mark"] = d["neg_mark"].fillna(DEFAULT_NEG)
    d["q"] = d["correct"] + d["wrong"] + d["skipped"]
    d = d[d["q"] > 0].copy()
    d["mpq"] = d["total_marks"] / d["q"]
    d["neg_lost"] = d["wrong"] * d["neg_mark"] * d["mpq"]
    d["gross_pct"] = d["correct"] * d["mpq"] / d["total_marks"] * 100
    d["attempt_rate"] = (d["correct"] + d["wrong"]) / d["q"] * 100
    att = d["correct"] + d["wrong"]
    d["accuracy"] = np.where(att > 0, d["correct"] / att.replace(0, np.nan) * 100, np.nan)
    d["sec_per_q"] = np.where(
        d["time_minutes"].notna(), d["time_minutes"] * 60 / d["q"], np.nan
    )
    return d


def breakeven_accuracy(neg: float) -> float:
    """অনুমান করে উত্তর দিলে কত % সঠিক হলে লাভ: neg/(1+neg)।"""
    return neg / (1 + neg) * 100


def build_units(df: pd.DataFrame) -> pd.DataFrame:
    """রিভিশন/প্রায়োরিটির জন্য 'ইউনিট' (টপিক, না থাকলে সাবজেক্ট) আলাদা সারিতে।"""
    rows = []
    for r in df.itertuples():
        subs = [s for s in (r.subjects or "").split(SEP) if s and s != ALL_SUBJECTS_LABEL]
        tops = [t for t in (r.topics or "").split(SEP) if t]
        if tops:
            for t in tops:
                label = f"{t} — {subs[0]}" if len(subs) == 1 else t
                rows.append((label, r.exam_date, r.percentage, r.gap, r.id))
        else:
            for s in subs:
                rows.append((s, r.exam_date, r.percentage, r.gap, r.id))
    return pd.DataFrame(rows, columns=["unit", "exam_date", "percentage", "gap", "id"])


def interval_days(pct: float) -> int:
    if pct < 55:
        return 2
    if pct < 70:
        return 5
    if pct < 85:
        return 10
    return 21


def unit_table(df: pd.DataFrame, today: date = None) -> pd.DataFrame:
    today = pd.Timestamp(today or date.today())
    u = build_units(df)
    if u.empty:
        return pd.DataFrame()
    rows = []
    for unit, g in u.sort_values(["exam_date", "id"]).groupby("unit"):
        pcts = g["percentage"].tolist()
        n = len(pcts)
        avg = float(np.mean(pcts))
        last_pct = float(np.mean(pcts[-2:]))
        decline = max(0.0, pcts[-2] - pcts[-1]) if n >= 2 else 0.0
        last_date = g["exam_date"].max()
        stale = max(0, (today - last_date).days)
        priority = 0.6 * (100 - avg) + 0.2 * min(decline * 2, 100) + 0.2 * min(stale / 30, 1) * 100
        due = last_date + pd.Timedelta(days=interval_days(last_pct))
        rows.append({
            "unit": unit, "n": n, "avg": round(avg, 1), "last_pct": round(pcts[-1], 1),
            "last_date": last_date, "stale_days": stale,
            "priority": round(priority, 1), "due": due,
            "days_to_due": (due - today).days,
        })
    return pd.DataFrame(rows).sort_values("priority", ascending=False).reset_index(drop=True)


def projection(df: pd.DataFrame, days_ahead: int = 30):
    """শেষ ১০টি পরীক্ষার রৈখিক ট্রেন্ড থেকে আনুমানিক ভবিষ্যৎ স্কোর। যথেষ্ট ডেটা না থাকলে None।"""
    d = df.sort_values(["exam_date", "id"]).tail(10)
    if len(d) < 5:
        return None
    x = d["exam_date"].map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    if np.ptp(x) < 7:
        return None
    y = d["percentage"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    future = x.max() + days_ahead
    pred = float(np.clip(slope * future + intercept, 0, 100))
    return {"slope_per_week": slope * 7, "pred": pred, "n": len(d)}


def consistency(df: pd.DataFrame):
    d = df.sort_values(["exam_date", "id"]).tail(10)
    if len(d) < 4:
        return None
    return float(d["percentage"].std(ddof=0))


def prelim_insights(df: pd.DataFrame, dd: pd.DataFrame) -> list:
    tips = []
    if dd.empty:
        return tips
    C, W, S = dd["correct"].sum(), dd["wrong"].sum(), dd["skipped"].sum()
    att = C + W
    acc = C / att * 100 if att else 0
    attempt = att / (C + W + S) * 100
    neg = float(dd["neg_mark"].mean())
    be = breakeven_accuracy(neg)
    lost = dd["neg_lost"].sum()
    tot = dd["total_marks"].sum()
    if acc < be:
        tips.append(f"🚨 অ্যাকুরেসি {acc:.0f}% — ব্রেক-ইভেন {be:.0f}%-এর নিচে। অর্থাৎ গড়ে অনুমান করে উত্তর দেওয়া আপনার স্কোর কমাচ্ছে; নিশ্চিত না হলে বাদ দিন।")
    elif acc < 70 and attempt > 90:
        tips.append(f"⚠️ প্রায় সব প্রশ্নে উত্তর দিচ্ছেন ({attempt:.0f}%) কিন্তু অ্যাকুরেসি {acc:.0f}%। নেগেটিভ মার্কিং থেকে বাঁচতে দ্বিধার প্রশ্নে আরও বেছে নিন।")
    if lost / tot * 100 >= 8:
        tips.append(f"➖ নেগেটিভ মার্কিংয়ে মোট নম্বরের {lost / tot * 100:.1f}% হারাচ্ছেন। এটা কমালেই স্কোর সরাসরি বাড়বে।")
    if attempt < 70 and acc >= 75:
        tips.append(f"📈 অ্যাকুরেসি ভালো ({acc:.0f}%) কিন্তু অ্যাটেম্পট মাত্র {attempt:.0f}%। যেসব প্রশ্নে ৫০%+ নিশ্চিত, সেগুলোতে ঝুঁকি নেওয়ার সুযোগ আছে।")
    mu, mc, mt = dd["m_unknown"].fillna(0).sum(), dd["m_careless"].fillna(0).sum(), dd["m_time"].fillna(0).sum()
    tagged = mu + mc + mt
    if tagged >= 10:
        top = max((mu, "অজানা প্রশ্ন (কনটেন্ট ঘাটতি)"), (mc, "অসাবধানতা/ভুল পড়া"), (mt, "সময়ের চাপ"))
        tips.append(f"🔎 ভুলের সবচেয়ে বড় কারণ: {top[1]} ({top[0] / tagged * 100:.0f}%)।")
    spq = dd["sec_per_q"].dropna()
    if len(spq) >= 3:
        recent, older = spq.tail(3).mean(), spq.head(max(1, len(spq) - 3)).mean()
        if recent > older * 1.2:
            tips.append("⏱️ সাম্প্রতিক পরীক্ষাগুলোতে প্রতি প্রশ্নে সময় বেড়েছে; সময় ধরে অনুশীলন করুন।")
    return tips


def rule_based_guidance(df, subj, topics, score) -> list:
    tips = []
    n = len(df)
    if n < 5:
        tips.append(f"এখন পর্যন্ত {n}টি পরীক্ষার ডেটা আছে। অন্তত ৫–১০টি পরীক্ষার পর বিশ্লেষণ আরও নির্ভরযোগ্য হবে।")
    d = df.sort_values(["exam_date", "id"])
    if n >= 6:
        last3, prev3 = d.tail(3)["percentage"].mean(), d.iloc[-6:-3]["percentage"].mean()
        diff = last3 - prev3
        if diff >= 3:
            tips.append(f"✅ সাম্প্রতিক ৩টি পরীক্ষায় আগের ৩টির চেয়ে গড়ে {diff:.1f}% উন্নতি হয়েছে। এই রুটিন ধরে রাখুন।")
        elif diff <= -3:
            tips.append(f"📉 সাম্প্রতিক ৩টি পরীক্ষায় আগের ৩টির চেয়ে গড়ে {abs(diff):.1f}% পতন হয়েছে। ভুলগুলো বিশ্লেষণ করে রিভিশন বাড়ান।")
    weak_sub = subj[subj["গড় %"] < WEAK_THRESHOLD] if not subj.empty else subj
    if not weak_sub.empty:
        names = ", ".join(weak_sub["সাবজেক্ট"].head(3))
        tips.append(f"🎯 অগ্রাধিকার: {names} — এগুলোতে প্রতিদিন নির্দিষ্ট সময় বরাদ্দ করুন।")
    weak_top = topics[(topics["গড় %"] < WEAK_THRESHOLD)] if not topics.empty else topics
    if not weak_top.empty:
        names = ", ".join(weak_top["টপিক"].head(5))
        tips.append(f"🔁 এই টপিকগুলো আবার পড়ে টপিক-বেজড পরীক্ষা দিন: {names}")
    neg_gap = (df["gap"] < 0).mean() * 100
    if neg_gap > 50:
        tips.append(f"⚠️ {neg_gap:.0f}% পরীক্ষায় কাট-অফের নিচে আছেন।")
    if score >= 70 and not tips:
        tips.append("চমৎকার! ফুল মডেল টেস্টের সংখ্যা বাড়িয়ে পরীক্ষার পরিবেশে অভ্যস্ত হোন।")
    if not tips:
        tips.append("পারফরম্যান্স স্থিতিশীল আছে। নিয়মিত রিভিশন ও মডেল টেস্ট চালিয়ে যান।")
    return tips


# ----------------------------------------------------------------------------
# লগইন (বুকমার্ক করা লিংকে ?user=নাম থাকলে সরাসরি ঢুকে পড়ে)
# ----------------------------------------------------------------------------
def do_login(name: str):
    st.session_state["user_id"] = get_or_create_user(name)
    st.session_state["user_name"] = name.strip()
    try:
        st.query_params["user"] = name.strip()
    except Exception:
        pass


def login_screen():
    st.title("📚 BCS Prep Tracker")
    st.caption("আপনার ব্যক্তিগত বিসিএস প্রস্তুতি ট্র্যাকার — ডেটা থাকে আপনার নিজের লোকাল SQLite ডেটাবেসে।")
    with st.form("login_form"):
        name = st.text_input("আপনার নাম বা আইডি", placeholder="যেমন: rahim")
        ok = st.form_submit_button("ড্যাশবোর্ডে প্রবেশ করুন", type="primary")
    if ok:
        if len(name.strip()) < 2:
            st.error("কমপক্ষে ২ অক্ষরের নাম বা আইডি দিন।")
        else:
            do_login(name)
            st.rerun()
    st.info(
        "একই নাম দিলে আপনার আগের সব ডেটা ফিরে আসে। লগইনের পর ব্রাউজারের লিংকটি বুকমার্ক করে রাখুন — "
        "পরে সেটি খুললে নাম না লিখেই ঢুকতে পারবেন। (পাসওয়ার্ড নেই; এটি ব্যক্তিগত ব্যবহারের জন্য।)"
    )


# ----------------------------------------------------------------------------
# পেজ: পরীক্ষা যোগ করুন
# ----------------------------------------------------------------------------
FORM_DEFAULTS = {
    "f_type": "topic_based",
    "f_date": None,  # নিচে date.today() বসবে
    "f_subject": SUBJECTS[0],
    "f_subjects": [],
    "f_topics": "",
    "f_total": 100.0,
    "f_obtained": 0.0,
    "f_cutoff": 0.0,
    "f_notes": "",
    "f_detail": False,
    "f_correct": 0,
    "f_wrong": 0,
    "f_skipped": 0,
    "f_neg": DEFAULT_NEG,
    "f_time": 0.0,
    "f_m_unknown": 0,
    "f_m_careless": 0,
    "f_m_time": 0,
}


def init_form_state():
    ss = st.session_state
    ss.setdefault("extra_subjects", [])
    pending = ss.pop("_prefill", None)
    if pending:
        ss.update(pending)
    for k, v in FORM_DEFAULTS.items():
        ss.setdefault(k, date.today() if k == "f_date" else v)


def on_type_change():
    if st.session_state.get("f_type") == "full_model":
        st.session_state["f_total"] = 200.0


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def to_int(v):
    f = to_float(v)
    return None if f is None else int(round(f))


def build_prefill(data: dict) -> dict:
    ss = st.session_state
    pre = {}
    code = data.get("exam_type")
    if code in EXAM_TYPES:
        pre["f_type"] = code
    subs = data.get("subjects") or []
    if isinstance(subs, str):
        subs = split_items(subs)
    subs = [str(s).strip() for s in subs if str(s).strip()]
    for s in subs:
        if s not in SUBJECTS and s not in ss["extra_subjects"]:
            ss["extra_subjects"].append(s)
    if subs:
        pre["f_subjects"] = subs
        pre["f_subject"] = subs[0]
    tops = data.get("topics") or []
    if isinstance(tops, str):
        tops = split_items(tops)
    if tops:
        pre["f_topics"] = ", ".join(str(t).strip() for t in tops if str(t).strip())
    for key, field in (("total_marks", "f_total"), ("obtained_marks", "f_obtained"),
                       ("cut_off_marks", "f_cutoff"), ("time_minutes", "f_time")):
        val = to_float(data.get(key))
        if val is not None:
            pre[field] = val
    c, w, s = (to_int(data.get("correct_answers")), to_int(data.get("wrong_answers")),
               to_int(data.get("skipped_answers")))
    if c is not None and w is not None:
        pre["f_detail"] = True
        pre["f_correct"], pre["f_wrong"], pre["f_skipped"] = c, w, s or 0
    if pre.get("f_type") == "full_model" and "f_total" not in pre:
        pre["f_total"] = 200.0
    return pre


def page_add_exam(user_id: int):
    init_form_state()
    ss = st.session_state
    st.header("➕ নতুন পরীক্ষার ফলাফল যোগ করুন")

    if ss.get("flash"):
        st.success(ss.pop("flash"))

    with st.expander("🤖 রেজাল্ট স্ক্রিনশট থেকে অটো-ফিল (Gemini)", expanded=False):
        up = st.file_uploader("স্ক্রিনশট আপলোড করুন", type=["png", "jpg", "jpeg", "webp"])
        if up is not None:
            try:
                img = Image.open(BytesIO(up.getvalue())).convert("RGB")
                img.thumbnail((2200, 2200))
                st.image(img, caption="আপলোড করা ছবি", width=360)
                if st.button("ছবি পড়ে ফর্ম পূরণ করুন", type="primary"):
                    with st.spinner("Gemini ছবি পড়ছে..."):
                        data = extract_from_image(img)
                    ss["_prefill"] = build_prefill(data)
                    ss["flash"] = "ফর্ম পূরণ হয়েছে। সংরক্ষণের আগে নিচের তথ্য মিলিয়ে নিন।"
                    st.rerun()
            except Exception as e:
                st.error(f"স্ক্রিনশট পড়তে সমস্যা হয়েছে: {e}")

    subject_options = list(dict.fromkeys(SUBJECTS + ss["extra_subjects"]))

    with st.expander("➕ তালিকায় নেই এমন সাবজেক্ট যোগ করুন"):
        c1, c2 = st.columns([3, 1])
        new_sub = c1.text_input("নতুন সাবজেক্টের নাম", key="new_subject_name", label_visibility="collapsed")
        if c2.button("যোগ করুন"):
            if new_sub.strip() and new_sub.strip() not in subject_options:
                ss["extra_subjects"].append(new_sub.strip())
                st.rerun()

    col_a, col_b = st.columns(2)
    exam_code = col_a.selectbox(
        "পরীক্ষার ধরন", list(EXAM_TYPES), format_func=EXAM_TYPES.get,
        key="f_type", on_change=on_type_change,
    )
    exam_date = col_b.date_input("পরীক্ষার তারিখ", key="f_date")

    topics_raw = ""
    if exam_code == "full_model":
        subjects_list = [ALL_SUBJECTS_LABEL]
        st.caption("ফুল মডেল টেস্টে সাবজেক্ট আলাদা করে বাছাই করতে হবে না।")
    elif exam_code == "revision":
        for s in ss["f_subjects"]:
            if s not in subject_options:
                subject_options.append(s)
        subjects_list = st.multiselect("সাবজেক্টসমূহ", subject_options, key="f_subjects")
        topics_raw = st.text_area("টপিকসমূহ (কমা দিয়ে আলাদা করুন)", key="f_topics", height=80)
    elif exam_code == "subject_test":
        if ss["f_subject"] not in subject_options:
            subject_options.append(ss["f_subject"])
        subjects_list = [st.selectbox("সাবজেক্ট", subject_options, key="f_subject")]
    else:
        if ss["f_subject"] not in subject_options:
            subject_options.append(ss["f_subject"])
        subjects_list = [st.selectbox("সাবজেক্ট", subject_options, key="f_subject")]
        topics_raw = st.text_input("টপিক (একাধিক হলে কমা দিন)", key="f_topics")

    m1, m3 = st.columns(2)
    total = m1.number_input("মোট নম্বর (কত নম্বরের পরীক্ষা)", min_value=1.0, step=1.0, key="f_total")
    cutoff = m3.number_input("কাট-অফ মার্কস", min_value=0.0, max_value=float(total), step=0.25, key="f_cutoff")

    st.checkbox(
        "প্রশ্নভিত্তিক বিস্তারিত দেব (সঠিক / ভুল / বাদ) — প্রিলি অ্যানালাইসিসের জন্য সুপারিশকৃত",
        key="f_detail",
    )

    extra = {}
    if ss["f_detail"]:
        d1, d2, d3, d4, d5 = st.columns(5)
        correct = d1.number_input("সঠিক", min_value=0, step=1, key="f_correct")
        wrong = d2.number_input("ভুল", min_value=0, step=1, key="f_wrong")
        skipped = d3.number_input("বাদ দিয়েছি", min_value=0, step=1, key="f_skipped")
        neg = d4.number_input("ভুলে কাটা (প্রতি প্রশ্নে)", min_value=0.0, step=0.05, key="f_neg",
                              help="বিসিএস প্রিলিতে সাধারণত ০.৫")
        tmin = d5.number_input("সময় (মিনিট)", min_value=0.0, step=1.0, key="f_time")

        q = int(correct + wrong + skipped)
        mpq = total / q if q > 0 else 0.0
        obtained = round((correct - wrong * neg) * mpq, 2)
        st.caption(f"মোট প্রশ্ন: {q} | প্রতি প্রশ্নে {mpq:.2f} নম্বর | প্রাপ্ত নম্বর স্বয়ংক্রিয়ভাবে হিসাব হয়েছে।")

        if wrong > 0:
            st.markdown("**ভুলগুলোর কারণ (ঐচ্ছিক, মোট ভুলের বেশি হবে না)**")
            e1, e2, e3 = st.columns(3)
            mu = e1.number_input("অজানা ছিল", min_value=0, step=1, key="f_m_unknown")
            mc = e2.number_input("অসাবধানতা/ভুল পড়া", min_value=0, step=1, key="f_m_careless")
            mt = e3.number_input("সময়ের অভাব/তাড়াহুড়া", min_value=0, step=1, key="f_m_time")
        else:
            mu = mc = mt = 0
        extra = {
            "correct": int(correct), "wrong": int(wrong), "skipped": int(skipped),
            "time_minutes": float(tmin) if tmin > 0 else None, "neg_mark": float(neg),
            "m_unknown": int(mu), "m_careless": int(mc), "m_time": int(mt),
        }
    else:
        obtained = st.number_input(
            "প্রাপ্ত নম্বর", min_value=-float(total), max_value=float(total), step=0.25, key="f_obtained"
        )

    gap = obtained - cutoff
    pct = obtained / total * 100 if total > 0 else 0.0
    p1, p2, p3 = st.columns(3)
    p1.metric("প্রাপ্ত নম্বর", f"{obtained:.2f}")
    p2.metric("Gap Marks (প্রাপ্ত − কাট-অফ)", f"{gap:+.2f}")
    p3.metric("Percentage", f"{pct:.2f}%")

    notes = st.text_input("নোট (ঐচ্ছিক) — যেমন: ভুলের কারণ, সময় সমস্যা", key="f_notes")

    if st.button("💾 সংরক্ষণ করুন", type="primary"):
        err = None
        if exam_code != "full_model" and not subjects_list:
            err = "কমপক্ষে একটি সাবজেক্ট বাছাই করুন।"
        elif exam_code in ("topic_based", "important_topic") and not topics_raw.strip():
            err = "এই ধরনের পরীক্ষার জন্য টপিক লিখুন।"
        elif ss["f_detail"]:
            if extra["correct"] + extra["wrong"] + extra["skipped"] == 0:
                err = "সঠিক/ভুল/বাদ — অন্তত একটিতে সংখ্যা দিন।"
            elif extra["m_unknown"] + extra["m_careless"] + extra["m_time"] > extra["wrong"]:
                err = "ভুলের কারণগুলোর যোগফল মোট ভুলের চেয়ে বেশি হতে পারে না।"
        if err:
            st.error(err)
        else:
            insert_exam(
                user_id, exam_date, exam_code,
                SEP.join(subjects_list), SEP.join(split_items(topics_raw)),
                float(total), float(obtained), float(cutoff), notes.strip(), extra,
            )
            ss["_prefill"] = {
                "f_topics": "", "f_obtained": 0.0, "f_cutoff": 0.0, "f_notes": "",
                "f_subjects": [], "f_correct": 0, "f_wrong": 0, "f_skipped": 0,
                "f_time": 0.0, "f_m_unknown": 0, "f_m_careless": 0, "f_m_time": 0,
            }
            ss["flash"] = f"✅ পরীক্ষা সংরক্ষিত হয়েছে — {pct:.1f}% | Gap {gap:+.2f}"
            st.rerun()


# ----------------------------------------------------------------------------
# পেজ: ড্যাশবোর্ড
# ----------------------------------------------------------------------------
def build_ai_summary(df, dd, score, subj, tops, units) -> str:
    d = df.sort_values(["exam_date", "id"])
    parts = [
        f"রেডিনেস স্কোর: {score}%",
        f"মোট পরীক্ষা: {len(df)}",
        f"ক্যাটাগরি সারাংশ:\n{category_summary(df).to_string(index=False)}",
        f"সাবজেক্ট গড় (কম থেকে বেশি):\n{subj.to_string(index=False) if not subj.empty else 'ডেটা নেই'}",
        f"দুর্বল টপিক:\n{tops.head(8).to_string(index=False) if not tops.empty else 'ডেটা নেই'}",
        f"সর্বশেষ ৫টি পরীক্ষার %: {d.tail(5)['percentage'].round(1).tolist()}",
        f"কাট-অফের উপরে থাকার হার: {(df['gap'] >= 0).mean() * 100:.0f}%",
    ]
    if not dd.empty:
        C, W, S = dd["correct"].sum(), dd["wrong"].sum(), dd["skipped"].sum()
        parts.append(
            f"প্রশ্নভিত্তিক ({len(dd)}টি পরীক্ষা): সঠিক {int(C)}, ভুল {int(W)}, বাদ {int(S)}; "
            f"অ্যাটেম্পট {(C + W) / (C + W + S) * 100:.0f}%, অ্যাকুরেসি {C / max(C + W, 1) * 100:.0f}%; "
            f"নেগেটিভে হারানো নম্বর: {dd['neg_lost'].sum():.1f}"
        )
    if not units.empty:
        parts.append("অগ্রাধিকার তালিকা (শীর্ষ ৫): " + "; ".join(
            f"{r.unit} (গড় {r.avg}%)" for r in units.head(5).itertuples()))
    pr = projection(df)
    if pr:
        parts.append(f"ট্রেন্ড: সপ্তাহে {pr['slope_per_week']:+.1f}%, ৩০ দিন পর আনুমানিক {pr['pred']:.0f}%")
    return "\n".join(parts)


def page_dashboard(user_id: int):
    st.header("📊 ড্যাশবোর্ড ও অ্যানালিটিক্স")
    df = fetch_exams(user_id)
    if df.empty:
        st.info("এখনো কোনো পরীক্ষার ডেটা নেই। মেনু থেকে 'পরীক্ষা যোগ করুন' পেজে গিয়ে প্রথম ফলাফল দিন।")
        return
    dd = detail_frame(df)

    score = readiness_score(df)
    left, right = st.columns([1, 1.4])
    with left:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score, number={"suffix": "%"},
            title={"text": "সার্বিক বিসিএস রেডিনেস স্কোর"},
            gauge={
                "axis": {"range": [0, 100]}, "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 40], "color": "#fecaca"},
                    {"range": [40, 55], "color": "#fde68a"},
                    {"range": [55, 70], "color": "#bbf7d0"},
                    {"range": [70, 100], "color": "#86efac"},
                ],
            },
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(readiness_label(score))
    with right:
        k1, k2 = st.columns(2)
        k1.metric("মোট পরীক্ষা", len(df))
        k2.metric("গড় স্কোর", f"{df['percentage'].mean():.1f}%")
        k3, k4 = st.columns(2)
        k3.metric("গড় Gap (কাট-অফের সাপেক্ষে)", f"{df['gap'].mean():+.2f}")
        k4.metric("কাট-অফ পার করা পরীক্ষা", f"{(df['gap'] >= 0).mean() * 100:.0f}%")
        pr = projection(df)
        cs = consistency(df)
        k5, k6 = st.columns(2)
        if pr:
            k5.metric("৩০ দিন পর আনুমানিক স্কোর", f"{pr['pred']:.0f}%",
                      f"{pr['slope_per_week']:+.1f}%/সপ্তাহ")
        else:
            k5.metric("৩০ দিন পর আনুমানিক স্কোর", "—", help="কমপক্ষে ৫টি পরীক্ষা ও ৭ দিনের ব্যবধান লাগবে")
        k6.metric("স্কোরের ওঠানামা (SD)", f"{cs:.1f}" if cs is not None else "—",
                  help="যত কম, তত ধারাবাহিক")
        st.caption(
            "রেডিনেস স্কোর = ৫০% সাম্প্রতিক ৫টি পরীক্ষার গড় + ৩০% সর্বমোট গড় + ২০% কাট-অফ পার করার হার। "
            "এটি আপনার নিজের ডেটার আনুমানিক সূচক; প্রকৃত ফলাফলের নিশ্চয়তা নয়। আনুমানিক স্কোর শেষ ১০টি পরীক্ষার সরলরেখা ট্রেন্ড।"
        )

    st.divider()

    st.subheader("ক্যাটাগরি অনুযায়ী রানিং এভারেজ")
    d = df.sort_values(["exam_date", "id"]).copy()
    d["running_avg"] = d.groupby("exam_type")["percentage"].transform(lambda s: s.expanding().mean())
    fig = px.line(d, x="exam_date", y="running_avg", color="type_label", markers=True,
                  labels={"exam_date": "তারিখ", "running_avg": "রানিং এভারেজ (%)", "type_label": "ক্যাটাগরি"})
    fig.add_hline(y=WEAK_THRESHOLD, line_dash="dot", annotation_text=f"{WEAK_THRESHOLD:.0f}%")
    fig.update_layout(height=380, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(category_summary(df), use_container_width=True, hide_index=True)

    t1, t2 = st.columns(2)
    with t1:
        st.subheader("পারফরম্যান্স ট্রেন্ড (%)")
        fig = px.line(d, x="exam_date", y="percentage", markers=True,
                      labels={"exam_date": "তারিখ", "percentage": "স্কোর (%)"})
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        st.subheader("কাট-অফ Gap (প্রতিটি পরীক্ষা)")
        d["ফলাফল"] = d["gap"].apply(lambda g: "কাট-অফের উপরে" if g >= 0 else "কাট-অফের নিচে")
        fig = px.bar(d, x="exam_date", y="gap", color="ফলাফল",
                     color_discrete_map={"কাট-অফের উপরে": "#16a34a", "কাট-অফের নিচে": "#dc2626"},
                     labels={"exam_date": "তারিখ", "gap": "Gap Marks"})
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("সাবজেক্ট ওয়াইজ পারফরম্যান্স")
    subj = subject_stats(df)
    if subj.empty:
        st.info("সাবজেক্ট ভিত্তিক বিশ্লেষণের জন্য টপিক/সাবজেক্ট টেস্ট/রিভিশন পরীক্ষার ডেটা দরকার।")
    else:
        weak = subj[subj["গড় %"] < WEAK_THRESHOLD]
        if weak.empty:
            st.success(f"কোনো সাবজেক্টের গড় {WEAK_THRESHOLD:.0f}%-এর নিচে নেই। 👏")
        for _, r in weak.iterrows():
            st.error(f"⚠️ Weakness Alert: **{r['সাবজেক্ট']}** — গড় {r['গড় %']:.1f}% ({int(r['পরীক্ষা'])}টি পরীক্ষা)")
        fig = px.bar(subj, x="গড় %", y="সাবজেক্ট", orientation="h",
                     color=subj["গড় %"] < WEAK_THRESHOLD,
                     color_discrete_map={True: "#dc2626", False: "#2563eb"})
        fig.add_vline(x=WEAK_THRESHOLD, line_dash="dot")
        fig.update_layout(height=max(280, 40 * len(subj)), showlegend=False, xaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(subj, use_container_width=True, hide_index=True)
        st.caption("মিক্সড রিভিশনে জড়িত সব সাবজেক্টকে একই স্কোর ধরা হয়, তাই সাবজেক্ট টেস্টের ডেটা বেশি নির্ভরযোগ্য। ফুল মডেল টেস্ট এই হিসাবের বাইরে।")

    tops = topic_stats(df)
    if not tops.empty:
        st.subheader("টপিক ওয়াইজ (সবচেয়ে দুর্বল ১০টি)")
        st.dataframe(tops.head(10), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("🧭 স্মার্ট দিকনির্দেশনা")
    for tip in prelim_insights(df, dd) + rule_based_guidance(df, subj, tops, score):
        st.markdown(f"- {tip}")

    units = unit_table(df)
    summary = build_ai_summary(df, dd, score, subj, tops, units)
    sig = md5(summary.encode("utf-8")).hexdigest()

    if st.button("🤖 Gemini থেকে বিস্তারিত গাইডেন্স নিন"):
        try:
            with st.spinner("বিশ্লেষণ করা হচ্ছে..."):
                st.session_state["ai_tip"] = (sig, ai_guidance(summary))
        except Exception as e:
            st.error(f"AI গাইডেন্স আনতে সমস্যা হয়েছে: {e}")
    cached = st.session_state.get("ai_tip")
    if cached and cached[0] == sig:
        st.info(cached[1])


# ----------------------------------------------------------------------------
# পেজ: প্রিলি অ্যানালাইসিস
# ----------------------------------------------------------------------------
def page_prelim(user_id: int):
    st.header("🎯 প্রিলি অ্যানালাইসিস")
    df = fetch_exams(user_id)
    if df.empty:
        st.info("এখনো কোনো পরীক্ষার ডেটা নেই।")
        return
    dd = detail_frame(df)
    if dd.empty:
        st.warning(
            "এই বিশ্লেষণের জন্য পরীক্ষা যোগ করার সময় 'প্রশ্নভিত্তিক বিস্তারিত দেব' টিক দিয়ে সঠিক/ভুল/বাদের সংখ্যা দিতে হবে। "
            "আগের পরীক্ষাগুলোতে এই তথ্য নেই, তাই সেগুলো এখানে ধরা হয়নি।"
        )
        return

    st.caption(f"{len(dd)}টি পরীক্ষার প্রশ্নভিত্তিক ডেটা ধরা হয়েছে (মোট {len(df)}টির মধ্যে)।")
    C, W, S = dd["correct"].sum(), dd["wrong"].sum(), dd["skipped"].sum()
    att = C + W
    acc = C / att * 100 if att else 0.0
    attempt = att / (C + W + S) * 100
    neg = float(dd["neg_mark"].mean())
    be = breakeven_accuracy(neg)
    lost = dd["neg_lost"].sum()
    tot = dd["total_marks"].sum()

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("অ্যাটেম্পট রেট", f"{attempt:.0f}%")
    a2.metric("অ্যাকুরেসি (উত্তর দেওয়াগুলোর)", f"{acc:.0f}%", f"ব্রেক-ইভেন {be:.0f}%", delta_color="off")
    a3.metric("নেগেটিভে হারানো নম্বর", f"{lost:.1f}", f"মোটের {lost / tot * 100:.1f}%", delta_color="off")
    a4.metric("নেগেটিভ ছাড়া গড় স্কোর", f"{dd['gross_pct'].mean():.1f}%",
              f"আসল {dd['percentage'].mean():.1f}%", delta_color="off")
    st.caption(
        f"ব্রেক-ইভেন = ভুলে {neg:g} কাটলে অনুমান করে উত্তর দেওয়া তখনই লাভজনক, যখন ওই প্রশ্নে সঠিক হওয়ার সম্ভাবনা "
        f"{be:.0f}%-এর বেশি। এর নিচে হলে বাদ দেওয়াই ভালো।"
    )

    for tip in prelim_insights(df, dd):
        st.markdown(f"- {tip}")

    st.divider()
    ds = dd.sort_values(["exam_date", "id"])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("প্রতিটি পরীক্ষায় সঠিক / ভুল / বাদ")
        long = ds.melt(id_vars=["id", "exam_date"], value_vars=["correct", "wrong", "skipped"],
                       var_name="ধরন", value_name="সংখ্যা")
        long["ধরন"] = long["ধরন"].map({"correct": "সঠিক", "wrong": "ভুল", "skipped": "বাদ"})
        long["পরীক্ষা"] = long["exam_date"].dt.strftime("%m-%d") + " #" + long["id"].astype(str)
        fig = px.bar(long, x="পরীক্ষা", y="সংখ্যা", color="ধরন",
                     color_discrete_map={"সঠিক": "#16a34a", "ভুল": "#dc2626", "বাদ": "#9ca3af"})
        fig.update_layout(height=320, barmode="stack")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("অ্যাকুরেসি ও অ্যাটেম্পট ট্রেন্ড")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ds["exam_date"], y=ds["accuracy"], mode="lines+markers", name="অ্যাকুরেসি %"))
        fig.add_trace(go.Scatter(x=ds["exam_date"], y=ds["attempt_rate"], mode="lines+markers", name="অ্যাটেম্পট %"))
        fig.add_hline(y=be, line_dash="dot", annotation_text="ব্রেক-ইভেন")
        fig.update_layout(height=320, yaxis_range=[0, 100], legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)

    m1, m2 = st.columns(2)
    with m1:
        st.subheader("ভুলের কারণ")
        mu, mc, mt = (dd["m_unknown"].fillna(0).sum(), dd["m_careless"].fillna(0).sum(),
                      dd["m_time"].fillna(0).sum())
        untagged = max(0, W - (mu + mc + mt))
        causes = pd.DataFrame({
            "কারণ": ["অজানা ছিল", "অসাবধানতা/ভুল পড়া", "সময়ের অভাব", "কারণ দেওয়া হয়নি"],
            "সংখ্যা": [mu, mc, mt, untagged],
        })
        causes = causes[causes["সংখ্যা"] > 0]
        if causes.empty:
            st.info("ভুলের কারণ এখনো দেওয়া হয়নি।")
        else:
            fig = px.pie(causes, names="কারণ", values="সংখ্যা", hole=0.45)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    with m2:
        st.subheader("প্রতি প্রশ্নে গড় সময় (সেকেন্ড)")
        tdf = ds.dropna(subset=["sec_per_q"])
        if tdf.empty:
            st.info("সময় (মিনিট) এখনো দেওয়া হয়নি।")
        else:
            fig = px.line(tdf, x="exam_date", y="sec_per_q", markers=True,
                          labels={"exam_date": "তারিখ", "sec_per_q": "সেকেন্ড/প্রশ্ন"})
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("সাবজেক্ট ভিত্তিক অ্যাকুরেসি ও নেগেটিভ")
    ex = explode_items(dd, "subjects", exclude=(ALL_SUBJECTS_LABEL,))
    if ex.empty:
        st.info("সাবজেক্ট ভিত্তিক প্রশ্নডেটা নেই (ফুল মডেল টেস্ট এখানে ধরা হয় না)।")
    else:
        g = ex.groupby("item").agg(
            পরীক্ষা=("id", "nunique"), সঠিক=("correct", "sum"), ভুল=("wrong", "sum"),
            বাদ=("skipped", "sum"), নেগেটিভে_হারানো=("neg_lost", "sum"),
        ).reset_index().rename(columns={"item": "সাবজেক্ট"})
        g["অ্যাকুরেসি %"] = (g["সঠিক"] / (g["সঠিক"] + g["ভুল"]).replace(0, np.nan) * 100).round(1)
        g["অ্যাটেম্পট %"] = ((g["সঠিক"] + g["ভুল"]) / (g["সঠিক"] + g["ভুল"] + g["বাদ"]) * 100).round(1)
        g["নেগেটিভে_হারানো"] = g["নেগেটিভে_হারানো"].round(1)
        st.dataframe(g.sort_values("অ্যাকুরেসি %"), use_container_width=True, hide_index=True)
        st.caption("মিক্সড রিভিশনের প্রশ্নসংখ্যা জড়িত সব সাবজেক্টে গণনা হয়, তাই সংখ্যাগুলো আনুমানিক।")

    st.divider()
    st.subheader("কাট-অফ প্রজেকশন")
    pr = projection(df)
    if pr is None:
        st.info("প্রজেকশনের জন্য কমপক্ষে ৫টি পরীক্ষা এবং ৭ দিনের ব্যবধান দরকার।")
    else:
        cut_pct = float((df["cutoff_marks"] / df["total_marks"] * 100).tail(10).mean())
        st.markdown(
            f"- সাম্প্রতিক {pr['n']}টি পরীক্ষার ট্রেন্ড: সপ্তাহে **{pr['slope_per_week']:+.1f}%**\n"
            f"- এই গতিতে ৩০ দিন পর আনুমানিক স্কোর: **{pr['pred']:.0f}%**\n"
            f"- আপনার দেওয়া কাট-অফের গড় (শতাংশে): **{cut_pct:.0f}%**"
        )
        if pr["pred"] >= cut_pct:
            st.success("এই গতি ধরে রাখলে কাট-অফের উপরে থাকার সম্ভাবনা আছে।")
        else:
            st.warning(f"এই গতিতে কাট-অফ থেকে প্রায় {cut_pct - pr['pred']:.0f}% পিছিয়ে থাকবেন; গতি বাড়াতে হবে।")
        st.caption("এটি সরল রৈখিক ট্রেন্ড, ভবিষ্যদ্বাণী নয়। পরীক্ষা যত বেশি, নির্ভরযোগ্যতা তত বেশি।")

    st.divider()
    st.subheader("📌 অগ্রাধিকার তালিকা (কোনটা আগে পড়বেন)")
    units = unit_table(df)
    if units.empty:
        st.info("ডেটা নেই।")
    else:
        show = pd.DataFrame({
            "টপিক/সাবজেক্ট": units["unit"], "পরীক্ষা": units["n"], "গড় %": units["avg"],
            "সর্বশেষ %": units["last_pct"], "শেষ পরীক্ষা": units["last_date"].dt.strftime("%Y-%m-%d"),
            "কত দিন আগে": units["stale_days"], "প্রায়োরিটি": units["priority"],
        })
        st.dataframe(show.head(15), use_container_width=True, hide_index=True)
        st.caption("প্রায়োরিটি = ৬০% দুর্বলতা (১০০−গড়) + ২০% সর্বশেষ পতন + ২০% কতদিন ধরে পরীক্ষা দেননি (৩০ দিনে সর্বোচ্চ)। বেশি মানে আগে পড়া উচিত।")


# ----------------------------------------------------------------------------
# পেজ: আজকের রিভিশন
# ----------------------------------------------------------------------------
def page_revision(user_id: int):
    st.header("🔁 আজকের রিভিশন")
    df = fetch_exams(user_id)
    units = unit_table(df) if not df.empty else pd.DataFrame()
    if units.empty:
        st.info("রিভিশন শিডিউলের জন্য টপিক/সাবজেক্টসহ কিছু পরীক্ষার ডেটা দরকার।")
        return

    overdue = units[units["days_to_due"] < 0].sort_values("days_to_due")
    today_due = units[units["days_to_due"] == 0]
    upcoming = units[(units["days_to_due"] > 0) & (units["days_to_due"] <= 7)].sort_values("days_to_due")

    k1, k2, k3 = st.columns(3)
    k1.metric("মেয়াদোত্তীর্ণ", len(overdue))
    k2.metric("আজ", len(today_due))
    k3.metric("আগামী ৭ দিনে", len(upcoming))

    def fmt(frame, status):
        return pd.DataFrame({
            "টপিক/সাবজেক্ট": frame["unit"], "সর্বশেষ %": frame["last_pct"],
            "শেষ পরীক্ষা": frame["last_date"].dt.strftime("%Y-%m-%d"),
            "রিভিশনের তারিখ": frame["due"].dt.strftime("%Y-%m-%d"),
            "অবস্থা": frame["days_to_due"].apply(
                lambda d: status if d == 0 else (f"{abs(d)} দিন দেরি" if d < 0 else f"{d} দিন পরে")),
        })

    if not overdue.empty or not today_due.empty:
        st.subheader("এখনই রিভিশন দিন")
        both = pd.concat([overdue, today_due])
        st.dataframe(fmt(both, "আজ"), use_container_width=True, hide_index=True)
    else:
        st.success("আজ কোনো রিভিশন বাকি নেই। 👏")
    if not upcoming.empty:
        st.subheader("আগামী ৭ দিনে")
        st.dataframe(fmt(upcoming, "আজ"), use_container_width=True, hide_index=True)

    st.caption(
        "নিয়ম: শেষ পরীক্ষার স্কোর ৫৫%-এর নিচে হলে ২ দিন, ৫৫–৭০% হলে ৫ দিন, ৭০–৮৫% হলে ১০ দিন, "
        "৮৫%-এর উপরে হলে ২১ দিন পর আবার রিভিশন/পরীক্ষা। রিভিশনের পর ওই টপিকের নতুন পরীক্ষা যোগ করলে শিডিউল নিজে আপডেট হবে।"
    )


# ----------------------------------------------------------------------------
# পেজ: হিস্ট্রি
# ----------------------------------------------------------------------------
def page_history(user_id: int):
    st.header("🗂️ পরীক্ষার ইতিহাস")
    df = fetch_exams(user_id)
    if df.empty:
        st.info("এখনো কোনো পরীক্ষা সংরক্ষিত নেই।")
        return

    cats = st.multiselect("ক্যাটাগরি ফিল্টার", list(EXAM_TYPES.values()),
                          default=list(EXAM_TYPES.values()))
    view = df[df["type_label"].isin(cats)].sort_values(["exam_date", "id"], ascending=False)

    def n(col):
        return view[col].astype("Int64") if col in view else None

    show = pd.DataFrame({
        "ID": view["id"],
        "তারিখ": view["exam_date"].dt.strftime("%Y-%m-%d"),
        "ধরন": view["type_label"],
        "সাবজেক্ট": view["subjects"].str.replace(SEP, ", ", regex=False),
        "টপিক": view["topics"].str.replace(SEP, ", ", regex=False),
        "মোট": view["total_marks"],
        "প্রাপ্ত": view["obtained_marks"],
        "কাট-অফ": view["cutoff_marks"],
        "Gap": view["gap"],
        "%": view["percentage"],
        "সঠিক": n("correct"),
        "ভুল": n("wrong"),
        "বাদ": n("skipped"),
        "সময় (মি.)": view["time_minutes"],
        "নোট": view["notes"],
    })
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button("⬇️ CSV ডাউনলোড", show.to_csv(index=False).encode("utf-8-sig"),
                       file_name="bcs_exam_history.csv", mime="text/csv")

    st.divider()
    st.subheader("🗑️ পরীক্ষা ডিলিট করুন")
    options = {
        int(r.id): f"ID {int(r.id)} — {r.exam_date:%Y-%m-%d} — {r.type_label} — {r.percentage:.1f}%"
        for r in df.sort_values(["exam_date", "id"], ascending=False).itertuples()
    }
    chosen = st.selectbox("ডিলিট করার পরীক্ষা বাছাই করুন", list(options), format_func=options.get)
    confirm = st.checkbox("আমি নিশ্চিত — এটি আর ফেরত আনা যাবে না")
    if st.button("ডিলিট করুন", disabled=not confirm):
        if delete_exam(user_id, chosen):
            st.session_state["flash_hist"] = f"ID {chosen} ডিলিট হয়েছে।"
            st.rerun()
        else:
            st.error("ডিলিট করা যায়নি।")


# ----------------------------------------------------------------------------
# মেইন
# ----------------------------------------------------------------------------
PAGES = ["📊 ড্যাশবোর্ড", "🎯 প্রিলি অ্যানালাইসিস", "🔁 আজকের রিভিশন", "➕ পরীক্ষা যোগ করুন", "🗂️ হিস্ট্রি"]


def sidebar() -> str:
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state['user_name']}")
        page = st.radio("মেনু", PAGES, label_visibility="collapsed")
        st.divider()
        if not secret_or_env("GEMINI_API_KEY"):
            st.text_input("Gemini API Key", type="password", key="api_key_input",
                          help="aistudio.google.com থেকে বিনামূল্যে নেওয়া যায়। শুধু এই সেশনে ব্যবহৃত হয়।")
        else:
            st.caption("🔑 Gemini API Key কনফিগার করা আছে")
        st.caption(f"AI মডেল: {get_model_name()}")
        if os.path.exists(DB_PATH):
            with open(DB_PATH, "rb") as f:
                st.download_button("💾 ডেটাবেস ব্যাকআপ", f.read(),
                                   file_name="bcs_tracker_backup.db",
                                   mime="application/octet-stream")
        if st.button("লগআউট"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            try:
                st.query_params.clear()
            except Exception:
                pass
            st.rerun()
    return page


def main():
    init_db()
    if "user_id" not in st.session_state:
        try:
            saved = st.query_params.get("user", "")
        except Exception:
            saved = ""
        if isinstance(saved, list):
            saved = saved[0] if saved else ""
        if len(str(saved).strip()) >= 2:
            do_login(str(saved))
            st.rerun()
        login_screen()
        return
    user_id = st.session_state["user_id"]
    page = sidebar()
    if msg := st.session_state.pop("flash_hist", None):
        st.success(msg)
    if page == PAGES[0]:
        page_dashboard(user_id)
    elif page == PAGES[1]:
        page_prelim(user_id)
    elif page == PAGES[2]:
        page_revision(user_id)
    elif page == PAGES[3]:
        page_add_exam(user_id)
    else:
        page_history(user_id)


if __name__ == "__main__":
    main()

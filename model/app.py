"""
app.py — Streamlit UI for YouTube Shorts virality predictor.
Run: streamlit run app.py
"""
import streamlit as st
import subprocess, sys, json, tempfile, os, re
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="Shorts Predictor",
    page_icon="🎬",
    layout="centered",
)

st.markdown("""
<style>
    /* ── Score ── */
    .score-wrap {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .score-num {
        font-size: 5rem;
        font-weight: 800;
        letter-spacing: -2px;
        line-height: 1;
    }
    .score-sup {
        font-size: 2rem;
        font-weight: 600;
        vertical-align: super;
    }
    .score-sub {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.4rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* ── Issue cards ── */
    .issue-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-left: 3px solid #ef4444;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
    }
    .issue-label {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        font-size: 0.7rem;
        font-weight: 700;
        color: #ef4444;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.35rem;
    }
    .issue-label::before {
        content: "";
        display: inline-block;
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background: #ef4444;
    }
    .issue-fix {
        font-size: 0.88rem;
        color: #cbd5e1;
        line-height: 1.5;
    }
    .issue-bench {
        font-size: 0.76rem;
        color: #475569;
        margin-top: 0.3rem;
    }


    /* ── Section label ── */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Shorts Predictor")
st.caption("Upload a Short and get specific, data-backed feedback before you post.")

with st.form("predict_form"):
    video_file = st.file_uploader("Upload your Short (MP4)", type=["mp4", "mov"])
    channel = st.text_input("Your channel (URL or @handle)", placeholder="@YourChannel — used to benchmark against your own avg views")
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Title", placeholder="Your video title")
        tags  = st.text_input("Tags (comma-separated)", placeholder="fitness, gym, workout")
    with col2:
        now = datetime.now()
        post_hour    = st.slider("Hour you plan to post", 0, 23, now.hour)
        post_weekday = st.selectbox(
            "Day you plan to post",
            options=list(range(7)),
            format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x],
            index=now.weekday(),
        )
    desc = st.text_area("Description (optional)", height=80, placeholder="Your caption / description...")
    submitted = st.form_submit_button("Analyse", use_container_width=True, type="primary")

if submitted:
    if video_file is None:
        st.error("Please upload a video file.")
        st.stop()

    suffix = Path(video_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    with st.spinner("Analysing… ~30-60 seconds"):
        cmd = [
            sys.executable, str(Path(__file__).parent / "predict.py"),
            "--video",   tmp_path,
            "--title",   title,
            "--tags",    tags,
            "--desc",    desc,
            "--hour",    str(post_hour),
            "--weekday", str(post_weekday),
            "--niche",   "fitness",
        ]
        if channel.strip():
            cmd += ["--channel", channel.strip()]
        result = subprocess.run(cmd, capture_output=True, text=True)

    os.unlink(tmp_path)
    raw = result.stdout + result.stderr

    if result.returncode != 0 and not raw.strip():
        st.error("Prediction failed. Check that all model files are present.")
        with st.expander("Error details"):
            st.code(raw)
        st.stop()

    pct_match          = re.search(r"Percentile:\s+(\d+)", raw)
    hook_frame_match   = re.search(r"HOOK_FRAME:(.+)", raw)
    hook_checks_match  = re.search(r"HOOK_CHECKS:(.+)", raw)
    recs_match         = re.search(r"RECOMMENDATIONS:(.+)", raw)

    display_pct      = int(pct_match.group(1))               if pct_match         else None
    hook_frame_path  = hook_frame_match.group(1).strip()     if hook_frame_match  else None
    hook_checks      = json.loads(hook_checks_match.group(1)) if hook_checks_match else []
    recommendations  = json.loads(recs_match.group(1))        if recs_match        else []

    if display_pct is not None:
        if display_pct >= 65:
            colour  = "#22c55e"
            verdict = "STRONG"
        elif display_pct >= 40:
            colour  = "#f97316"
            verdict = "AVERAGE"
        else:
            colour  = "#ef4444"
            verdict = "WEAK"

        st.divider()

        st.markdown(f"""
<div class="score-wrap">
  <div class="score-num" style="color:{colour}">
    {display_pct}<span class="score-sup" style="color:{colour}">th</span>
  </div>
  <div class="score-sub">Percentile vs {verdict.lower()} videos in your niche</div>
</div>
""", unsafe_allow_html=True)

        # ── Hook frame ────────────────────────────────────
        if hook_frame_path and Path(hook_frame_path).exists():
            st.markdown('<div class="section-label">Hook frame (first 0.5s)</div>', unsafe_allow_html=True)
            st.image(hook_frame_path, use_container_width=False, width=280)
            os.unlink(hook_frame_path)

        # ── Issues & fixes ────────────────────────────────
        if recommendations:
            st.markdown('<div class="section-label">Issues & fixes</div>', unsafe_allow_html=True)
            for rec in recommendations:
                color = "#ef4444" if rec.get("type") == "check" else "#f97316"
                bench_html = f'<div class="issue-bench">{rec["benchmark"]}</div>' if rec.get("benchmark") else ""
                st.markdown(f"""
<div class="issue-card" style="border-left-color:{color}">
  <div class="issue-label" style="color:{color}">{rec['label']}</div>
  <div class="issue-fix">{rec['message']}</div>
  {bench_html}
</div>""", unsafe_allow_html=True)
        else:
            st.success("No major issues found — your video looks solid!")

    elif display_pct is None:
        st.warning("Could not parse prediction output.")
        with st.expander("Raw output"):
            st.code(raw)

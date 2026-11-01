import html
import json
import logging
import os
import re
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from typing import Dict, List

from google import genai
from google.genai import types
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("MentorBot")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.json")
TASKS_FILE = os.path.join(SCRIPT_DIR, "tasks.json")

# ----------------- Configuration & Storage ----------------- #

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"AI Mentor Telegram Bot is running 24/7!")

    def log_message(self, format, *args):
        pass  # suppress clutter in logs

def start_health_server():
    """Starts a lightweight HTTP server on $PORT for cloud hosting platforms."""
    port = int(os.environ.get("PORT", 7860))
    try:
        server = HTTPServer(("0.0.0.0", port), HealthHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"Health check HTTP server active on port {port}")
    except Exception as e:
        logger.warning(f"Health server could not start on port {port}: {e}")

def load_config() -> dict:
    config = {
        "TELEGRAM_BOT_TOKEN": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
        "MODEL_NAME": os.environ.get("MODEL_NAME", "gemini-3.5-flash-lite"),
    }

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                file_cfg = json.load(f)
                for k, v in file_cfg.items():
                    if not config.get(k) or "PASTE_YOUR" in config.get(k, ""):
                        config[k] = v
        except Exception:
            pass

    return config

def load_tasks() -> List[dict]:
    if not os.path.exists(TASKS_FILE):
        return []
    try:
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_tasks(tasks: List[dict]):
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)

# Global in-memory conversation history: {user_id: [ {"role": "user"|"model", "text": "..."} ]}
conversation_histories: Dict[int, List[dict]] = {}

# ----------------- System Prompt for the Mentor ----------------- #

SYSTEM_INSTRUCTION = """
You are a disciplined, elite, high-standards, and deeply caring Personal Mentor & Teacher.
You are mentoring a 1st-semester Computer Science Engineering (CSE) student at Bihar Engineering University (BEU).
He is a day scholar who commutes 3.1 km each way by bicycle (10:00 AM to 5:00 PM college).
He comes from a Tier-3 college with no on-campus placements, but he has the ultimate ambition:
"A MAN WHO DOESN'T REQUIRE ANYONE'S HELP, NOT EVEN ANY AI" — to become completely self-reliant, disciplined, mentally and physically a 10/10.

His Master Roadmap:
1. Primary Long-Term Focus: GATE CSE (Engineering Math, C Programming, Discrete Math, DSA, OS, DBMS, Networks) + High University CGPA (>= 8.0).
2. Weekend Shift: SSC CGL (Saturday evenings: Reasoning + English Vocab; Sunday mornings: Speed Arithmetic + Static GK).
3. 10/10 Protocol: Break late-night screen addiction (phone 6ft away at 10:30 PM), eliminate bed laziness (wake at 6:00 AM with 3-sec rule), 20-min daily calisthenics + 6.2 km cycling, water, and soaked chana.

CRITICAL FORMATTING RULES FOR TELEGRAM MOBILE SCREEN:
The user reads your responses on a mobile phone screen. You MUST format every message for elite visual readability:
1. Visual Structure:
   - Start with a bold emoji header: e.g. "🎯 **Concept Title**" or "⚔️ **Mentor Brief**"
   - Use horizontal dividers: "---" between major sections.
   - Use short, punchy paragraphs (2-3 sentences maximum). Never dump walls of dense text.
2. Clear Lists:
   - Use distinct bullet points ("• **Bold Keyword:** explanation").
3. Code & Technical Topics:
   - Always put code inside standard markdown code blocks (```c ... ```) with clean indentation.
   - For short variables/syntax, use inline backticks (`int *p;`).
4. Math Equations:
   - Do NOT use raw LaTeX commands like \\frac or \\begin{matrix}.
   - Instead, use clean Unicode characters: λ, θ, ², ³, √, ×, ÷, π, ≤, ≥, ≠, ∈, →, ∫, Δ.
5. End every message with an actionable step:
   - A quick challenge, practice question, or next immediate step: "⚡ **Your Next Move:** ..."
"""

# ----------------- Helper Functions ----------------- #

def format_telegram_html(raw_text: str) -> str:
    """Converts markdown text to clean, rich Telegram HTML formatting."""
    if not raw_text:
        return ""

    # 1. Protect code blocks
    code_blocks = []
    def save_cb(m):
        code = html.escape(m.group(2).strip("\n"))
        idx = len(code_blocks)
        code_blocks.append(f"<pre><code>{code}</code></pre>")
        return f"___CODEBLOCK_{idx}___"
    
    text = re.sub(r"```([a-zA-Z0-9_]*)\n?(.*?)```", save_cb, raw_text, flags=re.DOTALL)

    # 2. Protect inline code
    inline_codes = []
    def save_ic(m):
        code = html.escape(m.group(1))
        idx = len(inline_codes)
        inline_codes.append(f"<code>{code}</code>")
        return f"___INLINE_{idx}___"
    text = re.sub(r"`([^`\n]+)`", save_ic, text)

    # 3. Escape HTML characters in normal text
    text = html.escape(text)

    # 4. Dividers
    text = re.sub(r"(?m)^(\-{3,}|\_{3,}|\*{3,})\s*$", "━━━━━━━━━━━━━━━━━━━━━", text)

    # 5. Headers with clean icons
    text = re.sub(r"(?m)^#\s+(.*?)$", r"\n<b>⚔️ \1</b>\n", text)
    text = re.sub(r"(?m)^##\s+(.*?)$", r"\n<b>⚡ \1</b>\n", text)
    text = re.sub(r"(?m)^###\s+(.*?)$", r"\n<b>📌 \1</b>\n", text)
    text = re.sub(r"(?m)^####\s+(.*?)$", r"\n<b>▫️ \1</b>", text)

    # 6. Bold & Italic
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!_)_([^_\n]+)_(?!_)", r"<i>\1</i>", text)

    # 7. Blockquotes (> note)
    text = re.sub(r"(?m)^&gt;\s+(.*?)$", r"<blockquote>\1</blockquote>", text)

    # 8. Bullets
    text = re.sub(r"(?m)^[\*\-]\s+", "• ", text)

    # 9. Restore code blocks and inline code
    for i, ic in enumerate(inline_codes):
        text = text.replace(f"___INLINE_{i}___", ic)
    for i, cb in enumerate(code_blocks):
        text = text.replace(f"___CODEBLOCK_{i}___", cb)

    # Clean redundant empty lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def split_text_cleanly(text: str, max_len: int = 3800) -> List[str]:
    """Splits a long message on paragraph boundaries so formatting isn't severed."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    paragraphs = text.split("\n\n")
    current_chunk = ""

    for p in paragraphs:
        if len(current_chunk) + len(p) + 2 <= max_len:
            current_chunk = f"{current_chunk}\n\n{p}" if current_chunk else p
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(p) <= max_len:
                current_chunk = p
            else:
                lines = p.split("\n")
                for line in lines:
                    if len(current_chunk) + len(line) + 1 <= max_len:
                        current_chunk = f"{current_chunk}\n{line}" if current_chunk else line
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = line

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

async def send_split_message(update: Update, text: str):
    """Formats markdown into clean Telegram HTML and sends safely."""
    formatted_html = format_telegram_html(text)
    chunks = split_text_cleanly(formatted_html, max_len=3800)

    for chunk in chunks:
        try:
            await update.message.reply_text(
                chunk,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
        except Exception as e:
            logger.warning(f"HTML send failed ({e}), sending plain text fallback")
            plain = re.sub(r"<[^>]+>", "", chunk)
            await update.message.reply_text(plain)

def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)

# ----------------- Command Handlers ----------------- #

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "<b>⚔️ WELCOME, WARRIOR</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "I am your 24/7 Personal Mentor, GATE/SSC Planner, and Daily Coach.\n\n"
        "<b>⚡ Quick Commands:</b>\n"
        "• <code>/homework &lt;text&gt;</code> — Log college homework or assignment\n"
        "• <code>/tasks</code> or <code>/todo</code> — View your pending assignments\n"
        "• <code>/done &lt;id&gt;</code> — Mark an assignment completed\n"
        "• <code>/schedule</code> — View your master timetable\n"
        "• <code>/habits</code> — View your 10/10 discipline protocol\n"
        "• <code>/clear</code> — Reset conversation context\n\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "<b>💬 Doubts & Questions:</b>\n"
        "Simply send any message! Ask about C programming, Engineering Math, GATE concepts, SSC GK, or check in with your daily progress."
    )
    await update.message.reply_text(welcome_text, parse_mode=ParseMode.HTML)

async def schedule_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    schedule_text = (
        "<b>📅 MASTER TIMETABLE (Day Scholar: 10 AM – 5 PM)</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "<b>⏰ MONDAY – FRIDAY (College + GATE Core):</b>\n\n"
        "• <b>06:00 AM – 06:45 AM:</b> Wake up (3-sec rule), 20-min Workout + Bath\n"
        "• <b>06:45 AM – 07:15 AM:</b> Soaked Chana / High-protein Breakfast\n"
        "• <b>07:15 AM – 09:15 AM:</b> 🎯 <i>Deep Study Block 1 (GATE Core / Math)</i>\n"
        "• <b>09:15 AM – 10:00 AM:</b> Cycle ride (3.1 km) to College\n"
        "• <b>10:00 AM – 05:00 PM:</b> 🏫 College Hours (Finish homework during breaks!)\n"
        "• <b>05:00 PM – 05:30 PM:</b> Cycle ride back (3.1 km)\n"
        "• <b>05:30 PM – 06:15 PM:</b> Rest, refresh, tea\n"
        "• <b>06:15 PM – 08:30 PM:</b> 🎯 <i>Deep Study Block 2 (C Programming & College Prep)</i>\n"
        "• <b>08:30 PM – 09:15 PM:</b> Dinner + 15m walk\n"
        "• <b>09:15 PM – 10:15 PM:</b> 🎯 <i>Block 3 (Revision / 30m Aptitude)</i>\n"
        "• <b>10:15 PM – 10:30 PM:</b> Pack bag, phone 6 feet away from bed\n"
        "• <b>10:30 PM:</b> 🛌 Lights out (7.5 hrs sleep)\n\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "<b>⚔️ WEEKEND SSC SHIFT:</b>\n\n"
        "• <b>Saturday Evening (6:30 PM – 9:00 PM):</b> Reasoning Puzzles + English Vocab/Grammar\n"
        "• <b>Sunday Morning (7:30 AM – 11:30 AM):</b> Speed Arithmetic + Static GK (Polity/History)"
    )
    await update.message.reply_text(schedule_text, parse_mode=ParseMode.HTML)

async def habits_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    habits_text = (
        "<b>🛡️ THE 10/10 DISCIPLINE PROTOCOL</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "1. ▫️ <b>10:30 PM Hard Cutoff:</b> Phone placed 6 feet away from bed\n"
        "2. ▫️ <b>06:00 AM Wake Up:</b> Count 3-2-1, feet on floor immediately\n"
        "3. ▫️ <b>Hydration:</b> 1 Large glass of cold water after waking up\n"
        "4. ▫️ <b>Calisthenics:</b> 20 mins (Push-ups, Squats, Planks, Surya Namaskar)\n"
        "5. ▫️ <b>Active Cardio:</b> 6.2 km daily cycle ride (3.1 km each way)\n"
        "6. ▫️ <b>Morning Block:</b> 2 hours uninterrupted study before college\n"
        "7. ▫️ <b>Nutrition:</b> Soaked chana + peanuts + jaggery daily\n"
        "8. ▫️ <b>Focus:</b> Zero social media / reels during study blocks\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "<i>'Discipline is choosing between what you want now and what you want most.'</i>"
    )
    await update.message.reply_text(habits_text, parse_mode=ParseMode.HTML)

async def add_homework_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "<b>Usage:</b> <code>/homework &lt;details&gt;</code>\n"
            "<i>Example:</i> <code>/homework C programming lab report on arrays</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    text = " ".join(context.args)
    tasks = load_tasks()
    task_id = len(tasks) + 1
    new_task = {
        "id": task_id,
        "text": text,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status": "pending",
    }
    tasks.append(new_task)
    save_tasks(tasks)

    msg = (
        f"<b>✅ Task #{task_id} Logged!</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        f"📝 <b>Details:</b> <code>{html.escape(text)}</code>\n\n"
        "💡 <i>Strategy: Try to knock this out during free periods at college tomorrow!</i>"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

async def list_tasks_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tasks = load_tasks()
    pending = [t for t in tasks if t.get("status") == "pending"]

    if not pending:
        await update.message.reply_text(
            "<b>🎉 No pending homework or assignments!</b>\n"
            "You are completely up-to-date. Keep the slate clean!",
            parse_mode=ParseMode.HTML,
        )
        return

    msg = "<b>📋 PENDING HOMEWORK & ASSIGNMENTS:</b>\n━━━━━━━━━━━━━━━━━━━━━\n"
    for t in pending:
        msg += f"• <b>#{t['id']}</b>: {html.escape(t['text'])}\n  <i>Added: {t['created_at']}</i>\n\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━\nTo mark as done, type: <code>/done &lt;id&gt;</code>"
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

async def done_task_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text(
            "<b>Usage:</b> <code>/done &lt;task_id&gt;</code>\n<i>Example:</i> <code>/done 1</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    task_id = int(context.args[0])
    tasks = load_tasks()
    found = False
    for t in tasks:
        if t["id"] == task_id:
            t["status"] = "completed"
            found = True
            break

    if found:
        save_tasks(tasks)
        await update.message.reply_text(
            f"<b>🔥 Great Execution!</b>\nTask #{task_id} marked as completed. Momentum is building!",
            parse_mode=ParseMode.HTML,
        )
    else:
        await update.message.reply_text(
            f"⚠️ Task #{task_id} not found in your list.",
            parse_mode=ParseMode.HTML,
        )

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conversation_histories[user_id] = []
    await update.message.reply_text("🧹 <b>Context Cleared.</b> Starting a fresh focus session. What's next?", parse_mode=ParseMode.HTML)

# ----------------- AI Message Handler ----------------- #

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text
    if not user_text:
        return

    config = load_config()
    api_key = config.get("GEMINI_API_KEY", "")
    model_name = config.get("MODEL_NAME", "gemini-3.5-flash-lite")

    if not api_key or "PASTE_YOUR" in api_key:
        await update.message.reply_text(
            "⚠️ <b>Gemini API Key missing!</b>\n"
            "Please paste your Gemini API key in <code>config.json</code> on your laptop, then restart the bot.",
            parse_mode=ParseMode.HTML,
        )
        return

    if user_id not in conversation_histories:
        conversation_histories[user_id] = []

    tasks = load_tasks()
    pending = [f"#{t['id']}: {t['text']}" for t in tasks if t.get("status") == "pending"]
    pending_summary = ("Current pending college tasks:\n" + "\n".join(pending)) if pending else "No pending college tasks."

    full_system_instruction = f"{SYSTEM_INSTRUCTION}\n\n[Active Student Context]\n{pending_summary}"

    history = conversation_histories[user_id]
    await update.message.chat.send_action("typing")

    try:
        client = get_gemini_client(api_key)
        recent_history = history[-8:]
        
        prompt_with_context = ""
        for turn in recent_history:
            role_label = "Student" if turn["role"] == "user" else "Mentor"
            prompt_with_context += f"{role_label}: {turn['text']}\n"
        prompt_with_context += f"Student: {user_text}\nMentor:"

        # Resilient multi-model fallback:
        models_to_try = [model_name]
        for fallback in ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.8-flash"]:
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        response = None
        last_error = None
        for current_model in models_to_try:
            try:
                response = client.models.generate_content(
                    model=current_model,
                    contents=prompt_with_context,
                    config=types.GenerateContentConfig(
                        system_instruction=full_system_instruction,
                        temperature=0.7,
                    ),
                )
                if response and response.text:
                    break
            except Exception as err:
                last_error = err
                logger.warning(f"Model {current_model} failed: {err}. Retrying with next model...")

        if not response or not response.text:
            if last_error:
                raise last_error
            reply_text = "Let's refocus. What is your next priority?"
        else:
            reply_text = response.text

        # Save to memory
        conversation_histories[user_id].append({"role": "user", "text": user_text})
        conversation_histories[user_id].append({"role": "model", "text": reply_text})

        if len(conversation_histories[user_id]) > 20:
            conversation_histories[user_id] = conversation_histories[user_id][-20:]

        await send_split_message(update, reply_text)

    except Exception as e:
        logger.error(f"Error querying Gemini: {e}")
        await update.message.reply_text(
            f"⚠️ An error occurred while communicating with Gemini: {str(e)}\n"
            "Check your API key and network connection."
        )

# ----------------- Main Entrypoint ----------------- #

def main():
    config = load_config()
    token = config.get("TELEGRAM_BOT_TOKEN", "").strip()

    if not token or "PASTE_YOUR" in token:
        print("\n" + "=" * 60)
        print("ERROR: Telegram Bot Token is not configured yet!")
        print(f"Please open: {CONFIG_FILE}")
        print("and replace 'PASTE_YOUR_TELEGRAM_BOT_TOKEN_HERE' with your token from @BotFather.")
        print("=" * 60 + "\n")
        sys.exit(1)

    print("=" * 60)
    print("🔥 AI Mentor Telegram Bot is starting (High Readability Mode)...")
    print("Connected to Telegram & Gemini API")
    print("Press Ctrl+C in this terminal to stop.")
    print("=" * 60)

    # Start health check server for cloud hosting (Render, Koyeb, Hugging Face)
    start_health_server()

    app = ApplicationBuilder().token(token).build()

    # Commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", start_command))
    app.add_handler(CommandHandler("schedule", schedule_command))
    app.add_handler(CommandHandler("habits", habits_command))
    app.add_handler(CommandHandler("homework", add_homework_command))
    app.add_handler(CommandHandler("assignment", add_homework_command))
    app.add_handler(CommandHandler("tasks", list_tasks_command))
    app.add_handler(CommandHandler("todo", list_tasks_command))
    app.add_handler(CommandHandler("done", done_task_command))
    app.add_handler(CommandHandler("clear", clear_command))

    # General Chat / Doubts
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling()

if __name__ == "__main__":
    main()

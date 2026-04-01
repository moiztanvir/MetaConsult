from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi import UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from supabase import create_client, Client
from datetime import datetime, timedelta
from typing import Optional
import os
from dotenv import load_dotenv
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from proposal_agent import ProposalAgent
from planner_agent import PlannerAgent
from agent_workflow import MultiAgentWorkflow
from langchain_core.messages import HumanMessage, AIMessage
import asyncio


# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI()

##
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")


from fastapi.responses import FileResponse

from fastapi.responses import FileResponse

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/login")
def login():
    return FileResponse("static/login.html")

@app.get("/signup")
def signup():
    return FileResponse("static/signup.html")

@app.get("/main")
def main_page():
    return FileResponse("static/main_page.html")
##


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Some of our frontend files used to issue manual OPTIONS "tests".
# Starlette's CORS middleware only treats OPTIONS as a CORS *preflight*
# if `Access-Control-Request-Method` is present; otherwise it reaches routing
# and can return 405. This catch-all keeps those OPTIONS requests from failing.
@app.options("/{full_path:path}")
async def options_ok(full_path: str, request: Request):
    return Response(status_code=204)

# Log all requests and catch errors
@app.middleware("http")
async def log_requests_and_errors(request: Request, call_next):
    logger.debug(f"Incoming request: {request.method} {request.url}")
    logger.debug(f"Request headers: {dict(request.headers)}")
    try:
        response = await call_next(request)
        logger.debug(f"Response status: {response.status_code}")
        logger.debug(f"Response headers: {dict(response.headers)}")
        return response
    except HTTPException as http_exc:
        # Let FastAPI handle HTTP exceptions (e.g., 401, 404) to avoid converting them to 500
        logger.error(
            f"HTTP error processing {request.method} {request.url}: {http_exc.status_code} {http_exc.detail}"
        )
        raise http_exc
    except Exception as e:
        # Re-raise other exceptions so global handlers and CORS middleware can apply
        logger.error(f"Error processing {request.method} {request.url}: {str(e)}")
        raise

# Supabase client
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# JWT settings
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # Changed to 24 hours

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# bcrypt truncates at 72 bytes; passlib raises if longer
MAX_BCRYPT_PASSWORD_BYTES = 72

# Pydantic models
class UserCreate(BaseModel):
    email: str
    password: str
    full_name: str
    company_name: str
    industry: str
    company_size: Optional[str] = None
    revenue: Optional[str] = None
    role: Optional[str] = None
    country: Optional[str] = None
    challenge: Optional[str] = None
    referral: Optional[str] = None
    terms: bool

class UserLogin(BaseModel):
    email: str
    company_name: str
    password: str

class ChatRequest(BaseModel):
    message: str

# LangChain setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
)
# Different prompts for different conversation stages
simple_chat_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a business consultant for Meta Consult. Use the user's company details (company_name: {company_name}, industry: {industry}, challenge: {challenge}) to ask DETAILED QUESTIONS about their business problem. Do NOT provide solutions or advice. Only ask 1-2 focused questions per response to understand their problem better. Keep responses SHORT (2-3 sentences max). Use simple HTML: <p> tags only. Do NOT use markdown or code fences."
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

proposal_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a business consultancy AI for Meta Consult specializing in business problem analysis. Use the user's company details (company_name: {company_name}, industry: {industry}, challenge: {challenge}) to create detailed problem analysis documents. Your role is to analyze and document PROBLEMS and their NEGATIVE IMPACTS on businesses - NOT to provide solutions. When generating problem analysis documents, follow the exact structure: Executive Summary (500 words, paragraphs), Problem Statement (300-500 words, paragraphs), Key Reasons (300-500 words, bullet points), Financial Impact (300 words, bullet points), Operational Impact (300 words, bullet points), Strategic Impact (300 words, bullet points). Total document should be 2000-2500 words. Sections marked as bullet points MUST use bullet point format with detailed points (2-3 sentences each). Focus exclusively on problems and negative impacts. Respond ONLY in Markdown format (not HTML). Use # for main headings, ## for section headings. Use proper markdown formatting with **bold** for emphasis. Do NOT use HTML tags. Do NOT wrap your answer in code fences. Never output ``` or ```html or ```markdown."
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
 
# Prompt template for generating comprehensive solution reports (used by generate_solution_report)
solution_prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert business solutions AI for Meta Consult. Using the provided COMPANY CONTEXT and WEB SEARCH RESULTS, generate a comprehensive, practical, and actionable solution report in Markdown. The report must follow the exact structure and word counts supplied in the input. Ground recommendations in the supplied web search results and be explicit about which sources inform each recommendation. Respond ONLY in Markdown. Do NOT use HTML."
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
memory = ConversationBufferWindowMemory(k=10, return_messages=True)

# Initialize agents
proposal_agent = ProposalAgent()
planner_agent = PlannerAgent()
workflow = MultiAgentWorkflow()

# JWT helper functions
def verify_password(plain_password, hashed_password):
    try:
        # Guard against bcrypt 72-byte limitation and any other verify-time errors
        if plain_password is None:
            return False
        if isinstance(plain_password, str) and len(plain_password.encode("utf-8")) > MAX_BCRYPT_PASSWORD_BYTES:
            return False
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verify error: {str(e)}")
        return False

def get_password_hash(password):
    return pwd_context.hash(password)

def _validate_password_length_or_400(password: str):
    if password is None:
        raise HTTPException(status_code=400, detail="Password is required")
    try:
        password_bytes = password.encode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid password encoding")
    if len(password_bytes) > MAX_BCRYPT_PASSWORD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Password too long. Must be at most {MAX_BCRYPT_PASSWORD_BYTES} bytes.",
        )

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            logger.error("No email in JWT payload")
            raise credentials_exception
        logger.debug(f"JWT decoded successfully for email: {email}")
    except JWTError as e:
        logger.error(f"JWT decode error: {str(e)}")
        raise credentials_exception
    user = supabase.table("users").select("*").eq("email", email).execute()
    if not user.data:
        logger.error(f"No user found for email: {email}")
        raise credentials_exception
    logger.debug(f"User found: {user.data[0]}")
    return user.data[0]

def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    logger.debug(f"Created JWT with expiry: {expire}")
    return encoded_jwt

# ---- Helpers (frontend uses these for Solution tab & downloads) ----
def _safe_email_for_filename(email: str) -> str:
    return (email or "").replace("@", "_").replace(".", "_").replace("+", "_")

def _latest_solution_pdf_for_email(email: str) -> Optional[str]:
    """Return absolute path to newest solution PDF for this email, or None."""
    safe_email = _safe_email_for_filename(email)
    if not safe_email:
        return None
    solution_folder = "problem_proposal"
    if not os.path.exists(solution_folder):
        return None

    import re
    candidates = []
    prefix = f"solution_{safe_email}_"
    for name in os.listdir(solution_folder):
        if not (name.startswith(prefix) and name.endswith(".pdf")):
            continue
        path = os.path.join(solution_folder, name)
        if not os.path.exists(path):
            continue
        try:
            # Prefer timestamp embedded in filename: solution_<safe_email>_YYYYMMDD_HHMMSS.pdf
            ts = None
            m = re.match(rf"^solution_{re.escape(safe_email)}_(\d{{8}}_\d{{6}})\.pdf$", name)
            if m:
                ts = m.group(1)  # lexicographically sortable
            candidates.append((path, ts, os.path.getmtime(path)))
        except OSError:
            continue

    if not candidates:
        return None
    # Sort by (has timestamp, timestamp value, mtime) descending
    candidates.sort(key=lambda x: ((x[1] is not None), (x[1] or ""), x[2]), reverse=True)
    return os.path.abspath(candidates[0][0])

def _solution_prefix_for_email(email: str) -> str:
    safe_email = _safe_email_for_filename(email)
    return f"solution_{safe_email}_" if safe_email else ""

# Routes
@app.post("/signup")
async def signup(user: UserCreate, request: Request):
    logger.debug(f"Received signup request: {user.dict()}")
    if not user.terms:
        logger.error("Terms not agreed")
        raise HTTPException(status_code=400, detail="You must agree to the terms")

    # Avoid bcrypt 72-byte limitation causing 500s
    _validate_password_length_or_400(user.password)

    try:
        existing_user = supabase.table("users").select("*").eq("email", user.email).execute()
        logger.debug(f"Existing user check: {existing_user.data}")
        if existing_user.data:
            logger.error("Email already registered")
            raise HTTPException(status_code=409, detail="Email already registered")
    except Exception as e:
        logger.error(f"Error checking existing user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    hashed_password = get_password_hash(user.password)
    data = {
        "email": user.email,
        "hashed_password": hashed_password,
        "full_name": user.full_name,
        "company_name": user.company_name,
        "industry": user.industry,
        "company_size": user.company_size,
        "revenue": user.revenue,
        "role": user.role,
        "country": user.country,
        "challenge": user.challenge,
        "referral": user.referral,
        "created_at": datetime.utcnow().isoformat(),
        "verified": False
    }
    try:
        logger.debug(f"Inserting user data: {data}")
        response = supabase.table("users").insert(data).execute()
        logger.debug(f"Insert response: {response.data}")
        if response.data:
            return {"message": "User created successfully"}
        logger.error("No data returned from insert")
        raise HTTPException(status_code=500, detail="Error creating user")
    except Exception as e:
        logger.error(f"Error inserting user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/login")
async def login(user: UserLogin, request: Request):
    logger.debug(f"Login attempt for email: {user.email}, company_name: {user.company_name}")
    # Avoid bcrypt 72-byte limitation causing 500s
    _validate_password_length_or_400(user.password)
    try:
        db_user = supabase.table("users").select("*")\
            .eq("email", user.email)\
            .eq("company_name", user.company_name)\
            .execute()
    except Exception as e:
        logger.error(f"Login DB error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    if not db_user.data or not verify_password(user.password, db_user.data[0]["hashed_password"]):
        logger.error("Invalid credentials")
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    logger.debug("Login successful")
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/refresh-token")
async def refresh_token(current_user: dict = Depends(get_current_user)):
    logger.debug(f"Refresh token request for user: {current_user['email']}")
    try:
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": current_user["email"]}, expires_delta=access_token_expires
        )
        logger.debug("Token refreshed successfully")
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Token refresh error: {str(e)}")

def _clean_llm_html(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("\n", 1)[0]
    cleaned = cleaned.replace("```html", "").replace("```HTML", "").replace("```", "").strip()

    # If not HTML-looking, convert basic markdown-ish patterns to HTML
    if "<" not in cleaned and "/>" not in cleaned and "></" not in cleaned and "</" not in cleaned:
        import re
        # Convert bullet blocks to <ul><li>
        lines = cleaned.splitlines()
        html_lines = []
        in_list = False
        for line in lines:
            l = line.strip()
            if l.startswith("- ") or l.startswith("* "):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                item = l[2:]
                item = re.sub(r"\*\*(.*?)\*\*", r"<strong>\\1</strong>", item)
                html_lines.append(f"<li>{item}</li>")
            else:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                if re.match(r"^\*\*(.+)\*\*:?$", l):
                    # Line like **Heading:** -> <h3>Heading</h3>
                    heading = re.sub(r"^\*\*(.+)\*\*:?$", r"\\1", l)
                    html_lines.append(f"<h3>{heading}</h3>")
                elif l:
                    l = re.sub(r"\*\*(.*?)\*\*", r"<strong>\\1</strong>", l)
                    html_lines.append(f"<p>{l}</p>")
                else:
                    html_lines.append("<br>")
        if in_list:
            html_lines.append("</ul>")
        cleaned = "\n".join(html_lines)
    return cleaned

# Removed detect_positive_sentiment function - now using explicit text matching

def count_user_messages() -> int:
    """Count actual user messages in conversation"""
    count = 0
    for msg in memory.chat_memory.messages:
        if hasattr(msg, 'content') and isinstance(msg, HumanMessage):
            count += 1
    return count

def _ensure_chat_session_for_user(current_user: dict) -> None:
    """
    Ensure chat memory and proposal flags are scoped per logged-in user.
    If a different user logs in, reset the conversational context so their
    chat starts fresh from the questioning phase.
    """
    try:
        email = (current_user or {}).get("email", "")
    except Exception:
        email = ""
    last_email = proposal_agent.gathered_info.get("last_chat_email")
    if email and last_email != email:
        logger.info(f"Resetting chat session context from {last_email} to {email}")
        # Clear conversational memory so new user starts fresh
        try:
            memory.chat_memory.messages.clear()
        except Exception as e:
            logger.warning(f"Failed to clear chat memory cleanly: {e}")
        # Reset per-chat flags so new user can go through questioning -> proposal flow
        proposal_agent.store_gathered_info("last_chat_email", email)
        proposal_agent.store_gathered_info("proposal_offered", "false")
        # Do NOT touch stored PDFs on disk; proposal_generated flag is only for flow control
        proposal_agent.store_gathered_info("proposal_generated", "false")

@app.post("/chat")
async def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    logger.debug(f"Chat request from user {current_user['email']}: {request.message}")
    try:
        # Ensure memory/flags are scoped to this user (new login => fresh chat)
        _ensure_chat_session_for_user(current_user)
        # Load chat history
        memory.load_memory_variables({})
        user_message_count = count_user_messages()
        
        # Workflow handling:
        msg_lower = request.message.lower().strip()

        # 0) User-triggered solution download (typed command).
        # IMPORTANT: do NOT include 'yes'/'y' here (those confirm proposal generation).
        if msg_lower in ["download solution", "download", "download pdf", "download report"]:
            solution_pdf_path = _latest_solution_pdf_for_email(current_user.get("email", ""))
            if solution_pdf_path and os.path.exists(solution_pdf_path):
                logger.info(f"User requesting solution PDF download: {solution_pdf_path}")
                filename = os.path.basename(solution_pdf_path)
                return FileResponse(
                    path=solution_pdf_path,
                    filename=filename,
                    media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
                )
            logger.warning("No solution PDF found for user download request")
            error_response = """
            <h3>❌ Solution PDF Not Found</h3>
            <p>Sorry, the solution PDF could not be found. Please try generating the solution again.</p>
            """
            return {"response": error_response, "response_html": error_response}

        # 1) If user manually asks to generate proposal (during conversation phase)
        if msg_lower in ['yes', 'y', 'generate proposal', 'create proposal'] and proposal_agent.gathered_info.get("proposal_offered") == "true" and proposal_agent.gathered_info.get("proposal_generated") != "true":
            # Clear the offered flag and auto-trigger the full workflow
            proposal_agent.store_gathered_info("proposal_offered", "false")
            return await generate_proposal(current_user)

        # 2) Offer proposal generation once conversation is mature.
        # After the threshold, *always* redirect user back to the proposal offer
        # until they confirm with 'yes' / 'generate proposal'.
        if user_message_count >= 3 and proposal_agent.gathered_info.get("proposal_generated") != "true":
            proposal_offer = """
            <p>Based on our detailed conversation, I have gathered comprehensive information about your business challenge.</p>
            <p><strong>Would you like Meta Consult to create a detailed proposal for solving your problem?</strong></p>
            <p>Type 'yes', So I can start the Agentic Workflow to find the solution of your problem. (It can take several minutes)</p>
            """
            memory.save_context({"input": request.message}, {"output": proposal_offer})
            proposal_agent.store_gathered_info("proposal_offered", "true")
            return {"response": proposal_offer, "response_html": proposal_offer}
        
        # Regular chat flow - ask detailed questions without solutions
        company_name = current_user.get("company_name", "your company")
        industry = current_user.get("industry", "your industry")
        challenge = current_user.get("challenge", "your business challenges")
        
        # Add user message to history
        memory.save_context({"input": request.message}, {"output": ""})
        
        # Use simple chat prompt for questioning phase
        chain = simple_chat_prompt | llm
        response = await chain.ainvoke({
            "company_name": company_name,
            "industry": industry,
            "challenge": challenge,
            "input": request.message,
            "history": memory.chat_memory.messages
        })
        
        # Save AI response to history
        memory.save_context({"input": request.message}, {"output": response.content})
        
        html_content = _clean_llm_html(response.content)
        logger.debug(f"Chat response (html): {html_content[:200]}...")
        return {"response": html_content, "response_html": html_content}
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

async def generate_proposal(current_user: dict):
    """Generate proposal PDF only - planner agent processing happens separately"""
    try:
        # Create filename with timestamp and ensure it's stored in problem_proposal folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_email = current_user['email'].replace('@', '_').replace('.', '_').replace('+', '_')
        filename = f"proposal_{safe_email}_{timestamp}.pdf"
        pdf_path = os.path.join("problem_proposal", filename)
        
        # Ensure directory exists
        os.makedirs("problem_proposal", exist_ok=True)
        
        # Generate structured draft from conversation using Gemini
        conversation_summary = await generate_structured_draft(current_user)
        
        # Generate PDF using absolute path
        abs_pdf_path = os.path.abspath(pdf_path)
        generated_path = proposal_agent.generate_proposal_pdf(
            current_user, 
            [conversation_summary],  # Pass structured draft instead of raw conversation
            abs_pdf_path
        )
        
        logger.info(f"PDF generated at: {generated_path}")
        
        # Store the proposal info for future reference
        proposal_agent.store_gathered_info("proposal_path", generated_path)
        # State flags for workflow
        proposal_agent.store_gathered_info("proposal_generated", "true")
        proposal_agent.store_gathered_info("proposal_offered", "false")
        proposal_agent.store_gathered_info("proposal_analyzed", "false")
        
        # Show proposal generation success
        response_html = """
        <h3>✅ Proposal Generated Successfully!</h3>
        <p>Your detailed business proposal has been created and saved.</p>
        <br>
        """
        
        memory.save_context(
            {"input": "Generate proposal"}, 
            {"output": response_html}
        )
        
        # Automatically trigger analysis and web search in background without asking the user
        logger.info("Scheduling background analysis and web search task...")
        status_message = """
        <h3>⚙️ Planner Agent is working...</h3>
        <p>Analyzing your proposal and identifying strategic domains...</p>
        """
        memory.save_context(
            {"input": "Analyzing proposal"},
            {"output": status_message}
        )

        # store initial workflow stage and last html for frontend polling
        proposal_agent.store_gathered_info("workflow_stage", "proposal_generated")
        proposal_agent.store_gathered_info("workflow_last_html", status_message)

        # Run analysis and web search in background so frontend can poll status
        async def _background_analysis_and_search(user_copy: dict):
            try:
                proposal_path = proposal_agent.gathered_info.get("proposal_path")
                if not proposal_path or not os.path.exists(proposal_path):
                    logger.warning("Background task: proposal path missing")
                    proposal_agent.store_gathered_info("workflow_stage", "error")
                    proposal_agent.store_gathered_info("workflow_last_html", "<p>Proposal file missing for background analysis.</p>")
                    return

                # Update stage
                proposal_agent.store_gathered_info("workflow_stage", "analyzing")
                proposal_agent.store_gathered_info("workflow_last_html", status_message)

                # Extract and classify
                proposal_content = planner_agent.extract_proposal_content(proposal_path)
                result = planner_agent.process_proposal(proposal_path)
                domains_list = result["classification"].get("domains", [])
                if not domains_list or not isinstance(domains_list, list):
                    primary_domain = result["classification"].get("primary_domain", "unknown")
                    domains_list = [primary_domain] if primary_domain != "unknown" else []
                valid_domains = [d for d in domains_list if d in planner_agent.domains]
                if not valid_domains:
                    valid_domains = [result["classification"].get("primary_domain", "strategic_planning")]

                domains_str = ",".join(valid_domains) if isinstance(valid_domains, list) else str(valid_domains)
                proposal_agent.store_gathered_info("domain_classification", domains_str)
                proposal_agent.store_gathered_info("primary_domain", valid_domains[0] if valid_domains else "unknown")
                proposal_agent.store_gathered_info("proposal_content_for_search", proposal_content)
                proposal_agent.store_gathered_info("proposal_analyzed", "true")

                # move to web search stage
                proposal_agent.store_gathered_info("workflow_stage", "web_searching")
                web_search_status = """
                <h3>🔍 Web Search Agents are working...</h3>
                <p>Searching the web for solutions across identified domains...</p>
                """
                proposal_agent.store_gathered_info("workflow_last_html", web_search_status)
                memory.save_context({"input": "Searching for solutions"}, {"output": web_search_status})

                # mark pending and execute web search (this will also generate solution PDF)
                proposal_agent.store_gathered_info("web_search_pending", "true")
                try:
                    await get_web_search_results(user_copy)
                except Exception as e:
                    logger.error(f"Background web search failed: {e}")
                    proposal_agent.store_gathered_info("workflow_stage", "error")
                    proposal_agent.store_gathered_info("workflow_last_html", f"<p>Error during web search: {str(e)}</p>")
                    return

                # final stage
                proposal_agent.store_gathered_info("workflow_stage", "ready")
                proposal_agent.store_gathered_info("workflow_last_html", "<p>Solution generation complete.</p>")
                proposal_agent.store_gathered_info("solution_ready_for_download", "true")
            except Exception as bg_e:
                logger.error(f"Background analysis error: {bg_e}", exc_info=True)
                proposal_agent.store_gathered_info("workflow_stage", "error")
                proposal_agent.store_gathered_info("workflow_last_html", f"<p>Background analysis error: {str(bg_e)}</p>")

        # create background task
        try:
            asyncio.create_task(_background_analysis_and_search(dict(current_user)))
        except Exception as task_err:
            logger.error(f"Failed to schedule background task: {task_err}")
            proposal_agent.store_gathered_info("workflow_stage", "error")
            proposal_agent.store_gathered_info("workflow_last_html", "<p>Failed to start background processing.</p>")

        # Return immediate response to frontend
        return {"response": response_html + status_message, "response_html": response_html + status_message, "auto_web_search": True}
        
    except Exception as e:
        logger.error(f"Proposal generation error: {str(e)}")
        error_html = f"""
        <h3>❌ Proposal Generation Failed</h3>
        <p>Sorry, there was an error generating your proposal</p>
        <p>Error details: {str(e)}</p>
        """
        return {"response": error_html, "response_html": error_html}

async def generate_structured_draft(current_user: dict) -> str:
    """Generate a structured business problem draft from conversation using Gemini - focuses on problem and negative impact"""
    try:
        # Get conversation history
        conversation_messages = []
        for msg in memory.chat_memory.messages:
            if hasattr(msg, 'content') and msg.content.strip():
                # Skip system messages and empty content
                if not msg.content.startswith("Based on our conversation") and not msg.content.startswith("Would you like"):
                    conversation_messages.append(msg.content)
        
        # Create a structured prompt for Gemini to generate a business problem draft
        company_name = current_user.get("company_name", "the company")
        industry = current_user.get("industry", "their industry")
        challenge = current_user.get("challenge", "their business challenges")
        
        draft_prompt = f"""
        Based on the following conversation with {company_name} in the {industry} industry, create a comprehensive business problem analysis document.
        
        Company Context:
        - Company: {company_name}
        - Industry: {industry}
        - Primary Challenge: {challenge}
        
        Conversation Summary:
        {chr(10).join(conversation_messages)}
        
        IMPORTANT: This document should focus ONLY on the PROBLEM and its NEGATIVE IMPACT on the company/business. DO NOT include solutions, requirements, or success criteria. This is purely a problem statement document.
        
        Create a detailed business problem analysis using proper Markdown formatting. The document should be approximately 2000-2500 words total.

        FORMAT REQUIREMENTS:
        1. Use # for main title (Executive Summary)
        2. Use ## for section headings
        3. Use proper line breaks between sections (double newline)
        4. Use **bold** for emphasis and key points
        5. Use bullet points (- or *) for listing items - REQUIRED for Key Reasons, Financial Impact, Operational Impact, and Strategic Impact
        6. Use proper indentation and spacing
        
        REQUIRED SECTIONS (Total word count: 2000-2500 words):
        
        1. Executive Summary (exactly 500 words)
           - Provide a comprehensive overview of the problem
           - Summarize the key challenges facing the company
           - Highlight the urgency and severity of the situation
           - Emphasize the negative implications if not addressed
           - Write in paragraph form (not bullet points)
           - Must be exactly 500 words
        
        2. Problem Statement (300-500 words)
           - Provide a detailed, in-depth description of the core problem
           - Explain what the problem is and why it exists
           - Describe how the problem has evolved or worsened over time
           - Include specific examples and evidence from the conversation
           - Detail the complexity and interconnected nature of the issue
           - Write in paragraph form (not bullet points)
        
        3. Key Reasons (300-500 words) - MUST BE IN BULLET POINTS
           - List the root causes and contributing factors that led to this problem
           - Each bullet point should be a clear, specific reason (2-3 sentences per point)
           - Explain why each reason is significant
           - Use bullet points (- or *) for formatting
           - Should have approximately 8-12 bullet points to reach 300-500 words
           - Format: Start with '## Key Reasons' then use bullet points
        
        4. Financial Impact (exactly 300 words) - MUST BE IN BULLET POINTS
           - Analyze how the problem negatively affects the company financially
           - Each bullet point should describe a specific financial impact (2-3 sentences per point)
           - Include: revenue losses, cost increases, profit margin erosion, budget constraints, cash flow problems, investment opportunities missed
           - Use bullet points (- or *) for formatting
           - Should have approximately 10-15 bullet points to reach 300 words
           - Format: Start with '## Financial Impact' then use bullet points
        
        5. Operational Impact (exactly 300 words) - MUST BE IN BULLET POINTS
           - Analyze how the problem negatively affects daily operations
           - Each bullet point should describe a specific operational impact (2-3 sentences per point)
           - Include: process inefficiencies, productivity decline, workflow disruptions, resource allocation issues, quality problems, service delivery failures
           - Use bullet points (- or *) for formatting
           - Should have approximately 10-15 bullet points to reach 300 words
           - Format: Start with '## Operational Impact' then use bullet points
        
        6. Strategic Impact (exactly 300 words) - MUST BE IN BULLET POINTS
           - Analyze how the problem negatively affects long-term strategic goals
           - Each bullet point should describe a specific strategic impact (2-3 sentences per point)
           - Include: growth limitations, expansion barriers, strategic goal delays, long-term sustainability concerns, innovation setbacks, competitive positioning issues
           - Use bullet points (- or *) for formatting
           - Should have approximately 10-15 bullet points to reach 300 words
           - Format: Start with '## Strategic Impact' then use bullet points
        
        CRITICAL INSTRUCTIONS:
        - Focus exclusively on PROBLEMS and NEGATIVE IMPACTS
        - Do NOT mention solutions, recommendations, requirements, or success criteria
        - Use specific, concrete examples and quantified impacts where possible
        - Write in professional, analytical business language
        - Total document must be 2000-2500 words
        - Executive Summary must be exactly 500 words
        - Financial, Operational, and Strategic Impact sections must each be exactly 300 words and in bullet point format
        - Key Reasons must be 300-500 words and in bullet point format
        - Problem Statement must be 300-500 words in paragraph form
        - Use data, metrics, and evidence from the conversation to support claims
        - Emphasize the severity and urgency of the problem
        
        IMPORTANT FORMATTING NOTES:
        - Start with '# Executive Summary'
        - Use '##' for all section headers (Problem Statement, Key Reasons, Financial Impact, Operational Impact, Strategic Impact)
        - Use proper markdown spacing (double newline between sections)
        - Make key points, metrics, and negative impacts bold using **text** within bullet points
        - Key Reasons, Financial Impact, Operational Impact, and Strategic Impact MUST use bullet points (not paragraphs)
        - Each bullet point should be detailed (2-3 sentences) to reach word count targets
        - Executive Summary and Problem Statement should be in paragraph form
        
        Write in professional business language, be specific, detailed, and emphasize the negative consequences.
        FORMAT THE RESPONSE ENTIRELY IN MARKDOWN. Do not use HTML tags.
        """
        
        # Use Gemini to generate structured draft
        chain = proposal_prompt | llm
        response = await chain.ainvoke({
            "company_name": company_name,
            "industry": industry,
            "challenge": challenge,
            "input": draft_prompt,
            "history": []
        })
        
        return response.content
        
    except Exception as e:
        logger.error(f"Error generating structured draft: {str(e)}")
        # Fallback to simple summary
        return f"Business Problem Analysis for {current_user.get('company_name', 'Company')}: {current_user.get('challenge', 'Business challenge identified through conversation')}"

async def analyze_proposal(current_user: dict):
    """Analyze the generated proposal using planner agent and automatically trigger web search"""
    try:
        # Get the stored proposal path
        proposal_path = proposal_agent.gathered_info.get("proposal_path")
        if not proposal_path or not os.path.exists(proposal_path):
            error_html = """
            <h3>❌ No Proposal Found</h3>
            <p>No proposal file found to analyze. Please generate a proposal first.</p>
            """
            return {"response": error_html, "response_html": error_html, "auto_web_search": False}
        
        logger.info(f"Analyzing proposal: {proposal_path}")
        
        # Extract proposal content
        proposal_content = planner_agent.extract_proposal_content(proposal_path)
        
        # Process with planner agent using DeepSeek-R1 for domain classification
        result = planner_agent.process_proposal(proposal_path)
        
        # Get domains from classification
        domains_list = result["classification"].get("domains", [])
        if not domains_list or not isinstance(domains_list, list):
            # Fallback to primary_domain if domains list is empty
            primary_domain = result["classification"].get("primary_domain", "unknown")
            domains_list = [primary_domain] if primary_domain != "unknown" else []
        
        # Filter to only valid domains
        valid_domains = [d for d in domains_list if d in planner_agent.domains]
        if not valid_domains:
            logger.warning("No valid domains found, using fallback")
            valid_domains = [result["classification"].get("primary_domain", "strategic_planning")]
        
        # Store the domain classification and prepare for web search
        domains_str = ",".join(valid_domains) if isinstance(valid_domains, list) else str(valid_domains)
        proposal_agent.store_gathered_info("domain_classification", domains_str)
        proposal_agent.store_gathered_info("primary_domain", valid_domains[0] if valid_domains else "unknown")
        proposal_agent.store_gathered_info("proposal_content_for_search", proposal_content)
        # Mark as analyzed and automatically start web search in background
        proposal_agent.store_gathered_info("proposal_analyzed", "true")
        proposal_agent.store_gathered_info("awaiting_search_confirmation", "false")
        proposal_agent.store_gathered_info("web_search_pending", "true")

        # Kick off background web search so endpoint returns immediately
        try:
            asyncio.create_task(get_web_search_results(dict(current_user)))
        except Exception as bg_err:
            logger.error(f"Failed to schedule background web search: {bg_err}")
            proposal_agent.store_gathered_info("workflow_last_html", "<p>Failed to start web search in background.</p>")

        logger.info(f"Classified domains: {valid_domains}")

        # Return domain explanation and notify that web search has started
        domain_explanation = result["explanation"]
        response_html = f"""
        <h3>🔍 Proposal Analysis Complete!</h3>
        <p>Your business proposal has been analyzed using advanced reasoning.</p>
        <br>
        {domain_explanation}
        """

        # Inform user that web search has been started automatically
        auto_start_msg = """
        <p><strong>Web search has been started automatically for the identified domain(s). You will be notified when results are ready.</strong></p>
        """
        full_response_html = response_html + "<br>" + auto_start_msg
        memory.save_context({"input": "Analyze proposal"}, {"output": full_response_html})
        return {"response": full_response_html, "response_html": full_response_html, "auto_web_search": True}

    except Exception as e:
        logger.error(f"Proposal analysis error: {str(e)}", exc_info=True)
        error_html = f"""
        <h3>❌ Analysis Failed</h3>
        <p>Sorry, there was an error analyzing your proposal: {str(e)}</p>
        <p>Please try again or contact support.</p>
        """
        return {"response": error_html, "response_html": error_html, "auto_web_search": False}

async def generate_solution_report(current_user: dict, detailed_search_results: dict, problem_statement: str, domains: list):
    """Generate a comprehensive solution report using Gemini with web search results
    
    Args:
        current_user: Current user information
        detailed_search_results: Dictionary of domain -> search results with descriptions
        problem_statement: The extracted problem statement from proposal
        domains: List of domains that were searched
        
    Returns:
        Markdown formatted solution report
    """
    try:
        # Build search results summary with descriptions (200-300 words each)
        search_results_summary = "## Web Search Results Summary\n\n"
        
        for domain in domains:
            if domain in detailed_search_results and detailed_search_results[domain]:
                results = detailed_search_results[domain]
                search_results_summary += f"### {domain.replace('_', ' ').title()}\n\n"
                
                for i, result in enumerate(results[:5], 1):  # Top 5 results per domain
                    title = result.get("title", "No title")
                    url = result.get("url", "")
                    description = result.get("description", result.get("content", "No description available"))
                    
                    # Ensure description is 200-300 words
                    words = description.split()
                    if len(words) > 300:
                        description = ' '.join(words[:300]) + "..."
                    
                    search_results_summary += f"**{i}. {title}**\n"
                    if url:
                        search_results_summary += f"Source: {url}\n\n"
                    search_results_summary += f"{description}\n\n"
        
        # Create comprehensive prompt for Gemini to generate solution
        company_name = current_user.get("company_name", "Company")
        industry = current_user.get("industry", "Industry")
        challenge = current_user.get("challenge", "Business challenge")
        
        solution_prompt = f"""Based on the following problem statement and web search results, generate a comprehensive solution report in Markdown format.

## PROBLEM STATEMENT
{problem_statement}

## COMPANY CONTEXT
- Company: {company_name}
- Industry: {industry}
- Challenge: {challenge}

## WEB SEARCH RESULTS (with detailed descriptions 200-300 words each)
{search_results_summary}

## INSTRUCTIONS FOR SOLUTION REPORT
Generate a detailed solution report in Markdown format with the following structure:

1. **Executive Summary** (300-400 words)
   - Overview of the solution approach
   - Key benefits and expected outcomes
   - How this addresses the problem

2. **Detailed Solution** (1000-1500 words in multiple paragraphs)
   - Break down the solution into logical phases/components
   - For each component, explain:
     - What it involves
     - Why it's necessary
     - How it addresses the specific problem
     - Expected outcomes and impact
   - Use data and insights from the web search results
   - Include practical implementation approaches
   - Reference specific solutions and approaches mentioned in search results

3. **Implementation Roadmap** (400-600 words)
   - Phase 1: Initial steps (timeline and actions)
   - Phase 2: Core implementation (timeline and actions)
   - Phase 3: Optimization and scaling (timeline and actions)
   - Include key milestones and success metrics

4. **Risk Mitigation & Challenges** (300-400 words)
   - Potential challenges in implementation
   - How to mitigate each risk
   - Alternative approaches if primary approach faces obstacles

5. **Expected Results & ROI** (300-400 words)
   - Quantifiable outcomes (where possible)
   - Timeline to see results
   - Long-term benefits
   - How success will be measured

CRITICAL REQUIREMENTS:
- Total length: 2500-3500 words
- Use proper Markdown formatting (# for main sections, ## for subsections)
- Include **bold** text for key points and metrics
- Use bullet points where appropriate for lists
- Make it specific to the problem and company context
- Ground all recommendations in the web search results provided
- Use concrete examples and best practices from the search results
- Be actionable and practical, not theoretical
- Assume no technical background unless industry suggests otherwise

Format: Respond ONLY in Markdown. Do not use HTML tags or code fences. Do not include any preamble or meta-commentary."""

        # Call Gemini to generate solution report using the solution-specific prompt template
        chain = solution_prompt_template | llm
        response = await chain.ainvoke({
            "company_name": company_name,
            "industry": industry,
            "challenge": challenge,
            "input": solution_prompt,
            "history": []
        })
        
        markdown_report = response.content
        
        # Store the report in proposal_agent for later access
        proposal_agent.store_gathered_info("solution_report_markdown", markdown_report)
        
        logger.info(f"Solution report generated: {len(markdown_report)} characters")
        return markdown_report
        
    except Exception as e:
        logger.error(f"Error generating solution report: {str(e)}", exc_info=True)
        # Return fallback report
        fallback_report = f"""# Comprehensive Solution Report

## Executive Summary
Based on the analysis of your business problem and web search results, a comprehensive solution has been identified. This report outlines the approach, implementation roadmap, and expected outcomes.

## Problem Analysis
**Problem:** {problem_statement[:500]}

**Company:** {current_user.get('company_name', 'Your Company')}

**Industry:** {current_user.get('industry', 'Your Industry')}

## Solution Overview
A detailed solution incorporating best practices and industry insights has been identified through web research across multiple domains:
{', '.join([d.replace('_', ' ').title() for d in domains])}

## Recommended Actions
1. Review the search results provided for specific solutions and approaches
2. Evaluate which solutions best fit your company's needs and constraints
3. Develop an implementation plan based on the identified solutions
4. Consider phased rollout for complex implementations
5. Monitor and optimize as you implement the solution

## Next Steps
Please contact our team for a detailed consultation on implementing the recommended solutions."""
        
        proposal_agent.store_gathered_info("solution_report_markdown", fallback_report)
        return fallback_report

async def get_web_search_results(current_user: dict):
    """Get web search results for the analyzed proposal - called after domain classification"""
    try:
        # Check if web search is pending
        web_search_pending = proposal_agent.gathered_info.get("web_search_pending")
        logger.info(f"Web search pending status: {web_search_pending}")
        
        if web_search_pending != "true":
            logger.warning(f"Web search not pending. Status: {web_search_pending}")
            return None
        
        proposal_path = proposal_agent.gathered_info.get("proposal_path")
        proposal_content = proposal_agent.gathered_info.get("proposal_content_for_search", "")
        domains_str = proposal_agent.gathered_info.get("domain_classification", "")
        
        logger.info(f"Proposal path: {proposal_path}, Content length: {len(proposal_content) if proposal_content else 0}, Domains: {domains_str}")
        
        if not proposal_path or not proposal_content or not domains_str:
            logger.error(f"Missing required data - path: {bool(proposal_path)}, content: {bool(proposal_content)}, domains: {bool(domains_str)}")
            return None
        
        # Parse domains
        valid_domains = [d.strip() for d in domains_str.split(",") if d.strip() in planner_agent.domains]
        if not valid_domains:
            logger.error(f"No valid domains found. Parsed: {domains_str.split(',')}")
            return None
        
        # Get domain classification from stored data
        domain_classification = {
            "domains": valid_domains,
            "primary_domain": valid_domains[0] if valid_domains else "unknown"
        }
        
        logger.info(f"Running web search for domains: {valid_domains}")
        
        # Run LangGraph workflow for web search
        try:
            workflow_result = workflow.run(
                proposal_path=proposal_path,
                proposal_content=proposal_content,
                domains=valid_domains,
                domain_classification=domain_classification
            )
            logger.info(f"Workflow completed. Result keys: {list(workflow_result.keys()) if isinstance(workflow_result, dict) else 'Not a dict'}")
        except Exception as workflow_error:
            logger.error(f"Workflow execution error: {str(workflow_error)}", exc_info=True)
            return None
        
        logger.info(f"Workflow result keys: {workflow_result.keys() if isinstance(workflow_result, dict) else 'Not a dict'}")
        
        # Mark web search as complete
        proposal_agent.store_gathered_info("web_search_pending", "false")
        
        web_search_results_html = workflow_result.get("final_response", "")
        detailed_search_results = workflow_result.get("detailed_search_results", {})
        problem_statement = workflow_result.get("problem_statement", "")
        
        logger.info(f"Web search results HTML length: {len(web_search_results_html)}")
        
        if web_search_results_html:
            # Generate solution report using Gemini
            logger.info("Generating comprehensive solution report...")
            solution_markdown = await generate_solution_report(
                current_user=current_user,
                detailed_search_results=detailed_search_results,
                problem_statement=problem_statement,
                domains=valid_domains
            )
            
            # Generate solution PDF from markdown report
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_email = current_user['email'].replace('@', '_').replace('.', '_').replace('+', '_')
                solution_filename = f"solution_{safe_email}_{timestamp}.pdf"
                solution_pdf_path = os.path.join("problem_proposal", solution_filename)
                os.makedirs("problem_proposal", exist_ok=True)
                
                abs_solution_pdf_path = os.path.abspath(solution_pdf_path)
                generated_solution_path = proposal_agent.generate_solution_pdf(
                    markdown_report=solution_markdown,
                    output_path=abs_solution_pdf_path
                )
                
                logger.info(f"Solution PDF generated at: {generated_solution_path}")
                
                # Store solution PDF path for download
                proposal_agent.store_gathered_info("solution_pdf_path", generated_solution_path)
                
                response_html = f"""
                <h3>🌐 Web Search Results & Comprehensive Solution Report</h3>
                <p>I've searched the web for solutions related to your problem in the identified domain(s). Here are the top results:</p>
                <br>
                {web_search_results_html}
                <br>
                <h3>📄 Comprehensive Solution Report Generated</h3>
                <p>Based on the web search results and your specific problem, I've generated a comprehensive solution report with:</p>
                <ul>
                    <li>Executive summary of the solution approach</li>
                    <li>Detailed solution with implementation details</li>
                    <li>Implementation roadmap with phases and milestones</li>
                    <li>Risk mitigation strategies</li>
                    <li>Expected results and ROI analysis</li>
                </ul>
                <br>
                    <p><strong>Your solution PDF is ready:</strong></p>
                    <p><a href="javascript:void(0)" onclick="openSolutionPDF()" style="display: inline-block; padding: 10px 15px; background-color: #3b82f6; color: white; border-radius: 5px; text-decoration: none; font-weight: bold; cursor: pointer;"> Download Solution Report (PDF)</a></p>
                """
                
                memory.save_context(
                    {"input": "Web search results"}, 
                    {"output": response_html}
                )

                # Save final html for frontend polling
                proposal_agent.store_gathered_info("workflow_final_html", response_html)
                proposal_agent.store_gathered_info("workflow_last_html", response_html)
                
                # Store state for download handling
                proposal_agent.store_gathered_info("solution_ready_for_download", "true")
                
                logger.info("Solution report generated and PDF created successfully")
                return {"response": response_html, "response_html": response_html}
                
            except Exception as pdf_error:
                logger.error(f"Error generating solution PDF: {str(pdf_error)}", exc_info=True)
                # Still return search results even if PDF generation fails
                response_html = f"""
                <h3>🌐 Web Search Results</h3>
                <p>I've searched the web for solutions related to your problem in the identified domain(s). Here are the top results:</p>
                <br>
                {web_search_results_html}
                <p><em>Note: Solution report PDF generation encountered an issue. Please contact support for assistance.</em></p>
                """
                
                memory.save_context(
                    {"input": "Web search results"}, 
                    {"output": response_html}
                )
                proposal_agent.store_gathered_info("workflow_final_html", response_html)
                proposal_agent.store_gathered_info("workflow_last_html", response_html)
                return {"response": response_html, "response_html": response_html}
        else:
            logger.warning("Web search results HTML is empty")
            return None
        
    except Exception as e:
        logger.error(f"Web search error: {str(e)}", exc_info=True)
        return None

# ---- File upload chat ----
def _read_text_from_upload(filename: str, content: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".txt"):
        return content.decode("utf-8", errors="ignore")
    if name.endswith(".csv"):
        import pandas as pd
        from io import StringIO
        try:
            df = pd.read_csv(StringIO(content.decode("utf-8", errors="ignore")))
            return df.to_csv(index=False)
        except Exception:
            return content.decode("utf-8", errors="ignore")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        import pandas as pd
        from io import BytesIO
        df = pd.read_excel(BytesIO(content))
        return df.to_csv(index=False)
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        from io import BytesIO
        reader = PdfReader(BytesIO(content))
        pages_text = []
        for page in reader.pages:
            pages_text.append(page.extract_text() or "")
        return "\n\n".join(pages_text).strip()
    if name.endswith(".docx") or name.endswith(".doc"):
        import docx
        from io import BytesIO
        doc = docx.Document(BytesIO(content))
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n".join(paragraphs)
    # Fallback: best-effort decode
    return content.decode("utf-8", errors="ignore")

@app.post("/chat-with-file")
async def chat_with_file(
    message: str = Form(""),
    file: UploadFile = File(None),
    current_user: dict = Depends(get_current_user),
):
    logger.debug(f"Chat-with-file request from user {current_user['email']} with message length {len(message)} and file {file.filename if file else 'None'}")
    try:
        # Ensure memory/flags are scoped to this user (new login => fresh chat)
        _ensure_chat_session_for_user(current_user)
        # Load chat history
        memory.load_memory_variables({})
        user_message_count = count_user_messages()

        # Workflow handling for file-based chat:
        msg_lower = message.lower().strip()

        # 0) User-triggered solution download (typed command).
        # IMPORTANT: do NOT include 'yes'/'y' here (those confirm proposal generation).
        if msg_lower in ["download solution", "download", "download pdf", "download report"]:
            solution_pdf_path = _latest_solution_pdf_for_email(current_user.get("email", ""))
            if solution_pdf_path and os.path.exists(solution_pdf_path):
                logger.info(f"User requesting solution PDF download: {solution_pdf_path}")
                filename = os.path.basename(solution_pdf_path)
                return FileResponse(
                    path=solution_pdf_path,
                    filename=filename,
                    media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
                )
            logger.warning("No solution PDF found for user download request")
            error_response = """
            <h3>❌ Solution PDF Not Found</h3>
            <p>Sorry, the solution PDF could not be found. Please try generating the solution again.</p>
            """
            return {"response": error_response, "response_html": error_response}

        # 1) / 2) Background automation: confirmations removed. Workflow runs automatically after proposal generation.

        # 3) Direct analysis command
        if msg_lower in ['analyze proposal', 'analyze', 'analysis'] and proposal_agent.gathered_info.get("proposal_path"):
            return await analyze_proposal(current_user)

        # 4) If user confirmed generation after being offered
        if msg_lower in ['yes', 'y', 'ye', 'generate proposal', 'create proposal'] and proposal_agent.gathered_info.get("proposal_offered") == "true" and proposal_agent.gathered_info.get("proposal_generated") != "true":
            # Clear the offered flag (we'll generate now)
            proposal_agent.store_gathered_info("proposal_offered", "false")
            return await generate_proposal(current_user)

        # 5) Offer proposal generation once conversation is mature.
        # After the threshold, keep directing the user back to the proposal offer
        # until they explicitly confirm with 'yes'.
        if user_message_count >= 3 and proposal_agent.gathered_info.get("proposal_generated") != "true":
            proposal_offer = """
            <p>Based on our conversation and the file you've shared, I have a comprehensive understanding of your business challenge.</p>
            <p><strong>Would you like Meta Consult to create a detailed proposal for solving your problem?</strong></p>
            <p>Type 'yes', So I can start the Agentic Workflow to find the solution of your problem. (It can take several minutes).</p>
            """
            memory.save_context({"input": message}, {"output": proposal_offer})
            proposal_agent.store_gathered_info("proposal_offered", "true")
            return {"response": proposal_offer, "response_html": proposal_offer}

        company_name = current_user.get("company_name", "your company")
        industry = current_user.get("industry", "your industry")
        challenge = current_user.get("challenge", "your business challenges")

        extracted_text = ""
        if file is not None:
            try:
                content = await file.read()
                extracted_text = _read_text_from_upload(file.filename, content)
                logger.info(f"Successfully extracted text from {file.filename}: {len(extracted_text)} characters")
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                extracted_text = f"[Error processing file: {str(e)}]"

        combined_input = message or ""
        if extracted_text:
            combined_input = (
                f"User message: {message}\n\n"
                f"Attached file '{file.filename}' content (first 7,000 characters):\n"
                f"{extracted_text[:7000]}"
            )

        memory.save_context({"input": combined_input}, {"output": ""})

        # Use simple chat prompt for questioning phase
        chain = simple_chat_prompt | llm
        response = await chain.ainvoke({
            "company_name": company_name,
            "industry": industry,
            "challenge": challenge,
            "input": combined_input,
            "history": memory.chat_memory.messages,
        })

        memory.save_context({"input": combined_input}, {"output": response.content})

        html_content = _clean_llm_html(response.content)
        logger.debug("Chat-with-file response generated")
        return {"response": html_content, "response_html": html_content}
    except Exception as e:
        logger.error(f"Chat-with-file error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat-with-file error: {str(e)}")

# Test endpoint for Supabase
@app.get("/test-supabase")
async def test_supabase():
    try:
        response = supabase.table("users").select("*").limit(1).execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# Download solution PDF endpoint
@app.get("/download-solution")
async def download_solution(current_user: dict = Depends(get_current_user)):
    """Download the generated solution PDF"""
    try:
        # Always resolve by CURRENT USER EMAIL (never shared in-memory path).
        user_email = current_user.get("email", "")
        solution_pdf_path = _latest_solution_pdf_for_email(user_email)
        
        if not solution_pdf_path or not os.path.exists(solution_pdf_path):
            logger.warning(f"No solution PDF found for user {current_user['email']}")
            raise HTTPException(
                status_code=404,
                detail=f"Solution PDF not found for account '{user_email}'. Please generate a solution first."
            )
        
        logger.info(f"Serving solution PDF for user {current_user['email']}: {solution_pdf_path}")
        
        filename = os.path.basename(solution_pdf_path)
        return FileResponse(
            path=solution_pdf_path,
            filename=filename,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=\"{filename}\"",
                "Cache-Control": "no-store",
                "Pragma": "no-cache",
                "X-MetaConsult-User": user_email,
            },
        )
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error downloading solution PDF: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading solution PDF: {str(e)}"
        )

# Workflow status endpoint for frontend polling
@app.get("/workflow-status")
async def workflow_status(current_user: dict = Depends(get_current_user)):
    try:
        stage = proposal_agent.gathered_info.get("workflow_stage", "idle")
        last_html = proposal_agent.gathered_info.get("workflow_last_html", "")
        solution_ready = proposal_agent.gathered_info.get("solution_ready_for_download") == "true"
        final_html = proposal_agent.gathered_info.get("workflow_final_html", "")
        solution_path = proposal_agent.gathered_info.get("solution_pdf_path")
        return JSONResponse({
            "stage": stage,
            "last_html": last_html,
            "final_html": final_html,
            "solution_ready": solution_ready,
            "solution_pdf_path": solution_path
        })
    except Exception as e:
        logger.error(f"Error getting workflow status: {e}")
        return JSONResponse({"stage": "error", "last_html": "", "final_html": "", "solution_ready": False})

# Debug endpoint to inspect generated solution markdown
@app.get("/debug-solution-markdown")
async def debug_solution_markdown(current_user: dict = Depends(get_current_user)):
    """Debug endpoint: return the generated solution markdown for inspection"""
    try:
        solution_markdown = proposal_agent.gathered_info.get("solution_report_markdown")
        if not solution_markdown:
            return {"error": "No solution markdown found. Generate a solution first."}
        
        return {
            "solution_markdown": solution_markdown,
            "length": len(solution_markdown),
            "preview": solution_markdown[:500] + "..." if len(solution_markdown) > 500 else solution_markdown
        }
    except Exception as e:
        logger.error(f"Error retrieving solution markdown: {str(e)}")
        return {"error": f"Error retrieving solution markdown: {str(e)}"}

# User profile endpoint for dashboard
@app.get("/user-profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    """Get current user profile data for dashboard"""
    try:
        # Return user data without sensitive information
        return {
            "full_name": current_user.get("full_name", ""),
            "company_name": current_user.get("company_name", ""),
            "industry": current_user.get("industry", ""),
            "company_size": current_user.get("company_size", ""),
            "revenue": current_user.get("revenue", ""),
            "role": current_user.get("role", ""),
            "country": current_user.get("country", ""),
            "challenge": current_user.get("challenge", ""),
            "email": current_user.get("email", ""),
            "created_at": current_user.get("created_at", "")
        }
    except Exception as e:
        logger.error(f"Error getting user profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving user profile: {str(e)}")

# Solution PDF status endpoint
@app.get("/solution-status")
async def get_solution_status(current_user: dict = Depends(get_current_user)):
    """Check if solution PDF exists for current user - returns the most recent one (scoped by email)."""
    try:
        user_email = current_user.get("email", "")
        solution_pdf_path = _latest_solution_pdf_for_email(user_email)
        if solution_pdf_path and os.path.exists(solution_pdf_path):
            return JSONResponse({
                "exists": True,
                "filename": os.path.basename(solution_pdf_path),
                "path": solution_pdf_path,
                "account_email": user_email,
                "expected_prefix": _solution_prefix_for_email(user_email),
            }, headers={"Cache-Control": "no-store"})
        
        return JSONResponse({
            "exists": False,
            "filename": None,
            "path": None,
            "account_email": user_email,
            "expected_prefix": _solution_prefix_for_email(user_email),
        }, headers={"Cache-Control": "no-store"})
    except Exception as e:
        logger.error(f"Error checking solution status: {str(e)}")
        return JSONResponse({
            "exists": False,
            "filename": None,
            "path": None,
            "account_email": current_user.get("email", ""),
            "expected_prefix": _solution_prefix_for_email(current_user.get("email", "")),
            "error": str(e),
        }, headers={"Cache-Control": "no-store"})

# MetaConsult — AI-Powered Business Consultancy Platform

> **Final Year Project** | Democratizing high-quality business consultancy through multi-agent AI workflows.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.27-orange)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-purple)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5--flash-red?logo=google)](https://deepmind.google/technologies/gemini/)
[![Supabase](https://img.shields.io/badge/Supabase-Auth%20%26%20DB-green?logo=supabase)](https://supabase.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?logo=vercel)](https://f25-236-r-meta-consult-final-ii.vercel.app)

---

## Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Agentic Workflow](#agentic-workflow)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Agent Descriptions](#agent-descriptions)
- [API Reference](#api-reference)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Variables](#environment-variables)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
- [Database Schema](#database-schema)
- [Frontend Pages](#frontend-pages)
- [Security](#security)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**MetaConsult** is an innovative AI-powered business consultancy platform that revolutionizes how companies approach problem-solving and strategic planning. Instead of replacing human consultants outright, MetaConsult augments the process by deploying a pipeline of specialized AI agents — each responsible for a distinct phase of the consultancy lifecycle — to produce comprehensive, data-driven solution reports tailored to each client's specific context.

The platform is built on a **multi-agent architecture** orchestrated with **LangGraph**, backed by **Google Gemini 2.5 Flash** for language understanding and generation, and enriched with **live web search** to incorporate the latest market trends, industry benchmarks, and real-world case studies.

### Our Mission

To democratize access to high-quality business consultancy by combining the power of artificial intelligence with deep domain expertise — helping businesses of all sizes identify problems, analyze root causes, and develop actionable solutions that drive growth, optimize operations, and enhance competitive positioning.

---

## Live Demo

> 🌐 **[https://f25-236-r-meta-consult-final-ii.vercel.app](https://f25-236-r-meta-consult-final-ii.vercel.app)**

---

## Key Features

- **Conversational Problem Discovery** — An AI consultant engages the user through a targeted Q&A session to fully understand the business challenge before any analysis begins.
- **Structured Problem Proposal (PDF)** — After sufficient information is gathered, a 2,000–2,500-word professionally formatted problem analysis PDF is auto-generated, covering the executive summary, problem statement, key reasons, and financial/operational/strategic impacts.
- **Intelligent Domain Classification** — A Planner Agent classifies the problem into one or more strategic domains (Market Research, Strategic Planning, Technology, Management) using advanced reasoning.
- **Parallel Web Search by Specialized Agents** — Domain-specific web search agents independently query the web using Tavily and DuckDuckGo to gather the latest solutions, case studies, and best practices.
- **Comprehensive Solution Report (PDF)** — A synthesis agent consolidates web research and domain expertise into a complete, downloadable solution report with implementation roadmaps, risk mitigation strategies, and expected ROI.
- **Background Async Processing** — Long-running agent tasks (web search, PDF generation) run asynchronously using Python's `asyncio`, so the user interface remains responsive throughout.
- **JWT Authentication** — Secure user registration and login backed by Supabase, with 24-hour JSON Web Tokens.
- **Per-User Session Isolation** — Chat memory and workflow state are scoped per logged-in user, preventing cross-session data leakage.
- **Repeat Workflow Support** — Users can trigger a new consultation cycle at any time without restarting the application.
- **Docker & Cloud Ready** — Containerized with Docker and deployed on cloud infrastructure; frontend served via Vercel.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Browser                           │
│          (HTML/CSS/JS — Served from /static via FastAPI)        │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTPS / REST API
┌────────────────────────────▼────────────────────────────────────┐
│                     FastAPI Backend (main.py)                    │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │  Auth Layer │  │  Chat Router │  │  Workflow Orchestrator  │  │
│  │  JWT + bcrypt│  │  LangChain   │  │  (asyncio background)  │  │
│  └──────┬──────┘  └──────┬───────┘  └────────────┬───────────┘  │
│         │                │                        │              │
└─────────┼────────────────┼────────────────────────┼─────────────┘
          │                │                        │
          ▼                ▼                        ▼
   ┌─────────────┐  ┌─────────────┐      ┌──────────────────────┐
   │  Supabase   │  │   Gemini    │      │  LangGraph Workflow   │
   │  (Auth + DB)│  │  2.5 Flash  │      │  (agent_workflow.py) │
   └─────────────┘  └─────────────┘      └──────────┬───────────┘
                                                     │
                         ┌───────────────────────────┼───────────────────────────┐
                         ▼                           ▼                           ▼
               ┌──────────────────┐      ┌─────────────────────┐     ┌──────────────────┐
               │  Proposal Agent  │      │   Planner Agent     │     │  Web Search      │
               │ (proposal_agent) │      │ (planner_agent.py)  │     │  Agents (x4)     │
               │  PDF via ReportLab│      │ Domain Classification│     │  Tavily/DDG Search│
               └──────────────────┘      └─────────────────────┘     └──────────────────┘
```

---

## Agentic Workflow

MetaConsult's core value is its **6-stage agentic pipeline** that transforms a business conversation into a downloadable solution report:

```
Stage 1: Conversation Phase
  └── Gemini 2.5 Flash + LangChain ConversationBufferWindowMemory
  └── AI asks targeted questions to understand the business challenge
  └── Threshold: after 3+ user messages, proposal is offered

Stage 2: Proposal Generation
  └── generate_structured_draft() → 2,000–2,500 word Markdown document
  └── ProposalAgent.generate_proposal_pdf() → PDF via ReportLab
  └── Sections: Executive Summary | Problem Statement | Key Reasons |
               Financial Impact | Operational Impact | Strategic Impact

Stage 3: Domain Classification (Planner Agent)
  └── PlannerAgent.extract_proposal_content() → reads PDF text
  └── PlannerAgent.process_proposal() → classifies into domains:
        • market_research
        • strategic_planning
        • management
        • technology

Stage 4: Web Search (Specialized Agents in LangGraph)
  └── MarketResearchWebSearchAgent
  └── StrategicPlanningWebSearchAgent
  └── ManagementWebSearchAgent
  └── TechnologyWebSearchAgent
  └── Each agent independently searches using Tavily + DuckDuckGo

Stage 5: Solution Report Generation
  └── generate_solution_report() synthesizes all web search results
  └── Gemini 2.5 Flash generates 4,000–5,000 word Markdown report
  └── Converted to PDF using ReportLab

Stage 6: Document Delivery
  └── PDF stored in /problem_proposal/ directory
  └── User can download via chat command or UI button
  └── Frontend polls /workflow-status endpoint for progress updates
```

All stages 3–6 run in the background via `asyncio.create_task()`, and the frontend polls the `/workflow-status` endpoint to display real-time progress indicators.

---

## Project Structure

```
MetaConsult/
│
├── main.py                            # FastAPI application entry point
│                                      #   - All REST endpoints (auth, chat, workflow)
│                                      #   - LangChain chat logic & prompt templates
│                                      #   - Background async orchestration
│                                      #   - JWT auth helpers, Supabase client
│
├── agent_workflow.py                  # LangGraph multi-agent workflow
│                                      #   - StateGraph definition (AgentState TypedDict)
│                                      #   - Nodes: extract_problem_statement,
│                                      #            classify_domains, web_search,
│                                      #            format_response
│                                      #   - Coordinates all 4 web search agents
│
├── proposal_agent.py                  # Proposal PDF generation agent
│                                      #   - Formats structured Markdown → ReportLab PDF
│                                      #   - Stores workflow state flags (gathered_info)
│                                      #   - Per-user PDF naming with timestamps
│
├── planner_agent.py                   # Domain classification agent
│                                      #   - Extracts text from proposal PDF
│                                      #   - Classifies problem into strategic domains
│                                      #   - Returns structured classification with explanation
│
├── web_search_agent_market_research.py    # Market Research web search agent
├── web_search_agent_strategic_planning.py # Strategic Planning web search agent
├── web_search_agent_management.py         # Management web search agent
├── web_search_agent_technology.py         # Technology web search agent
│                                          # Each agent:
│                                          #   - Constructs domain-specific search queries
│                                          #   - Queries Tavily API + DuckDuckGo fallback
│                                          #   - Returns top-5 results with descriptions
│                                          #   - Formats results as HTML for chat display
│
├── api/                               # API utility modules (helpers, schemas)
│
├── static/                            # Frontend static assets
│   ├── index.html                     # Landing / home page
│   ├── login.html                     # Login page
│   ├── signup.html                    # User registration (multi-step form)
│   └── main_page.html                 # Main consultancy chat interface
│
├── problem_proposal/                  # Generated PDF storage (auto-created at runtime)
│                                      # Files named: proposal_<email>_<timestamp>.pdf
│                                      #              solution_<email>_<timestamp>.pdf
│
├── index.html                         # Root HTML redirect
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker container definition
└── .env                               # Environment variables (not committed)
```

---

## Tech Stack

### Backend

| Technology | Purpose |
|---|---|
| **Python 3.11** | Core language |
| **FastAPI** | REST API framework with async support |
| **Uvicorn** | ASGI server |
| **LangChain 0.3.27** | LLM orchestration, prompt templates, conversation memory |
| **LangGraph** | Multi-agent state graph workflow |
| **Google Gemini 2.5 Flash** | Primary LLM (via `langchain-google-genai`) |
| **Tavily Python** | Primary web search API for agents |
| **DuckDuckGo Search** | Fallback web search |
| **ReportLab** | Programmatic PDF generation |
| **Supabase** | PostgreSQL database + Auth backend |
| **python-jose** | JWT encoding/decoding |
| **passlib + bcrypt** | Secure password hashing |
| **python-dotenv** | Environment variable management |
| **pypdf** | PDF text extraction (for Planner Agent) |
| **markdown2** | Markdown to HTML conversion |

### Frontend

| Technology | Purpose |
|---|---|
| **HTML5 / CSS3 / Vanilla JS** | Chat interface, login/signup forms |
| **Static file serving via FastAPI** | Pages served from `/static/` directory |

### Infrastructure

| Technology | Purpose |
|---|---|
| **Docker** | Containerization (Python 3.11-slim, port 10000) |
| **Vercel** | Frontend deployment |
| **Supabase** | Managed PostgreSQL + Auth |

---

## Agent Descriptions

### 1. Conversation Agent (Inline in `main.py`)
Powered by **Gemini 2.5 Flash** with `ConversationBufferWindowMemory` (last 10 messages). Uses the `simple_chat_prompt` template to ask 1–2 targeted questions per response without offering solutions. Scoped per user via session isolation logic.

### 2. Proposal Agent (`proposal_agent.py`)
Triggered after the user confirms with `"yes"` or `"generate proposal"`. Calls `generate_structured_draft()` which sends the full conversation to Gemini with a detailed Markdown-format prompt, producing a 2,000–2,500 word problem analysis document. The Markdown output is then rendered into a PDF using **ReportLab** and saved to `problem_proposal/`.

### 3. Planner Agent (`planner_agent.py`)
Reads the generated proposal PDF using **pypdf**, extracts the text, and sends it to the LLM to classify the business problem into one or more of four strategic domains: `market_research`, `strategic_planning`, `management`, `technology`. Returns a structured classification object with an explanation string for display in the chat.

### 4. Web Search Agents (`web_search_agent_*.py`)
Four specialized agents, one per domain. Each is instantiated in `MultiAgentWorkflow` and operates as a LangGraph node. They construct targeted search queries from the problem statement, call the Tavily API (with DuckDuckGo as fallback), retrieve the top 5 results per domain, and return both raw data (for report generation) and formatted HTML (for chat display).

### 5. Multi-Agent Workflow (`agent_workflow.py`)
A **LangGraph `StateGraph`** that wires the four web search agents into a sequential pipeline: `extract_problem_statement → classify_domains → web_search → format_response`. The shared `AgentState` TypedDict carries proposal content, domains, raw results, and detailed results through the graph. The workflow's `run()` method returns all data needed for solution report generation.

---

## API Reference

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/signup` | Register a new user. Accepts full profile (company, industry, challenge, etc.) |
| `POST` | `/login` | Authenticate with email, company name, and password. Returns JWT. |
| `POST` | `/refresh-token` | Refresh an active JWT token (requires valid Bearer token). |

### Chat & Workflow

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send a message to the AI consultant. Handles all workflow stages. |
| `GET` | `/workflow-status` | Poll for background task progress (stages: `idle`, `analyzing`, `web_searching`, `ready`, `error`). |
| `GET` | `/download-solution` | Download the latest solution PDF for the authenticated user. |
| `GET` | `/download-proposal` | Download the latest proposal PDF for the authenticated user. |

### Pages (HTML)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Landing page |
| `GET` | `/login` | Login page |
| `GET` | `/signup` | Signup page |
| `GET` | `/main_page` | Main consultancy chat interface |

### Chat Commands

The `/chat` endpoint recognizes several special text commands:

| Command | Action |
|---------|--------|
| `yes` / `y` / `generate proposal` | Confirms and triggers the full agentic workflow |
| `download solution` / `download pdf` | Downloads the latest solution PDF |
| `new report` / `another report` / `start over` | Resets all workflow state for a new consultation cycle |

---

## Getting Started

### Prerequisites

- Python 3.11+
- A [Supabase](https://supabase.com/) project with a `users` table
- A [Google AI Studio](https://aistudio.google.com/) API key for Gemini
- A [Tavily](https://app.tavily.com/) API key for web search
- Docker (optional, for containerized deployment)

### Environment Variables

Create a `.env` file in the project root with the following keys:

```env
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-or-service-key

# JWT
SECRET_KEY=your-random-secret-key-at-least-32-chars

# AI
GEMINI_API_KEY=your-google-gemini-api-key

# Web Search
TAVILY_API_KEY=your-tavily-api-key
```

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/moiztanvir/MetaConsult.git
cd MetaConsult

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create the .env file with your keys (see above)

# 5. Create the PDF output directory
mkdir -p problem_proposal

# 6. Run the application
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 7. Open your browser at:
#    http://localhost:8000
```

### Docker Setup

```bash
# Build the Docker image
docker build -t metaconsult .

# Run the container (pass your .env file)
docker run --env-file .env -p 10000:10000 metaconsult

# App will be available at:
# http://localhost:10000
```

---

## Database Schema

MetaConsult uses a single `users` table in Supabase (PostgreSQL):

```sql
CREATE TABLE users (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email           TEXT UNIQUE NOT NULL,
  hashed_password TEXT NOT NULL,
  full_name       TEXT,
  company_name    TEXT NOT NULL,
  industry        TEXT,
  company_size    TEXT,
  revenue         TEXT,
  role            TEXT,
  country         TEXT,
  challenge       TEXT,        -- Primary business challenge (pre-fills AI context)
  referral        TEXT,
  created_at      TIMESTAMPTZ DEFAULT NOW(),
  verified        BOOLEAN DEFAULT FALSE
);
```

The `company_name`, `industry`, and `challenge` fields are injected into every LLM prompt to personalize the AI consultant's questions and the generated reports.

---

## Frontend Pages

### Landing Page (`/`)
Marketing landing page introducing MetaConsult's capabilities and workflow.

### Signup Page (`/signup`)
Multi-field registration form collecting company profile information including industry, company size, revenue range, primary business challenge, and referral source. All fields feed into the AI context for personalized consultancy.

### Login Page (`/login`)
Email + company name + password authentication. Stores the JWT in browser local storage for subsequent API calls.

### Main Chat Interface (`/main_page`)
The primary consultancy interface. Features a real-time chat window, a workflow progress indicator, and download buttons for Proposal and Solution PDFs. The frontend polls `/workflow-status` every few seconds while background agents are running to display live status updates.

---

## Security

- **Passwords** are hashed with bcrypt (via passlib) with a 72-byte input guard to prevent bcrypt truncation vulnerabilities.
- **JWT tokens** are signed with HS256, expire after 24 hours, and are validated on every protected endpoint via the `Depends(get_current_user)` FastAPI dependency.
- **Per-user session isolation** — when a different user logs in on the same server instance, the conversation memory and all workflow flags are cleared for the new user.
- **CORS** is configured to allow all origins (`"*"`) for development; this should be restricted to your frontend domain in production.
- **PDF filenames** are sanitized using the authenticated user's email address (replacing `@`, `.`, `+` with `_`) and include a timestamp to prevent collisions.

---

## Contributing

This is a final year academic project. Contributions, suggestions, and feedback are welcome.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is developed as a Final Year Project. All rights reserved by the authors.

---

*Built with ❤️ using FastAPI, LangChain, LangGraph, and Google Gemini.*

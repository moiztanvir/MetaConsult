"""
LangGraph Workflow for Multi-Agent Orchestration
Coordinates Proposal Agent, Planner Agent, and Web Search Agents
"""
import logging
from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, END
from web_search_agent_market_research import MarketResearchWebSearchAgent
from web_search_agent_strategic_planning import StrategicPlanningWebSearchAgent
from web_search_agent_management import ManagementWebSearchAgent
from web_search_agent_technology import TechnologyWebSearchAgent

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State passed between agents in the workflow"""
    proposal_path: str
    proposal_content: str
    problem_statement: str
    domains: List[str]
    domain_classification: Dict
    web_search_results: Dict[str, List[Dict]]  # domain -> results
    detailed_search_results: Dict[str, List[Dict]]  # domain -> detailed results with descriptions
    final_response: str

class MultiAgentWorkflow:
    """LangGraph workflow for orchestrating multiple agents"""
    
    def __init__(self):
        # Initialize web search agents
        self.web_search_agents = {
            "market_research": MarketResearchWebSearchAgent(),
            "strategic_planning": StrategicPlanningWebSearchAgent(),
            "management": ManagementWebSearchAgent(),
            "technology": TechnologyWebSearchAgent()
        }
        
        # Build the graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("extract_problem_statement", self._extract_problem_statement)
        workflow.add_node("classify_domains", self._classify_domains)
        workflow.add_node("web_search", self._web_search)
        workflow.add_node("format_response", self._format_response)
        
        # Define the flow
        workflow.set_entry_point("extract_problem_statement")
        workflow.add_edge("extract_problem_statement", "classify_domains")
        workflow.add_edge("classify_domains", "web_search")
        workflow.add_edge("web_search", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile()
    
    def _extract_problem_statement(self, state: AgentState) -> AgentState:
        """Extract problem statement from proposal content"""
        try:
            proposal_content = state.get("proposal_content", "")
            
            # Extract Problem Statement section from markdown content
            problem_statement = self._parse_problem_statement(proposal_content)
            
            if not problem_statement:
                # Fallback: use first 500 characters of proposal
                problem_statement = proposal_content[:500]
                logger.warning("Could not extract Problem Statement section, using proposal excerpt")
            
            state["problem_statement"] = problem_statement
            logger.info(f"Extracted problem statement: {len(problem_statement)} characters")
            
        except Exception as e:
            logger.error(f"Error extracting problem statement: {str(e)}")
            state["problem_statement"] = state.get("proposal_content", "")[:500]
        
        return state
    
    def _parse_problem_statement(self, content: str) -> str:
        """Parse Problem Statement section from markdown content"""
        try:
            # Look for "## Problem Statement" or "Problem Statement" section
            lines = content.split('\n')
            problem_start = -1
            problem_end = -1
            
            for i, line in enumerate(lines):
                line_lower = line.lower().strip()
                # Check for Problem Statement heading
                if ("## problem statement" in line_lower or 
                    "# problem statement" in line_lower or
                    (line_lower.startswith("problem statement") and "##" in line)):
                    problem_start = i + 1
                    break
            
            if problem_start == -1:
                # Try to find it without markdown
                for i, line in enumerate(lines):
                    if "problem statement" in line.lower() and len(line.strip()) < 50:
                        problem_start = i + 1
                        break
            
            if problem_start > -1:
                # Find the end of the section (next heading or end of content)
                for i in range(problem_start, len(lines)):
                    line = lines[i].strip()
                    # Check if we hit another section heading
                    if line.startswith("##") or line.startswith("#"):
                        if "problem statement" not in line.lower():
                            problem_end = i
                            break
                
                if problem_end == -1:
                    problem_end = min(problem_start + 50, len(lines))  # Limit to ~50 lines
                
                problem_text = '\n'.join(lines[problem_start:problem_end]).strip()
                # Clean up markdown formatting
                problem_text = problem_text.replace('**', '').replace('*', '')
                return problem_text
            
            return ""
            
        except Exception as e:
            logger.error(f"Error parsing problem statement: {str(e)}")
            return ""
    
    def _classify_domains(self, state: AgentState) -> AgentState:
        """Classify domains using planner agent (this is a placeholder - actual classification happens in main.py)"""
        # This node expects domains to already be set in state
        # The actual classification is done in analyze_proposal before calling workflow
        domains = state.get("domains", [])
        if not domains:
            logger.warning("No domains provided to workflow")
            state["domains"] = []
        
        return state
    
    def _web_search(self, state: AgentState) -> AgentState:
        """Perform web search for each domain"""
        domains = state.get("domains", [])
        problem_statement = state.get("problem_statement", "")
        web_search_results = {}
        
        if not domains:
            logger.warning("No domains to search for")
            state["web_search_results"] = {}
            return state
        
        logger.info(f"Performing web search for domains: {domains}")
        
        # Search for each domain
        for domain in domains:
            if domain in self.web_search_agents:
                try:
                    agent = self.web_search_agents[domain]
                    results = agent.search(problem_statement)
                    web_search_results[domain] = results
                    logger.info(f"Found {len(results)} results for domain: {domain}")
                except Exception as e:
                    logger.error(f"Error searching domain {domain}: {str(e)}")
                    web_search_results[domain] = []
            else:
                logger.warning(f"Unknown domain: {domain}, skipping web search")
                web_search_results[domain] = []
        
        state["web_search_results"] = web_search_results
        return state
    
    def _format_response(self, state: AgentState) -> AgentState:
        """Format the final response with all web search results and store detailed search data for report generation"""
        web_search_results = state.get("web_search_results", {})
        domain_classification = state.get("domain_classification", {})
        domains = state.get("domains", [])
        problem_statement = state.get("problem_statement", "")
        
        # Start with domain classification explanation
        response_parts = []
        
        # Store detailed search results for later report generation
        detailed_results = {}
        
        # Add web search results for each domain
        for domain in domains:
            if domain in web_search_results and web_search_results[domain]:
                agent = self.web_search_agents.get(domain)
                if agent:
                    formatted = agent.format_results_for_chat(web_search_results[domain])
                    response_parts.append(formatted)
                    
                    # Store detailed results with descriptions for solution report generation
                    detailed_results[domain] = web_search_results[domain]
        
        # Combine all parts
        final_response = "\n<br><br>\n".join(response_parts)
        
        if not final_response:
            final_response = "<p>No web search results found. Please try again later.</p>"
        
        # Store detailed search results in state for solution report generation
        state["detailed_search_results"] = detailed_results
        state["final_response"] = final_response
        return state
    
    def run(self, proposal_path: str, proposal_content: str, domains: List[str], domain_classification: Dict) -> Dict:
        """
        Run the complete workflow
        
        Args:
            proposal_path: Path to the proposal PDF
            proposal_content: Extracted text content from proposal
            domains: List of domain keys (e.g., ["market_research", "strategic_planning"])
            domain_classification: Full classification result from planner agent
            
        Returns:
            Dictionary with final_response, web_search_results, detailed_search_results, and problem_statement
        """
        try:
            initial_state: AgentState = {
                "proposal_path": proposal_path,
                "proposal_content": proposal_content,
                "problem_statement": "",
                "domains": domains,
                "domain_classification": domain_classification,
                "web_search_results": {},
                "detailed_search_results": {},
                "final_response": ""
            }
            
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            
            return {
                "final_response": final_state.get("final_response", ""),
                "web_search_results": final_state.get("web_search_results", {}),
                "detailed_search_results": final_state.get("detailed_search_results", {}),
                "problem_statement": final_state.get("problem_statement", "")
            }
            
        except Exception as e:
            logger.error(f"Error running workflow: {str(e)}", exc_info=True)
            return {
                "final_response": f"<p>Error in workflow execution: {str(e)}</p>",
                "web_search_results": {},
                "detailed_search_results": {},
                "problem_statement": ""
            }


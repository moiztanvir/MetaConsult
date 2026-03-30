"""
Web Search Agent for Management & Operations Domain
"""
import logging
import os
import requests
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class ManagementWebSearchAgent:
    """Web search agent specialized in Management & Operations domain"""
    
    def __init__(self):
        self.domain_name = "Management & Operations"
        self.domain_key = "management"
        # Use Tavily API for web search (free tier available)
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.tavily_base_url = "https://api.tavily.com/search"
        
    def search(self, problem_statement: str, query: Optional[str] = None) -> List[Dict]:
        """
        Search the web for management and operations solutions related to the problem
        
        Args:
            problem_statement: The problem statement from the proposal PDF
            query: Optional specific search query (if not provided, will be generated)
            
        Returns:
            List of 5 best search results with title, url, and content
        """
        try:
            # Generate search query if not provided
            if not query:
                query = self._generate_search_query(problem_statement)
            
            logger.info(f"Management Agent: Searching for '{query}'")
            
            # Use Tavily API for web search
            if self.tavily_api_key:
                results = self._search_with_tavily(query)
            else:
                # Fallback to DuckDuckGo if Tavily not available
                logger.warning("Tavily API key not found, using DuckDuckGo fallback")
                results = self._search_with_duckduckgo(query)
            
            # Filter and rank results for management relevance
            filtered_results = self._filter_management_results(results, problem_statement)
            
            # Return top 5 results
            return filtered_results[:5]
            
        except Exception as e:
            logger.error(f"Error in Management web search: {str(e)}")
            return self._get_fallback_results(problem_statement)
    
    def _generate_search_query(self, problem_statement: str) -> str:
        """Generate a focused search query for management domain"""
        # Extract key terms from problem statement
        problem_lower = problem_statement.lower()
        
        # Management specific keywords
        keywords = []
        if any(term in problem_lower for term in ["management", "team", "employee", "leadership"]):
            keywords.append("management solutions")
        if any(term in problem_lower for term in ["operations", "operational", "process", "workflow"]):
            keywords.append("operations management")
        if any(term in problem_lower for term in ["efficiency", "productivity", "optimization"]):
            keywords.append("operational efficiency")
        if any(term in problem_lower for term in ["resource", "allocation", "capacity"]):
            keywords.append("resource management")
        
        # Build query
        base_query = "management operations solutions"
        if keywords:
            query = f"{' '.join(keywords[:2])} {base_query} business problem"
        else:
            query = f"{problem_statement[:100]} management operations solutions"
        
        return query
    
    def _search_with_tavily(self, query: str) -> List[Dict]:
        """Search using Tavily API and fetch detailed descriptions (200-300 words) for each result"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            data = {
                "api_key": self.tavily_api_key,
                "query": query,
                "search_depth": "advanced",
                "include_answer": True,
                "max_results": 10
            }
            
            response = requests.post(
                self.tavily_base_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                search_results = []
                
                # Extract results
                for item in result.get("results", []):
                    content = item.get("content", "")
                    # Ensure content is at least 200-300 words; if too short, fetch full page
                    if content and len(content.split()) < 150:
                        url = item.get("url", "")
                        if url:
                            expanded_content = self._fetch_page_description(url, content)
                            if expanded_content:
                                content = expanded_content
                    
                    search_results.append({
                        "title": item.get("title", "No title"),
                        "url": item.get("url", ""),
                        "content": content,
                        "description": self._create_detailed_description(content, 250),
                        "score": item.get("score", 0.0)
                    })
                
                # Also include answer if available
                if result.get("answer"):
                    search_results.insert(0, {
                        "title": "Management & Operations Insights",
                        "url": "",
                        "content": result["answer"],
                        "description": self._create_detailed_description(result["answer"], 250),
                        "score": 1.0
                    })
                
                return search_results
            else:
                logger.error(f"Tavily API error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Tavily search error: {str(e)}")
            return []
    
    def _fetch_page_description(self, url: str, existing_content: str) -> str:
        """Attempt to fetch and expand page description to 200-300 words"""
        try:
            response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code == 200:
                # Try to extract main content from HTML
                from html.parser import HTMLParser
                class MetaParser(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self.description = ""
                        self.capturing_text = False
                    
                    def handle_starttag(self, tag, attrs):
                        if tag in ['p', 'div', 'article']:
                            self.capturing_text = True
                    
                    def handle_data(self, data):
                        if self.capturing_text and len(self.description) < 500:
                            self.description += data + " "
                
                parser = MetaParser()
                try:
                    parser.feed(response.text)
                    if parser.description:
                        return existing_content + " " + parser.description[:300]
                except:
                    pass
            return existing_content
        except Exception as e:
            logger.warning(f"Failed to fetch page description from {url}: {str(e)}")
            return existing_content
    
    def _create_detailed_description(self, content: str, target_words: int = 250) -> str:
        """Create a detailed description (200-300 words) from content"""
        if not content:
            return "No description available."
        
        # Clean up content
        content = content.strip()
        content = ' '.join(content.split())  # Normalize whitespace
        
        # Split by sentences
        sentences = content.split('. ')
        
        # Build description by adding sentences until we reach target word count
        description_parts = []
        word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= target_words + 50:  # Allow some flexibility
                description_parts.append(sentence)
                word_count += sentence_words
            elif word_count > target_words - 50:  # Close enough to target
                break
        
        description = '. '.join(description_parts)
        if description and not description.endswith('.'):
            description += '.'
        
        # Fallback if description is too short
        if len(description.split()) < 100:
            description = content[:500] if len(content) >= 500 else content
        
        return description if description else "No description available."

    def _search_with_duckduckgo(self, query: str) -> List[Dict]:
        """Fallback search using DuckDuckGo (no API key required) with description expansion"""
        try:
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    max_results=10,
                    safesearch='moderate'
                ))
            
            search_results = []
            for item in results:
                content = item.get("body", "")
                description = self._create_detailed_description(content, 250)
                search_results.append({
                    "title": item.get("title", "No title"),
                    "url": item.get("href", ""),
                    "content": content,
                    "description": description,
                    "score": 0.8
                })
            
            return search_results
            
        except ImportError:
            logger.warning("duckduckgo_search not installed, returning empty results")
            return []
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {str(e)}")
            return []
    
    def _filter_management_results(self, results: List[Dict], problem_statement: str) -> List[Dict]:
        """Filter and rank results for management relevance"""
        # Keywords that indicate management relevance
        relevant_keywords = [
            "management", "operations", "operational efficiency", "process optimization",
            "team management", "resource management", "workflow", "productivity", "leadership"
        ]
        
        scored_results = []
        for result in results:
            score = result.get("score", 0.5)
            title_lower = result.get("title", "").lower()
            content_lower = result.get("content", "").lower()
            
            # Boost score for relevant keywords
            for keyword in relevant_keywords:
                if keyword in title_lower:
                    score += 0.2
                if keyword in content_lower:
                    score += 0.1
            
            scored_results.append({
                **result,
                "score": min(score, 1.0)  # Cap at 1.0
            })
        
        # Sort by score descending
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results
    
    def _get_fallback_results(self, problem_statement: str) -> List[Dict]:
        """Return fallback results if search fails"""
        return [
            {
                "title": "Management & Operations Solutions",
                "url": "",
                "content": f"Based on your problem: {problem_statement[:200]}... Management and operations solutions typically involve process optimization, team management, resource allocation, workflow automation, and operational efficiency improvements to enhance productivity and reduce costs.",
                "score": 0.5
            }
        ]
    
    def format_results_for_chat(self, results: List[Dict]) -> str:
        """Format search results as HTML for chat display with clickable links and 2-3 line descriptions"""
        if not results:
            return "<p>No search results found for Management & Operations domain.</p>"
        
        html = f"""
        <h3>🔍 Management & Operations - Top 5 Solutions</h3>
        <p>Here are the best solutions and insights I found for your management and operations challenges:</p>
        <ul style="list-style-type: none; padding-left: 0;">
        """
        
        for i, result in enumerate(results[:5], 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            content = result.get("content", "")
            
            # Extract 2-3 lines description (approximately 150-200 characters, 2-3 sentences)
            sentences = content.split('. ')
            description = '. '.join(sentences[:3])
            if description and not description.endswith('.'):
                description += '.'
            if len(description) > 200:
                description = description[:197] + '...'
            
            if url:
                html += f"""
                <li style="margin-bottom: 20px; padding: 15px; background-color: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6; border-radius: 4px;">
                    <strong style="font-size: 16px;">
                        <a href="{url}" target="_blank" rel="noopener noreferrer" style="color: #3b82f6; text-decoration: none; transition: color 0.3s;">
                            {title}
                        </a>
                    </strong>
                    <br>
                    <a href="{url}" target="_blank" rel="noopener noreferrer" style="color: #3b82f6; font-size: 14px; text-decoration: none; opacity: 0.8; transition: opacity 0.3s;">
                        {url}
                    </a>
                    <p style="margin-top: 8px; margin-bottom: 0; color: #d1d5db; line-height: 1.6;">
                        {description if description else 'No description available.'}
                    </p>
                </li>
                """
            else:
                html += f"""
                <li style="margin-bottom: 20px; padding: 15px; background-color: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6; border-radius: 4px;">
                    <strong style="font-size: 16px; color: #3b82f6;">{title}</strong>
                    <p style="margin-top: 8px; margin-bottom: 0; color: #d1d5db; line-height: 1.6;">
                        {description if description else 'No description available.'}
                    </p>
                </li>
                """
        
        html += "</ul>"
        return html


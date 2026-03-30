"""
Planner Agent - Uses DeepSeek-R1 for reasoning and domain classification
(Fixed for OpenRouter integration)
"""
import logging
import requests
import json
import os
from typing import Dict
from pypdf import PdfReader

# Optional gemini fallback via langchain Google adapter
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    ChatGoogleGenerativeAI = None
    ChatPromptTemplate = None

logger = logging.getLogger(__name__)

class PlannerAgent:
    def __init__(self):
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.deepseek_base_url = "https://openrouter.ai/api/v1"
        self.domains = {
            "market_research": "Market Research & Analysis",
            "strategic_planning": "Strategic Planning & Business Development",
            "management": "Management & Operations",
            "technology": "Technology & Digital Transformation"
        }

    def extract_proposal_content(self, pdf_path: str) -> str:
        """Extract text content from the proposal PDF with robust error handling"""
        import os
        
        # Validate path
        if not pdf_path or not isinstance(pdf_path, str):
            raise ValueError("Invalid PDF path provided")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            reader = PdfReader(pdf_path)
            text_content = []

            # Handle empty PDFs
            if not reader.pages:
                logger.warning(f"PDF file appears to be empty: {pdf_path}")
                return "PDF file is empty or could not be read."

            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and isinstance(page_text, str):
                        text_content.append(page_text.strip())
                except Exception as page_error:
                    logger.warning(f"Error extracting text from page {i+1}: {str(page_error)}")
                    continue

            if not text_content:
                logger.warning(f"No text content could be extracted from PDF: {pdf_path}")
                return "PDF file contains no extractable text content."

            extracted_text = "\n\n".join(text_content)
            
            # Ensure we have some content
            if not extracted_text or len(extracted_text.strip()) < 10:
                logger.warning(f"Extracted text is too short or empty from PDF: {pdf_path}")
                return "PDF file contains minimal or unreadable text content."

            return extracted_text
            
        except FileNotFoundError:
            raise
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error reading PDF: {str(e)}")
            raise Exception(f"Failed to read proposal PDF: {str(e)}")

    def classify_problem_domain(self, proposal_content: str) -> Dict:
        """Use DeepSeek-R1 (via OpenRouter) or Gemini to classify the problem domain(s) with reasoning. Returns 1-3 domains."""
        # Validate content
        if not proposal_content or not isinstance(proposal_content, str):
            logger.warning("Empty or invalid proposal content provided")
            return self._fallback_classification("")
        
        proposal_content = proposal_content.strip()
        if len(proposal_content) < 10:
            logger.warning("Proposal content is too short or minimal")
            return self._fallback_classification(proposal_content)
        
        # Try Gemini first if available (more reliable than DeepSeek via OpenRouter)
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key and ChatGoogleGenerativeAI is not None and ChatPromptTemplate is not None:
            logger.info("Attempting Gemini classification first")
            gemini_result = self._gemini_fallback_classification(proposal_content)
            if gemini_result and gemini_result.get("primary_domain") != "retry":
                logger.info("Gemini classification successful")
                return gemini_result
        
        # If Gemini failed or not available, try DeepSeek
        if not self.deepseek_api_key:
            logger.warning("DeepSeek API key not found, using fallback classification")
            return self._fallback_classification(proposal_content)

        try:
            logger.info("Using DeepSeek R1 model for domain classification")

            # Truncate content to avoid hitting token limits
            truncated_content = proposal_content[:3000]

            prompt = f"""
            You are an expert business consultant. Analyze the following business proposal and classify the problem domains.

            PROPOSAL CONTENT:
            {truncated_content}

            CLASSIFY the problem into ONE, TWO, OR THREE domains from these options:
            1. market_research - Problems related to market analysis, customer research, competitive intelligence
            2. strategic_planning - Problems related to business strategy, growth planning, market expansion
            3. management - Problems related to operations, team or employee management, process optimization
            4. technology - Problems related to digital transformation, software performance, automation, technical implementation

            IMPORTANT: Select 1-3 domains that best represent the problem. If the problem spans multiple areas, include all relevant domains. If it clearly fits one domain, select only that domain.

            Respond with a JSON object containing:
            {{
                "domains": ["array of 1-3 domain names from the four options above"],
                "primary_domain": "the most relevant domain (for backward compatibility)",
                "confidence": "confidence level (0-100)",
                "reasoning": "detailed explanation of at least 300 words explaining why these domain(s) were chosen",
                "key_indicators": ["list of key phrases or indicators that led to this classification"]
            }}

            Example response format:
            {{
                "domains": ["strategic_planning", "management"],
                "primary_domain": "strategic_planning",
                "confidence": 85,
                "reasoning": "The proposal indicates...",
                "key_indicators": ["business strategy", "team management"]
            }}
            """

            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json",
                # ✅ Required headers for OpenRouter attribution
                "HTTP-Referer": "http://localhost",  
                "X-Title": "PlannerAgent"
            }

            data = {
                "model": "deepseek/deepseek-r1:free",  # ✅ Correct OpenRouter model name
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7
            }

            response = requests.post(
                f"{self.deepseek_base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=90  # ✅ Increased timeout for DeepSeek reasoning
            )

            # ✅ Log full response text for debugging
            if response.status_code != 200:
                logger.error(f"DeepSeek API error {response.status_code}: {response.text}")
                # Try Gemini fallback before giving up
                if gemini_key and ChatGoogleGenerativeAI is not None:
                    logger.info("Retrying with Gemini after DeepSeek failure")
                    gemini_result = self._gemini_fallback_classification(proposal_content)
                    if gemini_result and gemini_result.get("primary_domain") != "retry":
                        return gemini_result
                # Use enhanced fallback classification instead of retry
                return self._fallback_classification(proposal_content)

            result = response.json()
            
            # Check if response has expected structure
            if "choices" not in result or not result["choices"]:
                logger.error(f"Unexpected DeepSeek response structure: {result}")
                if gemini_key and ChatGoogleGenerativeAI is not None:
                    gemini_result = self._gemini_fallback_classification(proposal_content)
                    if gemini_result and gemini_result.get("primary_domain") != "retry":
                        return gemini_result
                return self._fallback_classification(proposal_content)
            
            content = result["choices"][0]["message"]["content"]

            # ✅ Try parsing JSON safely
            try:
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_content = content[start_idx:end_idx]
                    classification = json.loads(json_content)
                    
                    # Validate and normalize the response
                    classification = self._normalize_classification(classification)
                    
                    # Ensure we have valid domains
                    if classification.get("primary_domain") in self.domains:
                        logger.info(f"Domain classification result: {classification}")
                        return classification
                    else:
                        logger.warning(f"Invalid domain in classification: {classification}")
                        # Try Gemini fallback
                        if gemini_key and ChatGoogleGenerativeAI is not None:
                            gemini_result = self._gemini_fallback_classification(proposal_content)
                            if gemini_result and gemini_result.get("primary_domain") != "retry":
                                return gemini_result
                        return self._fallback_classification(proposal_content)
                else:
                    logger.warning("Could not find JSON in DeepSeek response")
                    if gemini_key and ChatGoogleGenerativeAI is not None:
                        gemini_result = self._gemini_fallback_classification(proposal_content)
                        if gemini_result and gemini_result.get("primary_domain") != "retry":
                            return gemini_result
                    return self._fallback_classification(proposal_content)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to decode JSON from DeepSeek model response: {str(e)}, content: {content[:200]}")
                # Try Gemini fallback
                if gemini_key and ChatGoogleGenerativeAI is not None:
                    gemini_result = self._gemini_fallback_classification(proposal_content)
                    if gemini_result and gemini_result.get("primary_domain") != "retry":
                        return gemini_result
                return self._fallback_classification(proposal_content)

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error calling DeepSeek API: {str(e)}")
            # Try Gemini fallback
            if gemini_key and ChatGoogleGenerativeAI is not None:
                logger.info("Retrying with Gemini after network error")
                gemini_result = self._gemini_fallback_classification(proposal_content)
                if gemini_result and gemini_result.get("primary_domain") != "retry":
                    return gemini_result
            return self._fallback_classification(proposal_content)
        except Exception as e:
            logger.error(f"Error in domain classification: {str(e)}", exc_info=True)
            # Try Gemini fallback on unexpected exceptions from DeepSeek flow
            if gemini_key and ChatGoogleGenerativeAI is not None:
                logger.info("Retrying with Gemini after unexpected error")
                gemini_result = self._gemini_fallback_classification(proposal_content)
                if gemini_result and gemini_result.get("primary_domain") != "retry":
                    return gemini_result
            # Use enhanced fallback instead of retry message
            return self._fallback_classification(proposal_content)

    def _normalize_classification(self, classification: Dict) -> Dict:
        """Normalize classification response to ensure it has required fields and valid domains"""
        valid_domains = list(self.domains.keys())
        
        # Handle retry/error cases - don't normalize them, let caller handle
        if classification.get("primary_domain") == "retry":
            return classification
        
        # Ensure domains array exists
        if "domains" not in classification:
            # If only primary_domain exists, create domains array
            if "primary_domain" in classification:
                classification["domains"] = [classification["primary_domain"]]
            else:
                classification["domains"] = []
        
        # Ensure domains is a list
        if not isinstance(classification["domains"], list):
            if isinstance(classification["domains"], str):
                classification["domains"] = [classification["domains"]]
            else:
                classification["domains"] = []
        
        # Validate and filter domains (keep only 1-3 valid domains)
        valid_selected = [d for d in classification["domains"] if d in valid_domains]
        
        # Ensure we have at least 1 domain and at most 3
        if len(valid_selected) == 0:
            logger.warning("No valid domains found in classification, will use fallback")
            # Don't set to unknown here - return as-is so caller can use fallback
            return classification
        elif len(valid_selected) > 3:
            logger.warning(f"Too many domains ({len(valid_selected)}), keeping first 3")
            valid_selected = valid_selected[:3]
        
        classification["domains"] = valid_selected
        
        # Ensure primary_domain exists (use first domain)
        if "primary_domain" not in classification or classification["primary_domain"] not in valid_domains:
            classification["primary_domain"] = valid_selected[0]
        
        # Ensure other required fields exist
        if "confidence" not in classification or not isinstance(classification["confidence"], (int, float)):
            classification["confidence"] = 75
        if "reasoning" not in classification or not classification["reasoning"]:
            classification["reasoning"] = "Analysis completed based on proposal content."
        if "key_indicators" not in classification or not isinstance(classification["key_indicators"], list):
            classification["key_indicators"] = []
        
        return classification

    def _gemini_fallback_classification(self, proposal_content: str) -> Dict:
        """Use Gemini (gemini-2.0-flash) for domain classification.

        This attempts to use the same reasoning prompt but via the Gemini model. If Gemini is
        not available (missing adapter or API key) or the generation fails, return None so
        the caller can decide the next step.
        """
        try:
            if ChatGoogleGenerativeAI is None or ChatPromptTemplate is None:
                logger.warning("Gemini adapter not available in environment; cannot use Gemini.")
                return None

            gemini_key = os.getenv("GEMINI_API_KEY")
            if not gemini_key:
                logger.warning("GEMINI_API_KEY not set; cannot use Gemini.")
                return None

            logger.info("Attempting Gemini (gemini-2.0-flash) for domain classification")

            # Prepare prompt similar to DeepSeek's prompt but tailored for Gemini with multiple domains
            truncated = proposal_content[:4000] if len(proposal_content) > 4000 else proposal_content
            system_message = (
                "You are an expert business consultant. Analyze the following business proposal and classify the problem domains. "
                "Respond ONLY with a valid JSON object (no surrounding text, no code blocks, no markdown) that contains keys: domains, primary_domain, confidence, reasoning, key_indicators. "
                "domains must be an array of 1-3 domain names from: market_research, strategic_planning, management, technology. "
                "primary_domain must be one of the selected domains (the most relevant one). "
                "confidence must be a number between 0-100. "
                "Provide a detailed reasoning of at least 200 words explaining why these domain(s) were chosen. "
                "key_indicators must be an array of strings."
            )

            user_message = f"""PROPOSAL CONTENT:
{truncated}

CLASSIFY the problem into ONE, TWO, OR THREE domains from these options:
1. market_research - Problems related to market analysis, customer research, competitive intelligence
2. strategic_planning - Problems related to business strategy, growth planning, market expansion
3. management - Problems related to operations, team or employee management, process optimization
4. technology - Problems related to digital transformation, software performance, automation, technical implementation

Select 1-3 domains that best represent the problem. If the problem spans multiple areas, include all relevant domains. If it clearly fits one domain, select only that domain.

Respond with ONLY a JSON object, no other text."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", "{input}")
            ])

            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_key, temperature=0.1)
            chain = prompt | llm

            # Synchronous invoke (use .invoke when running synchronously)
            try:
                response = chain.invoke({"input": user_message})
            except Exception as e:
                logger.error(f"Gemini invoke failed with chain: {e}, trying direct invoke")
                # Some langchain adapters expose .generate or .invoke differently; try alternate call
                try:
                    response = llm.invoke(user_message)
                except Exception as ee:
                    logger.error(f"Gemini direct invoke also failed: {ee}")
                    return None

            content = None
            if isinstance(response, dict):
                content = response.get("content") or response.get("text")
            else:
                # Attempt to access attribute
                content = getattr(response, 'content', None) or getattr(response, 'text', None)

            if not content:
                logger.warning("Gemini returned empty content for classification.")
                return None

            # Clean content - remove markdown code blocks if present
            content_str = str(content).strip()
            if content_str.startswith("```"):
                # Remove code blocks
                lines = content_str.split("\n")
                content_str = "\n".join([line for line in lines if not line.strip().startswith("```")])

            # Parse JSON from content
            try:
                start = content_str.find('{')
                end = content_str.rfind('}') + 1
                if start != -1 and end > start:
                    json_text = content_str[start:end]
                    classification = json.loads(json_text)
                    
                    # Validate we got required fields
                    if "domains" not in classification and "primary_domain" not in classification:
                        logger.warning(f"Gemini response missing required fields: {classification}")
                        return None
                    
                    # Normalize the classification
                    classification = self._normalize_classification(classification)
                    
                    # Double-check we have valid domains
                    if classification.get("primary_domain") in self.domains or classification.get("primary_domain") == "retry":
                        logger.info(f"Gemini classification successful: {classification.get('domains', [classification.get('primary_domain')])}")
                        return classification
                    else:
                        logger.warning(f"Gemini returned invalid domain: {classification.get('primary_domain')}")
                        return None
                else:
                    logger.warning(f"Could not locate JSON object in Gemini response. Content: {content_str[:200]}")
                    return None
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing Gemini JSON: {e}, content: {content_str[:500]}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error parsing Gemini response: {e}")
                return None

        except Exception as e:
            logger.error(f"Unexpected error in Gemini classification: {e}", exc_info=True)
            return None

    def _handle_failed_analysis(self, proposal_content: str) -> Dict:
        """Handle cases where DeepSeek API fails and prompt user to retry."""
        # Internal warning only; do not expose service or technical failure messages to users.
        logger.warning("Model analysis failed to produce a valid classification. Asking user to retry.")
        # Return a neutral, user-friendly instruction to retry without revealing internal service names.
        return {
            "domains": ["retry"],
            "primary_domain": "retry",
            "confidence": 0,
            "reasoning": "I couldn't complete the analysis just now. Would you like me to try again?",
            "key_indicators": ["analysis_incomplete"]
        }

    def _fallback_classification(self, proposal_content: str) -> Dict:
        """Fallback classification when DeepSeek API is not available"""
        # Keep a non-technical fallback; do not mention DeepSeek or other services in user-facing text.
        logger.warning("Providing a generic fallback classification due to unavailable reasoning model.")
        
        # Try to do a simple keyword-based classification as fallback
        content_lower = proposal_content.lower()
        detected_domains = []
        
        if any(keyword in content_lower for keyword in ["market", "customer", "competitor", "research", "survey"]):
            detected_domains.append("market_research")
        if any(keyword in content_lower for keyword in ["strategy", "strategic", "growth", "expansion", "planning", "business plan"]):
            detected_domains.append("strategic_planning")
        if any(keyword in content_lower for keyword in ["management", "team", "employee", "operations", "process", "workflow"]):
            detected_domains.append("management")
        if any(keyword in content_lower for keyword in ["technology", "software", "digital", "automation", "system", "tech", "platform"]):
            detected_domains.append("technology")
        
        # Limit to 3 domains and ensure at least one
        if not detected_domains:
            detected_domains = ["unknown"]
        elif len(detected_domains) > 3:
            detected_domains = detected_domains[:3]
        
        return {
            "domains": detected_domains,
            "primary_domain": detected_domains[0],
            "confidence": 50 if detected_domains[0] != "unknown" else 0,
            "reasoning": "I couldn't determine the domain automatically with full analysis. Based on basic keyword matching, I identified relevant domains. You can retry the analysis or provide more context to help me classify it more accurately.",
            "key_indicators": []
        }

    def explain_domain_to_user(self, classification: Dict, proposal_content: str) -> str:
        """Generate a detailed explanation of the problem domain(s)"""
        # Get domains array (1-3 domains)
        domains_list = classification.get("domains", [])
        if not domains_list or not isinstance(domains_list, list):
            # Fallback to primary_domain if domains not available
            primary = classification.get("primary_domain", "unknown")
            domains_list = [primary]
        
        # Validate domains
        valid_domains = [d for d in domains_list if d in self.domains]
        if not valid_domains:
            valid_domains = [classification.get("primary_domain", "unknown")]
        
        # Get domain names
        domain_names = [self.domains.get(d, d.replace("_", " ").title()) for d in valid_domains]
        
        reasoning = classification.get("reasoning", "Based on the analysis of your business proposal.")
        confidence = classification.get("confidence", 75)
        key_indicators = classification.get("key_indicators", [])
        
        # Build explanation based on number of domains
        if len(valid_domains) == 1:
            explanation = f"""
            <h3>{domain_names[0]} Domain</h3>
            <p>Your business challenge falls under the <strong>{domain_names[0]}</strong> domain. Here's why:</p>
            <p><strong>Reasoning:</strong> {reasoning}</p>
            """
        elif len(valid_domains) == 2:
            explanation = f"""
            <h3>Multi-Domain Analysis</h3>
            <p>Your business challenge spans <strong>two</strong> key domains:</p>
            <ul>
                <li><strong>{domain_names[0]}</strong></li>
                <li><strong>{domain_names[1]}</strong></li>
            </ul>
            <p><strong>Reasoning:</strong> {reasoning}</p>
            """
        else:  # 3 domains
            explanation = f"""
            <h3>Multi-Domain Analysis</h3>
            <p>Your business challenge spans <strong>three</strong> key domains:</p>
            <ul>
                <li><strong>{domain_names[0]}</strong></li>
                <li><strong>{domain_names[1]}</strong></li>
                <li><strong>{domain_names[2]}</strong></li>
            </ul>
            <p><strong>Reasoning:</strong> {reasoning}</p>
            """
        
        if key_indicators:
            explanation += f'<p><strong>Key Indicators:</strong> {", ".join(key_indicators)}</p>'
        
        explanation += f'<p><strong>Analysis Confidence:</strong> {confidence}%</p><br>'
        
        return explanation

    def process_proposal(self, pdf_path: str) -> Dict:
        """Main method to process a proposal PDF and classify its domain(s)"""
        try:
            proposal_content = self.extract_proposal_content(pdf_path)
            
            # Validate extracted content
            if not proposal_content or len(proposal_content.strip()) < 10:
                logger.error(f"Proposal content is too short or empty: {len(proposal_content) if proposal_content else 0} characters")
                raise ValueError("Proposal PDF contains insufficient content for analysis")
            
            classification = self.classify_problem_domain(proposal_content)
            
            # If classification returned retry or invalid, use fallback
            if not classification or classification.get("primary_domain") == "retry":
                logger.warning("Classification returned retry, using enhanced fallback")
                classification = self._fallback_classification(proposal_content)
            
            # Ensure classification is normalized (but skip if retry)
            if classification.get("primary_domain") != "retry":
                normalized = self._normalize_classification(classification)
                # If normalization resulted in invalid domains, use fallback
                if normalized.get("primary_domain") not in self.domains:
                    logger.warning("Normalized classification has invalid domain, using fallback")
                    classification = self._fallback_classification(proposal_content)
                else:
                    classification = normalized
            
            explanation = self.explain_domain_to_user(classification, proposal_content)
            
            # Get domain names for all selected domains
            domains_list = classification.get("domains", [])
            if not domains_list or not isinstance(domains_list, list):
                domains_list = [classification.get("primary_domain", "unknown")]
            
            # Filter to valid domains only
            domains_list = [d for d in domains_list if d in self.domains]
            if not domains_list:
                domains_list = [classification.get("primary_domain", "unknown")]
            
            domain_names = [self.domains.get(d, d.replace("_", " ").title()) for d in domains_list]

            return {
                "classification": classification,
                "explanation": explanation,
                "domain_name": domain_names[0] if domain_names else "Business Strategy",  # Primary for backward compatibility
                "domain_names": domain_names  # All selected domains
            }

        except Exception as e:
            logger.error(f"Error processing proposal: {str(e)}", exc_info=True)
            raise Exception(f"Failed to process proposal: {str(e)}")

